"""Generate synthetic patient-weeks for training the classifier.

Two modes:
  - "distribution" : fully synthetic, no external data. Demographics and baseline
                     labs are drawn from Gaussians. Portable, 10 KB zero-dep.
  - "mimic"        : seeded from data/synthetic_CHF_visits_v2.parquet — real MIMIC
                     patient demographics + baseline labs + GDMT regimens + ECG
                     IDs. More realistic distributions / correlations, but
                     requires the 135 MB parquet to be present.

Mode is auto-selected based on parquet availability unless explicitly set.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import CLASSES_ALL, LABEL_COLS, CLASS_TO_COLKEY, DEFAULT_BASELINE_LABS
from .synthesize import synthesize_week
from .strategy import TARGETS


P_PRIMARY = dict(
    hypotension_detected=0.5, bradycardia_detected=0.5,
    volume_depletion_detected=0.5, worsening_HF_detected=0.5,
    hyperkalemia_detected=0.5, renal_dysfunction_detected=0.5,
    hyponatremia_detected=0.5, metabolic_acidosis_detected=0.5,
)
P_SEVERE_HYPO_GIVEN = 0.4
P_SEVERE_BRADY_GIVEN = 0.4

DEFAULT_MIMIC_PARQUET = Path(__file__).parent.parent.parent / "data" / "synthetic_CHF_visits_v2.parquet"

ITEMID_TO_KEY = {
    50971: "K", 50822: "K",
    50983: "Na", 50824: "Na",
    50912: "Cr", 50920: "eGFR", 51006: "BUN",
    50882: "HCO3", 50803: "HCO3",
    50820: "pH", 50818: "pCO2", 50821: "pO2",
    50931: "glucose", 50809: "glucose",
    50960: "Mg", 50970: "phosphate", 50993: "TSH",
}


def _sample_labels(rng: np.random.Generator) -> dict:
    labels = {k: bool(rng.random() < p) for k, p in P_PRIMARY.items()}
    labels["severe_hypotension_detected"] = (
        labels["hypotension_detected"] and rng.random() < P_SEVERE_HYPO_GIVEN
    )
    labels["severe_bradycardia_detected"] = (
        labels["bradycardia_detected"] and rng.random() < P_SEVERE_BRADY_GIVEN
    )
    labels["any_emergency_flag"] = bool(
        labels["severe_hypotension_detected"] or labels["severe_bradycardia_detected"]
    )
    return labels


def _vitals_for_labels(labels: dict, rng: np.random.Generator) -> dict:
    """Pick target vital stats that encode the sampled flags with realistic severity."""
    hr_mean = 72.0; hr_min = 62.0; sbp_mean = 128.0; sbp_min = 115.0
    spo2_min = 97.0; wt_delta = 0.0; qrs_max = 95.0; qtc_max = 415.0
    t_peaked = False; rhythm = "NSR"

    if labels["hypotension_detected"]:
        sbp_mean -= rng.uniform(12, 22); sbp_min -= rng.uniform(15, 30)
    if labels["severe_hypotension_detected"]:
        sbp_min = min(sbp_min, rng.uniform(65, 79))
    if labels["bradycardia_detected"]:
        hr_mean -= rng.uniform(8, 15); hr_min -= rng.uniform(12, 22)
    if labels["severe_bradycardia_detected"]:
        hr_min = min(hr_min, rng.uniform(30, 44))
    if labels["volume_depletion_detected"]:
        wt_delta -= rng.uniform(1.8, 3.5); hr_mean += rng.uniform(6, 14)
    if labels["worsening_HF_detected"]:
        wt_delta += rng.uniform(2.0, 4.0); spo2_min = min(spo2_min, rng.uniform(88, 93))
        hr_mean += rng.uniform(4, 10)
    if labels["hyperkalemia_detected"]:
        t_peaked = True; qrs_max = max(qrs_max, rng.uniform(130, 160))
        if rng.random() < 0.35:
            rhythm = "wide_complex"
    if labels["hyponatremia_detected"]:
        qtc_max = max(qtc_max, rng.uniform(460, 500))
    if labels["metabolic_acidosis_detected"]:
        hr_mean += rng.uniform(3, 8)
    if labels["renal_dysfunction_detected"]:
        wt_delta += rng.uniform(0.3, 1.5); sbp_mean += rng.uniform(3, 7)
    return dict(
        HR_mean=hr_mean, HR_min=hr_min,
        SBP_mean=sbp_mean, SBP_min=sbp_min,
        SpO2_min=spo2_min, weight_delta=wt_delta,
        QRS_max=qrs_max, QTc_max=qtc_max,
        T_peaked=t_peaked, rhythm=rhythm,
    )


def _as_list(v):
    if v is None: return []
    if hasattr(v, "__len__"):
        try: return list(v) if len(v) > 0 else []
        except Exception: return []
    return []


def _extract_baseline_labs(labs_list) -> dict:
    """Canonical baseline dict from MIMIC labs_2d."""
    out = dict(DEFAULT_BASELINE_LABS)
    recs = _as_list(labs_list)
    if not recs: return out
    buckets: dict[str, list[float]] = {}
    for rec in recs:
        if not isinstance(rec, dict): continue
        iid = rec.get("itemid")
        val = rec.get("valuenum", rec.get("value"))
        try:
            if val is None or pd.isna(val): continue
        except Exception: pass
        try:
            iid = int(iid) if iid is not None else None
        except Exception: continue
        key = ITEMID_TO_KEY.get(iid) if iid is not None else None
        if key is None: continue
        try: buckets.setdefault(key, []).append(float(val))
        except Exception: continue
    for k, vs in buckets.items():
        if vs: out[k] = float(np.median(vs))
    return out


def _med_state_from_gdmt_list(gdmt_list) -> dict:
    """Per-class dose fields from nested gdmt_prescriptions list.
    Picks the record with the highest dose_ratio per class."""
    per_class: dict = {}
    for rec in _as_list(gdmt_list):
        if not isinstance(rec, dict): continue
        cls = str(rec.get("class", "")).strip()
        key = CLASS_TO_COLKEY.get(cls)
        if not key: continue
        try:
            ratio = float(rec.get("dose_ratio")) if rec.get("dose_ratio") is not None else None
        except Exception: ratio = None
        existing = per_class.get(key)
        if existing is None or (ratio is not None and ratio > (existing.get("ratio") or -1)):
            per_class[key] = {
                "ratio": ratio,
                "actual": float(rec["dose_val_rx"]) if rec.get("dose_val_rx") not in (None, "") else None,
                "target": float(rec["target_dose_mg"]) if rec.get("target_dose_mg") not in (None, "") else None,
            }
    return per_class


def _build_row_common(
    age: float, gender: str, baseline: dict,
    labels: dict, vitals: dict, med_per_class: dict,
    meta: dict, rng: np.random.Generator,
) -> dict:
    fake_patient = dict(age=age, gender=gender, meds={}, vitals=vitals, baseline=baseline)
    tps = synthesize_week(fake_patient, seed=int(rng.integers(0, 10**9)))
    row = dict(
        subject_id=meta.get("subject_id", 0),
        hadm_id=meta.get("hadm_id", 0),
        gender=gender, anchor_age=age,
        baseline_labs=baseline,
        # Match baseline_labs keys (with None values) so pyarrow can infer a concrete
        # struct<K:double,Na:double,Cr:double,eGFR:double,HCO3:double> schema instead of
        # an empty struct (which parquet rejects). Downstream code treats empty/None
        # entries as "no confirmation lab available".
        confirmation_labs={k: None for k in baseline},
        timepoints=tps,
    )
    for cls in CLASSES_ALL:
        mc = med_per_class.get(cls)
        if mc is not None and mc.get("actual") is not None:
            row[f"on_{cls}"] = True
            row[f"actual_dose_mg_{cls}"] = float(mc["actual"])
            row[f"target_dose_mg_{cls}"] = float(mc.get("target") or TARGETS.get(cls) or 0.0)
            tgt = row[f"target_dose_mg_{cls}"]
            row[f"dose_ratio_{cls}"] = (row[f"actual_dose_mg_{cls}"] / tgt) if tgt else 0.0
        else:
            row[f"on_{cls}"] = False
            row[f"actual_dose_mg_{cls}"] = 0.0
            row[f"target_dose_mg_{cls}"] = float(TARGETS.get(cls) or 0.0)
            row[f"dose_ratio_{cls}"] = 0.0
    for k, v in labels.items():
        row[k] = bool(v)
    return row


def generate_distribution(n: int = 10000, seed: int = 42) -> pd.DataFrame:
    """Fully synthetic, no external data. Demographics and baseline labs from Gaussians."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        age = float(rng.integers(45, 85))
        gender = "M" if rng.random() < 0.5 else "F"
        baseline = dict(
            K=float(np.clip(rng.normal(4.3, 0.4), 3.3, 5.2)),
            Na=float(np.clip(rng.normal(140, 2.5), 132, 148)),
            Cr=float(np.clip(rng.normal(1.1, 0.25), 0.6, 2.0)),
            eGFR=float(np.clip(rng.normal(68, 18), 25, 110)),
            HCO3=float(np.clip(rng.normal(24, 1.5), 20, 28)),
        )
        labels = _sample_labels(rng)
        vitals = _vitals_for_labels(labels, rng)
        med_per_class = {}
        for cls in CLASSES_ALL:
            if rng.random() < 0.35:
                dose = float(rng.choice([5, 10, 20, 25, 40, 80]).item())
                med_per_class[cls] = {
                    "actual": dose,
                    "target": float(TARGETS.get(cls) or 40.0),
                    "ratio": dose / max(1e-9, float(TARGETS.get(cls) or 40.0)),
                }
        meta = dict(subject_id=10_000_000 + i, hadm_id=i)
        rows.append(_build_row_common(age, gender, baseline, labels, vitals, med_per_class, meta, rng))
    return pd.DataFrame(rows)


def generate_from_mimic(
    source_parquet: Path | str = DEFAULT_MIMIC_PARQUET,
    n: int = 10000, seed: int = 42,
) -> pd.DataFrame:
    """Seed synthetic weeks from real MIMIC patient contexts.

    Demographics, baseline labs, and per-class dose patterns come from the
    sampled MIMIC visit. Vital trajectories + labels remain synthetic to
    preserve per-label balance.
    """
    source_parquet = Path(source_parquet)
    if not source_parquet.exists():
        raise FileNotFoundError(
            f"MIMIC seed parquet not found at {source_parquet!s}. "
            f"Use mode='distribution' or place synthetic_CHF_visits_v2.parquet there."
        )
    src = pd.read_parquet(source_parquet)
    rng = np.random.default_rng(seed)
    idxs = rng.integers(0, len(src), size=n)

    rows = []
    for i, sidx in enumerate(idxs):
        sr = src.iloc[int(sidx)]
        age = float(sr.get("anchor_age") or 65)
        gender = str(sr.get("gender") or "M")
        if gender not in ("M", "F"):
            gender = "M" if rng.random() < 0.5 else "F"
        baseline = _extract_baseline_labs(sr.get("labs_2d"))
        labels = _sample_labels(rng)
        vitals = _vitals_for_labels(labels, rng)
        med_per_class = _med_state_from_gdmt_list(sr.get("gdmt_prescriptions"))
        meta = dict(
            subject_id=int(sr["subject_id"]) if "subject_id" in sr else 10_000_000 + i,
            hadm_id=int(sr["hadm_id"]) if "hadm_id" in sr else i,
        )
        rows.append(_build_row_common(age, gender, baseline, labels, vitals, med_per_class, meta, rng))
    return pd.DataFrame(rows)


def generate_patient_weeks(
    n: int = 10000, seed: int = 42,
    mode: str = "auto", source_parquet: Path | str | None = None,
) -> pd.DataFrame:
    """Unified entry point. mode ∈ {"auto", "distribution", "mimic"}.

    "auto" picks "mimic" if DEFAULT_MIMIC_PARQUET exists, else "distribution".
    """
    source = Path(source_parquet) if source_parquet else DEFAULT_MIMIC_PARQUET
    if mode == "auto":
        mode = "mimic" if source.exists() else "distribution"
    if mode == "mimic":
        return generate_from_mimic(source, n=n, seed=seed)
    if mode == "distribution":
        return generate_distribution(n=n, seed=seed)
    raise ValueError(f"unknown mode={mode!r}; options: auto, distribution, mimic")


def generate_and_save(
    n: int, out_path: str | Path, seed: int = 42,
    mode: str = "auto", source_parquet: Path | str | None = None,
) -> None:
    print(f"[data_gen] mode={mode} n={n:,}")
    df = generate_patient_weeks(n=n, seed=seed, mode=mode, source_parquet=source_parquet)
    final = Path(out_path)
    final.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temp file first, then os.replace → atomic on both POSIX and Windows.
    # If to_parquet crashes mid-write, the final path stays untouched (no 0-byte file
    # left behind for downstream readers to choke on).
    tmp = final.with_suffix(final.suffix + ".tmp")
    if tmp.exists():
        try: tmp.unlink()
        except Exception: pass
    try:
        df.to_parquet(tmp, index=False)
    except Exception:
        if tmp.exists():
            try: tmp.unlink()
            except Exception: pass
        raise
    os.replace(tmp, final)
    print(f"[data_gen] saved {len(df):,} patient-weeks → {final}")
    print("[data_gen]   label rates:")
    for c in LABEL_COLS:
        if c in df.columns:
            print(f"             {c:<34}  {df[c].mean() * 100:5.2f}%")
