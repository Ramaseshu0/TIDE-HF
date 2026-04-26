"""Aggregate 14 timepoints + patient context into the 108-feature vector the classifier expects."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import VITAL_KEYS, ECG_SCALAR_KEYS, RHYTHMS, CLASSES_ALL


def _arr(tps, key):
    return np.array([t.get(key) for t in tps if t.get(key) is not None], dtype=float)


def _ecg_arr(tps, key):
    return np.array(
        [t["ecg"].get(key) for t in tps if t.get("ecg") and t["ecg"].get(key) is not None],
        dtype=float,
    )


def featurize_week(tps: list[dict]) -> dict:
    """14-timepoint list → 95-key aggregated feature dict (before demographics + med state)."""
    feats: dict = {}
    for k in VITAL_KEYS:
        a = _arr(tps, k)
        if len(a):
            feats[f"{k}_mean"] = float(np.nanmean(a))
            feats[f"{k}_min"] = float(np.nanmin(a))
            feats[f"{k}_max"] = float(np.nanmax(a))
            feats[f"{k}_range"] = float(np.nanmax(a) - np.nanmin(a))
            feats[f"{k}_first"] = float(a[0])
            feats[f"{k}_last"] = float(a[-1])
            feats[f"{k}_slope"] = (
                float(np.polyfit(np.arange(len(a)), a, 1)[0]) if len(a) >= 2 else 0.0
            )
        else:
            for s in ("mean", "min", "max", "range", "first", "last", "slope"):
                feats[f"{k}_{s}"] = np.nan
        feats[f"{k}_n_missing"] = int(sum(1 for t in tps if t.get(k) is None))

    for k in ECG_SCALAR_KEYS:
        a = _ecg_arr(tps, k)
        if len(a):
            feats[f"ecg_{k}_mean"] = float(np.nanmean(a))
            feats[f"ecg_{k}_min"] = float(np.nanmin(a))
            feats[f"ecg_{k}_max"] = float(np.nanmax(a))
        else:
            feats[f"ecg_{k}_mean"] = feats[f"ecg_{k}_min"] = feats[f"ecg_{k}_max"] = np.nan

    t_peaks = [bool(t.get("ecg", {}).get("t_peaked")) for t in tps]
    feats["ecg_t_peaked_any"] = int(any(t_peaks))
    feats["ecg_t_peaked_frac"] = float(np.mean(t_peaks)) if t_peaks else 0.0

    rhy = [t.get("ecg", {}).get("rhythm", "NSR") for t in tps]
    for r in RHYTHMS:
        feats[f"rhythm_has_{r}"] = int(r in rhy)
    feats["rhythm_frac_nonNSR"] = float(
        sum(1 for x in rhy if x != "NSR") / max(1, len(rhy))
    )

    qual = [t.get("ecg", {}).get("quality", "good") for t in tps]
    feats["ecg_frac_good"] = float(sum(1 for q in qual if q == "good") / max(1, len(qual)))
    feats["ecg_any_poor"] = int("poor" in qual)
    return feats


def build_feature_row(
    patient: dict,
    tps: list[dict],
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Assemble a single-row DataFrame in the column order the classifier expects."""
    feats = featurize_week(tps)
    feats["gender_M"] = 1 if str(patient.get("gender")) == "M" else 0
    feats["anchor_age"] = float(patient.get("age") or patient.get("anchor_age") or 65)

    for cls in CLASSES_ALL:
        m = patient.get("meds", {}).get(cls, {"on": False, "dose": 0})
        feats[f"on_{cls}"] = int(bool(m.get("on")))
        if m.get("on") and m.get("dose"):
            target = patient.get("targets", {}).get(cls, 0) or 0
            feats[f"dose_ratio_{cls}"] = (m["dose"] / target) if target else 0.0
            feats[f"target_dose_mg_{cls}"] = float(target) if target else 0.0
            feats[f"actual_dose_mg_{cls}"] = float(m["dose"])
        else:
            feats[f"dose_ratio_{cls}"] = 0.0
            feats[f"target_dose_mg_{cls}"] = 0.0
            feats[f"actual_dose_mg_{cls}"] = 0.0

    if feature_cols is None:
        return pd.DataFrame([feats])
    return pd.DataFrame([feats]).reindex(columns=feature_cols, fill_value=0)
