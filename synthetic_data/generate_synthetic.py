"""
Synthetic CHF Weekly Dataset Generator
======================================

Builds one row per (subject_id, hadm_id) in patient_visit_codes.csv.
Each row represents one week of a CHF patient with twice-daily vitals
(HR, SBP/DBP, SpO2, weight) plus an ECG summary, current GDMT and
contraindication medications (with dose ratios), labs, and ground-truth
boolean flags for the adverse-effect classifier described in
synthetic_data_scheme.txt and overall_process.txt.

MIMIC-IV inputs:
  - hosp/patients.csv              -> gender, anchor_age
  - hosp/admissions.csv            -> admit time (for age at visit)
  - hosp/diagnoses_icd.csv         -> contraindication / comorbidity flags
  - hosp/prescriptions.csv         -> current GDMT + contraindication meds (streamed)
  - ecg/machine_measurements.csv   -> ECG summary per visit

Labs and vitals are synthesised so they are CONSISTENT with each row's
ground-truth adverse-effect flags. AE flags are sampled with target
prevalences so the dataset is balanced enough to train a classifier.
"""

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path("/Users/ramaseshu/Documents/CHF")
MIMIC = ROOT / "MIMIC-IV"
OUT_DIR = ROOT / "synthetic_data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VISITS_CSV = ROOT / "patient_visit_codes.csv"
PATIENTS_CSV = MIMIC / "hosp" / "patients.csv"
ADMISSIONS_CSV = MIMIC / "hosp" / "admissions.csv"
DIAG_CSV = MIMIC / "hosp" / "diagnoses_icd.csv"
PRESC_CSV = MIMIC / "hosp" / "prescriptions.csv"
ECG_CSV = MIMIC / "ecg" / "machine_measurements.csv"

OUT_CSV = OUT_DIR / "chf_weekly_synthetic.csv"
SUMMARY_TXT = OUT_DIR / "dataset_summary.txt"

RNG = np.random.default_rng(20260422)

# ---------------------------------------------------------------------------
# GDMT and contraindication drug dictionaries
# ---------------------------------------------------------------------------
GDMT_PATTERNS = {
    "ACEi": [
        "lisinopril", "enalapril", "captopril", "ramipril", "benazepril",
        "fosinopril", "quinapril", "perindopril", "trandolapril", "moexipril",
    ],
    "ARB": [
        "losartan", "valsartan", "irbesartan", "olmesartan", "candesartan",
        "telmisartan", "eprosartan", "azilsartan",
    ],
    "ARNi": ["sacubitril", "entresto"],
    "beta_blocker": [
        "carvedilol", "metoprolol succinate", "metoprolol tartrate", "metoprolol",
        "bisoprolol", "nebivolol",
    ],
    "MRA": ["spironolactone", "eplerenone", "finerenone"],
    "SGLT2i": [
        "dapagliflozin", "empagliflozin", "canagliflozin", "ertugliflozin",
        "sotagliflozin",
    ],
    "loop": ["furosemide", "bumetanide", "torsemide", "ethacrynic"],
}

# Drugs that are not GDMT but matter for AE / contraindication context
OTHER_PATTERNS = {
    "NSAID": [
        "ibuprofen", "naproxen", "ketorolac", "diclofenac", "celecoxib",
        "indomethacin", "meloxicam", "piroxicam",
    ],
    "potassium_supplement": ["potassium chloride", "potassium phosphate", "k-dur", "klor-con"],
    "other_diuretic": [
        "hydrochlorothiazide", "chlorthalidone", "metolazone",
        "indapamide", "amiloride", "triamterene",
    ],
}

# Approximate target daily doses (mg) for dose-ratio calculation
TARGET_DOSES_MG = {
    "lisinopril": 40, "enalapril": 40, "captopril": 150, "ramipril": 10,
    "benazepril": 40, "fosinopril": 40, "quinapril": 40, "perindopril": 16,
    "losartan": 150, "valsartan": 320, "candesartan": 32, "telmisartan": 80,
    "olmesartan": 40, "irbesartan": 300,
    "sacubitril": 400,  # sacubitril/valsartan target
    "carvedilol": 50, "metoprolol succinate": 200, "metoprolol tartrate": 200,
    "metoprolol": 200, "bisoprolol": 10, "nebivolol": 40,
    "spironolactone": 50, "eplerenone": 50, "finerenone": 40,
    "dapagliflozin": 10, "empagliflozin": 10, "canagliflozin": 100,
    "ertugliflozin": 15, "sotagliflozin": 200,
    "furosemide": 80, "bumetanide": 2, "torsemide": 40, "ethacrynic": 100,
}

DRUG_TO_CLASS: dict[str, str] = {}
for cls, names in GDMT_PATTERNS.items():
    for n in names:
        DRUG_TO_CLASS[n] = cls
for cls, names in OTHER_PATTERNS.items():
    for n in names:
        DRUG_TO_CLASS[n] = cls

ALL_DRUG_PATTERNS = sorted(DRUG_TO_CLASS.keys(), key=len, reverse=True)
DRUG_REGEX = re.compile("|".join(re.escape(p) for p in ALL_DRUG_PATTERNS), re.IGNORECASE)


def classify_drug(drug_name: str) -> tuple[str | None, str | None]:
    """Return (class, canonical pattern) for a drug string, else (None, None)."""
    if not isinstance(drug_name, str):
        return None, None
    m = DRUG_REGEX.search(drug_name.lower())
    if not m:
        return None, None
    pat = m.group(0).lower()
    return DRUG_TO_CLASS[pat], pat


# ---------------------------------------------------------------------------
# ICD-based contraindication / comorbidity dictionaries (substring match)
# ---------------------------------------------------------------------------
# Both ICD-9 and ICD-10 prefixes considered.
ICD_PATTERNS = {
    "angioedema_history": ["T783", "9953", "T7830", "T7831"],   # angioedema
    "pregnancy": ["O", "V22", "V23", "Z34", "Z32"],
    "bilateral_renal_artery_stenosis": ["I701", "44081", "Z9482"],
    "severe_asthma": ["J45", "493"],
    "av_block": ["I44", "4260", "4261", "4262", "4263"],
    "history_dka": ["E101", "E111", "E1310", "25010", "25011", "25012", "25013"],
    "type_1_diabetes": ["E10", "2500", "2501", "2503"],
    "chf": ["I50", "428"],
}


# ---------------------------------------------------------------------------
# Step 1: load visits
# ---------------------------------------------------------------------------
def load_visits() -> pd.DataFrame:
    print(f"[load_visits] reading {VISITS_CSV}")
    df = pd.read_csv(VISITS_CSV, usecols=["subject_id", "hadm_id"])
    df = df.dropna().drop_duplicates()
    df["subject_id"] = df["subject_id"].astype("int64")
    df["hadm_id"] = df["hadm_id"].astype("int64")
    print(f"[load_visits] {len(df):,} visits")
    return df


# ---------------------------------------------------------------------------
# Step 2: demographics (gender, age at admission)
# ---------------------------------------------------------------------------
def load_demographics(visits: pd.DataFrame) -> pd.DataFrame:
    print(f"[demographics] reading {PATIENTS_CSV}")
    pat = pd.read_csv(
        PATIENTS_CSV,
        usecols=["subject_id", "gender", "anchor_age", "anchor_year"],
        dtype={"subject_id": "int64", "anchor_age": "int16", "anchor_year": "int16"},
    )
    print(f"[demographics] reading admittime from {ADMISSIONS_CSV}")
    adm = pd.read_csv(
        ADMISSIONS_CSV,
        usecols=["subject_id", "hadm_id", "admittime"],
        parse_dates=["admittime"],
        dtype={"subject_id": "int64", "hadm_id": "int64"},
    )
    adm["admit_year"] = adm["admittime"].dt.year.astype("int16")
    df = visits.merge(pat, on="subject_id", how="left")
    df = df.merge(adm[["subject_id", "hadm_id", "admit_year"]],
                  on=["subject_id", "hadm_id"], how="left")
    # Age at this admission = anchor_age + (admit_year - anchor_year)
    df["age"] = (df["anchor_age"] + (df["admit_year"] - df["anchor_year"])).astype("Int16")
    df["gender"] = df["gender"].fillna("U")
    # Reasonable fallback for missing age
    df.loc[df["age"].isna(), "age"] = 65
    df["age"] = df["age"].astype("int16")
    df = df.drop(columns=["anchor_age", "anchor_year", "admit_year"])
    return df


# ---------------------------------------------------------------------------
# Step 3: diagnoses -> contraindication / comorbidity flags
# ---------------------------------------------------------------------------
def load_diagnoses(visits: pd.DataFrame) -> pd.DataFrame:
    print(f"[diagnoses] reading {DIAG_CSV}")
    diag = pd.read_csv(
        DIAG_CSV,
        usecols=["subject_id", "hadm_id", "icd_code"],
        dtype={"subject_id": "int64", "hadm_id": "int64", "icd_code": "string"},
    )
    diag["icd_code"] = diag["icd_code"].fillna("").str.strip().str.upper()

    # Build a long table of (subject_id, hadm_id, flag_name) for matched ICD codes.
    print("[diagnoses] matching ICD patterns")
    flag_cols = list(ICD_PATTERNS.keys())
    out = pd.DataFrame({
        "subject_id": visits["subject_id"].values,
        "hadm_id": visits["hadm_id"].values,
    })
    out_index = pd.MultiIndex.from_arrays([out["subject_id"], out["hadm_id"]])

    # For each flag, build a boolean indexed by (subject_id, hadm_id)
    for flag, prefixes in ICD_PATTERNS.items():
        regex = re.compile("^(" + "|".join(re.escape(p) for p in prefixes) + ")")
        mask = diag["icd_code"].str.match(regex, na=False)
        sub = diag.loc[mask, ["subject_id", "hadm_id"]].drop_duplicates()
        sub_idx = pd.MultiIndex.from_arrays([sub["subject_id"], sub["hadm_id"]])
        out[flag] = out_index.isin(sub_idx)
    return out


# ---------------------------------------------------------------------------
# Step 4: prescriptions -> per-visit medication state (streamed)
# ---------------------------------------------------------------------------
def load_prescriptions(visits: pd.DataFrame) -> pd.DataFrame:
    print(f"[prescriptions] streaming {PRESC_CSV}")
    visit_set = set(zip(visits["subject_id"].tolist(), visits["hadm_id"].tolist()))

    # Aggregator: (subject_id, hadm_id) -> {class: max_observed_dose_mg}
    agg: dict[tuple[int, int], dict[str, float]] = {}

    chunksize = 500_000
    cols = ["subject_id", "hadm_id", "drug", "dose_val_rx", "dose_unit_rx"]
    n_rows = 0
    n_kept = 0
    t0 = time.time()
    for chunk in pd.read_csv(
        PRESC_CSV,
        usecols=cols,
        dtype={"subject_id": "Int64", "hadm_id": "Int64",
               "drug": "string", "dose_val_rx": "string", "dose_unit_rx": "string"},
        chunksize=chunksize,
        low_memory=False,
    ):
        n_rows += len(chunk)
        chunk = chunk.dropna(subset=["subject_id", "hadm_id", "drug"])
        # Quick filter: keep only rows whose drug looks relevant
        chunk = chunk[chunk["drug"].str.contains(DRUG_REGEX, na=False)]
        if chunk.empty:
            if n_rows % (chunksize * 4) == 0:
                print(f"  [prescriptions] scanned {n_rows:,} rows, kept {n_kept:,} "
                      f"({time.time()-t0:.0f}s)")
            continue

        for sid, hid, drug, dose, unit in zip(
            chunk["subject_id"].astype("int64").values,
            chunk["hadm_id"].astype("int64").values,
            chunk["drug"].values,
            chunk["dose_val_rx"].values,
            chunk["dose_unit_rx"].values,
        ):
            if (sid, hid) not in visit_set:
                continue
            cls, pat = classify_drug(str(drug))
            if cls is None:
                continue
            try:
                dose_mg = float(dose) if dose is not None and dose == dose else 0.0
            except (ValueError, TypeError):
                dose_mg = 0.0
            # Only treat as mg if unit is mg or unspecified
            unit_l = (unit or "").lower() if isinstance(unit, str) else ""
            if unit_l and "mg" not in unit_l:
                dose_mg = 0.0
            d = agg.setdefault((sid, hid), {})
            d[cls] = max(d.get(cls, 0.0), dose_mg)
            # Track canonical pattern for target-dose lookup
            if cls in GDMT_PATTERNS:
                d.setdefault(f"_pat_{cls}", pat)
            n_kept += 1

        print(f"  [prescriptions] scanned {n_rows:,} rows, kept {n_kept:,} "
              f"({time.time()-t0:.0f}s)")

    print(f"[prescriptions] done. visits with any matched med: {len(agg):,}")

    # Build output dataframe
    rows = []
    gdmt_classes = list(GDMT_PATTERNS.keys())
    other_classes = list(OTHER_PATTERNS.keys())
    for (sid, hid), d in agg.items():
        row = {"subject_id": sid, "hadm_id": hid}
        for cls in gdmt_classes:
            on = cls in d
            row[f"on_{cls}"] = on
            if on:
                pat = d.get(f"_pat_{cls}")
                target = TARGET_DOSES_MG.get(pat, None)
                dose = d[cls]
                if target and dose > 0:
                    row[f"{cls}_dose_ratio"] = float(np.clip(dose / target, 0.05, 1.5))
                else:
                    row[f"{cls}_dose_ratio"] = 0.5  # unknown -> assume half-target
            else:
                row[f"{cls}_dose_ratio"] = 0.0
        for cls in other_classes:
            row[f"on_{cls}"] = cls in d
        rows.append(row)
    if not rows:
        rxs = pd.DataFrame(columns=["subject_id", "hadm_id"])
    else:
        rxs = pd.DataFrame(rows)
    # Left-merge onto visits with sensible defaults
    out = visits.merge(rxs, on=["subject_id", "hadm_id"], how="left")
    for cls in gdmt_classes:
        out[f"on_{cls}"] = out[f"on_{cls}"].fillna(False).astype(bool)
        out[f"{cls}_dose_ratio"] = out[f"{cls}_dose_ratio"].fillna(0.0).astype("float32")
    for cls in other_classes:
        out[f"on_{cls}"] = out[f"on_{cls}"].fillna(False).astype(bool)
    return out


# ---------------------------------------------------------------------------
# Step 5: ECG (one summary per visit)
# ---------------------------------------------------------------------------
ECG_RHYTHM_KEYWORDS = [
    "atrial fibrillation", "atrial flutter", "sinus rhythm",
    "sinus tachycardia", "sinus bradycardia", "paced rhythm",
    "junctional", "ventricular tachycardia",
]


def load_ecg(visits: pd.DataFrame) -> pd.DataFrame:
    print(f"[ecg] reading {ECG_CSV}")
    ecg = pd.read_csv(
        ECG_CSV,
        usecols=[
            "subject_id", "ecg_time",
            "report_0", "report_1", "report_2",
            "rr_interval", "qrs_onset", "qrs_end", "t_end",
            "p_axis", "qrs_axis", "t_axis",
        ],
        parse_dates=["ecg_time"],
        dtype={"subject_id": "int64"},
    )
    # Only keep ECGs for our subjects
    subj = set(visits["subject_id"].unique().tolist())
    ecg = ecg[ecg["subject_id"].isin(subj)]
    print(f"[ecg] {len(ecg):,} ECGs for our cohort")
    # Take the latest ECG per subject
    ecg = ecg.sort_values("ecg_time").drop_duplicates("subject_id", keep="last")
    ecg["qrs_duration"] = (ecg["qrs_end"] - ecg["qrs_onset"]).astype("float32")
    ecg["qt_interval"] = (ecg["t_end"] - ecg["qrs_onset"]).astype("float32")

    # Rhythm + abnormality keywords
    rep = (ecg["report_0"].fillna("") + " | " +
           ecg["report_1"].fillna("") + " | " +
           ecg["report_2"].fillna("")).str.lower()

    def first_match(s):
        for kw in ECG_RHYTHM_KEYWORDS:
            if kw in s:
                return kw
        return "other"

    ecg["ecg_rhythm"] = rep.map(first_match)
    ecg["ecg_afib"] = rep.str.contains("atrial fibrillation|atrial flutter", regex=True)
    ecg["ecg_lbbb"] = rep.str.contains("left bundle branch block")
    ecg["ecg_rbbb"] = rep.str.contains("right bundle branch block")
    ecg["ecg_lvh"] = rep.str.contains("left ventricular hypertrophy|lvh")
    ecg["ecg_st_changes"] = rep.str.contains("st elevation|st depression|ischemia")

    keep = ["subject_id", "ecg_rhythm", "rr_interval", "qrs_duration",
            "qt_interval", "p_axis", "qrs_axis", "t_axis",
            "ecg_afib", "ecg_lbbb", "ecg_rbbb", "ecg_lvh", "ecg_st_changes"]
    ecg = ecg[keep]
    out = visits.merge(ecg, on="subject_id", how="left")
    # Fill defaults for visits with no ECG
    out["ecg_rhythm"] = out["ecg_rhythm"].fillna("unknown")
    for c in ["rr_interval", "qrs_duration", "qt_interval", "p_axis", "qrs_axis", "t_axis"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float32")
    for b in ["ecg_afib", "ecg_lbbb", "ecg_rbbb", "ecg_lvh", "ecg_st_changes"]:
        out[b] = out[b].fillna(False).astype(bool)
    return out


# ---------------------------------------------------------------------------
# Step 6: assign ground-truth AE flags with class balance, then synthesize
#         consistent labs and twice-daily vitals.
# ---------------------------------------------------------------------------

# Target prevalences for adverse-effect flags. These keep enough positive
# examples for each class without making the dataset wildly unrealistic.
PREVALENCE = {
    "hypotension_detected": 0.18,
    "severe_hypotension_detected": 0.05,
    "bradycardia_detected": 0.12,
    "severe_bradycardia_detected": 0.03,
    "volume_depletion_detected": 0.13,
    "worsening_HF_detected": 0.15,
    "hyperkalemia_detected": 0.10,
    "renal_dysfunction_detected": 0.14,
    "hyponatremia_detected": 0.08,
    "metabolic_acidosis_detected": 0.06,
}

VITAL_TIMEPOINTS = 14  # twice daily for 7 days


def synthesize(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    print(f"[synthesize] generating AE flags + vitals + labs for {n:,} rows")

    # ---------- Ground-truth AE flags ----------
    flags = {k: RNG.random(n) < p for k, p in PREVALENCE.items()}
    # Severe implies non-severe
    flags["hypotension_detected"] |= flags["severe_hypotension_detected"]
    flags["bradycardia_detected"] |= flags["severe_bradycardia_detected"]
    flags["any_emergency_flag"] = (
        flags["severe_hypotension_detected"]
        | flags["severe_bradycardia_detected"]
        | flags["hyperkalemia_detected"] & (RNG.random(n) < 0.4)
        | flags["renal_dysfunction_detected"] & (RNG.random(n) < 0.3)
    )

    # ---------- Baseline vitals ----------
    base_hr = RNG.normal(72, 10, n).astype("float32")
    base_sbp = RNG.normal(120, 14, n).astype("float32")
    base_dbp = RNG.normal(74, 9, n).astype("float32")
    base_spo2 = RNG.normal(97, 1.2, n).astype("float32")
    base_weight = RNG.normal(82, 16, n).astype("float32")  # kg

    # Push baselines based on ground-truth flags
    bradi = flags["bradycardia_detected"]
    sbradi = flags["severe_bradycardia_detected"]
    hypo = flags["hypotension_detected"]
    shypo = flags["severe_hypotension_detected"]
    voldep = flags["volume_depletion_detected"]
    whf = flags["worsening_HF_detected"]

    base_hr[bradi] = RNG.normal(48, 4, bradi.sum())
    base_hr[sbradi] = RNG.normal(38, 3, sbradi.sum())
    base_sbp[hypo] = RNG.normal(86, 4, hypo.sum())
    base_dbp[hypo] = RNG.normal(56, 4, hypo.sum())
    base_sbp[shypo] = RNG.normal(74, 4, shypo.sum())
    base_dbp[shypo] = RNG.normal(46, 4, shypo.sum())
    # Worsening HF -> higher HR, slightly lower SpO2
    base_hr[whf] = RNG.normal(98, 8, whf.sum())
    base_spo2[whf] = RNG.normal(93, 1.5, whf.sum())

    # ---------- Per-timepoint vitals ----------
    T = VITAL_TIMEPOINTS
    out = {}
    # Drift over the week: weight rises for worsening HF, falls for volume depletion
    week_idx = np.arange(T, dtype="float32") / (T - 1)  # 0..1
    weight_drift = np.zeros((n, T), dtype="float32")
    weight_drift[whf] = np.outer(np.full(whf.sum(), 3.0, dtype="float32"), week_idx)
    weight_drift[voldep] = np.outer(np.full(voldep.sum(), -3.5, dtype="float32"), week_idx)

    hr_noise = RNG.normal(0, 4, (n, T)).astype("float32")
    sbp_noise = RNG.normal(0, 5, (n, T)).astype("float32")
    dbp_noise = RNG.normal(0, 4, (n, T)).astype("float32")
    spo2_noise = RNG.normal(0, 0.8, (n, T)).astype("float32")
    w_noise = RNG.normal(0, 0.4, (n, T)).astype("float32")

    HR = np.clip(base_hr[:, None] + hr_noise, 25, 180).astype("float32")
    SBP = np.clip(base_sbp[:, None] + sbp_noise, 50, 220).astype("float32")
    DBP = np.clip(base_dbp[:, None] + dbp_noise, 30, 130).astype("float32")
    SPO2 = np.clip(base_spo2[:, None] + spo2_noise, 70, 100).astype("float32")
    WT = np.clip(base_weight[:, None] + weight_drift + w_noise, 35, 220).astype("float32")

    for t in range(T):
        out[f"HR_t{t+1}"] = HR[:, t]
        out[f"SBP_t{t+1}"] = SBP[:, t]
        out[f"DBP_t{t+1}"] = DBP[:, t]
        out[f"SPO2_t{t+1}"] = SPO2[:, t]
        out[f"Weight_t{t+1}"] = WT[:, t]

    # Aggregate vitals (handy for classifier baselines)
    out["HR_mean"] = HR.mean(axis=1)
    out["HR_min"] = HR.min(axis=1)
    out["SBP_mean"] = SBP.mean(axis=1)
    out["SBP_min"] = SBP.min(axis=1)
    out["DBP_mean"] = DBP.mean(axis=1)
    out["SPO2_mean"] = SPO2.mean(axis=1)
    out["Weight_change"] = WT[:, -1] - WT[:, 0]

    # ---------- Labs (consistent with AE flags) ----------
    K = RNG.normal(4.2, 0.4, n).astype("float32")
    K[flags["hyperkalemia_detected"]] = RNG.normal(5.9, 0.3, flags["hyperkalemia_detected"].sum())
    Na = RNG.normal(139, 2.5, n).astype("float32")
    Na[flags["hyponatremia_detected"]] = RNG.normal(131, 2.0, flags["hyponatremia_detected"].sum())
    Cr = RNG.normal(1.0, 0.25, n).astype("float32")
    Cr[flags["renal_dysfunction_detected"]] = RNG.normal(2.1, 0.4, flags["renal_dysfunction_detected"].sum())
    eGFR = np.clip(120 - 50 * (Cr - 1.0) + RNG.normal(0, 8, n), 5, 120).astype("float32")
    HCO3 = RNG.normal(25, 2.0, n).astype("float32")
    HCO3[flags["metabolic_acidosis_detected"]] = RNG.normal(18, 1.5, flags["metabolic_acidosis_detected"].sum())
    base_Cr = (Cr - RNG.normal(0.3, 0.1, n)).clip(0.5).astype("float32")

    out["potassium"] = K
    out["sodium"] = Na
    out["creatinine"] = Cr
    out["egfr"] = eGFR
    out["bicarbonate"] = HCO3
    out["baseline_creatinine"] = base_Cr
    out["creatinine_pct_increase"] = ((Cr - base_Cr) / base_Cr * 100).astype("float32")

    # ---------- Baseline-condition contraindication booleans ----------
    out["baseline_potassium_more_than_5_5"] = (K > 5.5)
    out["sodium_less_than_130"] = (Na < 130)
    out["potassium_less_than_3_0"] = (K < 3.0)
    out["egfr_less_than_20"] = (eGFR < 20)
    out["egfr_less_than_30"] = (eGFR < 30)
    out["baseline_hr_less_than_50"] = (HR.min(axis=1) < 50)
    # severe baseline volume depletion: weight loss + low SBP
    out["severe_baseline_volume_depletion"] = (
        (WT[:, -1] - WT[:, 0] < -2.5) & (SBP.mean(axis=1) < 95)
    )

    # ---------- Ground-truth AE flag columns ----------
    for k in [
        "severe_hypotension_detected", "severe_bradycardia_detected",
        "any_emergency_flag", "hypotension_detected", "bradycardia_detected",
        "volume_depletion_detected", "worsening_HF_detected",
        "hyperkalemia_detected", "renal_dysfunction_detected",
        "hyponatremia_detected", "metabolic_acidosis_detected",
    ]:
        out[k] = flags[k]

    syn = pd.DataFrame(out)
    return pd.concat([df.reset_index(drop=True), syn.reset_index(drop=True)], axis=1)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    visits = load_visits()
    demo = load_demographics(visits)
    diag = load_diagnoses(visits)
    rxs = load_prescriptions(visits)
    ecg = load_ecg(visits)

    print("[merge] joining demographics + diagnoses + prescriptions + ECG")
    df = (demo
          .merge(diag, on=["subject_id", "hadm_id"], how="left")
          .merge(rxs, on=["subject_id", "hadm_id"], how="left")
          .merge(ecg, on=["subject_id", "hadm_id"], how="left"))

    # If a visit row was duplicated by merges (shouldn't happen but be safe)
    df = df.drop_duplicates(subset=["subject_id", "hadm_id"], keep="first")
    df = df.reset_index(drop=True)
    assert len(df) == len(visits), f"row count drifted: {len(df)} vs {len(visits)}"

    # Fill any leftover NaNs in flag columns produced by left-merges
    for col in list(ICD_PATTERNS.keys()):
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(bool)

    df = synthesize(df)

    # Re-order columns for readability
    front = [
        "subject_id", "hadm_id", "gender", "age",
    ]
    other_cols = [c for c in df.columns if c not in front]
    df = df[front + other_cols]

    print(f"[write] {OUT_CSV}  ({len(df):,} rows, {len(df.columns)} cols)")
    df.to_csv(OUT_CSV, index=False)

    # ---------- Summary ----------
    label_cols = [
        "severe_hypotension_detected", "severe_bradycardia_detected",
        "any_emergency_flag", "hypotension_detected", "bradycardia_detected",
        "volume_depletion_detected", "worsening_HF_detected",
        "hyperkalemia_detected", "renal_dysfunction_detected",
        "hyponatremia_detected", "metabolic_acidosis_detected",
    ]
    med_cols = [c for c in df.columns if c.startswith("on_")]
    summary_lines = [
        f"rows: {len(df):,}",
        f"columns: {len(df.columns)}",
        f"runtime_seconds: {time.time()-t0:.1f}",
        "",
        "label prevalence:",
    ]
    for c in label_cols:
        summary_lines.append(f"  {c}: {df[c].mean():.4f}  ({int(df[c].sum()):,})")
    summary_lines.append("")
    summary_lines.append("medication usage:")
    for c in med_cols:
        summary_lines.append(f"  {c}: {df[c].mean():.4f}  ({int(df[c].sum()):,})")
    SUMMARY_TXT.write_text("\n".join(summary_lines))
    print("[done]")
    print("\n".join(summary_lines[:25]))


if __name__ == "__main__":
    main()
