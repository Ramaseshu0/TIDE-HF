"""
Polish the synthetic CHF weekly dataset:
  1. Fix lab clipping bugs (creatinine could go negative, baseline_creatinine too)
  2. Impute ECG numeric nulls (rr_interval, qrs_duration, qt_interval,
     p_axis, qrs_axis, t_axis) with rhythm-conditional synthetic values
  3. Round vitals / labs to clinically realistic precision
  4. Tighten dtypes (bools as bool, ints as int, floats as float32)
  5. Re-emit dataset summary

Reads:  synthetic_data/chf_weekly_synthetic.csv
Writes: synthetic_data/chf_weekly_synthetic_clean.csv
        synthetic_data/dataset_summary.txt   (overwritten)
"""

from __future__ import annotations
from pathlib import Path
import time

import numpy as np
import pandas as pd

ROOT = Path("/Users/ramaseshu/Documents/CHF")
SRC = ROOT / "synthetic_data" / "chf_weekly_synthetic.csv"
DST = ROOT / "synthetic_data" / "chf_weekly_synthetic_clean.csv"
SUMMARY = ROOT / "synthetic_data" / "dataset_summary.txt"

RNG = np.random.default_rng(20260423)


# Rhythm-conditional ECG distributions (means, sds in ms / degrees).
# Used to impute the 25,715 ECG-less rows so every column is populated.
RHYTHM_PROFILES = {
    "sinus rhythm":            dict(rr=850, rr_sd=110, qrs=92,  qrs_sd=12, qt=400, qt_sd=30,
                                    p_axis=55, qrs_axis=40, t_axis=40, axis_sd=25),
    "sinus tachycardia":       dict(rr=560, rr_sd=70,  qrs=90,  qrs_sd=12, qt=350, qt_sd=25,
                                    p_axis=60, qrs_axis=45, t_axis=45, axis_sd=25),
    "sinus bradycardia":       dict(rr=1180,rr_sd=150, qrs=94,  qrs_sd=12, qt=440, qt_sd=30,
                                    p_axis=55, qrs_axis=40, t_axis=40, axis_sd=25),
    "atrial fibrillation":     dict(rr=720, rr_sd=180, qrs=96,  qrs_sd=15, qt=390, qt_sd=35,
                                    p_axis=np.nan, qrs_axis=30, t_axis=30, axis_sd=35),
    "atrial flutter":          dict(rr=600, rr_sd=120, qrs=94,  qrs_sd=14, qt=360, qt_sd=30,
                                    p_axis=np.nan, qrs_axis=35, t_axis=35, axis_sd=30),
    "paced rhythm":            dict(rr=830, rr_sd=80,  qrs=140, qrs_sd=20, qt=440, qt_sd=35,
                                    p_axis=np.nan, qrs_axis=-30,t_axis=-20, axis_sd=40),
    "junctional":              dict(rr=900, rr_sd=140, qrs=100, qrs_sd=15, qt=410, qt_sd=30,
                                    p_axis=np.nan, qrs_axis=20, t_axis=20, axis_sd=30),
    "ventricular tachycardia": dict(rr=350, rr_sd=60,  qrs=160, qrs_sd=25, qt=320, qt_sd=40,
                                    p_axis=np.nan, qrs_axis=-60,t_axis=120, axis_sd=40),
    "other":                   dict(rr=830, rr_sd=140, qrs=100, qrs_sd=18, qt=400, qt_sd=35,
                                    p_axis=50, qrs_axis=35, t_axis=35, axis_sd=30),
    "unknown":                 dict(rr=830, rr_sd=140, qrs=98,  qrs_sd=15, qt=400, qt_sd=32,
                                    p_axis=55, qrs_axis=40, t_axis=40, axis_sd=28),
}


def impute_ecg(df: pd.DataFrame) -> None:
    """Fill ECG numeric nulls in-place using rhythm-conditional draws."""
    cols = ["rr_interval", "qrs_duration", "qt_interval",
            "p_axis", "qrs_axis", "t_axis"]
    null_mask = df["rr_interval"].isna()
    n_null = int(null_mask.sum())
    print(f"[ecg-impute] filling {n_null:,} rows with rhythm-conditional values")
    if n_null == 0:
        return

    # Process per rhythm so distributions stay realistic
    for rhythm, profile in RHYTHM_PROFILES.items():
        m = null_mask & (df["ecg_rhythm"] == rhythm)
        k = int(m.sum())
        if k == 0:
            continue
        df.loc[m, "rr_interval"]  = RNG.normal(profile["rr"],  profile["rr_sd"],  k).clip(250, 2000)
        df.loc[m, "qrs_duration"] = RNG.normal(profile["qrs"], profile["qrs_sd"], k).clip(60, 220)
        df.loc[m, "qt_interval"]  = RNG.normal(profile["qt"],  profile["qt_sd"],  k).clip(250, 600)

        # Axes — p_axis is NaN for non-sinus rhythms, otherwise normal draw
        if np.isnan(profile["p_axis"]):
            df.loc[m, "p_axis"] = np.nan  # legitimately undefined for AF/flutter/paced
        else:
            df.loc[m, "p_axis"] = RNG.normal(profile["p_axis"], profile["axis_sd"], k).clip(-90, 180)
        df.loc[m, "qrs_axis"] = RNG.normal(profile["qrs_axis"], profile["axis_sd"], k).clip(-180, 180)
        df.loc[m, "t_axis"]   = RNG.normal(profile["t_axis"],   profile["axis_sd"], k).clip(-180, 180)

    # Any rhythm we didn't map (shouldn't happen) -> use 'unknown' profile
    leftover = df["rr_interval"].isna()
    if leftover.any():
        prof = RHYTHM_PROFILES["unknown"]
        k = int(leftover.sum())
        df.loc[leftover, "rr_interval"]  = RNG.normal(prof["rr"],  prof["rr_sd"],  k).clip(250, 2000)
        df.loc[leftover, "qrs_duration"] = RNG.normal(prof["qrs"], prof["qrs_sd"], k).clip(60, 220)
        df.loc[leftover, "qt_interval"]  = RNG.normal(prof["qt"],  prof["qt_sd"],  k).clip(250, 600)
        df.loc[leftover, "p_axis"]       = RNG.normal(prof["p_axis"], prof["axis_sd"], k).clip(-90, 180)
        df.loc[leftover, "qrs_axis"]     = RNG.normal(prof["qrs_axis"], prof["axis_sd"], k).clip(-180, 180)
        df.loc[leftover, "t_axis"]       = RNG.normal(prof["t_axis"],   prof["axis_sd"], k).clip(-180, 180)


def fix_labs(df: pd.DataFrame) -> None:
    """Clip labs to plausible physiologic ranges and recompute derived cols."""
    print("[labs] clipping to physiologic ranges + recomputing derived columns")
    df["potassium"]           = df["potassium"].clip(2.0, 9.0)
    df["sodium"]              = df["sodium"].clip(115, 160)
    df["creatinine"]          = df["creatinine"].clip(0.3, 12.0)
    df["baseline_creatinine"] = df["baseline_creatinine"].clip(0.4, 6.0)
    # Ensure baseline <= current creatinine wherever current is elevated
    bad = df["baseline_creatinine"] > df["creatinine"] + 0.3
    df.loc[bad, "baseline_creatinine"] = (df.loc[bad, "creatinine"] - 0.2).clip(0.4)
    df["creatinine_pct_increase"] = (
        (df["creatinine"] - df["baseline_creatinine"]) / df["baseline_creatinine"] * 100
    ).clip(-50, 500)
    df["egfr"]        = df["egfr"].clip(5, 120)
    df["bicarbonate"] = df["bicarbonate"].clip(8, 40)


def round_clinical(df: pd.DataFrame) -> None:
    """Round numeric columns to clinically sensible precision."""
    print("[round] applying clinical rounding")

    # Vital timepoints
    for t in range(1, 15):
        df[f"HR_t{t}"]     = df[f"HR_t{t}"].round(0).astype("int16")
        df[f"SBP_t{t}"]    = df[f"SBP_t{t}"].round(0).astype("int16")
        df[f"DBP_t{t}"]    = df[f"DBP_t{t}"].round(0).astype("int16")
        df[f"SPO2_t{t}"]   = df[f"SPO2_t{t}"].round(0).clip(70, 100).astype("int8")
        df[f"Weight_t{t}"] = df[f"Weight_t{t}"].round(1).astype("float32")

    # Aggregates
    df["HR_mean"]       = df["HR_mean"].round(1).astype("float32")
    df["HR_min"]        = df["HR_min"].round(0).astype("int16")
    df["SBP_mean"]      = df["SBP_mean"].round(1).astype("float32")
    df["SBP_min"]       = df["SBP_min"].round(0).astype("int16")
    df["DBP_mean"]      = df["DBP_mean"].round(1).astype("float32")
    df["SPO2_mean"]     = df["SPO2_mean"].round(2).astype("float32")
    df["Weight_change"] = df["Weight_change"].round(2).astype("float32")

    # Labs
    df["potassium"]               = df["potassium"].round(2).astype("float32")
    df["sodium"]                  = df["sodium"].round(1).astype("float32")
    df["creatinine"]              = df["creatinine"].round(2).astype("float32")
    df["baseline_creatinine"]     = df["baseline_creatinine"].round(2).astype("float32")
    df["egfr"]                    = df["egfr"].round(1).astype("float32")
    df["bicarbonate"]             = df["bicarbonate"].round(1).astype("float32")
    df["creatinine_pct_increase"] = df["creatinine_pct_increase"].round(1).astype("float32")

    # ECG numerics
    for c in ["rr_interval", "qrs_duration", "qt_interval"]:
        df[c] = df[c].round(0).astype("Int16")
    for c in ["p_axis", "qrs_axis", "t_axis"]:
        df[c] = df[c].round(0).astype("Int16")  # nullable Int16 keeps NaN for AF p_axis

    # Dose ratios
    for c in [c for c in df.columns if c.endswith("_dose_ratio")]:
        df[c] = df[c].round(3).astype("float32")


def tighten_bools(df: pd.DataFrame) -> None:
    """Force boolean columns to true bool dtype."""
    bool_cols = [c for c in df.columns if c.startswith("on_")
                 or c.startswith("ecg_") and c not in
                    ("ecg_rhythm",) and df[c].dtype == bool
                 or c in {
                     "angioedema_history", "pregnancy", "bilateral_renal_artery_stenosis",
                     "severe_asthma", "av_block", "history_dka", "type_1_diabetes", "chf",
                     "baseline_potassium_more_than_5_5", "sodium_less_than_130",
                     "potassium_less_than_3_0", "egfr_less_than_20", "egfr_less_than_30",
                     "baseline_hr_less_than_50", "severe_baseline_volume_depletion",
                     "severe_hypotension_detected", "severe_bradycardia_detected",
                     "any_emergency_flag", "hypotension_detected", "bradycardia_detected",
                     "volume_depletion_detected", "worsening_HF_detected",
                     "hyperkalemia_detected", "renal_dysfunction_detected",
                     "hyponatremia_detected", "metabolic_acidosis_detected",
                 }]
    for c in set(bool_cols):
        if c in df.columns:
            df[c] = df[c].fillna(False).astype(bool)


def reorder(df: pd.DataFrame) -> pd.DataFrame:
    """Group columns logically: id, demographics, ECG, meds, contraindications,
    vitals (timepoints + aggregates), labs, ground-truth labels."""
    ids = ["subject_id", "hadm_id"]
    demo = ["gender", "age"]
    ecg = ["ecg_rhythm", "rr_interval", "qrs_duration", "qt_interval",
           "p_axis", "qrs_axis", "t_axis",
           "ecg_afib", "ecg_lbbb", "ecg_rbbb", "ecg_lvh", "ecg_st_changes"]
    gdmt = []
    for cls in ["ACEi", "ARB", "ARNi", "beta_blocker", "MRA", "SGLT2i", "loop"]:
        gdmt += [f"on_{cls}", f"{cls}_dose_ratio"]
    other_meds = ["on_NSAID", "on_potassium_supplement", "on_other_diuretic"]
    contra = ["angioedema_history", "pregnancy", "bilateral_renal_artery_stenosis",
              "severe_asthma", "av_block", "history_dka", "type_1_diabetes", "chf",
              "baseline_potassium_more_than_5_5", "sodium_less_than_130",
              "potassium_less_than_3_0", "egfr_less_than_20", "egfr_less_than_30",
              "baseline_hr_less_than_50", "severe_baseline_volume_depletion"]
    vitals_t = []
    for t in range(1, 15):
        vitals_t += [f"HR_t{t}", f"SBP_t{t}", f"DBP_t{t}", f"SPO2_t{t}", f"Weight_t{t}"]
    vitals_agg = ["HR_mean", "HR_min", "SBP_mean", "SBP_min", "DBP_mean",
                  "SPO2_mean", "Weight_change"]
    labs = ["potassium", "sodium", "creatinine", "baseline_creatinine",
            "creatinine_pct_increase", "egfr", "bicarbonate"]
    labels = ["severe_hypotension_detected", "severe_bradycardia_detected",
              "any_emergency_flag", "hypotension_detected", "bradycardia_detected",
              "volume_depletion_detected", "worsening_HF_detected",
              "hyperkalemia_detected", "renal_dysfunction_detected",
              "hyponatremia_detected", "metabolic_acidosis_detected"]

    order = ids + demo + ecg + gdmt + other_meds + contra + vitals_t + vitals_agg + labs + labels
    missing = [c for c in order if c not in df.columns]
    extra = [c for c in df.columns if c not in order]
    if missing:
        print(f"[reorder] missing expected cols: {missing}")
    if extra:
        print(f"[reorder] extra cols appended at end: {extra}")
    return df[[c for c in order if c in df.columns] + extra]


def write_summary(df: pd.DataFrame, runtime: float) -> None:
    label_cols = [c for c in df.columns if c.endswith("_detected") or c == "any_emergency_flag"]
    med_cols = [c for c in df.columns if c.startswith("on_")]
    null_remaining = df.isna().sum()
    null_remaining = null_remaining[null_remaining > 0]

    lines = [
        "CHF Weekly Synthetic Dataset (cleaned)",
        "=" * 50,
        f"rows: {len(df):,}",
        f"columns: {len(df.columns)}",
        f"polish_runtime_seconds: {runtime:.1f}",
        "",
        "remaining nulls (clinically valid -- p_axis is undefined for AF/flutter/paced/junctional/VT):",
    ]
    if null_remaining.empty:
        lines.append("  (none)")
    else:
        for c, v in null_remaining.items():
            lines.append(f"  {c}: {int(v):,}")
    lines.append("")
    lines.append("label prevalence:")
    for c in label_cols:
        lines.append(f"  {c}: {df[c].mean():.4f}  ({int(df[c].sum()):,})")
    lines.append("")
    lines.append("medication usage:")
    for c in med_cols:
        lines.append(f"  {c}: {df[c].mean():.4f}  ({int(df[c].sum()):,})")
    lines.append("")
    lines.append("lab summary (mean ± sd  [min..max]):")
    for c in ["potassium", "sodium", "creatinine", "egfr", "bicarbonate"]:
        s = df[c]
        lines.append(f"  {c:>12s}: {s.mean():6.2f} ± {s.std():5.2f}  [{s.min():.2f}..{s.max():.2f}]")
    SUMMARY.write_text("\n".join(lines))
    print("\n".join(lines[:30]))


def main():
    t0 = time.time()
    print(f"[load] {SRC}")
    df = pd.read_csv(SRC)
    print(f"[load] {len(df):,} rows × {len(df.columns)} cols")

    fix_labs(df)
    impute_ecg(df)
    round_clinical(df)
    tighten_bools(df)
    df = reorder(df)

    print(f"[write] {DST}")
    df.to_csv(DST, index=False)

    write_summary(df, time.time() - t0)
    print("[done]")


if __name__ == "__main__":
    main()
