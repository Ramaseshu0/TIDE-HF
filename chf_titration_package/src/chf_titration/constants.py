"""Shared constants used across the pipeline."""

from importlib.resources import files

CLASSES = ["ACEi", "ARB", "ARNi", "beta_blocker", "MRA", "SGLT2i", "loop"]
CLASSES_ALL = CLASSES + ["thiazide"]

RHYTHMS = [
    "NSR", "Sinus_brady", "Sinus_tachy", "AF",
    "AV_block_1", "AV_block_2", "AV_block_3",
    "Junctional", "LBBB", "RBBB", "wide_complex",
]

VITAL_KEYS = ["HR", "SBP", "DBP", "SpO2", "weight_kg"]
ECG_SCALAR_KEYS = ["hr_ecg", "pr_ms", "qrs_ms", "qtc_ms", "t_amp_mv", "st_dev_mm"]

CLASS_TO_REP_DRUG = {
    "ACEi": "Lisinopril",
    "ARB": "Losartan",
    "ARNi": "Sacubitril-valsartan",
    "Beta blocker": "Carvedilol",
    "MRA": "Spironolactone",
    "SGLT2i": "Empagliflozin",
    "Loop diuretic": "Furosemide",
    "Thiazide diuretic": "Hydrochlorothiazide",
}

CLASS_TO_COLKEY = {
    "ACEi": "ACEi",
    "ARB": "ARB",
    "ARNi": "ARNi",
    "Beta blocker": "beta_blocker",
    "MRA": "MRA",
    "SGLT2i": "SGLT2i",
    "Loop diuretic": "loop",
    "Thiazide diuretic": "thiazide",
}

LABEL_COLS = [
    "hypotension_detected", "bradycardia_detected", "volume_depletion_detected",
    "worsening_HF_detected", "hyperkalemia_detected", "renal_dysfunction_detected",
    "hyponatremia_detected", "metabolic_acidosis_detected",
    "severe_hypotension_detected", "severe_bradycardia_detected", "any_emergency_flag",
]

DEFAULT_BASELINE_LABS = dict(
    K=4.2, Na=140, Cr=1.0, eGFR=75, BUN=15,
    HCO3=24, pH=7.40, pCO2=40, pO2=95,
    glucose=100, Mg=2.0, phosphate=3.5,
)


def data_path(filename: str) -> str:
    """Absolute path to a reference CSV bundled with the package."""
    return str(files("chf_titration.data") / filename)
