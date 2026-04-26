# CHF GDMT Titration Pipeline

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-research%20prototype-orange.svg)](#disclaimer)

A research pipeline that recommends **weekly guideline-directed medical therapy (GDMT) titration decisions** for patients with **congestive heart failure (CHF)**. It combines a synthetic-data generator (anchored on MIMIC-IV distributions), a LightGBM classifier for adverse-effect detection, a deterministic titration engine over five drug classes, and a Streamlit clinician UI.

> ⚠️ **Research prototype only.** Not a medical device. Do not use for real clinical decisions. See [Disclaimer](#disclaimer).

---

## Table of contents
- [Overview](#overview)
- [Repository layout](#repository-layout)
- [Quick start](#quick-start)
- [Pipeline](#pipeline)
- [Data sources](#data-sources)
- [Drug classes covered](#drug-classes-covered)
- [Adverse-effect flags](#adverse-effect-flags)
- [Development](#development)
- [Disclaimer](#disclaimer)
- [Citation](#citation)

---

## Overview

The pipeline executes four stages:

1. **Synthetic data generation** — produces realistic patient-week records (vitals × 14 timepoints, ECG, labs, GDMT meds, contraindications, ground-truth adverse-effect labels). Two modes: `distribution` (statistical priors) and `mimic` (anchored on MIMIC-IV).
2. **Adverse-effect classifier** — multi-output LightGBM model predicting 11 immediate / suspected / emergency flags from vitals + medication state.
3. **Titration logic engine** — deterministic, contraindication-aware decision tree over RAAS / β-blocker / MRA / SGLT2i / loop diuretic, emitting `increase / decrease / maintain / stop / start / hold + order_labs` per class.
4. **Clinician UI** — Streamlit app for reviewing the recommendation, exploring patient-week traces, and sweeping titration strategies.

---

## Repository layout

```
CHF/
├── README.md                       # this file
├── LICENSE                         # MIT + clinical disclaimer
├── CITATION.cff                    # academic citation metadata
├── .gitignore                      # excludes MIMIC raw, large parquets, venvs
│
├── chf_titration_package/          # the installable Python package
│   ├── pyproject.toml              # package metadata + console scripts
│   ├── requirements.txt
│   ├── README.md
│   ├── src/chf_titration/          # library code
│   │   ├── synthesize.py           # weekly synthetic data
│   │   ├── data_gen.py             # batch synth + MIMIC-anchored mode
│   │   ├── featurize.py            # vitals → feature row
│   │   ├── classifier.py           # LightGBM 11-flag classifier
│   │   ├── engine.py               # titration logic engine
│   │   ├── strategy.py             # traditional / strong_hf / rapid_sequence / sglt_mra_first
│   │   ├── ui.py                   # Streamlit app
│   │   ├── cli.py                  # console-script entry points
│   │   ├── constants.py            # drug classes, target doses, label cols
│   │   └── data/                   # small reference CSVs (GDMT, ICDs, labs)
│   ├── scripts/                    # thin CLI wrappers
│   │   ├── prepare.py              # one-shot setup (gen + train)
│   │   ├── generate_data.py
│   │   ├── train.py
│   │   └── run_ui.py
│   └── tests/test_smoke.py
│
├── synthetic_data/                 # standalone weekly-CSV generator (older)
│   ├── generate_synthetic.py
│   ├── polish_synthetic.py
│   └── dataset_summary.txt         # produced datasets are .gitignored
│
├── sample_synthetic_data/          # small XLSX samples for inspection
│   └── *.xlsx
│
├── er_diagram/                     # ER & flow diagrams
│   └── *.png
│
└── Other Info/                     # process notes, reference codes, PDF
    ├── overall_process.txt
    ├── synthetic_data_scheme.txt
    ├── Final.pdf
    ├── ecg_study_ids.csv
    └── patient_visit_codes.csv
```

> **Excluded from git** (see `.gitignore`):
> - `MIMIC-IV_Original_Data_Set/` — PhysioNet credentialed access only (~91 GB)
> - `synthetic_data/*.csv` — bulk generated CSVs (100s of MB)
> - `chf_titration_package/data/*.parquet` — generated datasets
> - `chf_titration_package/models/*.pkl` — trained model artefacts
> - `.venv/`, `__pycache__/`, `*.egg-info/`, `.pytest_cache/`, `.DS_Store`

---

## Quick start

```bash
# 1. clone
git clone https://github.com/<your-username>/chf-gdmt-titration.git
cd chf-gdmt-titration/chf_titration_package

# 2. install (editable, with UI extras)
python3.10 -m venv .venv
source .venv/bin/activate
pip install -e ".[ui]"

# 3. one-shot prep — generates synthetic weeks + trains the classifier
chf-prepare                       # ~1–2 min on a laptop

# 4. launch the Streamlit UI
chf-ui                            # http://localhost:8501
```

Console scripts installed by the package:

| Command            | Purpose                                                  |
|--------------------|----------------------------------------------------------|
| `chf-generate-data`| Produce N synthetic patient-weeks (parquet)              |
| `chf-train`        | Train the 11-flag LightGBM classifier                    |
| `chf-prepare`      | Generate + train (idempotent, used for first-run setup)  |
| `chf-ui`           | Launch the Streamlit clinician UI                        |

---

## Pipeline

```
                  ┌──────────────────────────────┐
MIMIC-IV  ─────►  │ 1. Synthetic-data generator  │ ─►  patient-week parquet
priors            └──────────────────────────────┘     (vitals×14 + ECG + labs +
                                                        meds + contraind + GT flags)
                                                                │
                                                                ▼
                  ┌──────────────────────────────┐
                  │ 2. LightGBM classifier (11)  │ ─►  adverse-effect flags
                  └──────────────────────────────┘     (immediate / suspected / emergency)
                                                                │
                                                                ▼
                  ┌──────────────────────────────┐
                  │ 3. Titration logic engine    │ ─►  per-class decision +
                  │    + strategy layer          │     order_labs recommendation
                  └──────────────────────────────┘
                                                                │
                                                                ▼
                  ┌──────────────────────────────┐
                  │ 4. Streamlit clinician UI    │
                  └──────────────────────────────┘
```

Detailed flow (including AE resolver, RAAS preference order, and per-class lab gates) is documented in [`Other Info/synthetic_data_scheme.txt`](Other%20Info/synthetic_data_scheme.txt) and visually in [`er_diagram/`](er_diagram/).

---

## Data sources

| Source                | Where                                        | Notes                                       |
|-----------------------|----------------------------------------------|---------------------------------------------|
| MIMIC-IV (raw)        | https://physionet.org/content/mimiciv/       | **Not in repo.** Credentialed access only.  |
| MIMIC-IV-derived      | `chf_titration_package/data/*.parquet`       | Generated locally; gitignored.              |
| GDMT / ICD reference  | `chf_titration_package/src/chf_titration/data/*.csv` | Small, shipped with the package.    |
| Synthetic samples     | `sample_synthetic_data/*.xlsx`               | Inspection-only XLSX previews.              |

To reproduce MIMIC-anchored mode: place the MIMIC-IV-derived CHF cohort parquet at `chf_titration_package/data/synthetic_CHF_visits_v2.parquet`, then run `chf-prepare --mode mimic`.

---

## Drug classes covered

The titration engine reasons over the five guideline pillars of HFrEF therapy:

| Class            | Representative drugs              |
|------------------|-----------------------------------|
| RAAS inhibitor   | ARNi → ACEi → ARB (preference order) |
| β-blocker        | carvedilol, metoprolol succ., bisoprolol |
| MRA              | spironolactone, eplerenone        |
| SGLT2 inhibitor  | dapagliflozin, empagliflozin      |
| Loop diuretic    | furosemide, torsemide, bumetanide |

Strategies available at the strategy layer: `traditional`, `strong_hf`, `rapid_sequence`, `sglt_mra_first`.

---

## Adverse-effect flags

The classifier predicts 11 boolean flags consumed by the engine:

```
severe_hypotension_detected      hypotension_detected
severe_bradycardia_detected      bradycardia_detected
any_emergency_flag               volume_depletion_detected
                                 worsening_HF_detected
                                 hyperkalemia_detected
                                 renal_dysfunction_detected
                                 hyponatremia_detected
                                 metabolic_acidosis_detected
```

Class prevalence in the reference 137,712-row synthetic dataset is reported in [`synthetic_data/dataset_summary.txt`](synthetic_data/dataset_summary.txt).

---

## Development

```bash
cd chf_titration_package
pip install -e ".[dev]"

pytest -v               # smoke tests
ruff check src/         # lint
```

Smoke tests live in `chf_titration_package/tests/test_smoke.py` and verify that:
- the synthetic generator produces 14 timepoints per patient-week,
- the engine + strategy run end-to-end without the classifier,
- a confirmed K=6.3 triggers the global-stop branch.

---

## Disclaimer

This software is a **research prototype**. It is **not a medical device**, has not been validated in clinical settings, and **must not be used for real patient care**. All recommendations produced by the titration engine are for research, education, and clinician review only. The authors accept no liability for any clinical use.

MIMIC-IV is provided by the MIT Laboratory for Computational Physiology under a credentialed-access PhysioNet license. No PHI is present in this repository.

---

## Citation

If you use this work, please cite via the included [`CITATION.cff`](CITATION.cff), or:

```
Ramaseshu (2026). CHF GDMT Titration Pipeline (v0.1.0).
https://github.com/<your-username>/chf-gdmt-titration
```

And cite MIMIC-IV:

> Johnson, A., Bulgarelli, L., Pollard, T., et al. *MIMIC-IV* (version 3.x). PhysioNet. https://doi.org/10.13026/6mm1-ek67
