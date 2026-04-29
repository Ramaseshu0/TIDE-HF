# TIDE-HF

Local pipeline for a CHF GDMT titration system: synthetic-data generator → LightGBM adverse-effect classifier → rule-based titration engine with lab gating → strategy applier → Streamlit UI.

No MIMIC dataset required. Everything runs locally.

## Setup (Python 3.10+)

```bash
git clone <this-repo>
cd TIDE-HF

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -U pip
pip install -e ".[ui]"
```

The `[ui]` extra pulls in Streamlit. If you only need the library + training, `pip install -e .` is enough.

## One-step run

```bash
# Launch the Streamlit UI locally on http://localhost:8501.
# On first launch, auto-generates 10,000 synthetic patient-weeks and trains the
# 11-flag LightGBM classifier (≈2 min on a modern laptop). Subsequent launches
# skip setup and load the cached bundle from models/chf_classifier_lgbm.pkl.
python scripts/run_ui.py
```

If a trained bundle is already committed with the repo (see the section on distributing the bundle below), the first launch skips training too.

To prepare the bundle ahead of time without launching the UI:

```bash
python scripts/prepare.py                  # 10,000 weeks, mimic mode if seed parquet present
python scripts/prepare.py --n 5000         # faster setup
python scripts/prepare.py --force          # regenerate + retrain from scratch
```

### Retraining or training with different settings

If you want to tweak the training set size, seed, or LightGBM hyperparameters, you can still run the two steps separately:

```bash
python scripts/generate_data.py --n 20000 --out data/synthetic_patient_weeks.parquet
python scripts/train.py --data data/synthetic_patient_weeks.parquet --out models/chf_classifier_lgbm.pkl --n-estimators 800
```

The UI hard-fails with a clear error if it can't find a trained bundle — there is no pseudo-classifier fallback.

## Two data-generation modes

The included `data/synthetic_CHF_visits_v2.parquet` (135 MB) is a pre-built snapshot of 137k MIMIC-IV CHF visits with real demographics, baseline labs, GDMT regimens, and ECG-study linkage. The generator uses it by default.

```bash
# MIMIC-seeded (default when the parquet is present): real demographics +
# baseline labs + per-class dose patterns come from sampled MIMIC visits.
python scripts/generate_data.py --n 10000 --mode mimic

# Pure distribution: Gaussian-sampled demographics and baselines. Runs with
# zero external data. Auto-selected if the MIMIC parquet is missing.
python scripts/generate_data.py --n 10000 --mode distribution

# Auto (default): "mimic" if data/synthetic_CHF_visits_v2.parquet exists, else "distribution".
python scripts/generate_data.py --n 10000 --mode auto
```

**What the modes do differently**

| aspect | `distribution` | `mimic` |
|---|---|---|
| demographics (age, gender) | Gaussian / uniform | sampled from real CHF visits |
| baseline labs (K, Na, Cr, eGFR) | independent Gaussians | extracted from real lab events |
| GDMT med state | random assignment, 35%/class | copied from real MIMIC prescriptions |
| vital trajectories + labels | synthesized from sampled labels | same |
| external data needed | none | data/synthetic_CHF_visits_v2.parquet |

Use `mimic` when you want realistic comorbidity correlations in your training set. Use `distribution` when portability matters more than realism or when the seed parquet isn't available.

## What's in the package

```
TIDE-HF/
├── src/chf_titration/
│   ├── constants.py       # class maps, rhythms, drug reps
│   ├── featurize.py       # featurize_week: 14 timepoints → 108 features
│   ├── synthesize.py      # synthesize_week + preset patients
│   ├── engine.py          # TitrationEngine v1.2 (lab-gated) + contraindications
│   ├── strategy.py        # apply_strategy: traditional / strong_hf / rapid_sequence / sglt_mra_first
│   ├── classifier.py      # train_classifier, load_bundle, predict_flags
│   ├── data_gen.py        # full synthetic-week batch generator
│   ├── ui.py              # Streamlit app
│   └── data/              # reference CSVs (GDMT, ICDs, lab maps)
├── scripts/
│   ├── generate_data.py
│   ├── train.py
│   ├── prepare.py         # one-shot: generate + train (idempotent)
│   ├── run_ui.py          # auto-prepares the bundle on first launch
│   └── run_mcp.py         # MCP server entrypoint
├── data/                  # synthetic_CHF_visits_v2.parquet, synthetic_patient_weeks.parquet
├── models/                # trained bundles land here (chf_classifier_lgbm.pkl)
├── synthetic_data/        # standalone synthetic CSV generators
├── tests/
├── website/               # Next.js marketing site (separate package)
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Preset scenarios in the UI

Seventeen presets cover the main decision paths. Each one is named after the pattern it showcases.

**Titration-state showcases**

| preset | what it demonstrates |
|---|---|
| `Newly diagnosed, stable` | Fresh patient. Engine recommends starting the preferred RAAS (ARNi) + other pillars per strategy. |
| `Partial titration, no AEs` | On ACEi 10/40, BB 12.5/50, loop. Shows `increase_dose` (below target) + `start_medication` (MRA, SGLT2i). |
| `Fully titrated, at target` | Quadruple therapy at target doses. Every class should `maintain_dose` with reason `at_or_above_target`. |

**Adverse-effect showcases**

| preset | what it demonstrates |
|---|---|
| `Suspected hyperkalemia` | ECG signs (T-peaked, wide QRS) but no labs → classifier fires `hyperkalemia_detected` → engine holds RAAS + MRA + BB, sets `order_labs`. |
| `Suspected renal dysfunction` | Weight drift + rising SBP but no labs → engine holds + requests labs for same classes. |
| `Confirmed hyperkalemia (K=6.3)` | Labs show K > 6.0 → `global_stop` triggers, all meds held, escalate. |
| `Volume depletion + hypotension` | AE resolver path: stop SGLT2i + decrease loop (volume on both). |
| `Worsening HF` | Weight gain + SpO2 drop → BB and loop get `immediate_ae`; RAAS/MRA still titrate. |
| `Bradycardia on beta blocker` | HR < 45 → AE resolver specifically down-titrates beta blocker first. |
| `Worsening HF + hyperkalemia (labs confirmed)` | Double-bind: MRA decreased (hyperK on MRA) AND loop increased (vol overload + hyperK). |
| `Hyponatremia + renal dysfunction (labs)` | Na < 130 + Cr Δ > 30% → `lab_ae` on loop (Na) + RAAS/MRA (Cr). Global stop via Cr Δ. |

**Contraindication showcases**

| preset | what it demonstrates |
|---|---|
| `Angioedema history (ARNi + ACEi blocked)` | Preferred RAAS falls to ARB. MRA + SGLT2i still start. |
| `Severe asthma (beta blocker blocked)` | BB contraindicated. RAAS + MRA + SGLT2i + loop all titrate normally. |
| `CKD stage 4 (eGFR=22)` | eGFR < 30 auto-derives the MRA contraindication from baseline. SGLT2i still allowed (> 20). |
| `Pregnant (RAAS blocked)` | All three RAAS agents blocked (preferred_raas = None). SGLT2i also contraindicated in pregnancy. |
| `AV block (beta blocker blocked)` | BB held. RAAS + MRA + loop continue. |
| `Severe hypotension (global stop)` | Classifier-triggered global_stop from severe hypotension flag (SBP < 80). |

Every preset carries a `note` field in `PATIENTS[...]` summarising the expected decision path — useful when reviewing the UI output.

## Using the library in your own code

```python
from chf_titration.synthesize import synthesize_week, PATIENTS
from chf_titration.classifier import load_bundle, predict_flags
from chf_titration.engine import TitrationEngine
from chf_titration.strategy import apply_strategy

bundle = load_bundle("models/chf_classifier_lgbm.pkl")
engine = TitrationEngine()

patient = PATIENTS["Newly diagnosed, stable"]
tps = synthesize_week(patient)

flags, probs = predict_flags(patient, tps, bundle)
result = engine.evaluate(patient, tps, flags, labs=None, awaiting_labs=set())
changes = apply_strategy(result, patient, "strong_hf")

for cls, change in changes.items():
    print(f"{cls:<14} {change['concrete_action']}  {change['current']} → {change['new_dose']} mg")
```

## The four titration strategies

| name | behavior |
|---|---|
| `traditional` | One new class started per week, RAAS→beta-blocker→MRA→SGLT2i→loop, single-rung titration |
| `strong_hf` | All eligible classes started at once, double-rung up-titration |
| `rapid_sequence` | All eligible classes started at once, single-rung steps (Greene 2021) |
| `sglt_mra_first` | Phase 1 starts SGLT2i + MRA; phase 2 then adds ARNi + beta-blocker |

## Running the UI alone

```bash
streamlit run src/chf_titration/ui.py
```

or

```bash
python scripts/run_ui.py
```

Opens at http://localhost:8501. Pick a preset, toggle between summary-stats input or per-timepoint CSV editing, pick a strategy, click Compute. No internet or tunnel required.

## Marketing site + Engine UI (`website/`)

The Vite + React + shadcn-ui app under `website/` hosts the landing page (`#/`) and the live Engine console (`#/engine`) where every patient field — vitals, baseline labs, recent labs, meds, contraindications — is editable.

The Engine page runs in **two modes**:

- **Python engine** — when the local FastAPI backend (`scripts/run_api.py`) is reachable at `http://127.0.0.1:8000`. Calls into `TitrationEngine` + the LightGBM classifier so you see real engine output.
- **Browser engine** — fallback that runs the rule logic entirely in the browser. This is what GitHub Pages serves; no backend required.

The header pill in the Engine console shows which mode is live.

### Local development

```bash
# Terminal 1 (optional) — Python API for the real engine
python scripts/run_api.py            # http://127.0.0.1:8000

# Terminal 2 — website
cd website
npm install
npm run dev                          # http://localhost:8901
```

Open http://localhost:8901/#/engine, pick a preset, edit anything, click **Compute**.

Override the API URL with `VITE_TIDE_API=…` at build/dev time. Other scripts: `npm run build`, `npm run preview`, `npm test`.

### Deploy to GitHub Pages

The repo ships [`.github/workflows/deploy.yml`](.github/workflows/deploy.yml). On every push to `main` that touches `website/`, the workflow builds and publishes to GitHub Pages.

One-time repo setup:

1. Push the repo to `https://github.com/Ramaseshu0/TIDE-HF`.
2. Repo → **Settings → Pages → Build and deployment → Source: GitHub Actions**.
3. (First push will trigger the workflow; subsequent ones auto-deploy.)

The site will be live at **https://ramaseshu0.github.io/TIDE-HF/** — landing at `#/`, Engine at `#/engine`. Hash-based routing means deep links (and refresh) work without a server-side SPA fallback.

GitHub Pages can't run Python, so the deployed Engine page automatically uses the in-browser engine. To run the real Python engine in production, host `scripts/run_api.py` somewhere reachable and rebuild with `VITE_TIDE_API=https://your-api.example.com npm run build`.

## Distributing the 135 MB MIMIC parquet and the trained bundle

Because `data/synthetic_CHF_visits_v2.parquet` is 135 MB, plain git will complain. Two clean options:

- **git-lfs** — `git lfs install && git lfs track "*.parquet" "*.pkl" && git add .gitattributes data/synthetic_CHF_visits_v2.parquet models/chf_classifier_lgbm.pkl` then commit normally.
- **Side-channel** — keep the parquet and/or pickle out of git (restore them to `.gitignore`) and ship them alongside the repo via a download link or direct copy. The package auto-detects the parquet's presence and falls back to `distribution` mode if absent. If the pickle is absent, `scripts/run_ui.py` will train one on first launch.

`.gitignore` already has an un-ignore rule for `models/chf_classifier_lgbm.pkl`, so once you've generated the bundle locally you can commit it directly:

```bash
python scripts/prepare.py
git add models/chf_classifier_lgbm.pkl
git commit -m "ship pretrained LightGBM bundle"
```

The trained pickle is ~5-15 MB depending on n_estimators and dataset size — small enough to commit without LFS in most cases.

## License

MIT
