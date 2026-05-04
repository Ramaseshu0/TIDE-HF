# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Build & Run Commands

**Python package (from project root):**
- Setup: `python3.14 -m venv .venv && source .venv/bin/activate && pip install -e ".[ui]"`
- First-time prep: `python scripts/prepare.py` (generates 10k synthetic weeks + trains classifier, ~2min)
- Run Streamlit UI: `python scripts/run_ui.py` (auto-prepares on first launch, opens :8501)
- Run FastAPI backend: `python scripts/run_api.py` (serves :8000 for website Engine console)
- Generate data only: `python scripts/generate_data.py --n 10000 --mode mimic`
- Train classifier only: `python scripts/train.py`

**Website (from website/ directory):**
- Setup: `cd website && npm install`
- Dev server: `npm run dev` (opens :8901, NOT standard :5173)
- Build: `npm run build` (outputs to website/dist with base="/TIDE-HF/" for GitHub Pages)
- Test: `npm test` or `npm run test:watch`

## Critical Non-Obvious Patterns

**Classifier bundle is mandatory:**
- [`predict_flags()`](src/chf_titration/classifier.py:24) raises `ValueError` if bundle is `None` (no silent fallback)
- Bundle must exist at `models/chf_classifier_lgbm.pkl` before calling predict
- Run `python scripts/prepare.py` to generate it (idempotent, skips if exists)

**RAAS class mapping is complex:**
- Engine uses `"ACEi"`, `"ARB"`, `"ARNi"` as separate classes in [`CLASSES`](src/chf_titration/constants.py:5)
- Website/TypeScript collapses them into single `"RAAS"` class in [`tide-engine.ts`](website/src/lib/tide-engine.ts:26)
- When calling Python API from website, must expand `"RAAS"` back to specific subclass
- [`CLASS_TO_COLKEY`](src/chf_titration/constants.py:28) maps display names to internal keys

**Dose ladders are auto-generated from CSV:**
- [`_build_ladder()`](src/chf_titration/strategy.py:26) creates doubling sequences from initial→target doses
- Loaded once at module import via [`_load_ladders()`](src/chf_titration/strategy.py:40)
- Stored in module-level `LADDERS` and `TARGETS` dicts
- CSV path resolved via [`data_path()`](src/chf_titration/constants.py:53) using `importlib.resources`

**Feature engineering requires exact 14 timepoints:**
- [`featurize_week()`](src/chf_titration/featurize.py:22) expects list of 14 dicts (2/day × 7 days)
- Generates 95 vital/ECG features + 13 demographic/med features = 108 total
- Missing timepoints handled via `_n_missing` counters, but structure must be 14-element list
- [`build_feature_row()`](src/chf_titration/featurize.py:68) returns single-row DataFrame in classifier's expected column order

**Contraindications derived from multiple sources:**
- [`derive_contraindications()`](src/chf_titration/engine.py:58) merges 3 sources: ICD categories, diagnosis codes, baseline labs
- ICD category mapping in [`CONTRA_ICD_CAT_MAP`](src/chf_titration/engine.py:49) (only 5 categories mapped)
- DKA/T1DM detected via hardcoded ICD-10 sets at [lines 80-86](src/chf_titration/engine.py:80)
- Lab thresholds: K>5.5, K<3.0, Na<130, eGFR<20 trigger flags
- Patient's `contras` dict overrides all derived values

**Website runs dual-mode engine:**
- Python mode: calls FastAPI at `:8000` when available (real LightGBM + TitrationEngine)
- Browser mode: pure TypeScript fallback in [`tide-engine.ts`](website/src/lib/tide-engine.ts:1) (no ML, rule-based only)
- Mode detected by pill indicator in Engine console header
- Set `VITE_TIDE_API` env var to override API URL

**Vite base path is production-specific:**
- [`vite.config.ts`](website/vite.config.ts:8) sets `base: "/TIDE-HF/"` in production mode only
- Dev mode uses `base: "/"` (standard)
- GitHub Pages deployment requires this for asset paths
- Hash routing (`#/engine`) works without server-side SPA config

**TypeScript strictness is relaxed:**
- [`tsconfig.json`](website/tsconfig.json:8) disables `noImplicitAny`, `strictNullChecks`, `noUnusedLocals`
- Allows rapid prototyping but means type safety is minimal
- `skipLibCheck: true` skips node_modules type checking

**Data generation has two modes:**
- `"mimic"` mode requires `data/synthetic_CHF_visits_v2.parquet` (135 MB, real MIMIC demographics/labs)
- `"distribution"` mode is fully synthetic (no external data, Gaussian sampling)
- Auto-selects based on parquet existence unless `--mode` specified
- MIMIC parquet path hardcoded at [`DEFAULT_MIMIC_PARQUET`](src/chf_titration/data_gen.py:36)

## Code Style

**Python:**
- Type hints: `from __future__ import annotations` enables forward refs
- Imports: standard lib → third-party → local (`.constants`, `.featurize`, etc.)
- Docstrings: module-level + public functions only (not every helper)
- Private helpers: prefix with `_` (e.g., `_as_list`, `_parse_numbers`)

**TypeScript/React:**
- Path alias: `@/` maps to `website/src/` via Vite config
- Component structure: shadcn/ui components in `website/src/components/ui/`
- No strict prop types (TypeScript strictness disabled)
- Hash routing via `react-router-dom` (not BrowserRouter)

## Testing

**Python:**
- Single test: `pytest tests/test_smoke.py::test_name -v`
- All tests: `pytest tests/`
- No test coverage configured

**Website:**
- Run once: `npm test` (from website/)
- Watch mode: `npm run test:watch`
- Uses vitest + @testing-library/react