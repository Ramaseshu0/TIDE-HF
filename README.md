# TIDE-HF

**Trajectory · Integrated · Decision · Engine.**
A local-first, guideline-directed AI pipeline for chronic heart-failure (CHF) titration: synthetic-data generator → LightGBM adverse-effect classifier → rule-based titration engine with lab gating → strategy applier → Streamlit & web UIs.

🌐 **Live site:** [https://ramaseshu0.github.io/TIDE-HF/](https://ramaseshu0.github.io/TIDE-HF/)
📂 **Repo:** [https://github.com/Ramaseshu0/TIDE-HF](https://github.com/Ramaseshu0/TIDE-HF)

No MIMIC dataset required. Everything runs locally on macOS.

---

## What it is — the four pillars

| Letter | Pillar | What it does |
|---|---|---|
| **T** | **Trajectory** | 14 timepoints per week → 108 features. Vitals, weight drift, rhythm, SpO₂, ECG signs become a continuous signal. |
| **I** | **Integrated** | Demographics, baseline labs (K, Na, Cr, eGFR), GDMT regimen and ECG-study linkage are fused — MIMIC-seeded or pure-distribution mode. |
| **D** | **Decision** | A LightGBM classifier predicts 11 adverse-effect flags. A rule-based titration engine turns those flags into a per-class action. |
| **E** | **Engine** | TitrationEngine v1.2 — lab-gated, contraindication-aware. `global_stop`, `hold + order_labs`, AE resolver and dose-rung logic are all explicit and auditable. |

**Pipeline:**
`synthetic-data generator` → `LightGBM AE classifier (11 flags)` → `TitrationEngine v1.2 (lab-gated)` → `strategy applier` → `Streamlit UI` / `Web Engine console`

---

## Quick start on macOS

### Prerequisites (Homebrew)

```bash
# Python 3.10+ (used 3.14 here) and Node 22+
brew install python@3.14 node
```

> ⚠️ macOS ships with Python 3.9 in `/Library/Developer/CommandLineTools`. Make sure your venv uses a 3.10+ interpreter — verify with `python3 --version` after activating.

### Clone and set up the Python package

```bash
git clone https://github.com/Ramaseshu0/TIDE-HF.git
cd TIDE-HF

python3.14 -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install -e ".[ui]"
```

The `[ui]` extra pulls in Streamlit. For library-only use (no UI), `pip install -e .` is enough.

---

## Run the local Streamlit UI

```bash
python scripts/run_ui.py
```

Opens at **http://localhost:8501**. On first launch, auto-generates 10,000 synthetic patient-weeks and trains the 11-flag LightGBM classifier (~2 min on Apple silicon). Subsequent launches load the cached bundle from `models/chf_classifier_lgbm.pkl`.

To prepare the bundle ahead of time without launching the UI:

```bash
python scripts/prepare.py                  # 10,000 weeks, mimic mode if seed parquet present
python scripts/prepare.py --n 5000         # faster setup
python scripts/prepare.py --force          # regenerate + retrain from scratch
```

---

## Run the website

The Vite + React + shadcn-ui app under `website/` hosts the landing page and the live **Engine console** where every patient field — vitals, baseline labs, recent labs, meds, contraindications — is editable.

```bash
cd website
npm install
npm run dev          # http://localhost:8901
```

Routes:
- `http://localhost:8901/#/`        — landing page (About · How it works · Strategies · Chat)
- `http://localhost:8901/#/engine`  — editable Engine console

The Engine page runs in **two modes**:

- **Python engine** — when the local FastAPI backend (`scripts/run_api.py`) is reachable on port 8000. Calls `TitrationEngine` + LightGBM directly so you see real engine output.
- **Browser engine** — fallback that runs the rule logic entirely in TypeScript. This is what the GitHub-hosted site uses.

The pill in the Engine console header shows which mode is live (green = Python, cyan = Browser).

### Optional: real Python engine for the web Engine console

```bash
# Terminal 1 — Python API (LightGBM classifier + engine + strategy + RAG /chat)
python scripts/run_api.py            # http://127.0.0.1:8000

# Terminal 2 — website
cd website && npm run dev            # http://localhost:8901
```

Override the API URL at build/dev time with `VITE_TIDE_API=https://…`.

---

## TIDE-HF Assistant (local RAG)

The website's **TIDE-HF Assistant** chat is a real retrieval-augmented LLM, not a canned-reply demo. It indexes the clinical PDFs you put in `rag_docs/` (AHA 2022 HF guideline · STRONG-HF · drug monographs · patient leaflets) and answers grounded questions about the **currently loaded patient** — labs, vitals, meds, contraindications, classifier flags, and the engine's per-class actions.

The stack is fully local — no API keys, no internet after first install:

| Component | Purpose | Disk |
|---|---|---|
| **ChromaDB** | Vector store for the chunked guideline PDFs | ~50 MB |
| **all-MiniLM-L6-v2** | Embedding model (CPU) | ~80 MB |
| **Ollama + mistral** | Generation model | ~4.4 GB |

### Setup (one-time)

```bash
# 1. Install RAG extras
pip install -e ".[rag]"

# 2. Install Ollama (the local LLM runtime) and pull mistral
brew install ollama
ollama serve &                      # leave this running
ollama pull mistral                 # 4.4 GB download — one time

# 3. Index the knowledge base under rag_docs/
python scripts/setup_rag.py         # ~1 min on Apple silicon
```

### Knowledge-base layout

```
rag_docs/
├── clinical/   ← shown to clinicians (AHA guideline, STRONG-HF, drug monographs)
├── patient/    ← shown to patients   (plain-language guides, symptom trackers)
└── shared/     ← shown to both       (optional)
```

Drop additional `*.pdf` or `*.docx` files into the right subfolder and re-run `python scripts/setup_rag.py --force` to re-index.

### Talking to the assistant

1. Launch the API + website (see "Optional: real Python engine" above).
2. Open **http://localhost:8901/#/engine**, pick a preset (or edit any field), click **Compute**. The latest patient state is cached in `localStorage`.
3. Scroll down to the **TIDE-HF Assistant** chat (or click "Chat" in the navbar).
4. Toggle **Patient** ⇄ **Clinician** in the chat header to switch tone and retrieval scope (clinical PDFs vs patient-friendly leaflets).
5. The chat header shows the loaded patient name + any `global_stop` / `order_labs` banners. Ask things like:
   - *"Why was my spironolactone reduced?"* (patient mode)
   - *"AHA 2022 threshold for MRA dose reduction in hyperkalemia?"* (clinician mode)
   - *"What does the engine recommend for this patient and why?"*

Without a running Python API, the chat falls back to a **deterministic offline summary** built from the engine state (no LLM, but still patient-aware). The header pill turns amber to indicate offline mode.

### Groq fallback (when Ollama is down)

If `GROQ_API_KEY` is set in the environment, the RAG layer automatically routes the same retrieved-chunks-plus-engine-context prompt through Groq's hosted Llama-3.3 whenever Ollama returns an error or is offline. Free key from [console.groq.com](https://console.groq.com).

```bash
export GROQ_API_KEY=gsk_...           # one line in your shell
python scripts/run_api.py             # restart the API
```

`GET /health` will report `"groq_fallback": true` so you know the safety net is active. Optional knob: `TIDE_GROQ_MODEL` (default `llama-3.3-70b-versatile`).

### Troubleshooting

- **`ggml_metal_init: failed to initialize the Metal library` / "llama runner process has terminated"** — Ollama 0.22.1 has a known incompatibility with macOS Tahoe (Darwin 25.x). Until a newer Ollama release lands, set `GROQ_API_KEY` (above) and the chat will keep working through Groq's hosted Llama. You can also swap models with `TIDE_OLLAMA_MODEL=llama3.2` or point `TIDE_OLLAMA_URL` at any OpenAI-compatible local runtime that exposes `/api/generate`.
- **`segfault` when starting `scripts/run_api.py`** — this is fixed by [scripts/run_api.py](scripts/run_api.py)'s pinned env vars (`TOKENIZERS_PARALLELISM=false`, `OMP_NUM_THREADS=1`, `LOKY_MAX_CPU_COUNT=1`); make sure you launch the server through that script, not raw `uvicorn`.
- **`Knowledge base is empty`** — run `python scripts/setup_rag.py` once after dropping PDFs/DOCX into `rag_docs/`.
- **`Missing RAG dependencies`** — run `pip install -e ".[rag]"`.

---

## Four titration strategies

| name | behavior |
|---|---|
| `traditional` | One new class per week · RAAS → BB → MRA → SGLT2i → loop · single-rung steps |
| `strong_hf` | All eligible classes started at once · double-rung up-titration |
| `rapid_sequence` | All eligible classes at once · single-rung steps (Greene 2021) |
| `sglt_mra_first` | Phase 1: SGLT2i + MRA · Phase 2: add ARNi + beta-blocker |

Pick the strategy at the top of either the Streamlit UI or the web Engine console before clicking **Compute**.

---

## Two data-generation modes

The included `data/synthetic_CHF_visits_v2.parquet` (135 MB) is a pre-built snapshot of 137k MIMIC-IV CHF visits with real demographics, baseline labs, GDMT regimens, and ECG-study linkage. The generator uses it by default.

```bash
# MIMIC-seeded (default when the parquet is present)
python scripts/generate_data.py --n 10000 --mode mimic

# Pure distribution: Gaussian-sampled demographics and baselines.
python scripts/generate_data.py --n 10000 --mode distribution

# Auto: "mimic" if data/synthetic_CHF_visits_v2.parquet exists, else "distribution".
python scripts/generate_data.py --n 10000 --mode auto
```

| aspect | `distribution` | `mimic` |
|---|---|---|
| demographics (age, gender) | Gaussian / uniform | sampled from real CHF visits |
| baseline labs (K, Na, Cr, eGFR) | independent Gaussians | extracted from real lab events |
| GDMT med state | random assignment, 35%/class | copied from real MIMIC prescriptions |
| vital trajectories + labels | synthesized from sampled labels | same |
| external data needed | none | `data/synthetic_CHF_visits_v2.parquet` |

---

## Preset scenarios in the UIs

Seventeen presets cover the main decision paths.

**Titration-state showcases** — `Newly diagnosed, stable` · `Partial titration, no AEs` · `Fully titrated, at target`

**Adverse-effect showcases** — `Suspected hyperkalemia` · `Confirmed hyperkalemia (K=6.3)` · `Volume depletion + hypotension` · `Worsening HF` · `Bradycardia on beta blocker` · `Worsening HF + hyperkalemia (labs confirmed)` · `Hyponatremia + renal dysfunction (labs)`

**Contraindication showcases** — `Angioedema history (ARNi + ACEi blocked)` · `Severe asthma (beta blocker blocked)` · `CKD stage 4 (eGFR=22)` · `Pregnant (RAAS blocked)` · `AV block (beta blocker blocked)` · `Severe hypotension (global stop)`

In the web Engine console you can pick any preset as a starting point, then **edit any field** — vitals, baseline labs, recent labs (toggle on/off), per-class meds + dose, contraindications — and click Compute.

---

## Project layout

```
TIDE-HF/
├── src/chf_titration/
│   ├── constants.py       # class maps, rhythms, drug reps
│   ├── featurize.py       # featurize_week: 14 timepoints → 108 features
│   ├── synthesize.py      # synthesize_week + 17 preset patients
│   ├── engine.py          # TitrationEngine v1.2 + contraindications
│   ├── strategy.py        # apply_strategy: 4 strategies
│   ├── classifier.py      # train_classifier, load_bundle, predict_flags
│   ├── api.py             # FastAPI backend (engine + /chat RAG)
│   ├── rag.py             # ChromaDB + sentence-transformers + Ollama bridge
│   ├── rag_ui.py          # Drop-in Streamlit "💬 Ask AI" tab
│   ├── ui.py              # Streamlit app
│   └── data/              # reference CSVs (GDMT, ICDs, lab maps)
├── scripts/
│   ├── generate_data.py
│   ├── train.py
│   ├── prepare.py         # one-shot: generate + train (idempotent)
│   ├── run_ui.py          # auto-prepares bundle on first launch
│   ├── run_api.py         # FastAPI server on :8000
│   ├── run_mcp.py
│   └── setup_rag.py       # index PDFs/DOCX in rag_docs/ → rag_db/
├── data/                  # synthetic parquets land here
├── models/                # trained bundles land here (chf_classifier_lgbm.pkl)
├── rag_docs/              # clinical/, patient/, shared/ — the RAG knowledge base
├── rag_db/                # ChromaDB persistent store (gitignored)
├── synthetic_data/        # standalone synthetic-CSV generators
├── tests/
├── website/               # Vite + React + shadcn-ui app (landing + Engine console)
├── .github/workflows/     # GitHub Pages deploy workflow
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

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

---

## Deploying the website to GitHub Pages

A workflow at [.github/workflows/deploy.yml](.github/workflows/deploy.yml) builds and publishes `website/` on every push to `main`.

One-time repo setup:

1. **Settings → Pages → Build and deployment → Source**: select **GitHub Actions**.
2. (Repo must be **public** unless you're on GitHub Pro/Team/Enterprise.)
3. Push to `main` — the workflow auto-runs.

The site goes live at **https://ramaseshu0.github.io/TIDE-HF/**. Hash-based routing means deep links (`#/engine`) and refresh both work without server-side SPA fallback. GitHub Pages can't run Python, so the deployed Engine page automatically uses the in-browser engine — to drive the deployed site with the real Python engine, host `scripts/run_api.py` somewhere reachable and rebuild with `VITE_TIDE_API=https://your-api npm run build`.

---

## Distributing the 135 MB MIMIC parquet and trained bundle

`data/synthetic_CHF_visits_v2.parquet` is 135 MB — plain git will complain. Two clean options:

- **git-lfs** — `git lfs install && git lfs track "*.parquet" "*.pkl" && git add .gitattributes data/synthetic_CHF_visits_v2.parquet models/chf_classifier_lgbm.pkl`, then commit normally.
- **Side-channel** — keep the parquet/pickle out of git (`.gitignore` them) and ship them alongside the repo via a download link. The package auto-detects the parquet and falls back to `distribution` mode if absent. If the pickle is absent, `scripts/run_ui.py` will train one on first launch.

The trained pickle is ~5–15 MB depending on `n_estimators` and dataset size — small enough to commit without LFS in most cases.

---

## License

MIT
