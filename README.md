# TIDE-HF

**Trajectory · Integrated · Decision · Engine** — a local-first, guideline-directed AI pipeline for chronic heart-failure (CHF) titration.

A complete clinical-decision-support stack built around four pillars:

| | | |
|---|---|---|
| **T**rajectory | 14 timepoints / week → 108 features | vitals, weight drift, rhythm, SpO₂, ECG signs |
| **I**ntegrated | demographics + baseline labs + current GDMT + ECG-study linkage | MIMIC-IV seeded or pure-distribution mode |
| **D**ecision | LightGBM classifier predicts 11 adverse-effect flags | calibrated probabilities |
| **E**ngine | rule-based TitrationEngine v1.2, lab-gated, contraindication-aware | every action carries an auditable reason |

The system runs entirely on your laptop — no MIMIC dataset required, no API keys required for the core pipeline. The TIDE-HF Assistant chat additionally uses local RAG (ChromaDB + sentence-transformers + Ollama), with an optional Groq fallback for higher answer quality.

🌐 **Live website:** https://ramaseshu0.github.io/TIDE-HF/
📂 **Repository:** https://github.com/Ramaseshu0/TIDE-HF

> Affiliated with **QAS.AI**. For clinical decision-support research and education only. Not a medical device.

---

## Table of contents

1. [Architecture at a glance](#architecture-at-a-glance)
2. [Quick start](#quick-start) — macOS / Windows / Linux
3. [Daily run — three terminals](#daily-run--three-terminals)
4. [The Streamlit UI](#the-streamlit-ui)
5. [The Website (Vite + React + shadcn-ui)](#the-website-vite--react--shadcn-ui)
6. [The TIDE-HF Assistant (local RAG)](#the-tide-hf-assistant-local-rag)
7. [Project layout — what every file does](#project-layout--what-every-file-does)
8. [API reference (FastAPI backend)](#api-reference-fastapi-backend)
9. [Using the library in your own code](#using-the-library-in-your-own-code)
10. [The four titration strategies](#the-four-titration-strategies)
11. [Two data-generation modes](#two-data-generation-modes)
12. [Preset scenarios](#preset-scenarios)
13. [Deploying the website to GitHub Pages](#deploying-the-website-to-github-pages)
14. [Troubleshooting](#troubleshooting)
15. [License](#license)

---

## Architecture at a glance

```
┌──────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│   Synthetic-data generator ──► LightGBM AE classifier (11 flags)         │
│           │                              │                               │
│           ▼                              ▼                               │
│   TitrationEngine v1.2  ──►  Strategy applier  ──►  per-class actions    │
│           │                              │                               │
│           └──────► engine_context ───────┘                               │
│                          │                                               │
│                          ▼                                               │
│              ChromaDB retrieval (606 chunks)                             │
│                          │                                               │
│                          ▼                                               │
│              Ollama (qwen3.5:4b) ──or──► Groq (Llama-3.3-70B)            │
│                          │                                               │
│                          ▼                                               │
│              Patient- or clinician-tuned RAG answer                      │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

Three runtimes:

- **Streamlit UI** — `python scripts/run_ui.py` on http://localhost:8501. Self-contained app for analysts.
- **FastAPI backend** — `python scripts/run_api.py` on http://127.0.0.1:8000. Exposes `/evaluate` and `/chat` to the website.
- **Website** — `cd website && npm run dev` on http://localhost:8901. Landing page + editable Engine console + RAG chat. Falls back to a browser-side engine and rule-based summary when the API is offline.

---

## Quick start

The project runs on **macOS**, **Windows**, and **Linux**. You'll need:

- **Python 3.10+** ([python.org/downloads](https://www.python.org/downloads/))
- **Node.js 20+** ([nodejs.org/en/download](https://nodejs.org/en/download/))
- **Ollama 0.23+** ([ollama.com/download](https://ollama.com/download)) — versions older than 0.23 crash on macOS Tahoe (Darwin 25.x) due to a Metal SDK incompatibility; on Windows/Linux this isn't an issue but the latest is still recommended.
- **Git** ([git-scm.com/downloads](https://git-scm.com/downloads))

### Install (cross-platform)

<details open>
<summary><b>macOS</b> — Homebrew (recommended)</summary>

```bash
brew install python@3.14 node ollama git

git clone https://github.com/Ramaseshu0/TIDE-HF.git
cd TIDE-HF

python3.14 -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install -e ".[ui,rag]"
```

> macOS ships with Python 3.9 in `/Library/Developer/CommandLineTools`. Make sure your venv uses 3.10+; verify with `python3 --version` after activating.

</details>

<details>
<summary><b>Windows</b> — PowerShell</summary>

After installing Python 3.10+, Node 20+, Ollama, and Git from the links above:

```powershell
git clone https://github.com/Ramaseshu0/TIDE-HF.git
cd TIDE-HF

py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install -U pip
pip install -e ".[ui,rag]"
```

> If PowerShell blocks `Activate.ps1`, run once: `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`.
> If you prefer `cmd.exe`, activate with `.\.venv\Scripts\activate.bat` instead.

</details>

<details>
<summary><b>Linux</b> — apt / dnf / pacman</summary>

```bash
# Ubuntu / Debian
sudo apt update && sudo apt install -y python3.12 python3.12-venv nodejs npm git
curl -fsSL https://ollama.com/install.sh | sh

git clone https://github.com/Ramaseshu0/TIDE-HF.git
cd TIDE-HF

python3.12 -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install -e ".[ui,rag]"
```

</details>

### One-time prep (all platforms)

With the venv activated:

```bash
# 1. Generate 10k synthetic patient-weeks + train the LightGBM classifier (~2 min)
python scripts/prepare.py

# 2. Pull a local LLM
ollama pull qwen3.5:4b              # 3.4 GB, project default
# ollama pull mistral               # 4.4 GB, alternative (also works on Ollama 0.23+)
# ollama pull qwen3:14b             # 8.2 GB, higher quality

# 3. Index the clinical PDFs + DOCX in rag_docs/  (~606 chunks → rag_db/)
python scripts/setup_rag.py
```

That's it for setup. Everything below is the day-to-day run.

---

## Daily run — three terminals

The same three commands on every platform — just swap `source .venv/bin/activate` for `.\.venv\Scripts\Activate.ps1` on Windows.

### Terminal 1 — local LLM runtime (skip if you only use Groq)

```bash
# macOS / Linux
ollama serve

# Windows (PowerShell)
ollama serve
```

### Terminal 2 — Python API (engine + RAG /chat)

```bash
# macOS / Linux
cd ~/Documents/TIDE-HF
source .venv/bin/activate
export GROQ_API_KEY=gsk_...         # optional but recommended (see "Picking a model")
python scripts/run_api.py           # http://127.0.0.1:8000
```

```powershell
# Windows (PowerShell)
cd $HOME\Documents\TIDE-HF
.\.venv\Scripts\Activate.ps1
$env:GROQ_API_KEY = "gsk_..."        # optional but recommended
python scripts/run_api.py
```

### Terminal 3 — website

```bash
# macOS / Linux
cd ~/Documents/TIDE-HF/website
npm install                         # first time only
npm run dev                         # http://localhost:8901 (or 8902 if 8901 is busy)
```

```powershell
# Windows (PowerShell)
cd $HOME\Documents\TIDE-HF\website
npm install                         # first time only
npm run dev
```

Open **http://localhost:8901/#/engine**, pick a preset, edit any field, click **Compute**, then scroll to the chat and ask away.

### Streamlit UI only (no website / no API)

```bash
python scripts/run_ui.py            # http://localhost:8501
```

The Streamlit UI is fully self-contained — it runs the engine in-process. RAG and Groq integration work there via the same env vars.

---

## The Streamlit UI

`python scripts/run_ui.py` → http://localhost:8501

- Auto-prepares the model bundle on first launch.
- 17 preset scenarios covering titration-state, adverse-effect, and contraindication paths.
- Per-timepoint CSV editing or summary-stats editing.
- "💬 Ask AI" tab driven by the same RAG engine ([src/chf_titration/rag_ui.py](src/chf_titration/rag_ui.py)).
- No internet required for the engine; chat uses local Ollama by default.

---

## The Website (Vite + React + shadcn-ui)

`cd website && npm run dev` → http://localhost:8901

| Route | Purpose |
|---|---|
| `#/` | Landing page — hero, About, How it works, Strategies, Chat, Footer |
| `#/engine` | Editable Engine console — vitals, labs, meds, contraindications, strategy → Compute → live engine output |

**Two execution modes** (auto-detected via `/health` probe):

- **Python engine (green pill)** — when `scripts/run_api.py` is reachable. Real LightGBM classifier + `TitrationEngine`.
- **Browser engine (cyan pill)** — fallback for static deployments (GitHub Pages). In-browser TypeScript engine produces the same `EvaluateResponse` shape.

The TIDE-HF Assistant chat works in both modes — Python mode does live RAG against the indexed PDFs, browser mode produces a deterministic patient-aware summary.

Other useful npm scripts:

- `npm run build` — production bundle (auto-uses `/TIDE-HF/` base for GitHub Pages)
- `npm run preview` — serve the production build locally
- `npm test` — vitest

---

## The TIDE-HF Assistant (local RAG)

The chat is a real retrieval-augmented LLM, not a canned-reply demo. It indexes the clinical PDFs you put in `rag_docs/` and answers questions grounded in the **currently loaded patient** (labs, vitals, meds, contraindications, classifier flags, engine actions).

### Stack

| Component | Purpose | Footprint |
|---|---|---|
| **ChromaDB** | Vector store (persistent) | ~50 MB on disk |
| **all-MiniLM-L6-v2** | Embedding model (CPU) | ~80 MB, downloads on first run |
| **Ollama + qwen3.5:4b** | Generation (default) | ~3.4 GB, downloads with `ollama pull` |
| **Groq + Llama-3.3-70B** | Optional fallback / higher-quality generation | hosted, free key |

### Knowledge-base layout

```
rag_docs/
├── clinical/   ← shown to clinicians  (AHA 2022 guideline, STRONG-HF, drug monographs)
├── patient/    ← shown to patients    (plain-language leaflets, symptom trackers)
└── shared/     ← shown to both        (optional)
```

Currently indexed: **20 documents → 606 chunks** (8 clinical + 11 patient + 1 shared). Drop more `*.pdf` or `*.docx` into the appropriate subfolder and re-run `python scripts/setup_rag.py --force` to re-index.

### Picking a model — answer-quality matters

Small local models occasionally hallucinate labs or refuse to answer. For real clinical-quality answers, pick the best option for your setup:

| Option | Disk / network | Quality |
|---|---|---|
| **qwen3.5:4b** (default) | 3.4 GB local | OK for quick replies, occasionally hallucinates |
| **qwen3:14b** | 8.2 GB local | Much better at following instructions and citing exact patient numbers |
| **mistral** | 4.4 GB local | Solid generalist (works on Ollama 0.23+; older Ollama crashes on macOS Tahoe) |
| **Groq Llama-3.3-70B** | hosted, free key | Best — fast, cites numbers precisely; auto-used as fallback when Ollama errors |

To switch model:

```bash
# Bigger local model
ollama pull qwen3:14b
export TIDE_OLLAMA_MODEL=qwen3:14b      # PowerShell: $env:TIDE_OLLAMA_MODEL = "qwen3:14b"

# Or: highest quality via Groq (free key)
open https://console.groq.com/keys      # macOS — on Windows just visit the URL
export GROQ_API_KEY=gsk_...
```

### How a question flows through the system

```
User edits patient on /engine ─► Compute ─► POST /evaluate
                                              │
                                              └─► localStorage cache
                                                       │
User asks question in chat ◄─────────────────────────┘
       │
       ▼
POST /chat { question, audience, patient, result }
       │
       ▼
build_engine_context(...)  →  multi-line plain-text summary of:
                                  • patient demographics
                                  • global_stop / order_labs status
                                  • active classifier flags
                                  • labs vs baseline (with deltas)
                                  • preferred RAAS, per-class engine actions
       │
       ▼
retrieval query  =  question + salient engine-context lines
       │
       ▼
ChromaDB top-10 chunks (filtered by audience: clinical | patient | shared)
       │
       ▼
LLM prompt  =  system_prompt
            +  <patient_data>engine_context</patient_data>
            +  <guidelines>retrieved_chunks</guidelines>
            +  <question>user_question</question>
            +  "Reference at least two patient numbers in your answer..."
       │
       ▼
Ollama (qwen3.5:4b, think:false, temp 0.45, top_p 0.9, repeat_penalty 1.15)
       │  → on failure ↓
Groq (Llama-3.3-70B, same temp/top_p) ─if GROQ_API_KEY set
       │
       ▼
Answer rendered in the chat with source label ("via local RAG" or "via Groq")
```

### Tunable env vars

| Variable | Default | What it does |
|---|---|---|
| `TIDE_OLLAMA_MODEL` | `qwen3.5:4b` | Which Ollama model to call |
| `TIDE_OLLAMA_URL` | `http://localhost:11434` | Where Ollama listens |
| `TIDE_GROQ_MODEL` | `llama-3.3-70b-versatile` | Which Groq model to use as fallback |
| `TIDE_LLM_TEMPERATURE` | `0.45` | ↑ for more variation, ↓ for more boilerplate |
| `TIDE_LLM_TOP_P` | `0.9` | Nucleus-sampling cutoff |
| `TIDE_LLM_MAX_TOKENS` | `900` | Max tokens to generate |
| `TIDE_RETRIEVE_K` | `10` | How many guideline chunks to retrieve |
| `TIDE_RAG_DB` | `<repo>/rag_db` | Override the ChromaDB location |
| `TIDE_RAG_DOCS` | `<repo>/rag_docs` | Override the docs folder |
| `TIDE_EMBED_MODEL` | `all-MiniLM-L6-v2` | Override the embedding model |
| `GROQ_API_KEY` | _(unset)_ | Enables Groq fallback when set |

Set on macOS/Linux: `export NAME=value`
Set on Windows PowerShell: `$env:NAME = "value"`

---

## Project layout — what every file does

```
TIDE-HF/
├── README.md                                   ← you are here
├── pyproject.toml                              ← Python package metadata + [ui,rag,mcp,dev] extras
├── requirements.txt                            ← pin-free deps (mirrors pyproject)
├── .gitignore                                  ← venvs, caches, rag_db/, tide_rag/, rag_knowledge_base/
│
├── src/chf_titration/                          ← Python package (installed as `chf_titration`)
│   │
│   ├── __init__.py                             ← package marker
│   ├── constants.py                            ← drug-class enum, rhythm strings, drug rep names
│   ├── synthesize.py                           ← synthesize_week(patient) → 14 timepoints
│   │                                             + PATIENTS dict (17 presets)
│   ├── data_gen.py                             ← full-batch synthetic-week generator
│   ├── featurize.py                            ← featurize_week(tps) → 108-feature vector
│   ├── classifier.py                           ← train_classifier, load_bundle, predict_flags
│   │                                             (LightGBM, 11 adverse-effect heads)
│   ├── engine.py                               ← TitrationEngine v1.2 — lab-gated, contraindication-aware
│   │                                             + derive_contraindications(patient)
│   ├── strategy.py                             ← apply_strategy: traditional / strong_hf /
│   │                                             rapid_sequence / sglt_mra_first
│   │                                             + TARGETS, LADDERS, step_up, step_down
│   │
│   ├── api.py                                  ← FastAPI app
│   │                                             • POST /evaluate    → run engine on a patient
│   │                                             • POST /chat        → RAG-grounded chat answer
│   │                                             • GET  /presets     → list PATIENTS dict
│   │                                             • GET  /health      → engine + RAG status
│   │
│   ├── rag.py                                  ← TideRAG class
│   │                                             • setup(docs_dir)   → chunk + embed + store
│   │                                             • ask(q, ctx, audience) → retrieve + call LLM
│   │                                             • _call_ollama / _call_groq fallback chain
│   │                                             + build_engine_context(...) helper
│   │
│   ├── rag_ui.py                               ← Drop-in 💬 Ask AI tab for Streamlit
│   ├── ui.py                                   ← Streamlit app (full clinical UI)
│   ├── cli.py                                  ← entry points exposed as `chf-*` console scripts
│   ├── mcp_server.py                           ← MCP (Model Context Protocol) server for IDE agents
│   │
│   └── data/                                   ← reference CSVs (GDMT, ICD codes, lab maps)
│
├── scripts/                                    ← user-facing runnable entry points
│   ├── prepare.py                              ← one-shot: generate + train (idempotent)
│   ├── generate_data.py                        ← synthetic-data generator only
│   ├── train.py                                ← classifier training only
│   ├── setup_rag.py                            ← index PDFs/DOCX in rag_docs/ → rag_db/
│   ├── run_ui.py                               ← Streamlit launcher (auto-preps bundle)
│   ├── run_api.py                              ← FastAPI launcher with macOS-safe env defaults
│   └── run_mcp.py                              ← MCP server launcher
│
├── data/                                       ← synthetic parquets
│   ├── synthetic_CHF_visits_v2.parquet         ← 135 MB MIMIC-IV seeded source (LFS-friendly)
│   └── synthetic_patient_weeks.parquet         ← generated training set (regenerated by prepare.py)
│
├── models/                                     ← trained model bundles
│   └── chf_classifier_lgbm.pkl                 ← pickled LightGBM bundle (~2 MB)
│
├── rag_docs/                                   ← RAG knowledge base (audience-segregated)
│   ├── clinical/                               ← 8 PDFs/DOCX for clinician answers
│   ├── patient/                                ← 11 PDFs/DOCX for patient answers
│   └── shared/                                 ← 1 DOCX shown to both
│
├── rag_db/                                     ← ChromaDB persistent store (auto-created, gitignored)
├── synthetic_data/                             ← standalone CSV generators (legacy)
├── er_diagram/                                 ← DB schema reference for the synthetic data
├── tests/
│   └── test_smoke.py                           ← engine + strategy smoke test
│
├── .github/workflows/
│   └── deploy.yml                              ← builds website/ on every push → GitHub Pages
│
└── website/                                    ← Vite + React + shadcn-ui app
    ├── package.json                            ← scripts: dev, build, preview, lint, test
    ├── vite.config.ts                          ← port 8901, base="/TIDE-HF/" in production
    ├── tailwind.config.ts
    ├── index.html
    ├── public/
    │   ├── favicon.ico
    │   ├── qas-ai.png                          ← QAS.AI affiliation logo (top-right of every page)
    │   └── .nojekyll                           ← keeps GitHub Pages from running Jekyll
    │
    └── src/
        ├── main.tsx                            ← React bootstrap + scrollRestoration override
        ├── App.tsx                             ← HashRouter + ScrollToTop + 3 routes
        ├── index.css                           ← Tailwind layers + design tokens
        │
        ├── pages/
        │   ├── Index.tsx                       ← landing page composition
        │   ├── Engine.tsx                      ← editable Engine console (every field editable,
        │   │                                     calls evaluatePatient(), persists via rememberPatient)
        │   └── NotFound.tsx
        │
        ├── components/
        │   ├── Navbar.tsx                      ← top nav with QAS.AI affiliation badge
        │   ├── Hero.tsx                        ← landing hero section
        │   ├── About.tsx                       ← four-pillar (T·I·D·E) explanation
        │   ├── HowItWorks.tsx                  ← pipeline steps + Four-Strategies section (id="strategies")
        │   ├── ChatBot.tsx                     ← TIDE-HF Assistant — calls /chat or offline summary
        │   ├── Footer.tsx
        │   ├── ScrollToTop.tsx                 ← resets scroll on every route change
        │   ├── NavLink.tsx
        │   └── ui/                             ← shadcn-ui primitives (button, dialog, switch, …)
        │
        ├── lib/
        │   ├── utils.ts                        ← cn() class merger
        │   ├── tide-engine.ts                  ← in-browser TS engine + 17 presets + flag rules
        │   │                                     (fallback for GitHub Pages — no Python required)
        │   └── tide-api.ts                     ← API client: evaluatePatient, askChat, apiHealth,
        │                                         rememberPatient/recallPatient (localStorage),
        │                                         deterministic offline-summary builder
        │
        ├── assets/                             ← hero / trajectory / decision-engine images
        ├── hooks/                              ← React custom hooks
        └── test/                               ← vitest setup
```

---

## API reference (FastAPI backend)

Base URL: `http://127.0.0.1:8000`

### `GET /health`

```json
{
  "ok":            true,
  "bundle_loaded": true,
  "rag_loaded":    true,
  "rag_chunks":    606,
  "groq_fallback": true
}
```

### `GET /presets`

Returns the full `PATIENTS` dict from `synthesize.py` — 17 preset patient objects ready to feed into `/evaluate`.

### `POST /evaluate`

Runs the engine on a patient payload.

```jsonc
// Request
{
  "patient":  { /* full patient object — vitals, labs, meds, contras, ... */ },
  "strategy": "strong_hf",                     // traditional | strong_hf | rapid_sequence | sglt_mra_first
  "labs":     { "K": 6.3, "Na": 138, ... } | null
}

// Response
{
  "flags":           { "hyperkalemia_detected": true, ... },
  "probs":           { "hyperkalemia_detected": 0.92, ... },
  "global_stop":     true,
  "order_labs":      false,
  "labs_requested":  [],
  "preferred_raas":  "ARNi",
  "decisions":       { "MRA": { "action": "hold_titration", "reason": "global_stop" }, ... },
  "changes":         { "MRA": { "concrete_action": "hold", "current": 25, "new_dose": 25, "target": 50, ... }, ... },
  "contras_derived": { ... }
}
```

### `POST /chat`

RAG-grounded chat answer, optionally personalized to the patient.

```jsonc
// Request
{
  "question": "Why was my spironolactone reduced this visit?",
  "audience": "patient",                       // patient | clinician
  "strategy": "strong_hf",
  "patient":  { ... } | null,                  // if omitted, answer is from guidelines only
  "labs":     { ... } | null,
  "result":   { ... } | null,                  // optional — cached /evaluate output (skips re-eval)
  "changes":  { ... } | null,
  "flags":    { ... } | null
}

// Response
{
  "answer":         "Your potassium of **6.3 mEq/L** is above the safety cutoff of 6.0 mEq/L, so the engine ...",
  "engine_context": "Patient: 72y M  |  HFrEF NYHA C/D\nSTATUS: GLOBAL STOP ...",
  "audience":       "patient"
}
```

---

## Using the library in your own code

```python
from chf_titration.synthesize import synthesize_week, PATIENTS
from chf_titration.classifier import load_bundle, predict_flags
from chf_titration.engine import TitrationEngine
from chf_titration.strategy import apply_strategy, TARGETS

bundle = load_bundle("models/chf_classifier_lgbm.pkl")
engine = TitrationEngine()

patient = {**PATIENTS["Newly diagnosed, stable"], "targets": TARGETS}
tps = synthesize_week(patient)

flags, probs = predict_flags(patient, tps, bundle)
result  = engine.evaluate(patient, tps, flags, labs=None, awaiting_labs=set())
changes = apply_strategy(result, patient, strategy="strong_hf")

for cls, change in changes.items():
    print(f"{cls:<14} {change['concrete_action']:<22}  {change['current']} → {change['new_dose']} mg")
```

---

## The four titration strategies

| Strategy | Behavior |
|---|---|
| `traditional` | One new class per week · RAAS → BB → MRA → SGLT2i → loop · single-rung steps |
| `strong_hf` | All eligible classes started at once · double-rung up-titration (default) |
| `rapid_sequence` | All eligible classes at once · single-rung steps (Greene 2021) |
| `sglt_mra_first` | Phase 1: SGLT2i + MRA · Phase 2: add ARNi + beta-blocker |

Pick the strategy at the top of either the Streamlit UI or the web Engine console before clicking **Compute**.

---

## Two data-generation modes

The included `data/synthetic_CHF_visits_v2.parquet` (135 MB) is a pre-built snapshot of 137k MIMIC-IV CHF visits with real demographics, baseline labs, GDMT regimens, and ECG-study linkage. The generator uses it by default.

```bash
python scripts/generate_data.py --n 10000 --mode mimic           # default when parquet is present
python scripts/generate_data.py --n 10000 --mode distribution    # pure Gaussian, no external data
python scripts/generate_data.py --n 10000 --mode auto            # mimic if parquet present, else distribution
```

| aspect | `distribution` | `mimic` |
|---|---|---|
| demographics (age, gender) | Gaussian / uniform | sampled from real CHF visits |
| baseline labs (K, Na, Cr, eGFR) | independent Gaussians | extracted from real lab events |
| GDMT med state | random assignment, 35%/class | copied from real MIMIC prescriptions |
| vital trajectories + labels | synthesized from sampled labels | same |
| external data needed | none | `data/synthetic_CHF_visits_v2.parquet` |

---

## Preset scenarios

17 presets, organized by decision path:

**Titration-state showcases** — `Newly diagnosed, stable` · `Partial titration, no AEs` · `Fully titrated, at target`

**Adverse-effect showcases** — `Suspected hyperkalemia` · `Confirmed hyperkalemia (K=6.3)` · `Volume depletion + hypotension` · `Worsening HF` · `Bradycardia on beta blocker` · `Worsening HF + hyperkalemia (labs confirmed)` · `Hyponatremia + renal dysfunction (labs)`

**Contraindication showcases** — `Angioedema history (ARNi + ACEi blocked)` · `Severe asthma (beta blocker blocked)` · `CKD stage 4 (eGFR=22)` · `Pregnant (RAAS blocked)` · `AV block (beta blocker blocked)` · `Severe hypotension (global stop)`

On the web Engine console you can pick any preset as a starting point, then **edit any field** (vitals, baseline labs, recent labs, meds, contraindications) before clicking Compute.

---

## Deploying the website to GitHub Pages

A workflow at [`.github/workflows/deploy.yml`](.github/workflows/deploy.yml) builds and publishes `website/` on every push to `main`.

One-time repo setup:

1. **Settings → Pages → Build and deployment → Source**: select **GitHub Actions**.
2. (Repo must be **public** unless you're on GitHub Pro/Team/Enterprise.)
3. Push to `main` — the workflow auto-runs.

The site goes live at https://ramaseshu0.github.io/TIDE-HF/. Hash-based routing means deep links (`#/engine`) and refresh both work without a server-side SPA fallback.

GitHub Pages can't run Python, so the deployed Engine page automatically uses the in-browser engine, and the chat uses the deterministic offline summary. To drive the deployed site with the real Python engine + RAG, host `scripts/run_api.py` somewhere reachable and rebuild with:

```bash
VITE_TIDE_API=https://your-api.example.com npm run build
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Chat says *"I can't reach the AI backend"* | Terminal 2 (the API) isn't running. Start `python scripts/run_api.py`. |
| `⚠ Local LLM is not reachable (500 …)` | `ollama serve` isn't running, **or** you're on Ollama 0.22.x on macOS Tahoe (Metal SDK bug — upgrade with `brew upgrade ollama` to 0.23+ or set `GROQ_API_KEY` to bypass). |
| `ggml_metal_init: failed to initialize the Metal library` | Pre-0.23 Ollama on macOS Tahoe (Darwin 25.x). Run `brew upgrade ollama` — 0.23.3+ fixes this. |
| `"Knowledge base is empty"` | Run `python scripts/setup_rag.py` once. |
| Chat hallucinates labs / refuses to answer | qwen3.5:4b is 4B params. Either `ollama pull qwen3:14b && export TIDE_OLLAMA_MODEL=qwen3:14b`, or set `GROQ_API_KEY`. |
| Segfault when starting `scripts/run_api.py` (macOS) | Always launch through `scripts/run_api.py` — it pins `TOKENIZERS_PARALLELISM=false`, `OMP_NUM_THREADS=1`, `LOKY_MAX_CPU_COUNT=1` and forces `spawn()`. Raw `uvicorn …` will fork-segfault under sentence-transformers on macOS Python 3.14. |
| `Missing RAG dependencies` | `pip install -e ".[rag]"` |
| Edits to `rag.py` / `api.py` aren't taking effect | The venv's editable install may point at a different copy. Check `cat .venv/lib/python3.14/site-packages/__editable__.chf_titration-0.1.0.pth` (or the Windows equivalent under `.venv\Lib\site-packages`). If wrong, `pip install --force-reinstall --no-deps -e .` from this folder. |
| `Port 8901 is in use` | Vite auto-falls-back to 8902. To kill the stuck process: `lsof -i :8901` (macOS/Linux) or `netstat -ano \| findstr :8901` (Windows). |
| `npm run build` produces stale assets | `rm -rf website/dist && npm run build` (macOS/Linux) or `rmdir /s /q website\dist && npm run build` (Windows). |
| `bundle_loaded: false` on `/health` | The classifier lazy-loads on first `/evaluate`; this is normal. |
| `PowerShell blocks Activate.ps1` (Windows) | Run once: `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`. |
| Ollama can't find a downloaded model | `ollama list` to confirm; if missing, `ollama pull <name>`. Models live under `~/.ollama/models` (macOS/Linux) or `%USERPROFILE%\.ollama\models` (Windows). |

---

## License

MIT
