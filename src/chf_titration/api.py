"""FastAPI backend exposing the TIDE-HF Python engine to the website.

GET  /presets       → list of preset patient dicts (from synthesize.PATIENTS)
POST /evaluate      → run classifier + engine + strategy on a patient payload
POST /chat          → patient-aware RAG answer (ChromaDB + Ollama)
GET  /health        → backend status (engine + RAG readiness)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from chf_titration.classifier import load_bundle, predict_flags
from chf_titration.engine import TitrationEngine, derive_contraindications
from chf_titration.strategy import TARGETS, apply_strategy
from chf_titration.synthesize import PATIENTS, synthesize_week

BUNDLE_PATH = Path(__file__).resolve().parents[2] / "models" / "chf_classifier_lgbm.pkl"

app = FastAPI(title="TIDE-HF API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_bundle = None
_engine = TitrationEngine()


def _get_bundle():
    global _bundle
    if _bundle is None:
        _bundle = load_bundle(BUNDLE_PATH)
        if _bundle is None:
            raise HTTPException(503, f"classifier bundle missing at {BUNDLE_PATH}; run scripts/prepare.py")
    return _bundle


class EvaluateRequest(BaseModel):
    patient: dict[str, Any]
    strategy: str = "strong_hf"
    labs: dict[str, float] | None = None


@app.get("/presets")
def presets() -> list[dict[str, Any]]:
    return [{"name": name, **p} for name, p in PATIENTS.items()]


@app.post("/evaluate")
def evaluate(req: EvaluateRequest) -> dict[str, Any]:
    bundle = _get_bundle()
    patient = {**req.patient, "targets": TARGETS}
    contras_derived = derive_contraindications(patient)
    patient = {**patient, "contras": {**patient.get("contras", {}), **contras_derived}}

    tps = synthesize_week(patient)
    flags, probs = predict_flags(patient, tps, bundle)

    labs = req.labs if req.labs else patient.get("labs")
    result = _engine.evaluate(patient, tps, flags, labs=labs, awaiting_labs=set())
    changes = apply_strategy(result, patient, strategy=req.strategy)

    return {
        "flags": flags,
        "probs": probs,
        "global_stop": result["global_stop"],
        "order_labs": result["order_labs"],
        "labs_requested": sorted(result["labs_requested"]),
        "preferred_raas": result["preferred_raas"],
        "decisions": result["decisions"],
        "changes": changes,
        "contras_derived": contras_derived,
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "bundle_loaded": _bundle is not None,
        "rag_loaded": _rag is not None,
        "rag_chunks": _rag.chunk_count if _rag is not None else 0,
    }


# ─── RAG / chat ──────────────────────────────────────────────────────────────

_rag = None  # lazily loaded TideRAG


def _get_rag():
    """Lazy-init the RAG engine. Returns None if dependencies aren't installed."""
    global _rag
    if _rag is not None:
        return _rag
    try:
        from chf_titration.rag import TideRAG
        _rag = TideRAG()
        return _rag
    except ImportError as e:
        raise HTTPException(
            503,
            f"RAG dependencies missing: {e}. Run `pip install -e '.[rag]'`.",
        )
    except Exception as e:
        raise HTTPException(503, f"RAG engine failed to initialize: {e}")


class ChatRequest(BaseModel):
    question: str
    audience: str = "patient"           # "patient" | "clinician"
    patient: dict[str, Any] | None = None
    strategy: str = "strong_hf"
    labs: dict[str, float] | None = None
    # if the caller already ran /evaluate they can pass result+changes+flags
    # to skip redoing the engine run for the chat call:
    result: dict[str, Any] | None = None
    changes: dict[str, Any] | None = None
    flags: dict[str, bool] | None = None


@app.post("/chat")
def chat(req: ChatRequest) -> dict[str, Any]:
    rag = _get_rag()

    if rag.chunk_count == 0:
        raise HTTPException(
            503,
            "Knowledge base is empty. Run `python scripts/setup_rag.py` to index PDFs.",
        )

    if req.audience not in ("patient", "clinician"):
        raise HTTPException(400, f"audience must be 'patient' or 'clinician', got {req.audience!r}")

    # Build the engine context if we have a patient. If the caller already
    # ran /evaluate and passed result/changes/flags, reuse them; otherwise
    # run the engine here so the chat is still patient-aware.
    engine_context = "(no patient selected — answer from guidelines only)"
    if req.patient:
        from chf_titration.rag import build_engine_context
        if req.result and req.changes and req.flags is not None:
            result, changes, flags = req.result, req.changes, req.flags
        else:
            bundle = _get_bundle()
            patient_full = {**req.patient, "targets": TARGETS}
            contras_derived = derive_contraindications(patient_full)
            patient_full = {**patient_full, "contras": {**patient_full.get("contras", {}), **contras_derived}}
            tps = synthesize_week(patient_full)
            flags, _ = predict_flags(patient_full, tps, bundle)
            labs = req.labs if req.labs else patient_full.get("labs")
            result = _engine.evaluate(patient_full, tps, flags, labs=labs, awaiting_labs=set())
            changes = apply_strategy(result, patient_full, strategy=req.strategy)
        engine_context = build_engine_context(
            result, changes, req.patient, flags, req.labs or req.patient.get("labs")
        )

    answer = rag.ask(req.question.strip(), engine_context, audience=req.audience)
    return {"answer": answer, "engine_context": engine_context, "audience": req.audience}
