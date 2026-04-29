"""FastAPI backend exposing the TIDE-HF Python engine to the website.

GET  /presets       → list of preset patient dicts (from synthesize.PATIENTS)
POST /evaluate      → run classifier + engine + strategy on a patient payload
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
    return {"ok": True, "bundle_loaded": _bundle is not None}
