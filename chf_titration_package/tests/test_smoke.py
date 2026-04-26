"""Minimal smoke test — verifies modules wire together. Classifier-free: uses a fixed flag dict."""

from chf_titration import (
    PATIENTS, synthesize_week, TitrationEngine, apply_strategy, TARGETS, LABEL_COLS,
)


def _all_false_flags() -> dict:
    return {lab: False for lab in LABEL_COLS}


def _hyperk_flags() -> dict:
    return {**_all_false_flags(),
            "hyperkalemia_detected": True, "renal_dysfunction_detected": True}


def test_synthesize_produces_14_timepoints():
    tps = synthesize_week(PATIENTS["Newly diagnosed, stable"])
    assert len(tps) == 14
    for t in tps:
        assert "HR" in t and "SBP" in t and "ecg" in t


def test_engine_runs_end_to_end_without_classifier():
    """Engine + strategy are classifier-free; we pass flags directly."""
    patient = PATIENTS["Volume depletion + hypotension"]
    patient_full = {**patient, "targets": TARGETS}
    tps = synthesize_week(patient)

    flags = _all_false_flags()
    flags["hypotension_detected"] = True
    flags["volume_depletion_detected"] = True

    engine = TitrationEngine()
    result = engine.evaluate(patient_full, tps, flags)
    assert "decisions" in result and len(result["decisions"]) == 7

    for strategy in ("traditional", "strong_hf", "rapid_sequence", "sglt_mra_first"):
        changes = apply_strategy(result, patient_full, strategy=strategy)
        assert len(changes) == 7
        for _cls, ch in changes.items():
            if ch["new_dose"] is not None and ch["target"] is not None:
                assert ch["new_dose"] <= ch["target"] + 1e-6


def test_confirmed_hyperk_triggers_global_stop():
    patient = PATIENTS["Confirmed hyperkalemia (K=6.3)"]
    patient_full = {**patient, "targets": TARGETS}
    tps = synthesize_week(patient)

    flags = _hyperk_flags()

    engine = TitrationEngine()
    result = engine.evaluate(patient_full, tps, flags, labs=patient["labs"])
    assert result["global_stop"] is True, "K=6.3 must trigger global_stop"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
