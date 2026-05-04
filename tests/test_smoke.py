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


def test_suspected_lab_ae_with_coincident_vital_ae_orders_labs():
    """Regression: when a class has BOTH a suspected lab AE (hyperK / renal) AND
    a vital-sign immediate AE (hypotension / brady) in the same cycle and labs
    are absent, the engine must hold the class and order labs rather than
    short-circuiting on the immediate AE and confidently down-titrating."""
    patient = PATIENTS["Suspected hyperkalemia"]
    patient_full = {**patient, "targets": TARGETS}
    tps = synthesize_week(patient)

    # Mirror the simulator's classifier output for this preset.
    flags = _all_false_flags()
    for k in (
        "hypotension_detected", "bradycardia_detected",
        "hyperkalemia_detected", "renal_dysfunction_detected",
        "metabolic_acidosis_detected",
    ):
        flags[k] = True

    engine = TitrationEngine()
    result = engine.evaluate(patient_full, tps, flags, labs=None, awaiting_labs=set())

    assert result["order_labs"] is True, \
        "labs must be ordered when a suspected lab AE is flagged without labs"
    assert not result["global_stop"]
    held_with_reason = {
        cls: result["decisions"][cls]["reason"]
        for cls in ("ACEi", "beta_blocker", "MRA", "loop")
    }
    for cls, reason in held_with_reason.items():
        assert reason == "suspected_ae_awaiting_labs", \
            f"{cls} should be holding for suspected_ae_awaiting_labs, got {reason!r}"
        assert result["decisions"][cls]["action"] == "hold_titration", \
            f"{cls} should be hold_titration, got {result['decisions'][cls]['action']!r}"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
