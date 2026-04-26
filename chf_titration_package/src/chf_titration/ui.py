"""Streamlit UI — run locally with `streamlit run src/chf_titration/ui.py` or `python scripts/run_ui.py`."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from chf_titration.constants import CLASSES, RHYTHMS
from chf_titration.synthesize import PATIENTS, synthesize_week
from chf_titration.engine import TitrationEngine
from chf_titration.strategy import apply_strategy, STRATEGIES, TARGETS
from chf_titration.classifier import load_bundle, predict_flags

BUNDLE_PATH = Path("models/chf_classifier_lgbm.pkl")


def _tps_to_stats(tps):
    def vals(k): return [t.get(k) for t in tps if t.get(k) is not None]
    hr, sbp, spo2, wt = vals("HR"), vals("SBP"), vals("SpO2"), vals("weight_kg")
    e_qrs = [t["ecg"].get("qrs_ms") for t in tps if t.get("ecg") and t["ecg"].get("qrs_ms") is not None]
    e_qtc = [t["ecg"].get("qtc_ms") for t in tps if t.get("ecg") and t["ecg"].get("qtc_ms") is not None]
    rhys = [t.get("ecg", {}).get("rhythm", "NSR") for t in tps]
    dom = max(set(rhys), key=rhys.count) if rhys else "NSR"
    return dict(
        HR_mean=round(float(np.mean(hr)), 1) if hr else 72,
        HR_min=round(float(min(hr)), 1) if hr else 60,
        SBP_mean=round(float(np.mean(sbp)), 1) if sbp else 125,
        SBP_min=round(float(min(sbp)), 1) if sbp else 110,
        SpO2_min=round(float(min(spo2)), 1) if spo2 else 96,
        weight_delta=round(float(wt[-1] - wt[0]), 2) if len(wt) >= 2 else 0.0,
        QRS_max=round(float(max(e_qrs)), 0) if e_qrs else 95,
        QTc_max=round(float(max(e_qtc)), 0) if e_qtc else 420,
        T_peaked=any(t.get("ecg", {}).get("t_peaked", False) for t in tps),
        rhythm=dom,
    )


def _stats_to_tps(s):
    fake = {"age": 65, "gender": "M", "meds": {}, "vitals": dict(s)}
    return synthesize_week(fake, seed=0)


def _tps_to_csv(tps) -> str:
    rows = [["time", "HR", "SBP", "DBP", "SpO2", "weight_kg", "QRS_ms", "QTc_ms", "T_peaked", "rhythm"]]
    for t in tps:
        e = t.get("ecg", {})
        rows.append([
            t.get("timestamp", ""),
            t.get("HR"), t.get("SBP"), t.get("DBP"), t.get("SpO2"), t.get("weight_kg"),
            e.get("qrs_ms"), e.get("qtc_ms"), e.get("t_peaked", False), e.get("rhythm", "NSR"),
        ])
    buf = io.StringIO()
    pd.DataFrame(rows[1:], columns=rows[0]).to_csv(buf, index=False)
    return buf.getvalue()


def _csv_to_tps(csv_text: str, base_tps: list[dict]) -> list[dict]:
    try:
        df = pd.read_csv(io.StringIO(csv_text))
    except Exception:
        return base_tps
    out = []
    for i, r in df.iterrows():
        if i >= 14:
            break
        old_ecg = base_tps[i].get("ecg", {}) if i < len(base_tps) else {}
        def f(col):
            v = r.get(col)
            if v is None or (isinstance(v, float) and pd.isna(v)):
                return None
            try: return float(v)
            except Exception: return None
        out.append({
            "timestamp": str(r.get("time", "")),
            "HR": f("HR"), "SBP": f("SBP"), "DBP": f("DBP"),
            "SpO2": f("SpO2"), "weight_kg": f("weight_kg"),
            "ecg": {
                **old_ecg,
                "qrs_ms": f("QRS_ms"), "qtc_ms": f("QTc_ms"),
                "t_peaked": str(r.get("T_peaked", False)).strip().lower() in ("true", "1", "yes"),
                "rhythm": str(r.get("rhythm", "NSR")).strip() or "NSR",
            },
        })
    return out if out else base_tps


def main():
    st.set_page_config(page_title="CHF Titration Simulator", layout="wide")
    st.title("CHF titration simulator")
    st.caption("Classifier → engine → strategy → recommendation. Local demo — no cloud tunnel required.")

    bundle = load_bundle(BUNDLE_PATH)
    if bundle is None:
        st.error(
            f"**No trained classifier found at `{BUNDLE_PATH}`.**\n\n"
            f"The UI requires a trained LightGBM bundle. The easiest way is:\n\n"
            f"```\n"
            f"python scripts/run_ui.py\n"
            f"```\n\n"
            f"which auto-prepares the bundle on first launch. If you started Streamlit "
            f"directly (e.g. `streamlit run src/chf_titration/ui.py`), run this once from "
            f"the package root first:\n\n"
            f"```\n"
            f"python scripts/prepare.py\n"
            f"```\n\n"
            f"Then reload this page."
        )
        st.stop()
    st.caption(f"✓ Loaded trained classifier from `{BUNDLE_PATH}` ({len(bundle['label_cols'])} flags)")

    if "tps" not in st.session_state:
        st.session_state.tps = synthesize_week(PATIENTS["Newly diagnosed, stable"])
        st.session_state.preset = "Newly diagnosed, stable"

    col1, col2 = st.columns([3, 1])
    with col1:
        preset = st.selectbox("Patient preset", list(PATIENTS.keys()),
                              index=list(PATIENTS.keys()).index(st.session_state.preset))
        if preset != st.session_state.preset:
            st.session_state.preset = preset
            st.session_state.tps = synthesize_week(PATIENTS[preset])
    with col2:
        if st.button("Resynthesize week"):
            st.session_state.tps = synthesize_week(
                PATIENTS[st.session_state.preset],
                seed=int(np.random.default_rng().integers(0, 10**9)),
            )

    patient = PATIENTS[st.session_state.preset]

    with st.expander("EHR snapshot", expanded=True):
        bl = patient["baseline"]
        meds_on = [f"{cls} {patient['meds'][cls]['dose']}mg" for cls in CLASSES if patient["meds"][cls]["on"]]
        st.write(f"**Demographics:** {patient['age']}yo {patient['gender']}")
        st.write(f"**Baseline labs:** K={bl['K']:.1f}, Na={bl['Na']:.0f}, Cr={bl['Cr']:.2f}, eGFR={bl['eGFR']:.0f}")
        st.write(f"**Current GDMT:** {', '.join(meds_on) if meds_on else '(none)'}")
        if patient.get("labs"):
            lb = patient["labs"]
            st.write(f"**Labs this cycle available:** K={lb['K']:.1f} Na={lb['Na']:.0f} Cr={lb['Cr']:.2f} eGFR={lb['eGFR']:.0f}")

    mode = st.radio("Input mode", ["Summary stats", "Individual measurements"], horizontal=True)
    stats = _tps_to_stats(st.session_state.tps)

    if mode == "Summary stats":
        c1, c2 = st.columns(2)
        with c1:
            stats["HR_mean"]   = st.number_input("HR mean (bpm)",   30.0, 200.0, float(stats["HR_mean"]), step=1.0)
            stats["SBP_mean"]  = st.number_input("SBP mean (mmHg)", 60.0, 240.0, float(stats["SBP_mean"]), step=1.0)
            stats["SpO2_min"]  = st.number_input("SpO2 min (%)",    60.0, 100.0, float(stats["SpO2_min"]), step=1.0)
            stats["QRS_max"]   = st.number_input("QRS max (ms)",    60.0, 250.0, float(stats["QRS_max"]), step=1.0)
            stats["T_peaked"]  = st.checkbox("T-peaked", value=bool(stats["T_peaked"]))
        with c2:
            stats["HR_min"]       = st.number_input("HR min (bpm)",          30.0, 200.0, float(stats["HR_min"]), step=1.0)
            stats["SBP_min"]      = st.number_input("SBP min (mmHg)",        50.0, 240.0, float(stats["SBP_min"]), step=1.0)
            stats["weight_delta"] = st.number_input("Δ weight over week (kg)", -10.0, 10.0, float(stats["weight_delta"]), step=0.1)
            stats["QTc_max"]      = st.number_input("QTc max (ms)",          300.0, 600.0, float(stats["QTc_max"]), step=1.0)
            stats["rhythm"]       = st.selectbox("Dominant rhythm", RHYTHMS, index=RHYTHMS.index(stats["rhythm"]) if stats["rhythm"] in RHYTHMS else 0)
        if st.button("Sync stats → timepoints"):
            st.session_state.tps = _stats_to_tps(stats)
    else:
        csv_text = st.text_area("14-row CSV of individual measurements", value=_tps_to_csv(st.session_state.tps), height=320)
        if st.button("Apply edits"):
            st.session_state.tps = _csv_to_tps(csv_text, st.session_state.tps)

    c1, c2, c3 = st.columns([2, 2, 2])
    with c1:
        strategy = st.selectbox("Titration strategy", list(STRATEGIES.keys()),
                                 index=list(STRATEGIES.keys()).index("strong_hf"))
    with c2:
        labs_flag = st.checkbox("Labs arrived this cycle",
                                 value=patient.get("labs") is not None)
    with c3:
        compute = st.button("Compute recommendations", type="primary")

    if compute:
        patient_full = {**patient, "targets": TARGETS}
        flags, probs = predict_flags(patient_full, st.session_state.tps, bundle)
        labs = patient.get("labs") if (labs_flag and patient.get("labs")) else None
        engine = TitrationEngine()
        result = engine.evaluate(patient_full, st.session_state.tps, flags, labs=labs, awaiting_labs=set())
        changes = apply_strategy(result, patient_full, strategy=strategy)

        st.subheader("Classifier flags")
        pos_flags = [lab for lab in flags if flags[lab]]
        st.write("Active:", ", ".join(pos_flags) if pos_flags else "(none above 0.5)")
        st.dataframe(pd.DataFrame([{"flag": k, "probability": round(probs[k], 3), "active": flags[k]} for k in probs]),
                     use_container_width=True, hide_index=True)

        st.subheader("Engine + strategy decisions")
        if result["global_stop"]:
            st.error("**Global stop** — hold all GDMT and escalate for urgent evaluation.")
        elif result["order_labs"]:
            req = ", ".join(sorted(result["labs_requested"]))
            st.warning(f"**Order labs** (K, Na, Cr, eGFR) — suspected AE in: {req}. Hold these classes until labs return.")

        rec_rows = []
        for cls in CLASSES:
            c = changes[cls]; cur = c["current"] or 0; nd = c["new_dose"]
            if c["concrete_action"] == "start_medication": dose_txt = f"start @ {nd}mg"
            elif c["concrete_action"] == "stop_medication": dose_txt = f"{cur}mg → 0 (stop)"
            elif c["concrete_action"] in ("maintain_dose", "hold_titration"):
                dose_txt = f"{cur}mg (no change)" if cur else "—"
            else: dose_txt = f"{cur}mg → {nd}mg"
            rec_rows.append({"class": cls, "action": c["concrete_action"], "dose": dose_txt, "reason": c["reason"]})
        st.dataframe(pd.DataFrame(rec_rows), use_container_width=True, hide_index=True)

        st.caption(f"preferred RAAS: **{result['preferred_raas'] or '(none)'}**  ·  "
                   f"active RAAS: **{result['active_raas'] or '(none)'}**  ·  "
                   f"order_labs: **{'YES' if result['order_labs'] else 'no'}**  ·  "
                   f"global_stop: **{'YES' if result['global_stop'] else 'no'}**")


if __name__ == "__main__":
    main()
