"""Synthetic vitals generator + preset patient definitions."""

from __future__ import annotations

from datetime import datetime, timedelta
import numpy as np

from .constants import CLASSES


def _meds_off() -> dict:
    return {cls: {"on": False, "dose": 0} for cls in CLASSES + ["thiazide"]}


def _meds_on(spec: dict) -> dict:
    m = _meds_off()
    for cls, d in spec.items():
        m[cls] = {"on": True, "dose": d}
    return m


PATIENTS = {
    # =============================================================
    # 1. BASELINE / TITRATION-STATE SCENARIOS
    # =============================================================
    "Newly diagnosed, stable": {
        "age": 55, "gender": "F",
        "baseline": {"K": 4.2, "Na": 140, "Cr": 1.0, "eGFR": 78, "BUN": 15, "HCO3": 24, "pH": 7.40},
        "contras": {}, "meds": _meds_off(),
        "vitals": {"HR_mean": 74, "HR_min": 65, "SBP_mean": 128, "SBP_min": 115, "SpO2_min": 97,
                   "weight_delta": 0.2, "QRS_max": 95, "QTc_max": 415, "T_peaked": False, "rhythm": "NSR"},
        "labs": None,
        "note": "No GDMT started. Engine should recommend starting preferred RAAS (ARNi) + other pillars per strategy.",
    },
    "Partial titration, no AEs": {
        "age": 68, "gender": "M",
        "baseline": {"K": 4.5, "Na": 139, "Cr": 1.1, "eGFR": 68, "BUN": 18, "HCO3": 24, "pH": 7.40},
        "contras": {}, "meds": _meds_on({"ACEi": 10, "beta_blocker": 12.5, "loop": 40}),
        "vitals": {"HR_mean": 76, "HR_min": 62, "SBP_mean": 125, "SBP_min": 110, "SpO2_min": 96,
                   "weight_delta": 0.1, "QRS_max": 100, "QTc_max": 425, "T_peaked": False, "rhythm": "NSR"},
        "labs": None,
        "note": "On ACEi 10/40, BB 12.5/50, loop. Should up-titrate ACEi + BB (ratio<1.0), start MRA + SGLT2i.",
    },
    "Fully titrated, at target": {
        "age": 64, "gender": "M",
        "baseline": {"K": 4.3, "Na": 140, "Cr": 1.1, "eGFR": 70, "BUN": 17, "HCO3": 24, "pH": 7.40},
        "contras": {}, "meds": _meds_on({"ARNi": 103, "beta_blocker": 50, "MRA": 50, "SGLT2i": 10, "loop": 40}),
        "vitals": {"HR_mean": 70, "HR_min": 62, "SBP_mean": 122, "SBP_min": 112, "SpO2_min": 97,
                   "weight_delta": 0.0, "QRS_max": 95, "QTc_max": 412, "T_peaked": False, "rhythm": "NSR"},
        "labs": None,
        "note": "At-target quadruple therapy. Every class should maintain (reason: at_or_above_target).",
    },

    # =============================================================
    # 2. ADVERSE-EFFECT SCENARIOS
    # =============================================================
    "Suspected hyperkalemia": {
        "age": 72, "gender": "M",
        "baseline": {"K": 4.8, "Na": 138, "Cr": 1.3, "eGFR": 58, "BUN": 24, "HCO3": 23, "pH": 7.38},
        "contras": {}, "meds": _meds_on({"ACEi": 20, "beta_blocker": 25, "MRA": 25, "loop": 40}),
        "vitals": {"HR_mean": 68, "HR_min": 55, "SBP_mean": 120, "SBP_min": 108, "SpO2_min": 96,
                   "weight_delta": 0.3, "QRS_max": 138, "QTc_max": 425, "T_peaked": True, "rhythm": "NSR"},
        "labs": None,
        "note": "ECG shows hyperK signs (t_peaked, QRS wide) but no labs. Engine holds RAAS+MRA+BB + orders labs.",
    },
    "Suspected renal dysfunction": {
        "age": 68, "gender": "M",
        "baseline": {"K": 4.4, "Na": 138, "Cr": 1.1, "eGFR": 62, "BUN": 18, "HCO3": 24, "pH": 7.40},
        "contras": {}, "meds": _meds_on({"ACEi": 20, "beta_blocker": 25, "MRA": 25, "loop": 40}),
        "vitals": {"HR_mean": 78, "HR_min": 68, "SBP_mean": 133, "SBP_min": 118, "SpO2_min": 96,
                   "weight_delta": 1.1, "QRS_max": 100, "QTc_max": 418, "T_peaked": False, "rhythm": "NSR"},
        "labs": None,
        "note": "Weight drift up + SBP rising (classic renal signature) but no labs. Engine holds + orders.",
    },
    "Confirmed hyperkalemia (K=6.3)": {
        "age": 72, "gender": "M",
        "baseline": {"K": 4.8, "Na": 138, "Cr": 1.3, "eGFR": 58, "BUN": 24, "HCO3": 23, "pH": 7.38},
        "contras": {}, "meds": _meds_on({"ACEi": 20, "beta_blocker": 25, "MRA": 25, "loop": 40}),
        "vitals": {"HR_mean": 62, "HR_min": 48, "SBP_mean": 115, "SBP_min": 100, "SpO2_min": 96,
                   "weight_delta": 0.4, "QRS_max": 145, "QTc_max": 435, "T_peaked": True, "rhythm": "wide_complex"},
        "labs": {"K": 6.3, "Na": 138, "Cr": 1.7, "eGFR": 48, "HCO3": 22},
        "note": "Labs confirm K=6.3 → global_stop fires (K>6.0). All meds held, escalate.",
    },
    "Volume depletion + hypotension": {
        "age": 70, "gender": "M",
        "baseline": {"K": 4.3, "Na": 138, "Cr": 1.1, "eGFR": 65, "BUN": 20, "HCO3": 24, "pH": 7.39},
        "contras": {}, "meds": _meds_on({"ACEi": 20, "beta_blocker": 25, "SGLT2i": 10, "loop": 80}),
        "vitals": {"HR_mean": 92, "HR_min": 80, "SBP_mean": 98, "SBP_min": 85, "SpO2_min": 97,
                   "weight_delta": -2.8, "QRS_max": 100, "QTc_max": 420, "T_peaked": False, "rhythm": "Sinus_tachy"},
        "labs": None,
        "note": "AE resolver: stop SGLT2i + decrease loop (volume depletion on both).",
    },
    "Worsening HF": {
        "age": 69, "gender": "F",
        "baseline": {"K": 4.2, "Na": 137, "Cr": 1.0, "eGFR": 70, "BUN": 16, "HCO3": 24, "pH": 7.39},
        "contras": {}, "meds": _meds_on({"ACEi": 10, "beta_blocker": 12.5, "loop": 40}),
        "vitals": {"HR_mean": 94, "HR_min": 82, "SBP_mean": 118, "SBP_min": 105, "SpO2_min": 92,
                   "weight_delta": 3.2, "QRS_max": 128, "QTc_max": 435, "T_peaked": False, "rhythm": "LBBB"},
        "labs": None,
        "note": "Weight gain + SpO2 drop. BB and loop get immediate_ae; RAAS/MRA can still titrate.",
    },
    "Bradycardia on beta blocker": {
        "age": 72, "gender": "M",
        "baseline": {"K": 4.3, "Na": 139, "Cr": 1.1, "eGFR": 68, "BUN": 17, "HCO3": 24, "pH": 7.40},
        "contras": {}, "meds": _meds_on({"ACEi": 20, "beta_blocker": 50, "MRA": 25}),
        "vitals": {"HR_mean": 52, "HR_min": 42, "SBP_mean": 118, "SBP_min": 105, "SpO2_min": 96,
                   "weight_delta": 0.0, "QRS_max": 98, "QTc_max": 420, "T_peaked": False, "rhythm": "Sinus_brady"},
        "labs": None,
        "note": "AE resolver: down-titrate beta blocker first. Other classes evaluate normally.",
    },
    "Worsening HF + hyperkalemia (labs confirmed)": {
        "age": 70, "gender": "M",
        "baseline": {"K": 4.5, "Na": 138, "Cr": 1.2, "eGFR": 62, "BUN": 20, "HCO3": 24, "pH": 7.39},
        "contras": {}, "meds": _meds_on({"ACEi": 20, "beta_blocker": 25, "MRA": 25, "loop": 40}),
        "vitals": {"HR_mean": 88, "HR_min": 78, "SBP_mean": 125, "SBP_min": 112, "SpO2_min": 91,
                   "weight_delta": 2.8, "QRS_max": 135, "QTc_max": 428, "T_peaked": True, "rhythm": "LBBB"},
        "labs": {"K": 5.8, "Na": 137, "Cr": 1.4, "eGFR": 50, "HCO3": 22},
        "note": "Classic double-bind: AE resolver decreases MRA (hyperK on MRA) AND increases loop (vol overload + hyperK).",
    },
    "Hyponatremia + renal dysfunction (labs)": {
        "age": 74, "gender": "F",
        "baseline": {"K": 4.5, "Na": 135, "Cr": 1.2, "eGFR": 55, "BUN": 22, "HCO3": 24, "pH": 7.38},
        "contras": {}, "meds": _meds_on({"ACEi": 10, "beta_blocker": 12.5, "MRA": 12.5, "loop": 80}),
        "vitals": {"HR_mean": 82, "HR_min": 72, "SBP_mean": 122, "SBP_min": 108, "SpO2_min": 95,
                   "weight_delta": 0.5, "QRS_max": 108, "QTc_max": 478, "T_peaked": False, "rhythm": "NSR"},
        "labs": {"K": 4.6, "Na": 126, "Cr": 1.65, "eGFR": 42, "HCO3": 24},
        "note": "Labs show Na=126 + Cr up 38%. lab_ae fires on loop (Na<130), RAAS/MRA (Cr Δ>30%).",
    },

    # =============================================================
    # 3. CONTRAINDICATION SCENARIOS
    # =============================================================
    "Angioedema history (ARNi + ACEi blocked)": {
        "age": 66, "gender": "F",
        "baseline": {"K": 4.3, "Na": 140, "Cr": 1.0, "eGFR": 72, "BUN": 15, "HCO3": 24, "pH": 7.40},
        "contras": {"angioedema_history": True},
        "meds": _meds_on({"beta_blocker": 12.5, "loop": 40}),
        "vitals": {"HR_mean": 74, "HR_min": 64, "SBP_mean": 126, "SBP_min": 112, "SpO2_min": 97,
                   "weight_delta": 0.1, "QRS_max": 95, "QTc_max": 415, "T_peaked": False, "rhythm": "NSR"},
        "labs": None,
        "note": "ARNi + ACEi both contraindicated. Engine should prefer ARB; recommend starting MRA + SGLT2i.",
    },
    "Severe asthma (beta blocker blocked)": {
        "age": 62, "gender": "F",
        "baseline": {"K": 4.3, "Na": 139, "Cr": 1.0, "eGFR": 70, "BUN": 16, "HCO3": 24, "pH": 7.40},
        "contras": {"severe_asthma": True},
        "meds": _meds_on({"ACEi": 10, "loop": 40}),
        "vitals": {"HR_mean": 78, "HR_min": 68, "SBP_mean": 125, "SBP_min": 115, "SpO2_min": 96,
                   "weight_delta": 0.1, "QRS_max": 95, "QTc_max": 412, "T_peaked": False, "rhythm": "NSR"},
        "labs": None,
        "note": "Beta blocker contraindicated. Everything else titrates normally; ACEi up-titrates, MRA+SGLT2i start.",
    },
    "CKD stage 4 (eGFR=22)": {
        "age": 75, "gender": "M",
        "baseline": {"K": 4.8, "Na": 138, "Cr": 2.3, "eGFR": 22, "BUN": 38, "HCO3": 22, "pH": 7.36},
        "contras": {}, "meds": _meds_on({"ACEi": 10, "beta_blocker": 25, "loop": 80}),
        "vitals": {"HR_mean": 78, "HR_min": 65, "SBP_mean": 135, "SBP_min": 120, "SpO2_min": 96,
                   "weight_delta": 0.3, "QRS_max": 105, "QTc_max": 428, "T_peaked": False, "rhythm": "NSR"},
        "labs": None,
        "note": "eGFR<30 → MRA contraindicated (derived from baseline). SGLT2i still allowed (eGFR>20).",
    },
    "Pregnant (RAAS blocked)": {
        "age": 32, "gender": "F",
        "baseline": {"K": 4.2, "Na": 140, "Cr": 0.9, "eGFR": 95, "BUN": 12, "HCO3": 24, "pH": 7.40},
        "contras": {"pregnancy": True},
        "meds": _meds_on({"beta_blocker": 12.5, "loop": 40}),
        "vitals": {"HR_mean": 84, "HR_min": 74, "SBP_mean": 118, "SBP_min": 108, "SpO2_min": 98,
                   "weight_delta": 0.5, "QRS_max": 92, "QTc_max": 415, "T_peaked": False, "rhythm": "NSR"},
        "labs": None,
        "note": "All three RAAS agents blocked. No RAAS available; beta blocker + loop continue, MRA allowed but SGLT2i contraindicated in pregnancy.",
    },
    "AV block (beta blocker blocked)": {
        "age": 78, "gender": "M",
        "baseline": {"K": 4.1, "Na": 140, "Cr": 1.1, "eGFR": 62, "BUN": 18, "HCO3": 24, "pH": 7.40},
        "contras": {"av_block": True},
        "meds": _meds_on({"ACEi": 20, "MRA": 25, "loop": 40}),
        "vitals": {"HR_mean": 62, "HR_min": 54, "SBP_mean": 128, "SBP_min": 115, "SpO2_min": 97,
                   "weight_delta": 0.0, "QRS_max": 112, "QTc_max": 422, "T_peaked": False, "rhythm": "AV_block_1"},
        "labs": None,
        "note": "Pacemaker/AV block history. Beta blocker held; RAAS+MRA+loop continue normally.",
    },
    "Severe hypotension (global stop)": {
        "age": 70, "gender": "M",
        "baseline": {"K": 4.3, "Na": 138, "Cr": 1.2, "eGFR": 58, "BUN": 21, "HCO3": 23, "pH": 7.38},
        "contras": {}, "meds": _meds_on({"ACEi": 20, "beta_blocker": 25, "loop": 40}),
        "vitals": {"HR_mean": 95, "HR_min": 88, "SBP_mean": 78, "SBP_min": 68, "SpO2_min": 95,
                   "weight_delta": -0.5, "QRS_max": 100, "QTc_max": 418, "T_peaked": False, "rhythm": "Sinus_tachy"},
        "labs": None,
        "note": "SBP<80 triggers severe_hypotension → classifier emergency → global_stop, escalate.",
    },
}


def synthesize_week(patient: dict, seed: int | None = None) -> list[dict]:
    """Generate 14 twice-daily timepoints matching the preset's clinical pattern."""
    if seed is None:
        seed = int(patient.get("age", 60)) * 1000 + sum(1 for m in patient.get("meds", {}).values() if m.get("on"))
    rng = np.random.default_rng(seed)
    v = patient.get("vitals_target", patient.get("vitals", {}))
    hr_m = v.get("HR_mean", 72); hr_lo = v.get("HR_min", hr_m - 10)
    sbp_m = v.get("SBP_mean", 125); sbp_lo = v.get("SBP_min", sbp_m - 15)
    spo_lo = v.get("SpO2_min", 96)
    wt_d = v.get("weight_delta", 0.0)
    qrs_peak = v.get("QRS_max", 95); qtc_peak = v.get("QTc_max", 420)
    t_rate = 0.7 if v.get("T_peaked", False) else 0.03
    dom_rhy = v.get("rhythm", "NSR")
    base_wt = float(v.get("baseline_weight_kg", 80.0))

    tps = []
    week_start = datetime(2024, 6, 3, 8, 0, 0)
    hr_trough_tp = rng.integers(3, 11)
    sbp_trough_tp = rng.integers(3, 11)
    onset = rng.integers(2, 5)
    for i in range(14):
        day = i // 2
        hour = 8 if i % 2 == 0 else 20
        ts = week_start + timedelta(days=day, hours=hour - 8)
        frac = max(0.0, (day - onset + 1) / 5.0)
        drift_hr = (hr_lo - hr_m) * (1.0 if i == hr_trough_tp else 0.3 * frac)
        drift_sbp = (sbp_lo - sbp_m) * (1.0 if i == sbp_trough_tp else 0.3 * frac)
        hr = float(np.clip(hr_m + drift_hr + rng.normal(0, 2.5), 30, 180))
        sbp = float(np.clip(sbp_m + drift_sbp + rng.normal(0, 3.5), 55, 220))
        dbp = float(np.clip(sbp * 0.58 + rng.normal(0, 2.5), 30, 130))
        spo2 = float(np.clip(spo_lo + rng.uniform(0.5, 3.0) - (1.0 if i == hr_trough_tp else 0), 80, 100))
        wt = float(np.clip(base_wt + wt_d * ((i + 1) / 14) + rng.normal(0, 0.25), 35, 260))
        t_peaked = bool(rng.random() < t_rate)
        t_amp = float(rng.uniform(0.55, 0.95) if t_peaked else rng.uniform(0.15, 0.45))
        rhythm = dom_rhy if (rng.random() < 0.55) else "NSR"
        qual = "good" if rng.random() < 0.87 else ("fair" if rng.random() < 0.9 else "poor")
        tps.append(dict(
            timestamp=ts.isoformat(),
            HR=round(hr, 1), SBP=round(sbp, 1), DBP=round(dbp, 1),
            SpO2=round(spo2, 1), weight_kg=round(wt, 2),
            ecg=dict(
                hr_ecg=round(hr + rng.normal(0, 1.5), 1),
                pr_ms=round(float(np.clip(160 + rng.normal(0, 10), 100, 340)), 1),
                qrs_ms=round(float(np.clip(qrs_peak - rng.uniform(0, 18) + rng.normal(0, 3), 65, 210)), 1),
                qtc_ms=round(float(np.clip(qtc_peak - rng.uniform(0, 18) + rng.normal(0, 4), 320, 560)), 1),
                t_amp_mv=round(t_amp, 2),
                t_peaked=t_peaked,
                st_dev_mm=round(float(np.clip(rng.normal(0, 0.25), -3, 3)), 2),
                rhythm=rhythm, quality=qual,
            ),
        ))
    return tps
