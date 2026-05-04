"""Titration engine v1.2 — strict lab gating, awaiting_labs carry-forward, AE resolver."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from .constants import CLASSES


def _as_list(v):
    if v is None:
        return []
    if hasattr(v, "__len__"):
        try:
            return list(v) if len(v) > 0 else []
        except Exception:
            return []
    return []


def _as_dict(v):
    if v is None:
        return {}
    if isinstance(v, dict):
        return v
    try:
        return dict(v)
    except Exception:
        return {}


def _safe_num(v):
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    try:
        return float(v)
    except Exception:
        return None


CONTRA_ICD_CAT_MAP = {
    "angioedema_history": "Angioedema",
    "pregnancy": "Pregnancy",
    "bilateral_renal_artery_stenosis": "Renal_artery_stenosis",
    "severe_asthma": "Asthma",
    "av_block": "AV_block",
}


def derive_contraindications(patient: dict) -> dict:
    """14 boolean contraindication flags from patient baseline + dx + meds."""
    c = {
        k: False for k in (
            "angioedema_history", "pregnancy", "acei_within_36_hours",
            "bilateral_renal_artery_stenosis", "baseline_potassium_more_than_5_5",
            "severe_asthma", "av_block", "baseline_hr_less_than_50",
            "history_dka", "type_1_diabetes", "egfr_less_than_20",
            "severe_baseline_volume_depletion", "sodium_less_than_130",
            "potassium_less_than_3_0",
        )
    }
    contra_dx = _as_list(patient.get("contraindication_diagnoses"))
    cats = {d.get("category") for d in contra_dx if isinstance(d, dict)}
    for flag, cat in CONTRA_ICD_CAT_MAP.items():
        if cat in cats:
            c[flag] = True
    contras_preset = patient.get("contras") or {}
    for k in c:
        if contras_preset.get(k):
            c[k] = True

    dka_codes = {"E101", "E111", "E131", "24910", "25010"}
    t1dm_codes = {"E101", "24910"}
    dcs = {str(x) for x in _as_list(patient.get("diagnosis_codes"))}
    if dcs & dka_codes:
        c["history_dka"] = True
    if dcs & t1dm_codes:
        c["type_1_diabetes"] = True

    bl = _as_dict(patient.get("baseline") or patient.get("baseline_labs"))
    K = bl.get("K"); Na = bl.get("Na"); eg = bl.get("eGFR")
    if K is not None and K > 5.5:
        c["baseline_potassium_more_than_5_5"] = True
    if K is not None and K < 3.0:
        c["potassium_less_than_3_0"] = True
    if Na is not None and Na < 130:
        c["sodium_less_than_130"] = True
    if eg is not None and eg < 20:
        c["egfr_less_than_20"] = True

    tps = _as_list(patient.get("timepoints"))
    early_hr = [t.get("HR") for t in tps[:2] if isinstance(t, dict) and t.get("HR") is not None]
    if early_hr and (sum(early_hr) / len(early_hr)) < 50:
        c["baseline_hr_less_than_50"] = True

    c["acei_within_36_hours"] = bool(patient.get("meds", {}).get("ACEi", {}).get("on"))
    return c


CMAP = {
    "ARNi": ["angioedema_history", "pregnancy", "acei_within_36_hours",
             "bilateral_renal_artery_stenosis", "baseline_potassium_more_than_5_5"],
    "ACEi": ["angioedema_history", "pregnancy",
             "bilateral_renal_artery_stenosis", "baseline_potassium_more_than_5_5"],
    "ARB":  ["pregnancy", "bilateral_renal_artery_stenosis",
             "baseline_potassium_more_than_5_5"],
    "beta_blocker": ["severe_asthma", "av_block", "baseline_hr_less_than_50"],
    "MRA": ["baseline_potassium_more_than_5_5", "egfr_less_than_20"],
    "SGLT2i": ["history_dka", "type_1_diabetes", "egfr_less_than_20",
               "severe_baseline_volume_depletion"],
    "loop": ["sodium_less_than_130", "potassium_less_than_3_0", "egfr_less_than_20"],
}


@dataclass
class ClassDecision:
    class_name: str
    action: str
    reason: str
    current_dose_mg: Optional[float] = None
    current_ratio: Optional[float] = None
    target_dose_mg: Optional[float] = None


@dataclass
class TitrationResult:
    global_stop: bool = False
    escalate: bool = False
    order_labs: bool = False
    has_labs_this_cycle: bool = False
    active_raas: Optional[str] = None
    preferred_raas: Optional[str] = None
    labs_requested: set = field(default_factory=set)
    awaiting_labs_next: set = field(default_factory=set)
    resolved_classes: set = field(default_factory=set)
    decisions: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    def as_dict(self) -> dict:
        """Plain-dict view for downstream strategy applier."""
        return dict(
            global_stop=self.global_stop, escalate=self.escalate, order_labs=self.order_labs,
            has_labs_this_cycle=self.has_labs_this_cycle,
            active_raas=self.active_raas, preferred_raas=self.preferred_raas,
            labs_requested=self.labs_requested, awaiting_labs_next=self.awaiting_labs_next,
            resolved_classes=self.resolved_classes,
            decisions={k: {"action": d.action, "reason": d.reason,
                           "current_dose_mg": d.current_dose_mg,
                           "target_dose_mg": d.target_dose_mg,
                           "current_ratio": d.current_ratio}
                       for k, d in self.decisions.items()},
            notes=self.notes,
        )


class TitrationEngine:
    """Rule-based titration decision engine (mirrors CHF_flow_mermaid_v3)."""

    def __init__(self):
        self.classes = CLASSES

    def evaluate(self, patient: dict, tps: list[dict], flags: dict,
                 labs: dict | None = None, awaiting_labs: set | None = None) -> dict:
        """Run one cycle. Returns a plain dict (compatible with apply_strategy)."""
        labs = _as_dict(labs)
        has_labs = len(labs) > 0
        awaiting_in = set(awaiting_labs or [])

        p = {**patient, "timepoints": tps}
        contras = derive_contraindications(p)
        baseline = _as_dict(patient.get("baseline") or patient.get("baseline_labs"))
        cr_pct = self._cr_pct(labs, baseline) if has_labs else None

        res = TitrationResult()
        res.has_labs_this_cycle = has_labs
        res.active_raas = self._active_raas(patient)
        res.preferred_raas = self._pref_raas(contras)

        if self._global_stop(flags, labs if has_labs else {}, cr_pct):
            for cls in self.classes:
                res.decisions[cls] = ClassDecision(
                    cls, "hold_titration", "global_stop",
                    current_dose_mg=self._cur(patient, cls),
                    current_ratio=self._cur_r(patient, cls),
                    target_dose_mg=self._tgt(patient, cls),
                )
            res.global_stop = True; res.escalate = True; res.order_labs = True
            return res.as_dict()

        for cls in self.classes:
            was_aw = cls in awaiting_in
            d = self._eval_class(cls, patient, flags, labs, has_labs, contras, cr_pct, res, was_awaiting=was_aw)
            res.decisions[cls] = d
            if d.reason == "suspected_ae_awaiting_labs":
                res.labs_requested.add(cls); res.awaiting_labs_next.add(cls)
            elif d.reason == "awaiting_labs_from_prior_request":
                res.awaiting_labs_next.add(cls)
            elif was_aw and has_labs:
                res.resolved_classes.add(cls)

        self._resolve(patient, flags, has_labs, res)
        res.order_labs = len(res.labs_requested) > 0
        return res.as_dict()

    def _global_stop(self, f, labs, crp):
        K = labs.get("K") if isinstance(labs, dict) else None
        return bool(
            f.get("severe_hypotension_detected") or f.get("severe_bradycardia_detected")
            or f.get("any_emergency_flag")
            or (K is not None and K > 6.0) or (crp is not None and crp > 30)
        )

    def _cr_pct(self, labs, bl):
        cr = labs.get("Cr"); bcr = bl.get("Cr") if isinstance(bl, dict) else None
        if cr is None or bcr is None or bcr <= 0:
            return None
        return float((cr - bcr) / bcr * 100)

    def _active_raas(self, p):
        m = p.get("meds", {})
        if m.get("ARNi", {}).get("on"): return "ARNi"
        if m.get("ACEi", {}).get("on"): return "ACEi"
        if m.get("ARB", {}).get("on"): return "ARB"
        return None

    def _pref_raas(self, c):
        if not (c["angioedema_history"] or c["pregnancy"] or c["acei_within_36_hours"]
                or c["bilateral_renal_artery_stenosis"] or c["baseline_potassium_more_than_5_5"]):
            return "ARNi"
        if not (c["angioedema_history"] or c["pregnancy"]
                or c["bilateral_renal_artery_stenosis"] or c["baseline_potassium_more_than_5_5"]):
            return "ACEi"
        if not (c["pregnancy"] or c["bilateral_renal_artery_stenosis"]
                or c["baseline_potassium_more_than_5_5"]):
            return "ARB"
        return None

    def _cur(self, p, cls):  return _safe_num(p.get("meds", {}).get(cls, {}).get("dose"))
    def _tgt(self, p, cls):  return _safe_num(p.get("targets", {}).get(cls))
    def _cur_r(self, p, cls):
        cur = self._cur(p, cls); tgt = self._tgt(p, cls)
        return (cur / tgt) if (cur and tgt) else None

    def _class_contra(self, cls, c):
        for f in CMAP.get(cls, []):
            if c[f]:
                return f
        return None

    def _lab_ae(self, cls, labs, crp):
        K = labs.get("K"); Na = labs.get("Na"); eg = labs.get("eGFR")
        if cls in ("ACEi", "ARB", "ARNi", "MRA"):
            return (K is not None and K > 5.5) or (crp is not None and crp > 30) or (eg is not None and eg < 30)
        if cls == "beta_blocker":
            return (crp is not None and crp > 30) or (eg is not None and eg < 30) \
                or (K is not None and K > 5.5) or (Na is not None and Na < 130)
        if cls == "SGLT2i":
            return (crp is not None and crp > 30) or (eg is not None and eg < 20) \
                or (Na is not None and Na < 130) or (K is not None and K > 5.5)
        if cls == "loop":
            return (Na is not None and Na < 130) or (K is not None and (K < 3.5 or K > 5.5)) \
                or (crp is not None and crp > 30) or (eg is not None and eg < 30)
        return False

    def _imm_ae(self, cls, f):
        if cls in ("ACEi", "ARB", "ARNi", "MRA"):
            return bool(f.get("hypotension_detected"))
        if cls == "beta_blocker":
            return bool(f.get("bradycardia_detected") or f.get("hypotension_detected")
                        or f.get("worsening_HF_detected"))
        if cls == "SGLT2i":
            return bool(f.get("volume_depletion_detected") or f.get("hypotension_detected"))
        if cls == "loop":
            return bool(f.get("volume_depletion_detected") or f.get("hypotension_detected")
                        or f.get("worsening_HF_detected"))
        return False

    def _sus_ae(self, cls, f):
        if cls in ("ACEi", "ARB", "ARNi", "MRA"):
            return bool(f.get("hyperkalemia_detected") or f.get("renal_dysfunction_detected"))
        if cls == "beta_blocker":
            return bool(f.get("renal_dysfunction_detected") or f.get("hyperkalemia_detected"))
        if cls == "SGLT2i":
            return bool(f.get("renal_dysfunction_detected") or f.get("hyperkalemia_detected"))
        if cls == "loop":
            return bool(f.get("hyperkalemia_detected") or f.get("hyponatremia_detected")
                        or f.get("renal_dysfunction_detected"))
        return False

    def _eval_class(self, cls, patient, flags, labs, has_labs, contras, cr_pct, res, was_awaiting=False):
        on = bool(patient.get("meds", {}).get(cls, {}).get("on"))
        cur = self._cur(patient, cls); rat = self._cur_r(patient, cls); tgt = self._tgt(patient, cls)
        d = ClassDecision(cls, "maintain_dose", "default", cur, rat, tgt)

        if was_awaiting and not has_labs:
            d.action = "hold_titration"; d.reason = "awaiting_labs_from_prior_request"
            return d
        ch = self._class_contra(cls, contras)
        if ch:
            d.action = "hold_titration" if on else "maintain_dose"
            d.reason = f"contraindication: {ch}"
            return d
        if has_labs and self._lab_ae(cls, labs, cr_pct):
            d.action = "decrease_dose" if on else "maintain_dose"; d.reason = "lab_ae"
            return d
        # When a suspected lab AE is present and we have no labs this cycle, hold the
        # class and order labs *before* falling through to the vital-sign immediate-AE
        # branch. Otherwise a co-occurring hypotension/bradycardia flag short-circuits
        # the lab request and the engine confidently down-titrates a class whose lab
        # status it can't actually verify (the "Suspected hyperkalemia" failure mode:
        # mild hypotension + brady from the synth pattern get treated as immediate AEs
        # and the hyperK/renal/acidosis flags never make it to labs_requested).
        if not has_labs and self._sus_ae(cls, flags):
            d.action = "hold_titration" if on else "maintain_dose"
            d.reason = "suspected_ae_awaiting_labs"
            return d
        if self._imm_ae(cls, flags):
            d.action = "decrease_dose" if on else "maintain_dose"; d.reason = "immediate_ae"
            return d
        if not on:
            if cls in ("ACEi", "ARB", "ARNi"):
                if res.preferred_raas == cls and res.active_raas is None:
                    d.action = "start_medication"; d.reason = "preferred_raas_not_started"
                else:
                    d.action = "maintain_dose"
                    d.reason = "not_preferred_raas" if res.active_raas else "no_raas_available"
            else:
                d.action = "start_medication"; d.reason = "below_target_not_started"
            return d
        if rat is None or rat >= 1.0:
            d.action = "maintain_dose"; d.reason = "at_or_above_target"
            return d
        d.action = "increase_dose"; d.reason = "below_target"
        return d

    def _resolve(self, patient, f, has_labs, res):
        on_sglt = bool(patient.get("meds", {}).get("SGLT2i", {}).get("on"))
        on_loop = bool(patient.get("meds", {}).get("loop", {}).get("on"))
        on_mra = bool(patient.get("meds", {}).get("MRA", {}).get("on"))
        on_bb = bool(patient.get("meds", {}).get("beta_blocker", {}).get("on"))
        act = res.active_raas

        def setA(cls, a, r):
            if cls not in res.decisions:
                return
            d = res.decisions[cls]
            if d.reason in ("awaiting_labs_from_prior_request", "suspected_ae_awaiting_labs"):
                return
            d.action = a; d.reason = f"ae_resolver:{r}"

        if f.get("hypotension_detected"):
            if f.get("volume_depletion_detected"):
                if on_loop: setA("loop", "decrease_dose", "hypo_volume")
                if on_sglt: setA("SGLT2i", "stop_medication", "hypo_volume")
            else:
                if act: setA(act, "decrease_dose", "hypo_vasodilatory")
                elif on_bb: setA("beta_blocker", "decrease_dose", "hypo_vasodilatory")
        if f.get("bradycardia_detected") and on_bb:
            setA("beta_blocker", "decrease_dose", "bradycardia")
        if f.get("hyperkalemia_detected") and has_labs:
            if on_mra: setA("MRA", "decrease_dose", "hyperK_on_mra")
            elif act: setA(act, "decrease_dose", "hyperK_on_raas")
            if f.get("worsening_HF_detected") and on_loop:
                setA("loop", "increase_dose", "vol_overload_with_hyperK")
        if f.get("renal_dysfunction_detected") and has_labs:
            if f.get("volume_depletion_detected") and on_loop:
                setA("loop", "decrease_dose", "renal_vol_depl")
            elif act:
                setA(act, "decrease_dose", "renal_not_volume")
            if f.get("hyperkalemia_detected") and on_mra:
                setA("MRA", "decrease_dose", "renal_with_hyperK")
        if f.get("volume_depletion_detected"):
            if on_sglt and on_loop:
                setA("SGLT2i", "stop_medication", "volD_on_sglt2_and_loop")
                setA("loop", "decrease_dose", "volD_on_sglt2_and_loop")
            elif on_loop:
                setA("loop", "decrease_dose", "volD_not_on_sglt2")
