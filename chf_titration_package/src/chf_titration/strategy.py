"""Dose ladders + strategy applier (traditional / strong_hf / rapid_sequence / sglt_mra_first)."""

from __future__ import annotations

import re
import pandas as pd

from .constants import CLASSES, CLASS_TO_REP_DRUG, CLASS_TO_COLKEY, data_path


def _parse_numbers(s):
    if not isinstance(s, str):
        return []
    s = s.replace("\u2013", "-").replace("\ufffd", "-")
    return [float(x) for x in re.findall(r"\d+\.?\d*", s)]


def _parse_min(s):
    n = _parse_numbers(s); return min(n) if n else None


def _parse_max(s):
    n = _parse_numbers(s); return max(n) if n else None


def _build_ladder(initial_str: str, target_str: str) -> list[float]:
    lo = _parse_min(initial_str); hi = _parse_max(target_str)
    if lo is None or hi is None or lo <= 0:
        return [hi] if hi else []
    if lo >= hi:
        return [hi]
    L = [lo]
    while L[-1] * 2 < hi:
        L.append(L[-1] * 2)
    if L[-1] != hi:
        L.append(hi)
    return L


def _load_ladders() -> tuple[dict, dict]:
    df = pd.read_csv(data_path("GDMT_drugs_ref.csv"), encoding="cp1252")
    for c in ("initial_dose", "target_dose"):
        df[c] = df[c].astype(str).str.replace("\ufffd", "-")
    df = df.drop_duplicates(subset=["Class", "GDMT_drug"])
    ladders: dict[str, list[float]] = {}
    for cls, drug in CLASS_TO_REP_DRUG.items():
        row = df[(df["Class"] == cls) & (df["GDMT_drug"] == drug)]
        if len(row):
            col_key = CLASS_TO_COLKEY[cls]
            ladders[col_key] = _build_ladder(row.iloc[0]["initial_dose"], row.iloc[0]["target_dose"])
    targets = {k: L[-1] for k, L in ladders.items()}
    return ladders, targets


LADDERS, TARGETS = _load_ladders()


def step_up(class_key: str, current: float | None, target: float | None, n_steps: int = 1) -> tuple[float, float]:
    """Step up N rungs, capped at target. Returns (new_mg, new_ratio)."""
    L = LADDERS.get(class_key, [target] if target else [])
    tgt = target or (L[-1] if L else current)
    if current is None or current <= 0:
        new = L[0] if L else tgt
        n_steps = max(0, n_steps - 1)
    else:
        new = current
    for _ in range(n_steps):
        higher = [x for x in L if x > new]
        if not higher:
            break
        new = higher[0]
    new = min(new, tgt) if tgt else new
    return float(new), (float(new / tgt) if tgt else 1.0)


def step_down(class_key: str, current: float | None, target: float | None, n_steps: int = 1) -> tuple[float, float]:
    L = LADDERS.get(class_key, [target] if target else [])
    tgt = target or (L[-1] if L else current)
    if current is None or current <= 0:
        return 0.0, 0.0
    new = current
    for _ in range(n_steps):
        lower = [x for x in L if x < new]
        if lower:
            new = lower[-1]
        else:
            new = 0.0
            break
    return float(new), (float(new / tgt) if tgt else 0.0)


def start_dose(class_key: str) -> float | None:
    L = LADDERS.get(class_key)
    return float(L[0]) if L else None


STRATEGIES = {
    "traditional":    dict(step=1, order=["ARNi", "ACEi", "ARB", "beta_blocker", "MRA", "SGLT2i", "loop"]),
    "strong_hf":      dict(step=2),
    "rapid_sequence": dict(step=1),
    "sglt_mra_first": dict(step=1, phase1=["SGLT2i", "MRA"], phase2=["ARNi", "ACEi", "ARB", "beta_blocker"]),
}


def apply_strategy(engine_res: dict, patient: dict, strategy: str = "strong_hf",
                   started_classes: set | None = None) -> dict[str, dict]:
    """Convert engine actions to concrete dose changes under the chosen strategy.
    Returns {class: {engine_action, concrete_action, reason, current, new_dose, new_ratio, target}}."""
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy: {strategy!r}; options: {list(STRATEGIES)}")
    s = STRATEGIES[strategy]
    started = set(started_classes or [])

    eligible = [c for c, d in engine_res["decisions"].items() if d["action"] == "start_medication"]
    if strategy == "traditional":
        allow: list[str] = []
        for c in s["order"]:
            if c in eligible and c not in started:
                allow = [c]; break
    elif strategy == "sglt_mra_first":
        p1 = [c for c in eligible if c in s["phase1"] and c not in started]
        p2 = [c for c in eligible if c in s["phase2"] and c not in started]
        allow = p1 if p1 else p2
    else:
        allow = list(eligible)

    out: dict[str, dict] = {}
    for cls in CLASSES:
        d = engine_res["decisions"][cls]
        cur = patient.get("meds", {}).get(cls, {}).get("dose", 0) or 0
        tgt = TARGETS.get(cls, 0)
        ca = d["action"]
        nd = cur
        nr = (cur / tgt) if (cur and tgt) else None

        if d["action"] == "increase_dose":
            new_mg, new_r = step_up(cls, cur, tgt, n_steps=s.get("step", 1))
            if new_mg <= cur:
                ca = "maintain_dose"
            else:
                nd, nr = new_mg, new_r
        elif d["action"] == "decrease_dose":
            new_mg, new_r = step_down(cls, cur, tgt, n_steps=s.get("step", 1))
            if new_mg == 0.0:
                ca = "stop_medication"
            nd, nr = new_mg, new_r
        elif d["action"] == "start_medication":
            if cls in allow:
                nd = start_dose(cls)
                nr = (nd / tgt) if (nd and tgt) else None
            else:
                ca = "maintain_dose"
                nd = cur
        elif d["action"] == "stop_medication":
            nd = 0.0; nr = 0.0

        out[cls] = dict(
            engine_action=d["action"], concrete_action=ca, reason=d["reason"],
            current=cur, new_dose=nd, new_ratio=nr, target=tgt,
        )
    return out
