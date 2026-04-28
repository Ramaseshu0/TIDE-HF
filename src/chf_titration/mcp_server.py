"""
MCP server — CHF GDMT Titration Assistant.

Clinical decision-support tools for cardiologists and allied health professionals
managing guideline-directed medical therapy (GDMT) in HFrEF patients.
Also exposes patient-facing summaries for shared decision-making.
"""

from __future__ import annotations

from pathlib import Path

from mcp.server.fastmcp import FastMCP

from chf_titration.synthesize import PATIENTS, synthesize_week
from chf_titration.classifier import load_bundle, predict_flags
from chf_titration.engine import TitrationEngine, derive_contraindications, CMAP
from chf_titration.strategy import apply_strategy, STRATEGIES, TARGETS, LADDERS
from chf_titration.constants import CLASSES, LABEL_COLS

BUNDLE_PATH = Path(__file__).resolve().parents[2] / "models" / "chf_classifier_lgbm.pkl"

# ── representative drugs and monitoring ───────────────────────────────────────

_REP_DRUG = {
    "ACEi":         "Lisinopril",
    "ARB":          "Losartan",
    "ARNi":         "Sacubitril-Valsartan (Entresto)",
    "beta_blocker": "Carvedilol",
    "MRA":          "Spironolactone",
    "SGLT2i":       "Empagliflozin (Jardiance)",
    "loop":         "Furosemide",
}

# Clinical thresholds the engine uses (mirrored from engine.py for accurate reporting)
_GLOBAL_STOP_THRESHOLDS = {
    "K > 6.0 mmol/L":              "Severe hyperkalaemia — immediate risk of life-threatening arrhythmia",
    "Cr Δ > 30% from baseline":    "Acute kidney injury — KDIGO stage ≥1 criterion",
    "severe_hypotension_detected":  "Classifier: SBP < 80 mmHg pattern",
    "severe_bradycardia_detected":  "Classifier: HR < 40 bpm pattern",
    "any_emergency_flag":           "Classifier: composite emergency flag",
}

_LAB_AE_THRESHOLDS = {
    "ACEi / ARB / ARNi / MRA": {
        "K > 5.5 mmol/L":      "Hyperkalaemia — hold RAAS/MRA, recheck in 48–72 h",
        "Cr Δ > 30%":          "Functional AKI — typically reversible on dose reduction",
        "eGFR < 30 mL/min":    "Severe CKD — reassess eligibility; ARNi requires eGFR ≥ 30",
    },
    "beta_blocker": {
        "K > 5.5 mmol/L":      "Hyperkalaemia — indirect concern; hold pending labs",
        "Na < 130 mmol/L":     "Hyponatraemia — low perfusion state; hold up-titration",
        "Cr Δ > 30%":          "Worsening renal function — hold BB up-titration",
    },
    "SGLT2i": {
        "K > 5.5 mmol/L":      "Hyperkalaemia — hold; SGLT2i can modestly raise K",
        "Na < 130 mmol/L":     "Hyponatraemia — osmotic diuresis may worsen",
        "eGFR < 20 mL/min":    "Absolute contraindication for glycaemic effect; some benefit persists for HF",
        "Cr Δ > 30%":          "AKI — hold SGLT2i",
    },
    "loop diuretic": {
        "Na < 130 mmol/L":     "Hyponatraemia — loop diuretics worsen; reduce dose",
        "K < 3.5 mmol/L":      "Hypokalaemia — loop-induced; add/increase MRA or KCl",
        "K > 5.5 mmol/L":      "Hyperkalaemia — unusual on loop alone; reassess regimen",
        "Cr Δ > 30%":          "Pre-renal AKI — reduce or hold loop",
    },
}

_MONITORING_SCHEDULE = {
    "ACEi":         "Renal panel + electrolytes: 1–2 wk after start/dose change, then 3-monthly",
    "ARB":          "Renal panel + electrolytes: 1–2 wk after start/dose change, then 3-monthly",
    "ARNi":         "Renal panel + electrolytes: 1–2 wk after start; avoid ACEi for ≥36 h before first dose",
    "beta_blocker": "HR and BP at each visit; no mandatory labs unless renal disease present",
    "MRA":          "K and Cr: 1 wk, 4 wk, 3 months after initiation; target K < 5.5 mmol/L",
    "SGLT2i":       "eGFR at baseline; HbA1c + renal panel 3-monthly; hold if eGFR < 20",
    "loop":         "Electrolytes (K, Na, Cr) 1–2 wk after dose change; daily weights at home",
}

_DRUG_PLAIN = {
    "ACEi":         "ACE inhibitor (e.g. Lisinopril) — reduces strain on your heart and lowers blood pressure",
    "ARB":          "angiotensin receptor blocker (e.g. Losartan) — protects your heart and lowers blood pressure",
    "ARNi":         "Entresto — reduces hospital admissions and helps your heart pump better",
    "beta_blocker": "beta blocker (e.g. Carvedilol) — slows your heart rate and reduces its workload",
    "MRA":          "spironolactone — removes excess fluid and keeps your potassium balanced",
    "SGLT2i":       "Empagliflozin — protects your kidneys and heart",
    "loop":         "Furosemide (water tablet) — removes excess fluid to reduce swelling",
}

_FLAG_PLAIN = {
    "hypotension_detected":        "Blood pressure lower than usual — dizziness when standing. Doctor may temporarily reduce blood pressure medicines.",
    "bradycardia_detected":        "Heart rate slower than normal — may be related to your beta blocker. Doctor may lower the dose slightly.",
    "volume_depletion_detected":   "Body may have lost too much fluid — thirst or dizziness. Doctor may reduce the water tablet.",
    "worsening_HF_detected":       "Heart failure may be getting worse — weight gain and lower oxygen readings. Doctor will review medicines.",
    "hyperkalemia_detected":       "Potassium level appears too high — can affect heart rhythm. Doctor will order a blood test and may pause some medicines.",
    "renal_dysfunction_detected":  "Kidneys under more stress. Doctor will order blood tests and may adjust medicines.",
    "hyponatremia_detected":       "Salt (sodium) level appears low — can cause tiredness or headaches. Doctor may adjust the water tablet.",
    "metabolic_acidosis_detected": "Blood chemistry imbalance detected. Doctor will check blood tests and adjust medicines.",
    "severe_hypotension_detected": "URGENT — blood pressure dangerously low. Contact your care team immediately or go to A&E.",
    "severe_bradycardia_detected": "URGENT — heart rate dangerously slow. Contact your care team immediately.",
    "any_emergency_flag":          "URGENT — serious warning detected. Contact your care team or go to hospital now.",
}

# ── server ─────────────────────────────────────────────────────────────────────

mcp = FastMCP(
    "CHF Titration Assistant",
    instructions=(
        "You are a clinical decision-support assistant for CHF GDMT titration.\n\n"
        "CLINICIAN MODE — use precise medical terminology:\n"
        "  • Report exact lab values and thresholds (K > 5.5 mmol/L, Cr Δ > 30%, eGFR mL/min/1.73m²)\n"
        "  • Use drug class names (RAAS, ARNi, MRA, SGLT2i, beta-blocker, loop diuretic)\n"
        "  • Reference dose ladders and titration ratios (current/target)\n"
        "  • Cite AE triggers and monitoring schedules\n"
        "  • Flag global-stop conditions explicitly with clinical rationale\n\n"
        "PATIENT MODE — plain language only:\n"
        "  • Say 'heart medicines', 'blood test', 'kidney check', 'water tablet'\n"
        "  • Never use: AUROC, titration, RAAS, ARNi, MRA, contraindication, eGFR\n"
        "  • Focus on what changes, what to watch for, and when to call the care team\n\n"
        "Always call list_patient_presets() first if you are unsure of the exact preset name."
    ),
)

_bundle_cache: dict | None = None


def _bundle() -> dict:
    global _bundle_cache
    if _bundle_cache is None:
        b = load_bundle(BUNDLE_PATH)
        if b is None:
            raise RuntimeError(
                f"Trained model not found at {BUNDLE_PATH}. "
                "Run `python scripts/prepare.py` from the package root."
            )
        _bundle_cache = b
    return _bundle_cache


def _validate_preset(name: str) -> str | None:
    if name not in PATIENTS:
        close = [k for k in PATIENTS if name.lower() in k.lower()]
        hint = f" Did you mean: {close[0]!r}?" if close else " Call list_patient_presets() to see all names."
        return f"Unknown preset {name!r}.{hint}"
    return None


def _validate_strategy(s: str) -> str | None:
    if s not in STRATEGIES:
        return f"Unknown strategy {s!r}. Valid: {', '.join(STRATEGIES.keys())}"
    return None


def _pipeline(preset: str, strategy: str, labs_available: bool) -> dict:
    err = _validate_preset(preset) or _validate_strategy(strategy)
    if err:
        raise ValueError(err)
    b = _bundle()
    patient = PATIENTS[preset]
    tps = synthesize_week(patient)
    pf = {**patient, "targets": TARGETS}
    flags, probs = predict_flags(pf, tps, b)
    labs = patient.get("labs") if (labs_available and patient.get("labs")) else None
    result = TitrationEngine().evaluate(pf, tps, flags, labs=labs, awaiting_labs=set())
    changes = apply_strategy(result, pf, strategy=strategy)
    return {"patient": patient, "tps": tps, "flags": flags,
            "probs": probs, "result": result, "changes": changes}


def _dose_line(action: str, cur, nd) -> str:
    cur = cur or 0
    if action == "start_medication":   return f"START @ {nd}mg"
    if action == "stop_medication":    return f"{cur}mg → STOP"
    if action == "hold_titration":     return f"{cur}mg (HOLD — awaiting labs/AE resolution)"
    if action == "maintain_dose":      return f"{cur}mg (maintain)"
    return f"{cur}mg → {nd}mg"


def _ratio_bar(ratio) -> str:
    if ratio is None:
        return "—"
    filled = int(round(ratio * 10))
    return "█" * filled + "░" * (10 - filled) + f"  {ratio:.0%}"


# ═══════════════════════════════════════════════════════════════════════════════
# CLINICIAN TOOLS
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def list_patient_presets() -> str:
    """
    List all 17 available patient preset scenarios grouped by clinical category.

    Returns the exact preset name (required by all other tools), category,
    and a one-line clinical description of what decision path each preset tests.

    Always call this first if you do not know the exact preset name.
    """
    cats = {
        "Titration state": [
            "Newly diagnosed, stable",
            "Partial titration, no AEs",
            "Fully titrated, at target",
        ],
        "Adverse effects (AE)": [
            "Suspected hyperkalemia",
            "Suspected renal dysfunction",
            "Confirmed hyperkalemia (K=6.3)",
            "Volume depletion + hypotension",
            "Worsening HF",
            "Bradycardia on beta blocker",
            "Worsening HF + hyperkalemia (labs confirmed)",
            "Hyponatremia + renal dysfunction (labs)",
        ],
        "Contraindications": [
            "Angioedema history (ARNi + ACEi blocked)",
            "Severe asthma (beta blocker blocked)",
            "CKD stage 4 (eGFR=22)",
            "Pregnant (RAAS blocked)",
            "AV block (beta blocker blocked)",
            "Severe hypotension (global stop)",
        ],
    }
    lines = []
    for cat, names in cats.items():
        lines.append(f"\n── {cat} ──")
        for name in names:
            p = PATIENTS[name]
            bl = p["baseline"]
            meds_on = [cls for cls in CLASSES if p["meds"][cls]["on"]]
            lines.append(f"  • {name!r}")
            lines.append(f"    {p['age']}yo {p['gender']}  |  K={bl['K']} Na={bl['Na']} Cr={bl['Cr']:.2f} eGFR={bl['eGFR']}")
            lines.append(f"    GDMT on: {', '.join(meds_on) if meds_on else '(none)'}")
            lines.append(f"    Note: {p.get('note', '')}")
    return "\n".join(lines)


@mcp.tool()
def get_patient_snapshot(preset_name: str) -> str:
    """
    Return a full clinical snapshot for a patient preset.

    Includes demographics, baseline metabolic panel (K, Na, Cr, eGFR, BUN, HCO3, pH),
    current GDMT medications with doses and dose ratios (current/target),
    derived contraindication flags, and labs available this cycle (if any).

    Use this to verify patient context before running a titration analysis.

    Args:
        preset_name: Exact preset name from list_patient_presets().
    """
    err = _validate_preset(preset_name)
    if err:
        return err

    p = PATIENTS[preset_name]
    bl = p["baseline"]
    labs = p.get("labs")

    lines = [
        f"Clinical Snapshot: {preset_name}",
        "═" * 60,
        f"  Age / Sex       : {p['age']}yo {p['gender']}",
        "",
        "  Baseline metabolic panel:",
        f"    K              : {bl['K']} mmol/L  {'⚠ HIGH' if bl['K'] > 5.0 else '⚠ LOW' if bl['K'] < 3.5 else 'normal (3.5–5.0)'}",
        f"    Na             : {bl['Na']} mmol/L  {'⚠ LOW' if bl['Na'] < 135 else 'normal (135–145)'}",
        f"    Creatinine     : {bl['Cr']:.2f} mg/dL",
        f"    eGFR           : {bl['eGFR']:.0f} mL/min/1.73m²  {_egfr_stage(bl['eGFR'])}",
        f"    BUN            : {bl.get('BUN', '?')} mg/dL",
        f"    HCO3           : {bl.get('HCO3', '?')} mmol/L",
        f"    pH             : {bl.get('pH', '?')}",
        "",
    ]

    if labs:
        cr_delta = None
        if bl.get("Cr") and labs.get("Cr"):
            cr_delta = (labs["Cr"] - bl["Cr"]) / bl["Cr"] * 100
        lines += [
            "  Labs this cycle (current values):",
            f"    K              : {labs['K']} mmol/L  {'⛔ > 6.0 → GLOBAL STOP' if labs['K'] > 6.0 else '⚠ HOLD RAAS/MRA' if labs['K'] > 5.5 else 'normal'}",
            f"    Na             : {labs['Na']} mmol/L  {'⚠ HYPONATRAEMIA' if labs['Na'] < 130 else 'normal'}",
            f"    Creatinine     : {labs['Cr']:.2f} mg/dL" + (f"  (Δ {cr_delta:+.1f}% from baseline {'⛔ > 30% → AKI criterion' if cr_delta > 30 else ''})" if cr_delta is not None else ""),
            f"    eGFR           : {labs['eGFR']:.0f} mL/min/1.73m²  {_egfr_stage(labs['eGFR'])}",
            "",
        ]
    else:
        lines.append("  Labs this cycle : not available\n")

    lines.append("  Current GDMT medications:")
    meds_any = False
    for cls in CLASSES:
        m = p["meds"][cls]
        if m["on"]:
            meds_any = True
            tgt = TARGETS.get(cls, 0) or 1
            ratio = m["dose"] / tgt
            ladder = LADDERS.get(cls, [])
            lines.append(f"    {cls:<14} {_REP_DRUG.get(cls, cls):<32} {m['dose']:>6}mg / {tgt}mg target  {_ratio_bar(ratio)}")
            if ladder:
                lines.append(f"    {'':14}   Ladder: {' → '.join(str(int(x)) + 'mg' for x in ladder)}")
    if not meds_any:
        lines.append("    (no GDMT started)")

    contras = derive_contraindications(p)
    active_c = [k for k, v in contras.items() if v]
    lines.append(f"\n  Active contraindications ({len(active_c)}):")
    if active_c:
        for flag in active_c:
            blocked = [cls for cls, flist in CMAP.items() if flag in flist]
            lines.append(f"    ⛔ {flag}  →  blocks: {', '.join(blocked) if blocked else 'none'}")
    else:
        lines.append("    none — all 7 drug classes eligible")

    lines.append(f"\n  Clinical note: {p.get('note', '')}")
    return "\n".join(lines)


def _egfr_stage(egfr: float) -> str:
    if egfr >= 90:   return "G1 (normal)"
    if egfr >= 60:   return "G2 (mildly reduced)"
    if egfr >= 45:   return "G3a (mild–moderate)"
    if egfr >= 30:   return "G3b (moderate–severe)"
    if egfr >= 15:   return "G4 (severe CKD)"
    return "G5 (kidney failure)"


@mcp.tool()
def run_titration(
    preset_name: str,
    strategy: str = "strong_hf",
    labs_available: bool = False,
) -> str:
    """
    Run the full GDMT titration pipeline: classifier → titration engine → strategy applier.

    Pipeline steps:
      1. LightGBM classifier scores 11 adverse-effect flags from 108 features
         derived from 14 vital-sign timepoints + patient demographics + current doses.
      2. Titration engine applies rule-based logic: global-stop check → contraindication
         derivation → per-class AE assessment (immediate AE / suspected AE / lab AE) →
         RAAS preference selection → titration decision per class.
      3. Strategy applier maps engine decisions to concrete dose changes using the
         class-specific dose ladder (doublings from initial to target dose).

    Returns:
      • Safety alerts: global-stop trigger + clinical rationale, or order-labs request
      • All 11 classifier flag probabilities (threshold 0.5) with probability bars
      • Per-class recommendation: engine action, concrete action, dose change,
        current/target ratio, and reason code
      • RAAS preference chain and active RAAS class
      • Weighted risk score

    Args:
        preset_name: Exact preset name from list_patient_presets().
        strategy: 'traditional' | 'strong_hf' | 'rapid_sequence' | 'sglt_mra_first'
        labs_available: True if K, Na, Cr, eGFR results are available this cycle.
                        If True and patient.labs is set, lab-AE gating is applied.
    """
    try:
        d = _pipeline(preset_name, strategy, labs_available)
    except Exception as e:
        return f"ERROR: {e}"

    result  = d["result"]
    changes = d["changes"]
    flags   = d["flags"]
    probs   = d["probs"]
    patient = d["patient"]
    labs    = patient.get("labs") if labs_available else None
    bl      = patient["baseline"]

    active_flags  = [k for k, v in flags.items() if v]
    SEVERE = {"severe_hypotension_detected", "severe_bradycardia_detected", "any_emergency_flag"}
    weighted_sum  = sum((3 if k in SEVERE else 1) * v for k, v in probs.items())
    max_possible  = sum(3 if k in SEVERE else 1 for k in probs)
    risk_pct      = round(weighted_sum / max_possible * 100, 1)
    severe_active = [k for k in active_flags if k in SEVERE]
    risk_level    = "CRITICAL" if severe_active else "HIGH" if len(active_flags) >= 3 else "MODERATE" if active_flags else "LOW"

    if labs:
        labs_str = "YES — K=" + str(labs["K"]) + " Na=" + str(labs["Na"]) + " Cr=" + f'{labs["Cr"]:.2f}' + " eGFR=" + str(labs["eGFR"])
    else:
        labs_str = "NO (baseline only)"

    lines = [
        f"GDMT Titration Report — {preset_name}",
        "═" * 70,
        f"  Strategy     : {strategy}",
        f"  Labs used    : {labs_str}",
        f"  Risk score   : {risk_pct:.1f}%  [{risk_level}]",
        "",
    ]

    # ── Safety alerts ──
    if result["global_stop"]:
        lines.append("  ╔══ ⛔ GLOBAL STOP ══════════════════════════════════════╗")
        lines.append("  ║  Hold ALL GDMT immediately. Escalate for urgent review. ║")
        # identify trigger
        if labs and labs.get("K", 0) > 6.0:
            lines.append(f"  ║  Trigger: K = {labs['K']} mmol/L (> 6.0 threshold)           ║")
        if severe_active:
            lines.append(f"  ║  Classifier: {', '.join(severe_active)}")
        lines.append("  ╚════════════════════════════════════════════════════════╝\n")
    elif result["order_labs"]:
        req = ", ".join(sorted(result["labs_requested"]))
        lines.append(f"  ⚠  ORDER LABS (K, Na, Cr, eGFR) — suspected AE in: {req}")
        lines.append(f"     Hold these classes until results return. Reassess next cycle.\n")

    # ── Classifier flags ──
    lines.append("── Classifier flags (LightGBM, threshold p ≥ 0.50) ──────────────────")
    lines.append(f"  Active ({len(active_flags)}/11): {', '.join(active_flags) if active_flags else 'none'}")
    lines.append(f"  {'Flag':<40}  {'p':>5}  {'Active':<7}  Bar")
    lines.append("  " + "─" * 68)
    for flag in LABEL_COLS:
        p_val  = probs[flag]
        active = flags[flag]
        marker = "▶ YES" if active else "  no "
        bar    = "█" * int(p_val * 20) + "░" * (20 - int(p_val * 20))
        sev    = " ⚡" if flag in SEVERE else ""
        lines.append(f"  {flag:<40}  {p_val:.3f}  {marker}    {bar}{sev}")

    # ── Drug recommendations ──
    lines.append("")
    lines.append("── Drug-by-drug recommendations ─────────────────────────────────────")
    lines.append(f"  {'Class':<14}  {'Drug (rep)':<32}  {'Action':<22}  {'Dose':<26}  {'Ratio':^12}  Reason")
    lines.append("  " + "─" * 120)
    for cls in CLASSES:
        c   = changes[cls]
        dt  = _dose_line(c["concrete_action"], c["current"], c["new_dose"])
        tgt = TARGETS.get(cls) or 0
        cur = c["current"] or 0
        new = c["new_dose"] or 0
        new_ratio = (new / tgt) if (new and tgt) else None
        ratio_str = _ratio_bar(new_ratio)
        drug = _REP_DRUG.get(cls, "")
        lines.append(
            f"  {cls:<14}  {drug:<32}  {c['concrete_action']:<22}  {dt:<26}  {ratio_str}  {c['reason']}"
        )

    # ── RAAS summary ──
    lines += [
        "",
        f"  Preferred RAAS : {result['preferred_raas'] or '(none — all RAAS agents contraindicated)'}",
        f"  Active RAAS    : {result['active_raas'] or '(none currently on)'}",
        f"  Order labs     : {'YES — classes: ' + ', '.join(sorted(result['labs_requested'])) if result['order_labs'] else 'no'}",
        f"  Global stop    : {'YES' if result['global_stop'] else 'no'}",
        f"  Escalate       : {'YES' if result['escalate'] else 'no'}",
    ]
    return "\n".join(lines)


@mcp.tool()
def get_gdmt_optimization_score(preset_name: str) -> str:
    """
    Calculate the GDMT optimization score for a patient's current regimen.

    For each drug class:
      • Reports current dose, target dose, and % of target achieved
      • Identifies classes not yet started, below target, or at target
      • Computes an overall optimization score = mean % of target across all 7 classes
      • Classifies optimization: Optimal (≥ 80%), Partial (40–79%), Suboptimal (< 40%)

    This score is independent of AE status — it reflects dosing completeness only.
    Use run_titration() to incorporate safety considerations.

    Args:
        preset_name: Exact preset name from list_patient_presets().
    """
    err = _validate_preset(preset_name)
    if err:
        return err

    p = PATIENTS[preset_name]
    contras = derive_contraindications(p)

    lines = [
        f"GDMT Optimization Score: {preset_name}",
        "═" * 60,
    ]

    ratios = []
    for cls in CLASSES:
        m   = p["meds"][cls]
        tgt = TARGETS.get(cls) or 0
        cur = m["dose"] if m["on"] else 0
        blocked = [k for k in CMAP.get(cls, []) if contras.get(k)]

        if blocked:
            status = f"CONTRAINDICATED ({blocked[0]})"
            ratio  = None
        elif not m["on"]:
            status = "not started"
            ratio  = 0.0
        elif tgt and cur >= tgt:
            status = "at target ✓"
            ratio  = 1.0
        elif tgt:
            status = f"below target ({cur}/{tgt}mg)"
            ratio  = cur / tgt
        else:
            status = f"on {cur}mg (no target defined)"
            ratio  = None

        ladder = LADDERS.get(cls, [])
        next_dose = None
        if ladder and cur is not None:
            higher = [x for x in ladder if x > cur]
            next_dose = higher[0] if higher else None

        bar = _ratio_bar(ratio) if ratio is not None else "N/A (contraindicated)"
        lines.append(f"\n  {cls} ({_REP_DRUG.get(cls, '')})")
        lines.append(f"    Current / Target : {cur}mg / {tgt}mg")
        lines.append(f"    Progress         : {bar}")
        lines.append(f"    Status           : {status}")
        if next_dose and not blocked and m["on"] and cur < tgt:
            lines.append(f"    Next dose rung   : {next_dose}mg")
        if ladder:
            lines.append(f"    Full ladder      : {' → '.join(str(int(x)) + 'mg' for x in ladder)}")

        if ratio is not None:
            ratios.append(ratio)

    if ratios:
        overall = sum(ratios) / len(ratios)
        grade   = "Optimal (≥ 80%)" if overall >= 0.8 else "Partial (40–79%)" if overall >= 0.4 else "Suboptimal (< 40%)"
        lines += [
            "",
            "═" * 60,
            f"  Overall optimization score : {overall:.0%}",
            f"  Grade                      : {grade}",
            f"  Classes at target          : {sum(1 for r in ratios if r >= 1.0)} / {len(ratios)}",
            f"  Classes not started        : {sum(1 for cls in CLASSES if not p['meds'][cls]['on'] and not any(contras.get(k) for k in CMAP.get(cls, [])))}",
        ]
    return "\n".join(lines)


@mcp.tool()
def compare_strategies(preset_name: str, labs_available: bool = True) -> str:
    """
    Run all four GDMT titration strategies for one patient and compare them side-by-side.

    For each strategy shows:
      • Any safety alerts (global stop / order labs)
      • Per-class action, dose change, and new dose ratio
      • Strategies may differ in how many classes start simultaneously and dose step size.

    Use this to select the most appropriate protocol given the patient's tolerance,
    urgency, and comorbidity profile.

    Args:
        preset_name: Exact preset name.
        labs_available: Whether lab results are available this cycle (default True).
    """
    err = _validate_preset(preset_name)
    if err:
        return err

    descs = {
        "traditional":    "1 class/week · order: RAAS→BB→MRA→SGLT2i→loop · single-rung steps",
        "strong_hf":      "all eligible classes at once · double-rung steps · aggressive",
        "rapid_sequence": "all eligible classes at once · single-rung steps · Greene et al. 2021",
        "sglt_mra_first": "phase 1: SGLT2i+MRA · phase 2: ARNi+BB · cardiorenal-first",
    }

    lines = [f"Strategy comparison: {preset_name}", "═" * 60]
    for strat, desc in descs.items():
        lines.append(f"\n┌─ {strat}")
        lines.append(f"│  {desc}")
        try:
            d = _pipeline(preset_name, strat, labs_available)
            result  = d["result"]
            changes = d["changes"]
            if result["global_stop"]:
                lines.append("│  ⛔ GLOBAL STOP — all strategies equally affected by global triggers")
            elif result["order_labs"]:
                req = ", ".join(sorted(result["labs_requested"]))
                lines.append(f"│  ⚠  ORDER LABS — hold: {req}")
            else:
                lines.append(f"│  {'Class':<14}  {'Action':<22}  Dose change")
                for cls in CLASSES:
                    c  = changes[cls]
                    dt = _dose_line(c["concrete_action"], c["current"], c["new_dose"])
                    tgt = TARGETS.get(cls) or 0
                    new = c["new_dose"] or 0
                    ratio_str = _ratio_bar(new / tgt) if (new and tgt) else "—"
                    marker = ("🟢" if c["concrete_action"] == "start_medication"  else
                              "🔼" if "increase" in c["concrete_action"] or "double" in c["concrete_action"] else
                              "✅" if c["concrete_action"] == "maintain_dose"     else
                              "⏸" if c["concrete_action"] == "hold_titration"    else
                              "🔽" if "decrease" in c["concrete_action"]          else
                              "🛑" if c["concrete_action"] == "stop_medication"   else "·")
                    lines.append(f"│  {marker} {cls:<14}  {c['concrete_action']:<22}  {dt}  [{ratio_str}]")
        except Exception as e:
            lines.append(f"│  ERROR: {e}")
        lines.append("└" + "─" * 58)
    return "\n".join(lines)


@mcp.tool()
def check_contraindications(preset_name: str) -> str:
    """
    Derive all contraindications for a patient and map them to blocked drug classes.

    Contraindications are derived from:
      • Baseline lab thresholds: K > 5.5 mmol/L, Na < 130, eGFR < 20 mL/min
      • Diagnosis codes: angioedema history, pregnancy, bilateral RAS, severe asthma, AV block
      • Historical codes: DKA (E101/E111), Type 1 DM
      • Early HR: mean HR < 50 bpm in first 2 timepoints → baseline_hr_less_than_50
      • ACEi current use → acei_within_36_hours (ARNi wash-out requirement)

    Returns all 14 contraindication flags, which are active, and which GDMT classes each blocks.

    Args:
        preset_name: Exact preset name.
    """
    err = _validate_preset(preset_name)
    if err:
        return err

    patient = PATIENTS[preset_name]
    c = derive_contraindications(patient)
    active = {k: v for k, v in c.items() if v}

    # Clinical rationale per flag
    _rationale = {
        "angioedema_history":               "Absolute contra for ACEi (class-wide) and ARNi; use ARB",
        "pregnancy":                        "All RAAS agents teratogenic; SGLT2i also avoided",
        "acei_within_36_hours":             "ARNi requires ≥ 36 h washout from ACEi to prevent angioedema",
        "bilateral_renal_artery_stenosis":  "RAAS inhibition drops GFR bilaterally; risk of renal failure",
        "baseline_potassium_more_than_5_5": "K > 5.5 mmol/L: hyperkalaemia risk with RAAS/MRA",
        "severe_asthma":                    "Non-selective beta-blockers (carvedilol) worsen bronchospasm",
        "av_block":                         "Beta-blockers worsen AV nodal conduction; risk of complete block",
        "baseline_hr_less_than_50":         "Resting HR < 50 bpm: beta-blocker will further suppress SAN",
        "history_dka":                      "SGLT2i increases euglycaemic DKA risk",
        "type_1_diabetes":                  "SGLT2i: DKA risk in absolute insulin deficiency",
        "egfr_less_than_20":                "MRA, SGLT2i, loop: inadequate filtration / electrolyte risk",
        "severe_baseline_volume_depletion": "SGLT2i osmotic diuresis worsens hypovolaemia",
        "sodium_less_than_130":             "Loop diuretics worsen hyponatraemia (↑ADH stimulus)",
        "potassium_less_than_3_0":          "Loop diuretics worsen hypokalaemia → arrhythmia risk",
    }

    lines = [f"Contraindication Analysis: {preset_name}", "═" * 60]
    if not active:
        lines.append("  ✅ No contraindications active — all 7 GDMT classes are eligible.")
    else:
        lines.append(f"  {len(active)} active contraindication(s):\n")
        for flag in sorted(active):
            blocked  = [cls for cls, flist in CMAP.items() if flag in flist]
            rationale = _rationale.get(flag, "—")
            lines.append(f"  ⛔ {flag}")
            lines.append(f"     Blocks    : {', '.join(blocked) if blocked else 'none directly'}")
            lines.append(f"     Rationale : {rationale}\n")

    lines.append("Full contraindication flag map (14 flags):")
    lines.append(f"  {'Flag':<45}  Status")
    lines.append("  " + "─" * 60)
    for flag, val in sorted(c.items()):
        status = "✓ ACTIVE" if val else "— clear"
        lines.append(f"  {flag:<45}  {status}")
    return "\n".join(lines)


@mcp.tool()
def list_strategies() -> str:
    """
    Describe the four GDMT titration strategies with clinical rationale and evidence base.

    Use this to select the most appropriate protocol for a specific patient presentation.
    """
    return (
        "GDMT Titration Strategies — Clinical Reference\n"
        "═" * 60 + "\n\n"

        "1. traditional\n"
        "   Initiation  : One new drug class per visit/week.\n"
        "   Order       : RAAS (ARNi preferred → ACEi → ARB) → beta-blocker → MRA → SGLT2i → loop.\n"
        "   Dose steps  : Single-rung (each rung = doubling from initial to target).\n"
        "   Time to full: 4–8 weeks to reach quadruple therapy at starting doses.\n"
        "   Evidence    : Classic ESC/ACC HF guideline approach; well-studied tolerability.\n"
        "   Indication  : Elderly, frail, or comorbid patients; new to GDMT; haemodynamic concerns.\n\n"

        "2. strong_hf\n"
        "   Initiation  : All eligible classes started simultaneously at first visit.\n"
        "   Dose steps  : Double-rung (skips one dose level; faster to target).\n"
        "   Time to full: 1–2 visits to starting doses; fewer visits to target.\n"
        "   Evidence    : Mirrors aggressive HF titration used in clinical trials (PARADIGM-HF starting arm).\n"
        "   Indication  : Young, haemodynamically stable HFrEF; no AE concerns; close follow-up feasible.\n"
        "   Caution     : Higher short-term risk of hypotension, hyperkalaemia, worsening renal function.\n\n"

        "3. rapid_sequence\n"
        "   Initiation  : All eligible classes started simultaneously.\n"
        "   Dose steps  : Single-rung per cycle.\n"
        "   Evidence    : Greene SJ et al. JAMA Cardiol 2021; Maddox et al. rapid up-titration protocol.\n"
        "   Indication  : Newly diagnosed HFrEF; balance of speed and safety preferred over aggressive.\n"
        "   Advantage   : Achieves quadruple therapy faster than traditional without double-rung risk.\n\n"

        "4. sglt_mra_first\n"
        "   Phase 1     : Start SGLT2i + MRA at first visit.\n"
        "   Phase 2     : Add ARNi (preferred RAAS) + beta-blocker.\n"
        "   Dose steps  : Single-rung throughout.\n"
        "   Rationale   : Early cardiorenal protection (SGLT2i: DAPA-HF, EMPEROR-Reduced) + "
        "aldosterone blockade (MRA: RALES, EMPHASIS-HF) before neurohormonal-axis agents.\n"
        "   Indication  : CKD G2–G3, diabetic cardiomyopathy, or where loop/MRA benefit is priority.\n"
        "   Note        : RAAS/BB lag one cycle; monitor K closely when MRA added."
    )


@mcp.tool()
def get_classifier_info() -> str:
    """
    Return metadata about the trained LightGBM adverse-effect classifier.

    Reports: number of features, flags, and per-flag test AUROCs computed on a
    held-out 15% test set (group-split by subject_id to avoid patient-level leakage).

    AUROC interpretation:
      ≥ 0.90 — Excellent discrimination
      0.80–0.90 — Good discrimination
      0.70–0.80 — Fair; use flag with caution; cross-check vitals
      < 0.70 — Poor; flag probability is informative but not reliable alone

    Feature set: 14 timepoints × vital signs (HR, SBP, DBP, SpO2, weight_kg)
    + ECG scalars (QRS_ms, QTc_ms, PR_ms, T_amp_mV, ST_dev_mm) + ECG categorical (rhythm,
    T_peaked) + patient demographics (age, gender) + per-class dose ratios (7 classes × 3 cols)
    = 108 features total.
    """
    try:
        b = _bundle()
    except Exception as e:
        return f"ERROR: {e}"

    lines = [
        "LightGBM Adverse-Effect Classifier",
        "═" * 60,
        f"  Feature columns      : {len(b['feature_cols'])}",
        f"  Adverse-effect flags : {len(b['label_cols'])}",
        f"  Training seed        : {b.get('random_state', '—')}",
        f"  Test set split       : 15% held-out, grouped by subject_id",
        "",
        "Per-flag test AUROCs:",
        f"  {'Flag':<40}  {'AUROC':>6}  Grade       Bar (20-char)",
        "  " + "─" * 72,
    ]
    for flag, auroc in b.get("test_metrics", {}).items():
        grade = ("Excellent" if auroc >= 0.90 else
                 "Good"      if auroc >= 0.80 else
                 "Fair"      if auroc >= 0.70 else "Poor")
        bar   = "█" * int(auroc * 20) + "░" * (20 - int(auroc * 20))
        lines.append(f"  {flag:<40}  {auroc:.3f}  {grade:<10}  {bar}")
    return "\n".join(lines)


@mcp.tool()
def get_vital_trends(preset_name: str) -> str:
    """
    Synthesize 14 vital-sign timepoints for a patient preset and return as a clinical table.

    These are the raw observations fed into the LightGBM classifier's feature extractor.
    14 timepoints represent twice-daily measurements over a 7-day period.

    Columns: Day, HR (bpm), SBP (mmHg), DBP (mmHg), SpO2 (%), Weight (kg),
             QRS (ms), QTc (ms), T-peaked (Y/N), Rhythm.

    Clinical reference ranges shown for each column.

    Args:
        preset_name: Exact preset name.
    """
    err = _validate_preset(preset_name)
    if err:
        return err

    tps = synthesize_week(PATIENTS[preset_name])
    lines = [
        f"Vital Trends: {preset_name}  (14 timepoints, ~twice-daily over 7 days)",
        "═" * 80,
        f"  Reference ranges: HR 60–100 bpm · SBP 90–140 mmHg · SpO2 ≥ 95% · QRS < 120ms · QTc < 450ms (M) / 460ms (F)",
        "",
        f"  {'Day':>3}  {'HR':>5}  {'SBP':>5}  {'DBP':>5}  {'SpO2':>6}  {'Wt(kg)':>8}  {'QRS':>5}  {'QTc':>5}  {'T-pk':<5}  Rhythm",
        "  " + "─" * 74,
    ]
    for i, t in enumerate(tps):
        e   = t.get("ecg", {})
        hr  = t.get("HR")  or 0
        sbp = t.get("SBP") or 0
        spo = t.get("SpO2") or 0

        hr_flag  = " ⚠" if hr < 50 or hr > 120 else ""
        sbp_flag = " ⚠" if sbp < 90 or sbp > 150 else ""
        spo_flag = " ⚠" if spo < 94 else ""
        qrs      = e.get("qrs_ms") or 0
        qtc      = e.get("qtc_ms") or 0
        qrs_flag = " ⚠" if qrs > 120 else ""
        qtc_flag = " ⚠" if qtc > 450 else ""

        lines.append(
            f"  {i+1:>3}  {hr:>5.0f}{hr_flag:<2}  {sbp:>5.0f}{sbp_flag:<2}  "
            f"{(t.get('DBP') or 0):>5.0f}  {spo:>6.1f}{spo_flag:<2}  "
            f"{(t.get('weight_kg') or 0):>8.1f}  "
            f"{qrs:>5.0f}{qrs_flag:<2}  {qtc:>5.0f}{qtc_flag:<2}  "
            f"{'Y' if e.get('t_peaked') else 'N':<5}  {e.get('rhythm', 'NSR')}"
        )
    return "\n".join(lines)


@mcp.tool()
def assess_clinical_urgency(preset_name: str, labs_available: bool = False) -> str:
    """
    Classify the clinical urgency tier for a patient and state the required action timeframe.

    Urgency tiers:
      TIER 1 — IMMEDIATE (< 4 hours): Global-stop conditions.
                K > 6.0 mmol/L, Cr Δ > 30%, severe hypotension/bradycardia, emergency flag.
      TIER 2 — URGENT (< 24 hours):   Labs required; hold affected classes.
                Suspected hyperK/renal (ECG signs or weight drift), order panel.
      TIER 3 — THIS VISIT:             AE flags active but haemodynamically stable.
                Titrate conservatively, plan close follow-up in 1–2 weeks.
      TIER 4 — ROUTINE:               No active AE flags; titrate per strategy.
                Standard follow-up 4–8 weeks after dose change.

    Args:
        preset_name: Exact preset name.
        labs_available: Whether labs are available this cycle.
    """
    err = _validate_preset(preset_name)
    if err:
        return err

    try:
        d = _pipeline(preset_name, "strong_hf", labs_available)
    except Exception as e:
        return f"ERROR: {e}"

    result = d["result"]
    flags  = d["flags"]
    probs  = d["probs"]
    patient = d["patient"]
    labs = patient.get("labs") if labs_available else None
    active = [k for k, v in flags.items() if v]

    SEVERE = {"severe_hypotension_detected", "severe_bradycardia_detected", "any_emergency_flag"}
    SUSPECTED = {"hyperkalemia_detected", "renal_dysfunction_detected",
                 "hyponatremia_detected", "worsening_HF_detected"}
    STABLE_AE = {"hypotension_detected", "bradycardia_detected",
                 "volume_depletion_detected", "metabolic_acidosis_detected"}

    if result["global_stop"]:
        tier = 1; label = "IMMEDIATE (< 4 hours)"
        actions = [
            "Hold ALL GDMT classes immediately",
            "Obtain urgent ECG + stat labs (K, Cr, BMP)",
            "IV access; prepare for haemodynamic support if SBP < 80",
            "Cardiology/ICU consult within 1–2 hours",
            "Do NOT restart GDMT until labs reviewed and haemodynamics stable",
        ]
    elif result["order_labs"]:
        tier = 2; label = "URGENT (within 24 hours)"
        held = sorted(result["labs_requested"])
        actions = [
            f"Hold: {', '.join(held)} pending lab results",
            "Order: K, Na, Cr, eGFR (stat or same-day)",
            "Resume and titrate only after labs reviewed",
            "Follow-up in 48–72 hours to review results",
            "If labs not returned within 24 h, repeat clinical assessment",
        ]
    elif any(k in STABLE_AE for k in active):
        tier = 3; label = "THIS VISIT"
        actions = [
            "Proceed with conservative titration (single-rung) for unaffected classes",
            "Reduce dose of affected class by 1 rung; do not stop unless haemodynamically unstable",
            "Arrange follow-up in 1–2 weeks with repeat vitals",
            "Counsel patient: symptom diary, daily weights, contact threshold",
        ]
    else:
        tier = 4; label = "ROUTINE"
        actions = [
            "Titrate per selected strategy",
            "Routine follow-up 4 weeks after each dose change",
            "Repeat metabolic panel 1–2 weeks after RAAS/MRA dose change",
            "Annual review of GDMT targets and tolerability",
        ]

    # Risk breakdown
    weighted_sum = sum((3 if k in SEVERE else 1) * probs[k] for k in probs)
    max_possible = sum(3 if k in SEVERE else 1 for k in probs)
    risk_pct = round(weighted_sum / max_possible * 100, 1)

    lines = [
        f"Clinical Urgency Assessment: {preset_name}",
        "═" * 60,
        f"  Urgency tier : TIER {tier} — {label}",
        f"  Risk score   : {risk_pct:.1f}%",
        f"  Active flags : {', '.join(active) if active else 'none'}",
        f"  Global stop  : {'YES' if result['global_stop'] else 'no'}",
        f"  Order labs   : {'YES' if result['order_labs'] else 'no'}",
        "",
        "  Required actions:",
    ]
    for i, action in enumerate(actions, 1):
        lines.append(f"    {i}. {action}")

    lines += [
        "",
        "  Flag probability summary:",
        f"  {'Category':<28}  Flags active",
        "  " + "─" * 50,
        f"  Severe / emergency       :  {', '.join(k for k in active if k in SEVERE) or 'none'}",
        f"  Suspected AE (labs req.) :  {', '.join(k for k in active if k in SUSPECTED) or 'none'}",
        f"  Stable AE (haemo. OK)    :  {', '.join(k for k in active if k in STABLE_AE) or 'none'}",
    ]
    return "\n".join(lines)


@mcp.tool()
def get_lab_monitoring_plan(preset_name: str, strategy: str = "strong_hf") -> str:
    """
    Generate a lab monitoring plan for all active GDMT classes after a titration cycle.

    For each drug class that is active or being started, states:
      • Which labs to monitor and the exact threshold that would require action
      • Timing of first recheck (1 wk, 2 wk, or 4 wk depending on class and event)
      • Ongoing monitoring interval
      • Class-specific safety thresholds (K, Na, Cr, eGFR) from ESC/ACC guidelines

    Args:
        preset_name: Exact preset name.
        strategy: Titration strategy to determine which classes are being started/changed.
    """
    err = _validate_preset(preset_name) or _validate_strategy(strategy)
    if err:
        return err

    try:
        d = _pipeline(preset_name, strategy, labs_available=False)
    except Exception as e:
        return f"ERROR: {e}"

    changes = d["changes"]
    patient = d["patient"]

    _schedule = {
        "ACEi": {
            "labs":     "K, Na, Creatinine, eGFR",
            "timing":   "1–2 wk after start or dose change; 3-monthly thereafter",
            "thresholds": "Stop/hold if K > 5.5 · Cr rise > 30% from baseline · eGFR < 30",
            "notes":    "If eGFR drops > 30% → reduce dose; > 50% → stop. Check for ACEi cough (→ switch to ARB).",
        },
        "ARB": {
            "labs":     "K, Na, Creatinine, eGFR",
            "timing":   "1–2 wk after start or dose change; 3-monthly thereafter",
            "thresholds": "Stop/hold if K > 5.5 · Cr rise > 30% · eGFR < 30",
            "notes":    "Same AKI monitoring as ACEi. No angioedema class effect — safe after ACEi-angioedema.",
        },
        "ARNi": {
            "labs":     "K, Na, Creatinine, eGFR, BP",
            "timing":   "1–2 wk after first dose; 4-wkly during up-titration",
            "thresholds": "Stop/hold if K > 5.5 · Cr rise > 30% · SBP < 90 · eGFR < 30",
            "notes":    "Requires ≥ 36 h ACEi-free washout before first dose. Symptomatic hypotension most common AE.",
        },
        "beta_blocker": {
            "labs":     "HR and BP at each visit; no mandatory labs unless CKD",
            "timing":   "Clinical review 2 wk after each dose change",
            "thresholds": "Hold if HR < 50 bpm · SBP < 90 mmHg · worsening HF symptoms",
            "notes":    "Start at very low dose (carvedilol 3.125 mg bd) only when patient is euvolaemic (no acute HF).",
        },
        "MRA": {
            "labs":     "K, Creatinine, eGFR",
            "timing":   "1 wk, 4 wk, 3 months after initiation; then 3-monthly",
            "thresholds": "Hold if K > 5.5 · Stop if K > 6.0 · eGFR < 30 is relative contra",
            "notes":    "Gynaecomastia in men (spironolactone): consider eplerenone. Avoid NSAIDs concurrently.",
        },
        "SGLT2i": {
            "labs":     "eGFR, HbA1c (diabetics), urine ketones if symptomatic",
            "timing":   "eGFR at baseline; 3-monthly in CKD; HbA1c 3-monthly if diabetic",
            "thresholds": "Hold if eGFR < 20 · Stop if recurrent UTI or DKA history · Discontinue peri-operatively",
            "notes":    "Hold 3–4 days before elective surgery (DKA risk). Genital mycotic infections: patient education.",
        },
        "loop": {
            "labs":     "K, Na, Creatinine, eGFR",
            "timing":   "1–2 wk after dose change; daily home weights",
            "thresholds": "Reduce if Na < 130 · Reduce/hold if K < 3.5 · Reduce if Cr rises > 30%",
            "notes":    "Titrate to euvolaemia (no JVP elevation, no oedema, weight at dry weight). Avoid over-diuresis.",
        },
    }

    lines = [
        f"Lab Monitoring Plan: {preset_name}  (strategy: {strategy})",
        "═" * 70,
    ]
    for cls in CLASSES:
        c      = changes[cls]
        action = c["concrete_action"]
        cur    = c["current"] or 0
        nd     = c["new_dose"] or 0
        sched  = _schedule.get(cls, {})
        if not sched:
            continue
        is_active = cur > 0 or action == "start_medication"
        change_flag = action in ("start_medication", "increase_dose", "double_dose", "decrease_dose")
        if not is_active:
            continue

        lines.append(f"\n  {cls} ({_REP_DRUG.get(cls, '')})  — {action}: {_dose_line(action, cur, nd)}")
        lines.append(f"    Labs       : {sched['labs']}")
        lines.append(f"    Timing     : {'PRIORITY: ' if change_flag else ''}{sched['timing']}")
        lines.append(f"    Thresholds : {sched['thresholds']}")
        lines.append(f"    Notes      : {sched['notes']}")

    lines += [
        "",
        "═" * 70,
        "  General monitoring:",
        "  • After any RAAS or MRA dose change: recheck K and Cr in 1–2 weeks",
        "  • After any loop diuretic change: electrolytes in 1–2 weeks",
        "  • After beta-blocker change: HR and BP check in 2 weeks (clinic or phone)",
        "  • Any symptom deterioration: same-day assessment; hold up-titration",
    ]
    return "\n".join(lines)


@mcp.tool()
def calculate_risk_score(preset_name: str, labs_available: bool = False) -> str:
    """
    Calculate a weighted adverse-effect risk score from all 11 classifier flags.

    Weighting rationale:
      • Severe flags (severe_hypotension_detected, severe_bradycardia_detected,
        any_emergency_flag): weight × 3 — these indicate immediate haemodynamic threat.
      • Standard flags: weight × 1.

    Score = Σ(weight × probability) / Σ(max possible weight) × 100%

    Risk levels:
      CRITICAL : any severe flag active (p ≥ 0.5)
      HIGH     : ≥ 3 standard flags active
      MODERATE : 1–2 standard flags active
      LOW      : no flags active

    Args:
        preset_name: Exact preset name.
        labs_available: Whether labs are available this cycle.
    """
    err = _validate_preset(preset_name)
    if err:
        return err

    try:
        d = _pipeline(preset_name, "strong_hf", labs_available)
    except Exception as e:
        return f"ERROR: {e}"

    probs  = d["probs"]
    flags  = d["flags"]
    active = [k for k, v in flags.items() if v]
    SEVERE = {"severe_hypotension_detected", "severe_bradycardia_detected", "any_emergency_flag"}

    weighted_sum = sum((3 if k in SEVERE else 1) * v for k, v in probs.items())
    max_possible = sum(3 if k in SEVERE else 1 for k in probs)
    score        = round(weighted_sum / max_possible * 100, 1)
    severe_active = [k for k in active if k in SEVERE]
    level = ("CRITICAL" if severe_active else
             "HIGH"     if len(active) >= 3 else
             "MODERATE" if active else "LOW")

    lines = [
        f"Adverse-Effect Risk Score: {preset_name}",
        "═" * 60,
        f"  Weighted score   : {score:.1f}%",
        f"  Risk level       : {level}",
        f"  Active flags     : {len(active)} / 11",
        f"  Severe active    : {', '.join(severe_active) if severe_active else 'none'}",
    ]
    if active:
        lines.append(f"  All active flags : {', '.join(active)}")

    lines += [
        "",
        f"  {'Flag':<40}  {'Wt':>3}  {'p':>6}  {'Contrib':>7}  Active  Bar",
        "  " + "─" * 72,
    ]
    for flag in LABEL_COLS:
        w       = 3 if flag in SEVERE else 1
        p_val   = probs[flag]
        contrib = w * p_val
        act     = "▶ YES" if flags[flag] else "  no "
        bar     = "█" * int(p_val * 10) + "░" * (10 - int(p_val * 10))
        sev_m   = " ⚡" if flag in SEVERE else "   "
        lines.append(
            f"  {flag:<40}  {w:>3}  {p_val:.3f}  {contrib:>7.3f}  {act}  {bar}{sev_m}"
        )
    lines += [
        "",
        f"  Weighted sum     : {weighted_sum:.3f}",
        f"  Max possible     : {max_possible:.3f}",
        f"  Final score      : {weighted_sum:.3f} / {max_possible:.3f} × 100 = {score:.1f}%",
    ]
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# PATIENT-FACING TOOLS
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def get_my_medications(preset_name: str) -> str:
    """
    Get a plain-English list of a patient's current heart failure medications.

    Explains what each drug does in everyday language, current dose, and latest
    blood test results without clinical jargon. Suitable to read to or by the patient.

    Args:
        preset_name: Exact preset name.
    """
    err = _validate_preset(preset_name)
    if err:
        return f"Sorry, that profile wasn't found. {err}"

    p  = PATIENTS[preset_name]
    bl = p["baseline"]
    meds = [(cls, p["meds"][cls]["dose"]) for cls in CLASSES if p["meds"][cls]["on"]]

    lines = [
        "Your Heart Failure Medications",
        "─" * 40,
        f"You are a {p['age']}-year-old {'woman' if p['gender'] == 'F' else 'man'} being treated for heart failure.",
        "",
    ]
    if not meds:
        lines += [
            "You are not yet taking any heart failure medications.",
            "Your doctor will discuss starting treatment at your next appointment.",
        ]
    else:
        lines.append(f"You are currently taking {len(meds)} heart failure medicine(s):\n")
        for cls, dose in meds:
            lines.append(f"  • {_DRUG_PLAIN.get(cls, cls)}")
            lines.append(f"    Your current dose: {dose} mg\n")

    lines += [
        "Your most recent blood results:",
        f"  Potassium  : {bl['K']} mmol/L",
        f"  Sodium     : {bl['Na']} mmol/L",
        f"  Creatinine : {bl['Cr']:.2f} mg/dL",
        f"  Kidney     : {bl['eGFR']:.0f} mL/min (kidney function score)",
        "",
        "If you feel dizzy, short of breath, notice swelling, or have any concerns,",
        "contact your care team before your next appointment.",
    ]
    return "\n".join(lines)


@mcp.tool()
def get_my_medication_changes(preset_name: str, strategy: str = "strong_hf") -> str:
    """
    Explain recommended medication changes to a patient in plain English.

    Describes what is starting, what dose is changing, what is staying the same,
    and what to watch for — with zero clinical jargon. Suitable for patient handout.

    Args:
        preset_name: Exact preset name.
        strategy: Titration strategy (default 'strong_hf').
    """
    try:
        d = _pipeline(preset_name, strategy, labs_available=True)
    except Exception as e:
        return f"Sorry, an error occurred: {e}"

    p       = d["patient"]
    changes = d["changes"]
    result  = d["result"]

    lines = [
        "Your Medication Plan — What's Changing This Week",
        "═" * 50,
        f"You are a {p['age']}-year-old {'woman' if p['gender'] == 'F' else 'man'}.",
        "",
    ]
    if result["global_stop"]:
        lines += [
            "IMPORTANT: Your doctor needs to see you urgently.",
            "  All heart medicines are being paused until you are assessed.",
            "  Please contact your care team or go to hospital today.",
        ]
        return "\n".join(lines)

    if result["order_labs"]:
        lines += [
            "Blood tests needed first.",
            "  Please get your blood tests done as soon as possible.",
            "  Your medicines stay the same until the results come back.",
            "",
        ]

    starts, ups, downs, same = [], [], [], []
    for cls in CLASSES:
        c   = changes[cls]; a = c["concrete_action"]
        pl  = _DRUG_PLAIN.get(cls, cls)
        cur = c["current"] or 0; nd = c["new_dose"]
        if a == "start_medication":
            starts.append(f"  + {pl}\n    Starting dose: {nd} mg")
        elif a in ("increase_dose", "double_dose"):
            ups.append(f"  ↑ {pl}\n    {cur} mg → {nd} mg")
        elif a in ("decrease_dose", "stop_medication"):
            label = f"{cur} mg → {nd} mg (reduced for safety)" if nd else f"{cur} mg → stopped for safety"
            downs.append(f"  ↓ {pl}\n    {label}")
        elif a in ("maintain_dose", "hold_titration") and cur > 0:
            same.append(f"  = {pl} — {cur} mg (no change)")

    if starts: lines += ["New medicines starting:", *starts, ""]
    if ups:    lines += ["Doses being increased:", *ups, ""]
    if same:   lines += ["Staying the same:", *same, ""]
    if downs:  lines += ["Adjustments for your safety:", *downs, ""]
    if not starts and not ups and not downs:
        lines += ["No medication changes this week — everything looks stable.", ""]

    lines += [
        "Things to watch for and tell your doctor:",
        "  • Dizziness or faintness when standing",
        "  • Weight gain of more than 1–2 kg in 2 days",
        "  • Increased swelling in legs or ankles",
        "  • Feeling more breathless than usual",
        "  • Muscle weakness or cramps",
        "",
        "If any of these happen, call your care team before your next appointment.",
    ]
    return "\n".join(lines)


@mcp.tool()
def explain_warning_sign(flag_name: str) -> str:
    """
    Explain what an adverse-effect flag means in plain English for a patient.

    Valid flag names (use exactly as written):
      hypotension_detected, bradycardia_detected, volume_depletion_detected,
      worsening_HF_detected, hyperkalemia_detected, renal_dysfunction_detected,
      hyponatremia_detected, metabolic_acidosis_detected,
      severe_hypotension_detected, severe_bradycardia_detected, any_emergency_flag

    Args:
        flag_name: Exact flag name from the list above.
    """
    exp = _FLAG_PLAIN.get(flag_name)
    if not exp:
        valid = "\n  ".join(LABEL_COLS)
        return f"Unknown flag {flag_name!r}.\n\nValid names:\n  {valid}"
    title    = flag_name.replace("_detected", "").replace("_", " ").title()
    is_urgent = "severe" in flag_name or flag_name == "any_emergency_flag"
    prefix   = "URGENT — " if is_urgent else ""
    return (
        f"{prefix}{title}\n"
        f"{'─' * 50}\n"
        f"{exp}\n\n"
        "Always speak to your doctor or nurse before changing any medicines."
    )


if __name__ == "__main__":
    mcp.run()
