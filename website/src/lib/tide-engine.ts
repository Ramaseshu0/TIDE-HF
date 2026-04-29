/**
 * TIDE-HF presets — mirrored 1:1 from src/chf_titration/synthesize.py (PATIENTS dict).
 * Logic in evaluate() mirrors the README's decision paths and the engine's per-class actions.
 */

export type Med = { on: boolean; dose: number };
export type Patient = {
  name: string;
  group: "Titration state" | "Adverse effect" | "Contraindication";
  age: number;
  gender: "M" | "F";
  baseline: { K: number; Na: number; Cr: number; eGFR: number };
  contras: Record<string, boolean>;
  meds: Record<string, Med>;
  vitals: {
    HR_mean: number; HR_min: number;
    SBP_mean: number; SBP_min: number;
    SpO2_min: number; weight_delta: number;
    QRS_max: number; QTc_max: number;
    T_peaked: boolean; rhythm: string;
  };
  labs: { K?: number; Na?: number; Cr?: number; eGFR?: number } | null;
  note: string;
};

export const CLASSES = ["RAAS", "beta_blocker", "MRA", "SGLT2i", "loop"] as const;
export type DrugClass = (typeof CLASSES)[number];

export const REP_DRUG: Record<DrugClass, string> = {
  RAAS: "Sacubitril-Valsartan (Entresto)",
  beta_blocker: "Carvedilol",
  MRA: "Spironolactone",
  SGLT2i: "Empagliflozin (Jardiance)",
  loop: "Furosemide",
};

export const STRATEGIES = ["traditional", "strong_hf", "rapid_sequence", "sglt_mra_first"] as const;
export type Strategy = (typeof STRATEGIES)[number];

export const STRATEGY_DESC: Record<Strategy, string> = {
  traditional: "One new class per week · RAAS → BB → MRA → SGLT2i → loop · single-rung",
  strong_hf: "All eligible classes started at once · double-rung up-titration",
  rapid_sequence: "All eligible classes at once · single-rung steps (Greene 2021)",
  sglt_mra_first: "Phase 1: SGLT2i + MRA · Phase 2: ARNi + beta-blocker",
};

const m = (spec: Record<string, number>) => {
  const all = ["RAAS", "beta_blocker", "MRA", "SGLT2i", "loop"];
  const out: Record<string, Med> = {};
  for (const c of all) out[c] = { on: false, dose: 0 };
  for (const [k, d] of Object.entries(spec)) out[k] = { on: true, dose: d };
  return out;
};

export const PATIENTS: Patient[] = [
  // ---- titration state ----
  {
    name: "Newly diagnosed, stable", group: "Titration state",
    age: 55, gender: "F",
    baseline: { K: 4.2, Na: 140, Cr: 1.0, eGFR: 78 },
    contras: {}, meds: m({}),
    vitals: { HR_mean: 74, HR_min: 65, SBP_mean: 128, SBP_min: 115, SpO2_min: 97, weight_delta: 0.2, QRS_max: 95, QTc_max: 415, T_peaked: false, rhythm: "NSR" },
    labs: null,
    note: "No GDMT started. Engine should recommend starting preferred RAAS (ARNi) + other pillars per strategy.",
  },
  {
    name: "Partial titration, no AEs", group: "Titration state",
    age: 68, gender: "M",
    baseline: { K: 4.5, Na: 139, Cr: 1.1, eGFR: 68 },
    contras: {}, meds: m({ RAAS: 10, beta_blocker: 12.5, loop: 40 }),
    vitals: { HR_mean: 76, HR_min: 62, SBP_mean: 125, SBP_min: 110, SpO2_min: 96, weight_delta: 0.1, QRS_max: 100, QTc_max: 425, T_peaked: false, rhythm: "NSR" },
    labs: null,
    note: "On ACEi 10/40, BB 12.5/50, loop. Should up-titrate, start MRA + SGLT2i.",
  },
  {
    name: "Fully titrated, at target", group: "Titration state",
    age: 64, gender: "M",
    baseline: { K: 4.3, Na: 140, Cr: 1.1, eGFR: 70 },
    contras: {}, meds: m({ RAAS: 103, beta_blocker: 50, MRA: 50, SGLT2i: 10, loop: 40 }),
    vitals: { HR_mean: 70, HR_min: 62, SBP_mean: 122, SBP_min: 112, SpO2_min: 97, weight_delta: 0, QRS_max: 95, QTc_max: 412, T_peaked: false, rhythm: "NSR" },
    labs: null,
    note: "At-target quadruple therapy. Every class should maintain (reason: at_or_above_target).",
  },

  // ---- adverse effects ----
  {
    name: "Suspected hyperkalemia", group: "Adverse effect",
    age: 72, gender: "M",
    baseline: { K: 4.8, Na: 138, Cr: 1.3, eGFR: 58 },
    contras: {}, meds: m({ RAAS: 20, beta_blocker: 25, MRA: 25, loop: 40 }),
    vitals: { HR_mean: 68, HR_min: 55, SBP_mean: 120, SBP_min: 108, SpO2_min: 96, weight_delta: 0.3, QRS_max: 138, QTc_max: 425, T_peaked: true, rhythm: "NSR" },
    labs: null,
    note: "ECG shows hyperK signs but no labs. Engine holds RAAS+MRA+BB + orders labs.",
  },
  {
    name: "Confirmed hyperkalemia (K=6.3)", group: "Adverse effect",
    age: 72, gender: "M",
    baseline: { K: 4.8, Na: 138, Cr: 1.3, eGFR: 58 },
    contras: {}, meds: m({ RAAS: 20, beta_blocker: 25, MRA: 25, loop: 40 }),
    vitals: { HR_mean: 62, HR_min: 48, SBP_mean: 115, SBP_min: 100, SpO2_min: 96, weight_delta: 0.4, QRS_max: 145, QTc_max: 435, T_peaked: true, rhythm: "wide_complex" },
    labs: { K: 6.3, Na: 138, Cr: 1.7, eGFR: 48 },
    note: "Labs confirm K=6.3 → global_stop fires. All meds held, escalate.",
  },
  {
    name: "Volume depletion + hypotension", group: "Adverse effect",
    age: 70, gender: "M",
    baseline: { K: 4.3, Na: 138, Cr: 1.1, eGFR: 65 },
    contras: {}, meds: m({ RAAS: 20, beta_blocker: 25, SGLT2i: 10, loop: 80 }),
    vitals: { HR_mean: 92, HR_min: 80, SBP_mean: 98, SBP_min: 85, SpO2_min: 97, weight_delta: -2.8, QRS_max: 100, QTc_max: 420, T_peaked: false, rhythm: "Sinus_tachy" },
    labs: null,
    note: "AE resolver: stop SGLT2i + decrease loop.",
  },
  {
    name: "Worsening HF", group: "Adverse effect",
    age: 69, gender: "F",
    baseline: { K: 4.2, Na: 137, Cr: 1.0, eGFR: 70 },
    contras: {}, meds: m({ RAAS: 10, beta_blocker: 12.5, loop: 40 }),
    vitals: { HR_mean: 94, HR_min: 82, SBP_mean: 118, SBP_min: 105, SpO2_min: 92, weight_delta: 3.2, QRS_max: 128, QTc_max: 435, T_peaked: false, rhythm: "LBBB" },
    labs: null,
    note: "Weight gain + SpO2 drop. BB and loop get immediate_ae; RAAS/MRA can still titrate.",
  },
  {
    name: "Bradycardia on beta blocker", group: "Adverse effect",
    age: 72, gender: "M",
    baseline: { K: 4.3, Na: 139, Cr: 1.1, eGFR: 68 },
    contras: {}, meds: m({ RAAS: 20, beta_blocker: 50, MRA: 25 }),
    vitals: { HR_mean: 52, HR_min: 42, SBP_mean: 118, SBP_min: 105, SpO2_min: 96, weight_delta: 0, QRS_max: 98, QTc_max: 420, T_peaked: false, rhythm: "Sinus_brady" },
    labs: null,
    note: "AE resolver: down-titrate beta blocker first.",
  },
  {
    name: "Severe hypotension (global stop)", group: "Adverse effect",
    age: 70, gender: "M",
    baseline: { K: 4.3, Na: 138, Cr: 1.2, eGFR: 58 },
    contras: {}, meds: m({ RAAS: 20, beta_blocker: 25, loop: 40 }),
    vitals: { HR_mean: 95, HR_min: 88, SBP_mean: 78, SBP_min: 68, SpO2_min: 95, weight_delta: -0.5, QRS_max: 100, QTc_max: 418, T_peaked: false, rhythm: "Sinus_tachy" },
    labs: null,
    note: "SBP<80 → severe_hypotension → global_stop, escalate.",
  },

  // ---- contraindications ----
  {
    name: "Angioedema (ARNi + ACEi blocked)", group: "Contraindication",
    age: 66, gender: "F",
    baseline: { K: 4.3, Na: 140, Cr: 1.0, eGFR: 72 },
    contras: { angioedema: true }, meds: m({ beta_blocker: 12.5, loop: 40 }),
    vitals: { HR_mean: 74, HR_min: 64, SBP_mean: 126, SBP_min: 112, SpO2_min: 97, weight_delta: 0.1, QRS_max: 95, QTc_max: 415, T_peaked: false, rhythm: "NSR" },
    labs: null,
    note: "ARNi + ACEi contraindicated. Prefer ARB; start MRA + SGLT2i.",
  },
  {
    name: "Severe asthma (BB blocked)", group: "Contraindication",
    age: 62, gender: "F",
    baseline: { K: 4.3, Na: 139, Cr: 1.0, eGFR: 70 },
    contras: { asthma: true }, meds: m({ RAAS: 10, loop: 40 }),
    vitals: { HR_mean: 78, HR_min: 68, SBP_mean: 125, SBP_min: 115, SpO2_min: 96, weight_delta: 0.1, QRS_max: 95, QTc_max: 412, T_peaked: false, rhythm: "NSR" },
    labs: null,
    note: "Beta-blocker contraindicated. Everything else titrates normally.",
  },
  {
    name: "CKD stage 4 (eGFR=22)", group: "Contraindication",
    age: 75, gender: "M",
    baseline: { K: 4.8, Na: 138, Cr: 2.3, eGFR: 22 },
    contras: {}, meds: m({ RAAS: 10, beta_blocker: 25, loop: 80 }),
    vitals: { HR_mean: 78, HR_min: 65, SBP_mean: 135, SBP_min: 120, SpO2_min: 96, weight_delta: 0.3, QRS_max: 105, QTc_max: 428, T_peaked: false, rhythm: "NSR" },
    labs: null,
    note: "eGFR<30 → MRA contraindicated. SGLT2i still allowed (eGFR>20).",
  },
  {
    name: "AV block (BB blocked)", group: "Contraindication",
    age: 78, gender: "M",
    baseline: { K: 4.1, Na: 140, Cr: 1.1, eGFR: 62 },
    contras: { av_block: true }, meds: m({ RAAS: 20, MRA: 25, loop: 40 }),
    vitals: { HR_mean: 62, HR_min: 54, SBP_mean: 128, SBP_min: 115, SpO2_min: 97, weight_delta: 0, QRS_max: 112, QTc_max: 422, T_peaked: false, rhythm: "AV_block_1" },
    labs: null,
    note: "Beta blocker held; RAAS+MRA+loop continue normally.",
  },
];

// ───────────────────────────────────────────────────────────
// Classifier flag predictor (mirrors LightGBM thresholds qualitatively)
// ───────────────────────────────────────────────────────────
export type FlagKey =
  | "hypotension_detected" | "bradycardia_detected" | "volume_depletion_detected"
  | "worsening_HF_detected" | "hyperkalemia_detected" | "renal_dysfunction_detected"
  | "hyponatremia_detected" | "severe_hypotension_detected" | "severe_bradycardia_detected"
  | "any_emergency_flag";

export const FLAG_LABELS: Record<FlagKey, string> = {
  hypotension_detected: "Hypotension",
  bradycardia_detected: "Bradycardia",
  volume_depletion_detected: "Volume depletion",
  worsening_HF_detected: "Worsening HF",
  hyperkalemia_detected: "Hyperkalemia",
  renal_dysfunction_detected: "Renal dysfunction",
  hyponatremia_detected: "Hyponatremia",
  severe_hypotension_detected: "Severe hypotension",
  severe_bradycardia_detected: "Severe bradycardia",
  any_emergency_flag: "Emergency",
};

export const predictFlags = (p: Patient): { flag: FlagKey; prob: number }[] => {
  const v = p.vitals;
  const out: { flag: FlagKey; prob: number }[] = [];
  const push = (f: FlagKey, prob: number) => out.push({ flag: f, prob });

  if (v.SBP_min < 90) push("hypotension_detected", 0.84);
  if (v.SBP_min < 80) push("severe_hypotension_detected", 0.95);
  if (v.HR_min < 50) push("bradycardia_detected", 0.78);
  if (v.HR_min < 45) push("severe_bradycardia_detected", 0.92);
  if (v.weight_delta < -2) push("volume_depletion_detected", 0.81);
  if (v.weight_delta > 2 || v.SpO2_min < 93) push("worsening_HF_detected", 0.86);
  if (v.T_peaked || v.QRS_max > 130) push("hyperkalemia_detected", 0.74);
  if (p.labs?.K && p.labs.K > 5.5) push("hyperkalemia_detected", 0.97);
  if (p.labs?.Cr && p.labs.Cr / p.baseline.Cr > 1.3) push("renal_dysfunction_detected", 0.9);
  if (p.labs?.Na && p.labs.Na < 130) push("hyponatremia_detected", 0.88);
  if (v.SBP_min < 80 || (p.labs?.K ?? 0) > 6.0) push("any_emergency_flag", 0.99);

  const dedup = new Map<FlagKey, number>();
  for (const { flag, prob } of out) dedup.set(flag, Math.max(dedup.get(flag) ?? 0, prob));
  return [...dedup.entries()].map(([flag, prob]) => ({ flag, prob }));
};

// ───────────────────────────────────────────────────────────
// Engine evaluate + strategy applier
// ───────────────────────────────────────────────────────────
export type Action =
  | "start_medication" | "increase_dose" | "maintain_dose"
  | "decrease_dose" | "hold" | "discontinue";

export type Decision = {
  cls: DrugClass;
  drug: string;
  current: number;
  newDose: number;
  action: Action;
  reason: string;
};

const TARGETS: Record<DrugClass, number> = {
  RAAS: 200, beta_blocker: 50, MRA: 50, SGLT2i: 10, loop: 80,
};

export const evaluate = (p: Patient, strategy: Strategy): {
  decisions: Decision[];
  globalStop: boolean;
  awaitingLabs: boolean;
  flags: { flag: FlagKey; prob: number }[];
} => {
  const flags = predictFlags(p);
  const flagSet = new Set(flags.map((f) => f.flag));
  const labsConfirmed = !!p.labs;

  // Global stop conditions
  const globalStop =
    flagSet.has("severe_hypotension_detected") ||
    (labsConfirmed && (p.labs?.K ?? 0) > 6.0);

  // "Holds awaiting labs" path: ECG-suspected hyperK or vitals-suspected renal w/o labs
  const awaitingHyperK = flagSet.has("hyperkalemia_detected") && !labsConfirmed;
  const awaitingRenal = !labsConfirmed && p.vitals.weight_delta > 0.8 && p.vitals.SBP_mean > 130;
  const awaitingLabs = awaitingHyperK || awaitingRenal;

  const decisions: Decision[] = [];

  for (const cls of CLASSES) {
    const med = p.meds[cls];
    const current = med.dose;
    const target = TARGETS[cls];
    let action: Action = "maintain_dose";
    let newDose = current;
    let reason = "stable";

    // Contraindications first
    const contraindicated =
      (cls === "RAAS" && (p.contras.angioedema || p.contras.pregnancy)) ||
      (cls === "beta_blocker" && (p.contras.asthma || p.contras.av_block)) ||
      (cls === "MRA" && p.baseline.eGFR < 30) ||
      (cls === "SGLT2i" && p.contras.pregnancy);

    if (contraindicated) {
      decisions.push({
        cls, drug: REP_DRUG[cls], current, newDose: med.on ? 0 : 0,
        action: med.on ? "discontinue" : "hold",
        reason: "contraindicated",
      });
      continue;
    }

    if (globalStop) {
      decisions.push({
        cls, drug: REP_DRUG[cls], current, newDose: 0,
        action: cls === "loop" ? "maintain_dose" : "hold",
        reason: cls === "loop" ? "volume_status_driven" : "global_stop",
      });
      continue;
    }

    if (awaitingLabs && (cls === "RAAS" || cls === "MRA" || cls === "beta_blocker")) {
      decisions.push({
        cls, drug: REP_DRUG[cls], current, newDose: current,
        action: "hold", reason: "order_labs",
      });
      continue;
    }

    // AE resolver — class-specific down-titrations
    if (cls === "beta_blocker" && (flagSet.has("bradycardia_detected") || flagSet.has("severe_bradycardia_detected"))) {
      action = "decrease_dose"; newDose = Math.max(0, current / 2); reason = "bradycardia"; 
      decisions.push({ cls, drug: REP_DRUG[cls], current, newDose, action, reason }); continue;
    }
    if (cls === "SGLT2i" && flagSet.has("volume_depletion_detected")) {
      action = "discontinue"; newDose = 0; reason = "volume_depletion";
      decisions.push({ cls, drug: REP_DRUG[cls], current, newDose, action, reason }); continue;
    }
    if (cls === "loop" && flagSet.has("volume_depletion_detected")) {
      action = "decrease_dose"; newDose = Math.max(0, current / 2); reason = "volume_depletion";
      decisions.push({ cls, drug: REP_DRUG[cls], current, newDose, action, reason }); continue;
    }
    if (cls === "loop" && flagSet.has("worsening_HF_detected")) {
      action = "increase_dose"; newDose = Math.min(target, current * 2 || 40); reason = "volume_overload";
      decisions.push({ cls, drug: REP_DRUG[cls], current, newDose, action, reason }); continue;
    }
    if (cls === "MRA" && labsConfirmed && (p.labs?.K ?? 0) > 5.5) {
      action = "decrease_dose"; newDose = Math.max(0, current / 2); reason = "hyperkalemia";
      decisions.push({ cls, drug: REP_DRUG[cls], current, newDose, action, reason }); continue;
    }
    if ((cls === "RAAS" || cls === "MRA") && labsConfirmed && p.labs?.Cr && p.labs.Cr / p.baseline.Cr > 1.3) {
      action = "hold"; newDose = current; reason = "renal_dysfunction";
      decisions.push({ cls, drug: REP_DRUG[cls], current, newDose, action, reason }); continue;
    }
    if (cls === "loop" && labsConfirmed && (p.labs?.Na ?? 999) < 130) {
      action = "decrease_dose"; newDose = Math.max(0, current / 2); reason = "hyponatremia";
      decisions.push({ cls, drug: REP_DRUG[cls], current, newDose, action, reason }); continue;
    }

    // Standard titration logic per strategy
    const rung = strategy === "strong_hf" ? 2 : 1;

    if (!med.on && cls !== "loop") {
      // sglt_mra_first phase logic — simplified: SGLT2i + MRA in phase 1, others phase 2
      const phase1Eligible = cls === "SGLT2i" || cls === "MRA";
      const startable =
        strategy === "traditional"
          ? cls === "RAAS" // start RAAS first in traditional
          : strategy === "sglt_mra_first"
          ? phase1Eligible
          : true; // strong_hf, rapid_sequence start everything eligible
      if (startable) {
        const startDose =
          cls === "RAAS" ? 24 : cls === "beta_blocker" ? 3.125 : cls === "MRA" ? 12.5 : 10;
        decisions.push({
          cls, drug: REP_DRUG[cls], current, newDose: startDose,
          action: "start_medication", reason: "GDMT_initiation",
        });
        continue;
      }
    }

    if (med.on && current < target) {
      const next = Math.min(target, current * (1 + 0.5 * rung) || current + 5);
      decisions.push({
        cls, drug: REP_DRUG[cls], current, newDose: Math.round(next * 10) / 10,
        action: "increase_dose", reason: "below_target",
      });
      continue;
    }
    if (med.on && current >= target) {
      decisions.push({
        cls, drug: REP_DRUG[cls], current, newDose: current,
        action: "maintain_dose", reason: "at_or_above_target",
      });
      continue;
    }

    decisions.push({ cls, drug: REP_DRUG[cls], current, newDose, action, reason });
  }

  return { decisions, globalStop, awaitingLabs, flags };
};
