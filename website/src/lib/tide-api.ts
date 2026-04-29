import {
  evaluate as localEvaluate,
  predictFlags,
  REP_DRUG,
  type Patient,
  type Strategy,
  type DrugClass,
} from "./tide-engine";

const API_BASE = (import.meta.env.VITE_TIDE_API as string | undefined) ?? "http://127.0.0.1:8000";

export type EngineDecision = { action: string; reason: string };
export type StrategyChange = {
  engine_action: string;
  concrete_action: string;
  reason: string;
  current: number;
  new_dose: number;
  new_ratio: number | null;
  target: number;
};

export type EvaluateResponse = {
  flags: Record<string, boolean>;
  probs: Record<string, number>;
  global_stop: boolean;
  order_labs: boolean;
  labs_requested: string[];
  preferred_raas: string | null;
  decisions: Record<string, EngineDecision>;
  changes: Record<string, StrategyChange>;
  contras_derived: Record<string, boolean>;
  source: "python" | "browser";
};

const TARGETS: Record<DrugClass, number> = {
  RAAS: 200, beta_blocker: 50, MRA: 50, SGLT2i: 10, loop: 80,
};

function browserEvaluate(patient: Patient, strategy: Strategy): EvaluateResponse {
  const r = localEvaluate(patient, strategy);
  const flagList = predictFlags(patient);

  const flags: Record<string, boolean> = {};
  const probs: Record<string, number> = {};
  for (const f of flagList) { flags[f.flag] = true; probs[f.flag] = f.prob; }

  const decisions: Record<string, EngineDecision> = {};
  const changes: Record<string, StrategyChange> = {};
  for (const d of r.decisions) {
    decisions[d.cls] = { action: d.action, reason: d.reason };
    changes[d.cls] = {
      engine_action: d.action,
      concrete_action: d.action,
      reason: d.reason,
      current: d.current,
      new_dose: d.newDose,
      new_ratio: TARGETS[d.cls] ? d.newDose / TARGETS[d.cls] : null,
      target: TARGETS[d.cls] ?? 0,
    };
  }

  return {
    flags, probs,
    global_stop: r.globalStop,
    order_labs: r.awaitingLabs,
    labs_requested: r.awaitingLabs ? ["RAAS", "MRA", "beta_blocker"] : [],
    preferred_raas: patient.contras?.angioedema_history ? "ARB" : "ARNi",
    decisions, changes,
    contras_derived: {},
    source: "browser",
  };
}

let pythonBackendAvailable: boolean | null = null;

export async function apiHealth(): Promise<{ ok: boolean; bundle_loaded: boolean; source: "python" | "browser" }> {
  try {
    const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(1500) });
    const data = await res.json();
    pythonBackendAvailable = !!(data.ok && data.bundle_loaded);
    return { ...data, source: "python" };
  } catch {
    pythonBackendAvailable = false;
    return { ok: true, bundle_loaded: true, source: "browser" };
  }
}

export async function evaluatePatient(patient: Patient, strategy: Strategy): Promise<EvaluateResponse> {
  if (pythonBackendAvailable) {
    try {
      const res = await fetch(`${API_BASE}/evaluate`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ patient, strategy, labs: patient.labs ?? null }),
        signal: AbortSignal.timeout(8000),
      });
      if (res.ok) {
        const data = (await res.json()) as EvaluateResponse;
        return { ...data, source: "python" };
      }
    } catch {
      pythonBackendAvailable = false;
    }
  }
  return browserEvaluate(patient, strategy);
}

// REP_DRUG re-export so Engine.tsx can label decisions even in browser-only mode
export { REP_DRUG };
