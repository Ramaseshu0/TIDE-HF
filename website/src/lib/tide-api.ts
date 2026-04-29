import type { Patient, Strategy } from "./tide-engine";

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
};

export async function evaluatePatient(
  patient: Patient,
  strategy: Strategy,
): Promise<EvaluateResponse> {
  const res = await fetch(`${API_BASE}/evaluate`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ patient, strategy, labs: patient.labs ?? null }),
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`API ${res.status}: ${text || res.statusText}`);
  }
  return (await res.json()) as EvaluateResponse;
}

export async function apiHealth(): Promise<{ ok: boolean; bundle_loaded: boolean }> {
  const res = await fetch(`${API_BASE}/health`);
  return await res.json();
}
