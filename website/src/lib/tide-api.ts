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

// ── chat (RAG) ──────────────────────────────────────────────────────────────

export type ChatAudience = "patient" | "clinician";

export type ChatRequest = {
  question: string;
  audience: ChatAudience;
  patient: Patient | null;
  strategy: Strategy;
  // Optionally pass a previously-computed engine response so the backend
  // doesn't re-run the engine for this chat call.
  result?: EvaluateResponse;
};

export type ChatResponse = {
  answer: string;
  engine_context: string;
  audience: ChatAudience;
  source: "python" | "browser";
};

const PATIENT_CTX_KEY = "tide-hf:lastPatient";
const PATIENT_RES_KEY = "tide-hf:lastResult";
const PATIENT_STR_KEY = "tide-hf:lastStrategy";

export function rememberPatient(patient: Patient, result: EvaluateResponse, strategy: Strategy) {
  try {
    localStorage.setItem(PATIENT_CTX_KEY, JSON.stringify(patient));
    localStorage.setItem(PATIENT_RES_KEY, JSON.stringify(result));
    localStorage.setItem(PATIENT_STR_KEY, strategy);
  } catch { /* localStorage may be disabled */ }
}

export function recallPatient():
  | { patient: Patient; result: EvaluateResponse; strategy: Strategy }
  | null {
  try {
    const p = localStorage.getItem(PATIENT_CTX_KEY);
    const r = localStorage.getItem(PATIENT_RES_KEY);
    const s = localStorage.getItem(PATIENT_STR_KEY);
    if (!p || !r) return null;
    return {
      patient: JSON.parse(p) as Patient,
      result: JSON.parse(r) as EvaluateResponse,
      strategy: (s as Strategy) ?? "strong_hf",
    };
  } catch {
    return null;
  }
}

export async function askChat(req: ChatRequest): Promise<ChatResponse> {
  // Try the Python backend first.
  if (pythonBackendAvailable !== false) {
    try {
      const body: Record<string, unknown> = {
        question: req.question,
        audience: req.audience,
        strategy: req.strategy,
      };
      if (req.patient) {
        body.patient = req.patient;
        body.labs = req.patient.labs ?? null;
      }
      if (req.result) {
        body.result = {
          global_stop: req.result.global_stop,
          order_labs: req.result.order_labs,
          labs_requested: req.result.labs_requested,
          preferred_raas: req.result.preferred_raas,
          decisions: req.result.decisions,
          active_raas: null,
        };
        body.changes = req.result.changes;
        body.flags = req.result.flags;
      }
      const res = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(120000),
      });
      if (res.ok) {
        const data = (await res.json()) as Omit<ChatResponse, "source">;
        pythonBackendAvailable = true;
        return { ...data, source: "python" };
      }
      const errText = await res.text().catch(() => "");
      throw new Error(`API ${res.status}: ${errText || res.statusText}`);
    } catch (e) {
      if (pythonBackendAvailable === null) pythonBackendAvailable = false;
      // Fall through to offline fallback.
      const msg = (e as Error).message;
      if (!/abort|fetch|network|Failed/i.test(msg)) {
        // Surface real backend errors (e.g. RAG empty) to the user verbatim.
        return {
          answer: `⚠ Backend error: ${msg}`,
          engine_context: "",
          audience: req.audience,
          source: "python",
        };
      }
    }
  }

  // Offline fallback: produce a deterministic patient-aware summary from the
  // engine state we have. No real LLM, but still personalised.
  return {
    answer: offlineChatReply(req),
    engine_context: req.result ? offlineEngineContext(req.result) : "",
    audience: req.audience,
    source: "browser",
  };
}

function offlineEngineContext(r: EvaluateResponse): string {
  const lines: string[] = [];
  if (r.global_stop) lines.push("STATUS: global_stop");
  const active = Object.entries(r.flags).filter(([, v]) => v).map(([k]) => k);
  if (active.length) lines.push("Active flags: " + active.join(", "));
  if (r.preferred_raas) lines.push("Preferred RAAS: " + r.preferred_raas);
  for (const [cls, ch] of Object.entries(r.changes)) {
    lines.push(`  ${cls}: ${ch.concrete_action} ${ch.current}→${ch.new_dose} mg [${ch.reason}]`);
  }
  return lines.join("\n");
}

function offlineChatReply({ question, audience, patient, result }: ChatRequest): string {
  const intro = audience === "patient"
    ? "I can't reach the AI backend right now, so this is a quick rule-based summary."
    : "Backend unreachable — falling back to a deterministic engine summary.";

  if (!patient || !result) {
    return `${intro}\n\nNo patient is loaded yet. Open the **Engine** tab, edit a preset, click **Compute**, then come back to ask about that patient. To enable full AI answers, run \`python scripts/run_api.py\` and \`python scripts/setup_rag.py\` locally.\n\nQuestion: *${question}*`;
  }

  const flags = Object.entries(result.flags).filter(([, v]) => v).map(([k]) => k);
  const changes = Object.entries(result.changes).map(([cls, ch]) =>
    `- **${cls}** — ${ch.concrete_action} · ${ch.current} mg → ${ch.new_dose} mg · *${ch.reason.replace(/_/g, " ")}*`,
  );
  const banners: string[] = [];
  if (result.global_stop) banners.push("**GLOBAL STOP** — all GDMT held; escalate.");
  if (result.order_labs) banners.push(`**Order labs** for: ${result.labs_requested.join(", ")}.`);

  return [
    intro,
    "",
    `**Patient:** ${patient.age}${patient.gender}, *${patient.note}*`,
    flags.length ? `**Active flags:** ${flags.join(", ")}` : "**No adverse-effect flags fired.**",
    `**Preferred RAAS:** ${result.preferred_raas ?? "none"}`,
    ...banners,
    "",
    "**Engine actions:**",
    ...changes,
    "",
    audience === "patient"
      ? "Talk to your care team if anything here looks new — they'll explain in plain language."
      : "Confirm against AHA 2022 GDMT thresholds and STRONG-HF up-titration windows before acting.",
  ].join("\n");
}
