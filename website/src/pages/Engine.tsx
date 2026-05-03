import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import {
  Activity, Play, AlertTriangle, ShieldAlert, FlaskConical,
  TrendingUp, TrendingDown, Minus, Power, Pause, ArrowLeft, Loader2,
} from "lucide-react";
import {
  PATIENTS, STRATEGIES, STRATEGY_DESC, FLAG_LABELS, CLASSES, REP_DRUG,
  type Patient, type Strategy, type DrugClass,
} from "@/lib/tide-engine";
import { evaluatePatient, apiHealth, rememberPatient, type EvaluateResponse } from "@/lib/tide-api";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";

type ConcreteAction =
  | "start_medication" | "increase_dose" | "maintain_dose" | "decrease_dose"
  | "hold" | "discontinue" | "stop_medication";

const ACTION_META: Record<string, { label: string; cls: string; icon: typeof TrendingUp }> = {
  start_medication: { label: "START", cls: "text-emerald-300 bg-emerald-400/10 border-emerald-400/30", icon: Play },
  increase_dose:    { label: "INCREASE", cls: "text-cyan-300 bg-cyan-400/10 border-cyan-400/30", icon: TrendingUp },
  maintain_dose:    { label: "MAINTAIN", cls: "text-sky-300 bg-sky-400/10 border-sky-400/30", icon: Minus },
  decrease_dose:    { label: "DECREASE", cls: "text-amber-300 bg-amber-400/10 border-amber-400/30", icon: TrendingDown },
  hold:             { label: "HOLD", cls: "text-orange-300 bg-orange-400/10 border-orange-400/30", icon: Pause },
  discontinue:      { label: "DISCONTINUE", cls: "text-rose-300 bg-rose-400/10 border-rose-400/30", icon: Power },
  stop_medication:  { label: "STOP", cls: "text-rose-300 bg-rose-400/10 border-rose-400/30", icon: Power },
};

const groupColors: Record<Patient["group"], string> = {
  "Titration state": "from-cyan-400 to-sky-500",
  "Adverse effect":  "from-amber-400 to-rose-500",
  "Contraindication": "from-violet-400 to-fuchsia-500",
};

const RHYTHMS = ["NSR", "AFib", "Sinus_brady", "Sinus_tachy", "wide_complex", "LBBB", "AV_block"];
const CONTRA_KEYS = [
  "asthma_severe", "av_block", "angioedema_history", "pregnancy",
  "hyperkalemia_chronic", "ckd_stage5", "egfr_below_20",
] as const;

const clone = <T,>(v: T): T => JSON.parse(JSON.stringify(v));
const numOrZero = (v: string) => (v === "" || isNaN(+v) ? 0 : +v);

const Engine = () => {
  const [selectedName, setSelectedName] = useState<string>(PATIENTS[0].name);
  const [patient, setPatient] = useState<Patient>(clone(PATIENTS[0]));
  const [strategy, setStrategy] = useState<Strategy>("strong_hf");
  const [result, setResult] = useState<EvaluateResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [engineSource, setEngineSource] = useState<"python" | "browser" | null>(null);

  useEffect(() => {
    apiHealth().then((h) => setEngineSource(h.source)).catch(() => setEngineSource("browser"));
  }, []);

  const grouped = useMemo(() => {
    const g: Record<string, Patient[]> = {};
    for (const p of PATIENTS) (g[p.group] ||= []).push(p);
    return g;
  }, []);

  const loadPreset = (p: Patient) => {
    setSelectedName(p.name);
    setPatient(clone(p));
    setResult(null);
    setError(null);
  };

  const compute = async () => {
    setLoading(true);
    setError(null);
    try {
      const r = await evaluatePatient(patient, strategy);
      setResult(r);
      rememberPatient(patient, r, strategy);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  };

  // ── tiny field helpers ────────────────────────────────────────────────────
  const update = (mut: (p: Patient) => void) => {
    const next = clone(patient);
    mut(next);
    setPatient(next);
    setResult(null);
  };

  const NumField = ({ label, value, step = 1, onChange }: { label: string; value: number; step?: number; onChange: (n: number) => void }) => (
    <div className="space-y-1">
      <Label className="text-[10px] font-mono uppercase tracking-widest text-muted-foreground">{label}</Label>
      <Input
        type="number"
        step={step}
        value={value}
        onChange={(e) => onChange(numOrZero(e.target.value))}
        className="h-8 font-mono text-sm"
      />
    </div>
  );

  return (
    <main className="min-h-screen bg-background text-foreground">
      {/* Header */}
      <header className="sticky top-0 z-40 border-b border-border bg-background/80 backdrop-blur-xl">
        <div className="mx-auto max-w-[1400px] px-6 h-16 flex items-center justify-between">
          <Link to="/" className="flex items-center gap-2.5 text-sm text-muted-foreground hover:text-foreground transition-colors">
            <ArrowLeft className="w-4 h-4" />
            <span>Back to TIDE-HF</span>
          </Link>
          <div className="flex items-center gap-3">
            <span className={`text-[10px] font-mono uppercase tracking-widest px-2 py-1 rounded border ${
              engineSource === null ? "border-border text-muted-foreground" :
              engineSource === "python" ? "border-emerald-400/40 text-emerald-300 bg-emerald-400/5"
                                       : "border-cyan-400/40 text-cyan-300 bg-cyan-400/5"
            }`}>
              {engineSource === null ? "checking…" : engineSource === "python" ? "Python engine" : "Browser engine"}
            </span>
            <div className="w-8 h-8 rounded-lg bg-gradient-primary grid place-items-center shadow-glow-soft">
              <Activity className="w-4 h-4 text-primary-foreground" strokeWidth={2.5} />
            </div>
            <div className="leading-tight">
              <div className="font-display font-semibold text-sm">TIDE-HF Engine</div>
              <div className="text-[10px] font-mono uppercase tracking-[0.18em] text-muted-foreground">
                lab-gated · contraindication-aware
              </div>
            </div>
          </div>
        </div>
      </header>

      <div className="mx-auto max-w-[1400px] px-6 py-10 grid lg:grid-cols-[340px_1fr] gap-8">
        {/* ───── Sidebar: presets ───── */}
        <aside className="space-y-6">
          <div>
            <div className="text-xs font-mono uppercase tracking-[0.2em] text-primary mb-3">01 — Pick a preset</div>
            <p className="text-xs text-muted-foreground mb-4">Edit any field on the right after selecting a starting point.</p>
          </div>

          {Object.entries(grouped).map(([group, list]) => (
            <section key={group}>
              <div className="flex items-center gap-2 mb-2">
                <span className={`w-1.5 h-1.5 rounded-full bg-gradient-to-r ${groupColors[group as Patient["group"]]}`} />
                <h3 className="text-[11px] font-mono uppercase tracking-widest text-muted-foreground">{group}</h3>
              </div>
              <div className="space-y-1.5">
                {list.map((p) => (
                  <button
                    key={p.name}
                    onClick={() => loadPreset(p)}
                    className={`w-full text-left px-3 py-2.5 rounded-lg border text-sm transition-all ${
                      selectedName === p.name
                        ? "border-primary/60 bg-primary/10 text-foreground shadow-glow-soft"
                        : "border-border bg-card/40 hover:border-primary/30 hover:bg-card text-muted-foreground hover:text-foreground"
                    }`}
                  >
                    {p.name}
                  </button>
                ))}
              </div>
            </section>
          ))}
        </aside>

        {/* ───── Main panel ───── */}
        <section className="space-y-6 min-w-0">
          {/* Patient header (editable) */}
          <div className="glass-panel rounded-2xl p-6 glow-border">
            <div className="flex flex-wrap items-start justify-between gap-4 mb-5">
              <div className="flex-1 min-w-[280px]">
                <div className="flex items-center gap-2 mb-1.5">
                  <span className={`text-[10px] font-mono uppercase tracking-widest px-2 py-0.5 rounded bg-gradient-to-r ${groupColors[patient.group]} text-background`}>
                    {patient.group}
                  </span>
                </div>
                <h1 className="font-display text-2xl md:text-3xl font-semibold tracking-tight">{patient.name}</h1>
                <p className="text-sm text-muted-foreground mt-1.5">
                  {patient.age}{patient.gender} · {patient.note}
                </p>
              </div>
              <button
                onClick={compute}
                disabled={loading}
                className="inline-flex items-center gap-2 rounded-xl bg-gradient-primary px-5 py-3 text-sm font-medium text-primary-foreground shadow-elegant hover:shadow-glow transition-all disabled:opacity-60 disabled:cursor-not-allowed"
              >
                {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
                {loading ? "Computing…" : "Compute"}
              </button>
            </div>

            {/* Demographics */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
              <NumField label="Age" value={patient.age} onChange={(v) => update((p) => { p.age = Math.max(0, Math.round(v)); })} />
              <div className="space-y-1">
                <Label className="text-[10px] font-mono uppercase tracking-widest text-muted-foreground">Gender</Label>
                <select
                  value={patient.gender}
                  onChange={(e) => update((p) => { p.gender = e.target.value as "M" | "F"; })}
                  className="h-8 w-full rounded-md border border-input bg-background px-2 text-sm font-mono"
                >
                  <option value="M">M</option>
                  <option value="F">F</option>
                </select>
              </div>
              <div className="space-y-1 md:col-span-2">
                <Label className="text-[10px] font-mono uppercase tracking-widest text-muted-foreground">Clinical note</Label>
                <Input value={patient.note} onChange={(e) => update((p) => { p.note = e.target.value; })} className="h-8 text-sm" />
              </div>
            </div>

            {/* Vitals */}
            <div className="rounded-xl border border-border bg-secondary/30 p-4 mb-3">
              <div className="text-[11px] font-mono uppercase tracking-widest text-muted-foreground mb-3">Vitals (week summary)</div>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <NumField label="HR mean (bpm)" value={patient.vitals.HR_mean} onChange={(v) => update((p) => { p.vitals.HR_mean = v; })} />
                <NumField label="HR min (bpm)" value={patient.vitals.HR_min} onChange={(v) => update((p) => { p.vitals.HR_min = v; })} />
                <NumField label="SBP mean (mmHg)" value={patient.vitals.SBP_mean} onChange={(v) => update((p) => { p.vitals.SBP_mean = v; })} />
                <NumField label="SBP min (mmHg)" value={patient.vitals.SBP_min} onChange={(v) => update((p) => { p.vitals.SBP_min = v; })} />
                <NumField label="SpO₂ min (%)" value={patient.vitals.SpO2_min} onChange={(v) => update((p) => { p.vitals.SpO2_min = v; })} />
                <NumField label="Δ Weight (kg)" step={0.1} value={patient.vitals.weight_delta} onChange={(v) => update((p) => { p.vitals.weight_delta = v; })} />
                <NumField label="QRS max (ms)" value={patient.vitals.QRS_max} onChange={(v) => update((p) => { p.vitals.QRS_max = v; })} />
                <NumField label="QTc max (ms)" value={patient.vitals.QTc_max} onChange={(v) => update((p) => { p.vitals.QTc_max = v; })} />
                <div className="space-y-1">
                  <Label className="text-[10px] font-mono uppercase tracking-widest text-muted-foreground">Rhythm</Label>
                  <select
                    value={patient.vitals.rhythm}
                    onChange={(e) => update((p) => { p.vitals.rhythm = e.target.value; })}
                    className="h-8 w-full rounded-md border border-input bg-background px-2 text-sm font-mono"
                  >
                    {RHYTHMS.map((r) => <option key={r} value={r}>{r}</option>)}
                  </select>
                </div>
                <label className="flex items-center gap-2 px-2 mt-5">
                  <Switch checked={patient.vitals.T_peaked} onCheckedChange={(v) => update((p) => { p.vitals.T_peaked = v; })} />
                  <span className="text-xs text-muted-foreground">T-peaked</span>
                </label>
              </div>
            </div>

            {/* Baseline labs */}
            <div className="rounded-xl border border-border bg-secondary/30 p-4 mb-3">
              <div className="text-[11px] font-mono uppercase tracking-widest text-muted-foreground mb-3">Baseline labs</div>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <NumField label="K (mEq/L)" step={0.1} value={patient.baseline.K} onChange={(v) => update((p) => { p.baseline.K = v; })} />
                <NumField label="Na (mEq/L)" value={patient.baseline.Na} onChange={(v) => update((p) => { p.baseline.Na = v; })} />
                <NumField label="Cr (mg/dL)" step={0.01} value={patient.baseline.Cr} onChange={(v) => update((p) => { p.baseline.Cr = v; })} />
                <NumField label="eGFR" value={patient.baseline.eGFR} onChange={(v) => update((p) => { p.baseline.eGFR = v; })} />
              </div>
            </div>

            {/* Recent labs (toggle) */}
            <div className="rounded-xl border border-border bg-secondary/30 p-4 mb-3">
              <div className="flex items-center justify-between mb-3">
                <div className="text-[11px] font-mono uppercase tracking-widest text-muted-foreground">Recent labs (this cycle)</div>
                <label className="flex items-center gap-2">
                  <Switch
                    checked={patient.labs !== null}
                    onCheckedChange={(v) => update((p) => {
                      p.labs = v ? { K: p.baseline.K, Na: p.baseline.Na, Cr: p.baseline.Cr, eGFR: p.baseline.eGFR } : null;
                    })}
                  />
                  <span className="text-xs text-muted-foreground">labs available</span>
                </label>
              </div>
              {patient.labs ? (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  <NumField label="K" step={0.1} value={patient.labs.K ?? 0} onChange={(v) => update((p) => { p.labs!.K = v; })} />
                  <NumField label="Na" value={patient.labs.Na ?? 0} onChange={(v) => update((p) => { p.labs!.Na = v; })} />
                  <NumField label="Cr" step={0.01} value={patient.labs.Cr ?? 0} onChange={(v) => update((p) => { p.labs!.Cr = v; })} />
                  <NumField label="eGFR" value={patient.labs.eGFR ?? 0} onChange={(v) => update((p) => { p.labs!.eGFR = v; })} />
                </div>
              ) : (
                <div className="text-sm text-muted-foreground italic">No labs yet — engine may order labs based on ECG/vitals.</div>
              )}
            </div>

            {/* Meds */}
            <div className="rounded-xl border border-border bg-secondary/30 p-4 mb-3">
              <div className="text-[11px] font-mono uppercase tracking-widest text-muted-foreground mb-3">Medications</div>
              <div className="space-y-2">
                {CLASSES.map((cls) => {
                  const med = patient.meds[cls];
                  return (
                    <div key={cls} className="flex flex-wrap items-center gap-3 rounded-lg border border-border bg-background/40 px-3 py-2">
                      <Switch
                        checked={med.on}
                        onCheckedChange={(v) => update((p) => { p.meds[cls].on = v; if (!v) p.meds[cls].dose = 0; })}
                      />
                      <div className="min-w-[120px]">
                        <code className="font-mono text-xs text-primary">{cls}</code>
                        <div className="text-xs text-muted-foreground">{REP_DRUG[cls as DrugClass]}</div>
                      </div>
                      <div className="flex items-center gap-2 ml-auto">
                        <Label className="text-[10px] font-mono uppercase tracking-widest text-muted-foreground">Dose (mg)</Label>
                        <Input
                          type="number"
                          step={0.5}
                          value={med.dose}
                          disabled={!med.on}
                          onChange={(e) => update((p) => { p.meds[cls].dose = numOrZero(e.target.value); })}
                          className="h-8 w-24 font-mono text-sm"
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Contraindications */}
            <div className="rounded-xl border border-border bg-secondary/30 p-4">
              <div className="text-[11px] font-mono uppercase tracking-widest text-muted-foreground mb-3">Contraindications</div>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
                {CONTRA_KEYS.map((k) => (
                  <label key={k} className="flex items-center gap-2 rounded-md border border-border bg-background/30 px-2 py-1.5">
                    <Switch
                      checked={!!patient.contras[k]}
                      onCheckedChange={(v) => update((p) => { p.contras[k] = v; })}
                    />
                    <span className="text-xs font-mono">{k}</span>
                  </label>
                ))}
              </div>
            </div>
          </div>

          {/* Strategy picker */}
          <div className="glass-panel rounded-2xl p-5">
            <div className="text-xs font-mono uppercase tracking-[0.2em] text-primary mb-3">02 — Titration strategy</div>
            <div className="grid sm:grid-cols-2 gap-2.5">
              {STRATEGIES.map((s) => (
                <button
                  key={s}
                  onClick={() => { setStrategy(s); setResult(null); }}
                  className={`text-left rounded-xl border p-3.5 transition-all ${
                    strategy === s
                      ? "border-primary/60 bg-primary/10 shadow-glow-soft"
                      : "border-border bg-card/40 hover:border-primary/30"
                  }`}
                >
                  <code className="font-mono text-sm text-primary">{s}</code>
                  <p className="text-xs text-muted-foreground mt-1.5 leading-relaxed">{STRATEGY_DESC[s]}</p>
                </button>
              ))}
            </div>
          </div>

          {/* ───── Output panel ───── */}
          {error && (
            <div className="rounded-2xl border border-rose-400/40 bg-rose-500/10 p-4 text-sm text-rose-200">
              <strong>API error:</strong> {error}
              <div className="mt-1 text-xs text-rose-100/70">
                Make sure the Python backend is running: <code className="font-mono">python scripts/run_api.py</code>
              </div>
            </div>
          )}

          {result ? (
            <div className="space-y-5 animate-fade-up">
              {result.global_stop && (
                <div className="rounded-2xl border border-rose-400/40 bg-rose-500/10 p-4 flex items-start gap-3">
                  <ShieldAlert className="w-5 h-5 text-rose-300 mt-0.5 shrink-0" />
                  <div>
                    <div className="font-display font-semibold text-rose-200">global_stop triggered</div>
                    <div className="text-sm text-rose-100/70 mt-0.5">
                      All disease-modifying therapy held. Escalate care.
                    </div>
                  </div>
                </div>
              )}
              {result.order_labs && !result.global_stop && (
                <div className="rounded-2xl border border-amber-400/40 bg-amber-500/10 p-4 flex items-start gap-3">
                  <FlaskConical className="w-5 h-5 text-amber-300 mt-0.5 shrink-0" />
                  <div>
                    <div className="font-display font-semibold text-amber-200">order_labs — titration gated</div>
                    <div className="text-sm text-amber-100/70 mt-0.5">
                      Labs requested for: {result.labs_requested.join(", ") || "—"}
                    </div>
                  </div>
                </div>
              )}

              {/* Classifier flags */}
              <div className="glass-panel rounded-2xl p-5">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="font-display font-semibold text-lg">Classifier output</h3>
                  <span className="text-[10px] font-mono uppercase tracking-widest text-muted-foreground">
                    LightGBM · 11 flags · preferred RAAS = {result.preferred_raas ?? "none"}
                  </span>
                </div>
                {Object.entries(result.flags).filter(([, v]) => v).length === 0 ? (
                  <p className="text-sm text-muted-foreground">No adverse-effect flags fired.</p>
                ) : (
                  <div className="flex flex-wrap gap-2">
                    {Object.entries(result.flags)
                      .filter(([, v]) => v)
                      .map(([flag]) => {
                        const prob = result.probs[flag] ?? 0;
                        const severe = flag === "any_emergency_flag" || flag.startsWith("severe_");
                        return (
                          <div key={flag} className={`rounded-lg border px-3 py-2 ${
                            severe ? "border-rose-400/40 bg-rose-500/10" : "border-primary/30 bg-primary/5"
                          }`}>
                            <div className="flex items-center gap-2">
                              <AlertTriangle className="w-3.5 h-3.5 text-current" />
                              <span className="text-sm font-medium">{FLAG_LABELS[flag] ?? flag}</span>
                              <span className="font-mono text-xs text-muted-foreground">p={prob.toFixed(2)}</span>
                            </div>
                          </div>
                        );
                      })}
                  </div>
                )}
              </div>

              {/* Per-class actions */}
              <div className="glass-panel rounded-2xl overflow-hidden">
                <div className="p-5 border-b border-border flex items-center justify-between">
                  <h3 className="font-display font-semibold text-lg">Engine actions</h3>
                  <code className="text-[10px] font-mono uppercase tracking-widest text-muted-foreground">
                    strategy = {strategy}
                  </code>
                </div>
                <div className="divide-y divide-border">
                  {Object.entries(result.changes).map(([cls, ch]) => {
                    const actionKey: string = ch.concrete_action;
                    const meta = ACTION_META[actionKey] ?? ACTION_META.maintain_dose;
                    const Icon = meta.icon;
                    return (
                      <div key={cls} className="p-5 flex flex-wrap items-center gap-4 hover:bg-secondary/30 transition-colors">
                        <div className="min-w-[180px]">
                          <code className="font-mono text-xs text-primary">{cls}</code>
                          <div className="font-medium text-sm mt-0.5">{REP_DRUG[cls as DrugClass]}</div>
                        </div>
                        <div className={`inline-flex items-center gap-1.5 rounded-md border px-2.5 py-1 text-xs font-mono ${meta.cls}`}>
                          <Icon className="w-3.5 h-3.5" />
                          {meta.label}
                        </div>
                        <div className="font-mono text-sm text-muted-foreground flex items-center gap-2">
                          <span>{ch.current} mg</span>
                          <span className="text-primary">→</span>
                          <span className="text-foreground">{ch.new_dose} mg</span>
                          {ch.target ? <span className="text-xs text-muted-foreground">/ tgt {ch.target}</span> : null}
                        </div>
                        <div className="ml-auto text-xs text-muted-foreground">
                          reason: <code className="font-mono text-foreground/80">{ch.reason}</code>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              <p className="text-center text-xs text-muted-foreground font-mono">
                {result.source === "python"
                  ? "Driven by Python TIDE-HF engine via local API · clinical decision support only."
                  : "Driven by in-browser engine (no Python backend running) · clinical decision support only."}
              </p>
            </div>
          ) : (
            !error && (
              <div className="glass-panel rounded-2xl p-12 text-center">
                <div className="w-12 h-12 rounded-2xl bg-gradient-primary mx-auto grid place-items-center shadow-glow animate-pulse-glow">
                  <Play className="w-5 h-5 text-primary-foreground" />
                </div>
                <p className="mt-5 text-muted-foreground text-sm">
                  Edit any field above, then click <span className="text-foreground font-medium">Compute</span> to run the
                  LightGBM classifier and lab-gated titration engine on the live patient.
                </p>
              </div>
            )
          )}
        </section>
      </div>
    </main>
  );
};

export default Engine;
