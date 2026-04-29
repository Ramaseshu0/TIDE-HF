import trajectoryImg from "@/assets/trajectory.jpg";
import decisionImg from "@/assets/decision-engine.jpg";

const steps = [
  {
    n: "01",
    title: "Synthesize the patient-week",
    body: "10,000 synthetic patient-weeks are generated from MIMIC-IV CHF visits (mimic mode) or from Gaussian distributions (distribution mode). Each week is featurized into 108 features across 14 timepoints — vitals, weight, SpO₂, rhythm and ECG signs.",
    img: trajectoryImg,
    chips: ["mimic mode", "distribution mode", "108 features"],
  },
  {
    n: "02",
    title: "Classify, gate, titrate",
    body: "A LightGBM classifier predicts 11 adverse-effect flags (hyperkalemia, renal dysfunction, hypotension, bradycardia…). The TitrationEngine then applies lab gating, contraindication checks and per-class actions: maintain, increase, decrease, hold + order_labs, or global_stop.",
    img: decisionImg,
    chips: ["11 AE flags", "lab gating", "contraindication-aware"],
  },
];

const STRATEGIES = [
  { name: "traditional", desc: "One new class per week · RAAS → BB → MRA → SGLT2i → loop · single-rung steps" },
  { name: "strong_hf", desc: "All eligible classes started at once · double-rung up-titration" },
  { name: "rapid_sequence", desc: "All eligible classes at once · single-rung steps (Greene 2021)" },
  { name: "sglt_mra_first", desc: "Phase 1: SGLT2i + MRA · Phase 2: add ARNi + beta-blocker" },
];

const HowItWorks = () => {
  return (
    <section id="how" className="relative py-28 bg-gradient-to-b from-background via-secondary/20 to-background">
      <div className="mx-auto max-w-7xl px-6">
        <div className="max-w-3xl mb-20">
          <div className="text-xs font-mono uppercase tracking-[0.2em] text-primary mb-4">
            02 — How it works
          </div>
          <h2 className="font-display text-4xl md:text-5xl font-semibold tracking-tight">
            From a synthetic patient-week to a <span className="text-gradient">defensible per-class action</span>.
          </h2>
        </div>

        <div className="space-y-24">
          {steps.map((s, i) => (
            <div
              key={s.n}
              className={`grid lg:grid-cols-2 gap-12 items-center ${i % 2 === 1 ? "lg:[&>*:first-child]:order-2" : ""}`}
            >
              <div className="relative rounded-3xl overflow-hidden glow-border group">
                <img
                  src={s.img}
                  alt={s.title}
                  loading="lazy"
                  width={1200}
                  height={800}
                  className="w-full h-auto transition-transform duration-700 group-hover:scale-105"
                />
                <div className="absolute inset-0 bg-gradient-to-t from-background/80 via-transparent to-transparent" />
                <div className="absolute bottom-5 left-5 font-mono text-xs text-primary tracking-widest">
                  /// {s.title.toUpperCase()}
                </div>
              </div>
              <div>
                <div className="font-mono text-sm text-primary mb-3">{s.n}</div>
                <h3 className="font-display text-3xl md:text-4xl font-semibold mb-5 tracking-tight">
                  {s.title}
                </h3>
                <p className="text-muted-foreground text-lg leading-relaxed">{s.body}</p>
                <div className="mt-5 flex flex-wrap gap-2">
                  {s.chips.map((c) => (
                    <span
                      key={c}
                      className="text-xs font-mono px-2.5 py-1 rounded-md border border-border bg-secondary/60 text-muted-foreground"
                    >
                      {c}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Strategies */}
        <div id="engine" className="mt-32">
          <div className="max-w-3xl mb-10">
            <div className="text-xs font-mono uppercase tracking-[0.2em] text-primary mb-4">
              03 — Four titration strategies
            </div>
            <h3 className="font-display text-3xl md:text-4xl font-semibold tracking-tight">
              Pick how aggressively to <span className="text-gradient">build the regimen</span>.
            </h3>
          </div>
          <div className="grid md:grid-cols-2 gap-5">
            {STRATEGIES.map((s, i) => (
              <div
                key={s.name}
                className="glass-panel rounded-2xl p-6 hover:border-primary/40 transition-colors animate-fade-up"
                style={{ animationDelay: `${i * 0.08}s` }}
              >
                <div className="flex items-baseline justify-between mb-3">
                  <code className="font-mono text-primary text-lg">{s.name}</code>
                  <span className="text-[10px] font-mono uppercase tracking-widest text-muted-foreground">
                    strategy {String(i + 1).padStart(2, "0")}
                  </span>
                </div>
                <p className="text-sm text-muted-foreground leading-relaxed">{s.desc}</p>
              </div>
            ))}
          </div>

          {/* Engine ECG */}
          <div className="mt-16 glass-panel rounded-3xl p-10 md:p-14 relative overflow-hidden">
            <div className="absolute inset-0 grid-bg opacity-30" />
            <div className="relative grid lg:grid-cols-[1fr_1.2fr] gap-10 items-center">
              <div>
                <div className="text-xs font-mono uppercase tracking-[0.2em] text-primary mb-4">
                  Engine internals
                </div>
                <h3 className="font-display text-3xl font-semibold tracking-tight">
                  Lab-gated. Contraindication-aware. <span className="text-gradient">Auditable.</span>
                </h3>
                <p className="mt-5 text-muted-foreground leading-relaxed">
                  Every action carries a reason: <code className="font-mono text-primary text-xs">at_or_above_target</code>,{" "}
                  <code className="font-mono text-primary text-xs">hyperkalemia_detected</code>,{" "}
                  <code className="font-mono text-primary text-xs">order_labs</code>,{" "}
                  <code className="font-mono text-primary text-xs">global_stop</code>. Seventeen presets in the UI cover the
                  main decision paths — newly diagnosed, partial titration, AE resolution, and contraindication
                  edge cases.
                </p>
              </div>
              <svg viewBox="0 0 600 200" className="w-full h-auto">
                <defs>
                  <linearGradient id="ecgGrad" x1="0" x2="1">
                    <stop offset="0%" stopColor="hsl(184 100% 55%)" stopOpacity="0" />
                    <stop offset="50%" stopColor="hsl(184 100% 55%)" />
                    <stop offset="100%" stopColor="hsl(210 100% 60%)" stopOpacity="0" />
                  </linearGradient>
                </defs>
                <path
                  d="M0 100 L120 100 L140 100 L150 60 L160 140 L170 80 L180 100 L300 100 L320 100 L330 40 L340 160 L350 70 L360 100 L500 100 L520 100 L530 50 L540 150 L550 90 L560 100 L600 100"
                  fill="none"
                  stroke="url(#ecgGrad)"
                  strokeWidth="2.5"
                  className="animate-ecg"
                />
                <path
                  d="M0 100 L600 100"
                  fill="none"
                  stroke="hsl(var(--border))"
                  strokeWidth="1"
                  strokeDasharray="4 6"
                />
              </svg>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default HowItWorks;
