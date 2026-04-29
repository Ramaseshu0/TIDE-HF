import { Activity, FlaskConical, ShieldAlert, GitBranch } from "lucide-react";

const pillars = [
  {
    letter: "T",
    word: "Trajectory",
    icon: Activity,
    desc: "14 timepoints per week → 108 features. Vitals, weight drift, rhythm, SpO₂ and ECG signs become a continuous signal that the engine reasons over.",
    color: "from-cyan-400 to-sky-500",
  },
  {
    letter: "I",
    word: "Integrated",
    icon: FlaskConical,
    desc: "Demographics, baseline labs (K, Na, Cr, eGFR), current GDMT regimen and ECG-study linkage are fused — MIMIC-seeded or synthetic distribution mode.",
    color: "from-sky-400 to-blue-500",
  },
  {
    letter: "D",
    word: "Decision",
    icon: GitBranch,
    desc: "A LightGBM classifier predicts 11 adverse-effect flags. A rule-based titration engine then turns those flags into a per-class action.",
    color: "from-blue-400 to-indigo-500",
  },
  {
    letter: "E",
    word: "Engine",
    icon: ShieldAlert,
    desc: "TitrationEngine v1.2 — lab-gated, contraindication-aware. global_stop, hold + order_labs, AE resolver and dose-rung logic are all explicit and auditable.",
    color: "from-indigo-400 to-cyan-500",
  },
];

const About = () => {
  return (
    <section id="about" className="relative py-28">
      <div className="mx-auto max-w-7xl px-6">
        <div className="max-w-3xl">
          <div className="text-xs font-mono uppercase tracking-[0.2em] text-primary mb-4">
            01 — What is TIDE-HF
          </div>
          <h2 className="font-display text-4xl md:text-5xl font-semibold tracking-tight">
            A local-first GDMT titration system, <span className="text-gradient">built on four ideas</span>.
          </h2>
          <p className="mt-5 text-muted-foreground text-lg leading-relaxed">
            TIDE-HF is a complete pipeline for chronic heart-failure care: a synthetic-data
            generator, a LightGBM adverse-effect classifier, and a rule-based titration engine
            with lab gating. No MIMIC dataset required — everything runs locally.
          </p>
        </div>

        <div className="mt-16 grid sm:grid-cols-2 lg:grid-cols-4 gap-5">
          {pillars.map((p, i) => (
            <article
              key={p.word}
              className="group relative glass-panel rounded-2xl p-6 hover:-translate-y-1 transition-all duration-500 glow-border animate-fade-up"
              style={{ animationDelay: `${i * 0.1}s` }}
            >
              <div className="flex items-start justify-between mb-6">
                <div className={`w-12 h-12 rounded-xl bg-gradient-to-br ${p.color} grid place-items-center shadow-glow-soft`}>
                  <p.icon className="w-6 h-6 text-background" strokeWidth={2.5} />
                </div>
                <span className="font-display text-5xl font-semibold text-muted-foreground/30 group-hover:text-primary/60 transition-colors">
                  {p.letter}
                </span>
              </div>
              <h3 className="font-display text-xl font-semibold mb-2">{p.word}</h3>
              <p className="text-sm text-muted-foreground leading-relaxed">{p.desc}</p>
            </article>
          ))}
        </div>

        {/* Pipeline strip */}
        <div className="mt-16 glass-panel rounded-2xl p-6 overflow-x-auto">
          <div className="flex items-center gap-3 min-w-max font-mono text-xs">
            {[
              "synthetic-data generator",
              "LightGBM AE classifier (11 flags)",
              "TitrationEngine v1.2 (lab-gated)",
              "strategy applier",
              "Streamlit UI",
            ].map((step, i, arr) => (
              <div key={step} className="flex items-center gap-3">
                <span className="px-3 py-1.5 rounded-lg bg-secondary border border-border text-foreground">
                  {step}
                </span>
                {i < arr.length - 1 && <span className="text-primary">→</span>}
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
};

export default About;
