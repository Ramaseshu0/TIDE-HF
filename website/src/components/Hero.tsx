import heroImg from "@/assets/hero-tide.jpg";
import { ArrowRight, Sparkles } from "lucide-react";
import { Link } from "react-router-dom";

const Hero = () => {
  return (
    <section id="top" className="relative min-h-screen pt-32 pb-20 overflow-hidden bg-gradient-hero">
      <div className="absolute inset-0 grid-bg opacity-60 pointer-events-none" />
      <div className="absolute -top-20 left-1/2 -translate-x-1/2 w-[800px] h-[800px] rounded-full blur-3xl opacity-30 bg-primary/30 pointer-events-none" />

      <div className="relative mx-auto max-w-7xl px-6">
        <div className="grid lg:grid-cols-2 gap-12 items-center">
          <div className="animate-fade-up">
            <div className="inline-flex items-center gap-2 rounded-full border border-border bg-card/50 backdrop-blur px-3.5 py-1.5 text-xs text-muted-foreground mb-6">
              <Sparkles className="w-3.5 h-3.5 text-primary" />
              <span className="font-mono uppercase tracking-wider">Clinical AI · Heart Failure</span>
            </div>

            <h1 className="font-display text-5xl md:text-6xl lg:text-7xl font-semibold leading-[1.02] tracking-tight">
              Guideline-directed
              <span className="block text-gradient">GDMT titration, end-to-end.</span>
            </h1>

            <p className="mt-6 text-lg text-muted-foreground max-w-xl leading-relaxed">
              TIDE-HF — Trajectory · Integrated · Decision · Engine — is a local-first pipeline for
              chronic heart-failure care: a synthetic patient-week generator, an 11-flag LightGBM
              adverse-effect classifier, and a lab-gated rule-based titration engine.
            </p>

            <div className="mt-9 flex flex-wrap gap-3">
              <Link
                to="/engine"
                className="inline-flex items-center gap-2 rounded-xl bg-gradient-primary px-6 py-3.5 text-sm font-medium text-primary-foreground shadow-elegant hover:shadow-glow transition-all"
              >
                Open the Engine
                <ArrowRight className="w-4 h-4" />
              </Link>
              <a
                href="#chat"
                className="inline-flex items-center gap-2 rounded-xl border border-border bg-secondary/40 px-6 py-3.5 text-sm font-medium hover:bg-secondary transition-colors"
              >
                Talk to TIDE AI
              </a>
              <a
                href="https://github.com/Ramaseshu0/TIDE-HF"
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-2 rounded-xl border border-border bg-secondary/40 px-6 py-3.5 text-sm font-medium hover:bg-secondary transition-colors"
              >
                GitHub
              </a>
            </div>

            <dl className="mt-12 grid grid-cols-3 gap-6 max-w-lg">
              {[
                { v: "11", l: "AE flags classified" },
                { v: "17", l: "Preset scenarios" },
                { v: "4", l: "Titration strategies" },
              ].map((s) => (
                <div key={s.v}>
                  <dt className="font-display text-2xl text-foreground">{s.v}</dt>
                  <dd className="text-xs text-muted-foreground mt-1">{s.l}</dd>
                </div>
              ))}
            </dl>
          </div>

          <div className="relative animate-fade-up" style={{ animationDelay: "0.2s" }}>
            <div className="absolute inset-0 bg-primary/20 blur-3xl rounded-full animate-pulse-glow" />
            <div className="relative rounded-3xl overflow-hidden glow-border">
              <img
                src={heroImg}
                alt="Holographic visualization of a heart with ECG waveform representing TIDE-HF AI"
                width={1600}
                height={1200}
                className="w-full h-auto animate-float"
              />
              <div className="absolute inset-0 bg-gradient-to-t from-background via-transparent to-transparent" />

              {/* Floating data chips */}
              <div className="absolute top-6 left-6 glass-panel rounded-xl px-3 py-2 text-xs font-mono">
                <div className="flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-primary animate-pulse" />
                  <span className="text-muted-foreground">EF</span>
                  <span className="text-foreground">28%</span>
                </div>
              </div>
              <div className="absolute bottom-6 right-6 glass-panel rounded-xl px-3 py-2 text-xs font-mono">
                <div className="flex items-center gap-2">
                  <span className="text-muted-foreground">NT-proBNP</span>
                  <span className="text-primary">1,842</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Hero;
