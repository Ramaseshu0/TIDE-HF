"use client"

import { motion } from "framer-motion"
import {
  Activity,
  ArrowRight,
  ChevronRight,
  HeartPulse,
  Menu,
  Sparkles,
} from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"

export default function Hero() {
  return (
    <section className="relative isolate min-h-screen w-full overflow-hidden bg-black text-white">
      {/* Background — dotted grid + cyan radial glow */}
      <div className="absolute inset-0 -z-10 bg-grid-fade" />
      <div className="pointer-events-none absolute -top-32 left-1/2 -z-10 h-[500px] w-[900px] -translate-x-1/2 rounded-full bg-cyan-500/20 blur-[120px]" />
      <div className="pointer-events-none absolute -bottom-40 right-0 -z-10 h-[420px] w-[600px] rounded-full bg-cyan-400/10 blur-[120px]" />
      <div className="pointer-events-none absolute inset-x-0 top-0 -z-10 h-px bg-gradient-to-r from-transparent via-cyan-400/40 to-transparent" />

      {/* Header */}
      <header className="relative z-20 mx-auto flex max-w-7xl items-center justify-between px-6 py-5">
        <a href="#" className="group flex items-center gap-2">
          <span className="relative inline-flex size-9 items-center justify-center rounded-lg border border-white/10 bg-white/5 backdrop-blur">
            <HeartPulse className="size-5 text-cyan-400" />
            <span className="absolute inset-0 rounded-lg ring-1 ring-cyan-400/0 transition group-hover:ring-cyan-400/40" />
          </span>
          <div className="leading-none">
            <span className="block text-base font-semibold tracking-tight">TIDE</span>
            <span className="block text-[10px] uppercase tracking-[0.18em] text-zinc-500">CHF Titration</span>
          </div>
        </a>

        <nav className="hidden items-center gap-1 md:flex">
          {["Platform", "Titration", "Patients", "Clinicians", "Docs"].map((l) => (
            <a
              key={l}
              href={`#${l.toLowerCase()}`}
              className="rounded-full px-3 py-2 text-sm text-zinc-400 transition hover:bg-white/5 hover:text-white"
            >
              {l}
            </a>
          ))}
        </nav>

        <div className="flex items-center gap-2">
          <Button variant="ghost" size="sm" className="hidden text-zinc-300 hover:text-white md:inline-flex">
            Sign in
          </Button>
          <Button size="sm" className="gap-1 bg-cyan-400 text-black hover:bg-cyan-300">
            Get started <ArrowRight className="size-3.5" />
          </Button>
          <Button size="icon" variant="ghost" className="md:hidden">
            <Menu className="size-5" />
          </Button>
        </div>
      </header>

      {/* Hero body */}
      <div className="relative z-10 mx-auto flex max-w-7xl flex-col items-center gap-12 px-6 pt-16 pb-24 lg:flex-row lg:items-stretch lg:gap-16 lg:pt-24 lg:pb-32">
        <motion.div
          className="flex-1"
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <Badge
            variant="outline"
            className="border-cyan-400/30 bg-cyan-400/5 text-cyan-300"
          >
            <Sparkles className="size-3" /> Guideline‑directed AI titration
          </Badge>

          <h1 className="mt-6 text-5xl font-semibold leading-[1.05] tracking-tight md:text-6xl lg:text-7xl">
            Heart failure care,{" "}
            <span className="bg-gradient-to-br from-white via-white to-cyan-300 bg-clip-text text-transparent">
              measured
            </span>
            <br />
            and{" "}
            <span className="relative">
              <span className="bg-gradient-to-r from-cyan-300 to-cyan-500 bg-clip-text text-transparent">
                titrated
              </span>
              <span className="absolute -bottom-1 left-0 h-px w-full bg-gradient-to-r from-cyan-400/0 via-cyan-400 to-cyan-400/0" />
            </span>
            .
          </h1>

          <p className="mt-6 max-w-xl text-base leading-relaxed text-zinc-400 md:text-lg">
            TIDE pairs a LightGBM safety classifier with explicit GDMT rules to titrate the four
            pillars of heart‑failure therapy — safely, transparently, and with your clinician in the
            loop.
          </p>

          <div className="mt-8 flex flex-wrap items-center gap-3">
            <Button size="lg" className="gap-2 bg-cyan-400 text-black hover:bg-cyan-300">
              Launch console <ArrowRight className="size-4" />
            </Button>
            <Button
              size="lg"
              variant="outline"
              className="gap-2 border-white/15 bg-white/5 text-white hover:bg-white/10 hover:text-white"
            >
              Read the protocol <ChevronRight className="size-4" />
            </Button>
          </div>

          {/* trust strip */}
          <div className="mt-12 grid max-w-md grid-cols-3 gap-6 border-t border-white/5 pt-6 text-sm">
            {[
              { k: "10,000+", v: "patient‑weeks" },
              { k: "11", v: "safety flags" },
              { k: "4", v: "GDMT pillars" },
            ].map((s) => (
              <div key={s.v}>
                <div className="text-2xl font-semibold tracking-tight text-white">{s.k}</div>
                <div className="text-xs uppercase tracking-wider text-zinc-500">{s.v}</div>
              </div>
            ))}
          </div>
        </motion.div>

        {/* Right side — monitor card */}
        <motion.div
          className="flex-1 lg:max-w-xl"
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.7, delay: 0.1 }}
        >
          <MonitorCard />
        </motion.div>
      </div>
    </section>
  )
}

function MonitorCard() {
  return (
    <div className="relative">
      <div className="absolute -inset-1 rounded-3xl bg-gradient-to-br from-cyan-500/20 via-transparent to-transparent blur-xl" />
      <div className="relative overflow-hidden rounded-3xl border border-white/10 bg-zinc-950/80 backdrop-blur-xl shadow-2xl shadow-cyan-500/10">
        {/* top window bar */}
        <div className="flex items-center justify-between border-b border-white/5 bg-zinc-900/60 px-4 py-2.5">
          <div className="flex items-center gap-1.5">
            <span className="size-2.5 rounded-full bg-zinc-700" />
            <span className="size-2.5 rounded-full bg-zinc-700" />
            <span className="size-2.5 rounded-full bg-zinc-700" />
          </div>
          <div className="flex items-center gap-1.5 text-[10px] font-mono uppercase tracking-widest text-zinc-500">
            <span className="size-1.5 animate-pulse rounded-full bg-cyan-400" />
            tide://patient/1042
          </div>
          <div className="w-12" />
        </div>

        <div className="grid grid-cols-4 gap-px bg-white/5">
          {[
            { l: "Age", v: "67", u: "yrs" },
            { l: "HR", v: "72", u: "bpm" },
            { l: "SBP", v: "118", u: "mmHg" },
            { l: "K⁺", v: "4.2", u: "mEq/L" },
          ].map((m) => (
            <div key={m.l} className="bg-zinc-950 px-4 py-3">
              <div className="text-[10px] uppercase tracking-widest text-zinc-500">{m.l}</div>
              <div className="mt-1 flex items-baseline gap-1">
                <span className="text-2xl font-semibold tabular-nums text-white">{m.v}</span>
                <span className="text-[10px] text-zinc-500">{m.u}</span>
              </div>
            </div>
          ))}
        </div>

        {/* ECG */}
        <div className="relative h-36 overflow-hidden bg-zinc-950">
          <ECGGrid />
          <ECGTrace />
          <div className="absolute left-3 top-3 flex items-center gap-1.5 text-[10px] font-mono uppercase tracking-widest text-cyan-300/80">
            <Activity className="size-3" /> ECG · LEAD II
          </div>
          <div className="absolute right-3 top-3 text-[10px] font-mono uppercase tracking-widest text-zinc-500">
            sinus rhythm
          </div>
        </div>

        {/* Titration footer */}
        <div className="border-t border-white/5 px-4 py-4">
          <div className="mb-2 flex items-center justify-between text-xs">
            <span className="text-zinc-400">Bisoprolol — week 4</span>
            <span className="font-mono tabular-nums text-cyan-300">2.5 → 5 mg</span>
          </div>
          <div className="h-1.5 w-full overflow-hidden rounded-full bg-white/5">
            <div className="h-full w-2/3 rounded-full bg-gradient-to-r from-cyan-500 to-cyan-300" />
          </div>
        </div>
      </div>
    </div>
  )
}

function ECGGrid() {
  return (
    <svg className="absolute inset-0 h-full w-full" preserveAspectRatio="none" viewBox="0 0 800 144">
      <defs>
        <pattern id="ecg-grid-sm" width="10" height="10" patternUnits="userSpaceOnUse">
          <path d="M 10 0 L 0 0 0 10" fill="none" stroke="rgba(34,211,238,0.06)" strokeWidth="0.5" />
        </pattern>
        <pattern id="ecg-grid-lg" width="50" height="50" patternUnits="userSpaceOnUse">
          <path d="M 50 0 L 0 0 0 50" fill="none" stroke="rgba(34,211,238,0.12)" strokeWidth="1" />
        </pattern>
      </defs>
      <rect width="800" height="144" fill="url(#ecg-grid-sm)" />
      <rect width="800" height="144" fill="url(#ecg-grid-lg)" />
    </svg>
  )
}

function ECGTrace() {
  return (
    <svg className="absolute inset-0 h-full w-full" viewBox="0 0 800 144" preserveAspectRatio="none">
      <defs>
        <linearGradient id="ecg-glow" x1="0%" x2="100%">
          <stop offset="0%" stopColor="#22d3ee" stopOpacity="0" />
          <stop offset="50%" stopColor="#22d3ee" stopOpacity="1" />
          <stop offset="100%" stopColor="#22d3ee" stopOpacity="0" />
        </linearGradient>
      </defs>
      <motion.path
        d="M0 72 L60 72 L80 72 L88 60 L96 96 L104 24 L112 110 L120 72 L200 72 L260 72 L280 60 L288 96 L296 24 L304 110 L312 72 L400 72 L460 72 L480 60 L488 96 L496 24 L504 110 L512 72 L600 72 L660 72 L680 60 L688 96 L696 24 L704 110 L712 72 L800 72"
        fill="none"
        stroke="#22d3ee"
        strokeWidth="2"
        strokeLinejoin="round"
        strokeLinecap="round"
        initial={{ pathLength: 0 }}
        animate={{ pathLength: 1 }}
        transition={{ duration: 3.2, repeat: Infinity, ease: "linear" }}
        style={{ filter: "drop-shadow(0 0 6px rgba(34,211,238,0.6))" }}
      />
    </svg>
  )
}
