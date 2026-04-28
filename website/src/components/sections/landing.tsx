"use client"

import { motion, useInView, useMotionValue, useTransform, animate } from "framer-motion"
import { useEffect, useRef } from "react"
import {
  Activity,
  ArrowRight,
  Brain,
  Calendar,
  CheckCircle,
  ChevronRight,
  ClipboardList,
  Clock,
  HeartPulse,
  Mail,
  MapPin,
  Phone,
  Pill,
  ShieldCheck,
  Sparkles,
  Stethoscope,
  Users,
  Zap,
} from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Progress } from "@/components/ui/progress"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"

/* --------------------------------- helpers --------------------------------- */

function SectionHeader({
  eyebrow,
  title,
  sub,
}: {
  eyebrow: string
  title: React.ReactNode
  sub?: string
}) {
  return (
    <div className="mx-auto max-w-3xl text-center">
      <Badge
        variant="outline"
        className="border-cyan-400/30 bg-cyan-400/5 text-cyan-300"
      >
        <span className="size-1.5 rounded-full bg-cyan-400 shadow-[0_0_6px_rgba(34,211,238,0.9)]" />
        {eyebrow}
      </Badge>
      <h2 className="mt-5 text-3xl font-semibold leading-[1.1] tracking-tight text-white md:text-5xl">
        {title}
      </h2>
      {sub && (
        <p className="mt-4 text-base leading-relaxed text-zinc-400 md:text-lg">{sub}</p>
      )}
    </div>
  )
}

function Counter({ to, suffix = "" }: { to: number; suffix?: string }) {
  const ref = useRef<HTMLSpanElement>(null)
  const inView = useInView(ref, { once: true, margin: "-20%" })
  const mv = useMotionValue(0)
  const text = useTransform(mv, (v) => `${Math.round(v).toLocaleString()}${suffix}`)
  useEffect(() => {
    if (inView) {
      const c = animate(mv, to, { duration: 1.5, ease: "easeOut" })
      return () => c.stop()
    }
  }, [inView, to, mv])
  return <motion.span ref={ref}>{text}</motion.span>
}

/* ---------------------------------- Stats ---------------------------------- */

export function Stats() {
  const items = [
    { v: 10000, suffix: "+", label: "patient‑weeks", icon: Activity },
    { v: 11, label: "safety flags", icon: ShieldCheck },
    { v: 98, suffix: "%", label: "rule coverage", icon: CheckCircle },
    { v: 4, label: "GDMT pillars", icon: Pill },
  ]
  return (
    <section className="relative z-10 px-6 py-20">
      <div className="mx-auto max-w-7xl rounded-3xl border border-white/10 bg-zinc-950/60 p-8 backdrop-blur md:p-12">
        <div className="grid grid-cols-2 gap-8 md:grid-cols-4">
          {items.map((it) => (
            <div key={it.label} className="flex flex-col items-start gap-3">
              <div className="inline-flex size-10 items-center justify-center rounded-lg border border-cyan-400/20 bg-cyan-400/5 text-cyan-300">
                <it.icon className="size-5" />
              </div>
              <div className="text-3xl font-semibold tracking-tight text-white md:text-4xl">
                <Counter to={it.v} suffix={it.suffix ?? ""} />
              </div>
              <div className="text-xs uppercase tracking-widest text-zinc-500">{it.label}</div>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}

/* --------------------------------- Features -------------------------------- */

const featureCards = [
  {
    icon: Brain,
    title: "LightGBM safety classifier",
    body: "Eleven adverse‑effect flags trained on 10k+ synthetic patient‑weeks. TIDE only proposes a step when the rule engine allows it.",
  },
  {
    icon: ClipboardList,
    title: "Guideline‑directed rules",
    body: "Lab gating on K⁺, eGFR, SCr and SBP. Every uptitration is auditable — TIDE shows the exact rule that fired.",
  },
  {
    icon: Activity,
    title: "Real‑time monitoring",
    body: "Vitals, weights and labs feed the engine continuously. Drift triggers a re‑evaluation, not a missed week.",
  },
  {
    icon: Calendar,
    title: "Smart scheduling",
    body: "Adherence reminders that respect a patient's routine. Cadence syncs to clinic visits and lab draws automatically.",
  },
  {
    icon: ShieldCheck,
    title: "Local‑first & private",
    body: "Runs entirely on your machine. No MIMIC dataset required — synthetic generator is built in.",
  },
  {
    icon: Sparkles,
    title: "Strategy applier",
    body: "Conservative, standard, and aggressive titration strategies — compared on the same patient, side by side.",
  },
]

export function Features() {
  return (
    <section id="platform" className="px-6 py-28">
      <div className="mx-auto max-w-7xl">
        <SectionHeader
          eyebrow="Platform"
          title={
            <>
              Care that's <em className="not-italic text-cyan-300">safe by design</em>,
              <br className="hidden md:block" /> not just smart on average.
            </>
          }
          sub="TIDE wraps a learned safety model in transparent guideline rules, so every dose change has a reason a patient — and a regulator — can read."
        />

        <div className="mt-14 grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
          {featureCards.map((f, i) => (
            <motion.div
              key={f.title}
              initial={{ opacity: 0, y: 24 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-10%" }}
              transition={{ duration: 0.4, delay: i * 0.05 }}
            >
              <Card className="group h-full border-white/10 bg-zinc-950/60 transition hover:border-cyan-400/30 hover:bg-zinc-950">
                <CardHeader>
                  <div className="mb-3 inline-flex size-11 items-center justify-center rounded-lg border border-cyan-400/20 bg-cyan-400/5 text-cyan-300 transition group-hover:bg-cyan-400/10">
                    <f.icon className="size-5" />
                  </div>
                  <CardTitle className="text-lg text-white">{f.title}</CardTitle>
                </CardHeader>
                <CardContent>
                  <CardDescription className="text-sm leading-relaxed text-zinc-400">
                    {f.body}
                  </CardDescription>
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  )
}

/* ----------------------------- How it works -------------------------------- */

export function HowItWorks() {
  const steps = [
    { n: "01", title: "Generate or load", body: "Spin up 10,000 synthetic CHF patient‑weeks locally — or plug in your own parquet.", icon: Activity },
    { n: "02", title: "Score risk", body: "LightGBM predicts 11 adverse effects per patient‑week with calibrated probabilities.", icon: Brain },
    { n: "03", title: "Apply rules", body: "Lab gates and clinical thresholds decide whether each pillar moves up, holds, or steps down.", icon: ClipboardList },
    { n: "04", title: "Explain & track", body: "Patient and clinician see the same trajectory — with reasons, not black‑box decisions.", icon: Stethoscope },
  ]

  return (
    <section className="border-y border-white/5 bg-zinc-950/40 px-6 py-28">
      <div className="mx-auto max-w-7xl">
        <SectionHeader
          eyebrow="How it works"
          title={
            <>
              Four steps. <span className="text-cyan-300">Each one transparent.</span>
            </>
          }
          sub="From raw labs to a kinder titration — every decision reversible, every reason on the page."
        />

        <div className="relative mt-16 grid grid-cols-1 gap-4 md:grid-cols-4">
          <div className="absolute left-[12%] right-[12%] top-7 hidden h-px bg-gradient-to-r from-transparent via-cyan-400/30 to-transparent md:block" />
          {steps.map((s, i) => (
            <motion.div
              key={s.n}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-10%" }}
              transition={{ duration: 0.4, delay: i * 0.08 }}
              className="relative"
            >
              <div className="relative z-10 mx-auto mb-4 inline-flex size-12 items-center justify-center rounded-full border border-cyan-400/30 bg-zinc-950 text-cyan-300">
                <s.icon className="size-5" />
              </div>
              <div className="rounded-2xl border border-white/10 bg-zinc-950/80 p-5 text-center">
                <div className="font-mono text-xs tracking-widest text-cyan-400/70">{s.n}</div>
                <h3 className="mt-1 font-semibold text-white">{s.title}</h3>
                <p className="mt-2 text-sm leading-relaxed text-zinc-400">{s.body}</p>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  )
}

/* ----------------------- Titration: tabs + dose ramp ----------------------- */

export function Titration() {
  return (
    <section id="titration" className="px-6 py-28">
      <div className="mx-auto max-w-7xl">
        <SectionHeader
          eyebrow="Medication titration"
          title={
            <>
              Doses that adjust to the patient,{" "}
              <span className="text-cyan-300">not the other way around.</span>
            </>
          }
          sub="TIDE walks each pillar up to its target — pausing on lab gates, retrying on the next safe window, and explaining itself in plain English."
        />

        <div className="mt-14 grid grid-cols-1 gap-8 lg:grid-cols-2">
          {/* Left: tabs */}
          <motion.div
            initial={{ opacity: 0, x: -24 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true, margin: "-10%" }}
            transition={{ duration: 0.5 }}
            className="rounded-2xl border border-white/10 bg-zinc-950/60 p-6 md:p-8"
          >
            <Tabs defaultValue="process" className="w-full">
              <TabsList className="grid w-full grid-cols-3 border border-white/10 bg-zinc-900">
                <TabsTrigger value="process">Process</TabsTrigger>
                <TabsTrigger value="benefits">Benefits</TabsTrigger>
                <TabsTrigger value="safety">Safety</TabsTrigger>
              </TabsList>

              <TabsContent value="process" className="mt-6 space-y-3">
                {[
                  "Initial assessment and baseline labs",
                  "Gradual dose adjustment based on response",
                  "Continuous monitoring and re‑evaluation",
                  "Regular follow‑ups with your care team",
                ].map((step) => (
                  <Row key={step} icon={CheckCircle} text={step} tint="text-cyan-300" />
                ))}
              </TabsContent>

              <TabsContent value="benefits" className="mt-6 space-y-3">
                {[
                  "Personalized treatment plans per patient",
                  "Reduced side‑effects via gated uptitration",
                  "Improved medication efficacy at target dose",
                  "Better long‑term outcomes — fewer admits",
                ].map((b) => (
                  <Row key={b} icon={Zap} text={b} tint="text-cyan-300" />
                ))}
              </TabsContent>

              <TabsContent value="safety" className="mt-6 space-y-3">
                {[
                  "Guideline‑aligned protocols (ACC/AHA, ESC)",
                  "Continuous adverse‑effect risk scoring",
                  "Clinician override on every step",
                  "Full rule trace for audit and review",
                ].map((s) => (
                  <Row key={s} icon={ShieldCheck} text={s} tint="text-cyan-300" />
                ))}
              </TabsContent>
            </Tabs>
          </motion.div>

          {/* Right: dose ramp */}
          <motion.div
            initial={{ opacity: 0, x: 24 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true, margin: "-10%" }}
            transition={{ duration: 0.5 }}
          >
            <Card className="relative overflow-hidden border-white/10 bg-zinc-950/60">
              <div className="pointer-events-none absolute -right-20 -top-20 h-56 w-56 rounded-full bg-cyan-500/10 blur-3xl" />
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-white">
                  <Pill className="size-4 text-cyan-300" />
                  Bisoprolol — titration trajectory
                </CardTitle>
                <CardDescription className="text-zinc-400">
                  Sample 8‑week ramp from 25 → 100% target dose, with a single rule‑gated hold.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-5">
                {[
                  { label: "Week 1–2", dose: "1.25 mg", pct: 25, ok: true },
                  { label: "Week 3–4", dose: "2.5 mg", pct: 50, ok: true },
                  { label: "Week 5–6", dose: "3.75 mg", pct: 75, ok: false, note: "Hold · SBP 88" },
                  { label: "Week 7+", dose: "5 mg (target)", pct: 100, ok: true },
                ].map((r) => (
                  <div key={r.label} className="space-y-2">
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-zinc-300">{r.label}</span>
                      <span className="flex items-center gap-2 font-mono tabular-nums text-cyan-300">
                        {r.dose}
                        {r.note && (
                          <Badge
                            variant="outline"
                            className="border-amber-400/30 bg-amber-400/5 text-[10px] uppercase tracking-wider text-amber-300"
                          >
                            {r.note}
                          </Badge>
                        )}
                      </span>
                    </div>
                    <Progress value={r.pct} className="h-1.5 bg-white/5 [&>div]:bg-gradient-to-r [&>div]:from-cyan-500 [&>div]:to-cyan-300" />
                  </div>
                ))}

                <div className="mt-2 flex items-center gap-2 rounded-lg border border-cyan-400/15 bg-cyan-400/5 p-3 text-xs text-cyan-200/90">
                  <CheckCircle className="size-4 shrink-0 text-cyan-300" />
                  Target reached across the trajectory.
                </div>
              </CardContent>
            </Card>
          </motion.div>
        </div>
      </div>
    </section>
  )
}

function Row({
  icon: Icon,
  text,
  tint,
}: {
  icon: typeof CheckCircle
  text: string
  tint: string
}) {
  return (
    <div className="flex items-start gap-3 rounded-lg border border-white/5 bg-zinc-900/40 px-3 py-2.5">
      <Icon className={`mt-0.5 size-4 shrink-0 ${tint}`} />
      <span className="text-sm text-zinc-200">{text}</span>
    </div>
  )
}

/* --------------------- Patient / Clinician split --------------------------- */

export function PatientClinicianSplit() {
  return (
    <section className="border-y border-white/5 bg-zinc-950/40 px-6 py-28">
      <div className="mx-auto max-w-7xl">
        <SectionHeader
          eyebrow="Built for both sides"
          title={
            <>
              Two views. <span className="text-cyan-300">One trajectory.</span>
            </>
          }
          sub="The same titration data — translated for whoever is reading it."
        />

        <div className="mt-14 grid grid-cols-1 gap-5 lg:grid-cols-2">
          <SplitCard
            id="patients"
            tone="patient"
            title="For patients"
            tagline="Calm, plain‑language guidance. No jargon, no panic."
            icon={Users}
            bullets={[
              "Daily weight check‑ins with friendly nudges",
              "Plain explanations of every dose change",
              "Red‑flag symptom guide — when to call your team",
              "Med reminders that respect your routine",
            ]}
            cta="Open patient app"
          />
          <SplitCard
            id="clinicians"
            tone="doctor"
            title="For clinicians"
            tagline="Auditable decisions. Trajectory at a glance. You stay in control."
            icon={Stethoscope}
            bullets={[
              "Risk‑gated next‑step recommendations",
              "Rule trace for every uptitration or hold",
              "Cohort dashboards across strategies",
              "Override anytime — TIDE logs the reason",
            ]}
            cta="Open clinician console"
          />
        </div>
      </div>
    </section>
  )
}

function SplitCard({
  id,
  tone,
  title,
  tagline,
  icon: Icon,
  bullets,
  cta,
}: {
  id: string
  tone: "patient" | "doctor"
  title: string
  tagline: string
  icon: typeof Users
  bullets: string[]
  cta: string
}) {
  return (
    <motion.div
      id={id}
      initial={{ opacity: 0, y: 24 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-10%" }}
      transition={{ duration: 0.5 }}
      className={
        "group relative overflow-hidden rounded-3xl border p-8 md:p-10 " +
        (tone === "patient"
          ? "border-cyan-400/20 bg-gradient-to-b from-cyan-500/5 to-transparent"
          : "border-white/10 bg-zinc-950/80")
      }
    >
      <div className="pointer-events-none absolute -right-20 -top-20 h-56 w-56 rounded-full bg-cyan-500/10 blur-3xl" />
      <div className="mb-5 inline-flex size-12 items-center justify-center rounded-2xl border border-cyan-400/30 bg-cyan-400/10 text-cyan-300">
        <Icon className="size-6" />
      </div>
      <h3 className="text-2xl font-semibold tracking-tight text-white">{title}</h3>
      <p className="mt-2 text-sm text-zinc-400">{tagline}</p>
      <ul className="mt-6 space-y-3">
        {bullets.map((b) => (
          <li key={b} className="flex items-start gap-3 text-sm text-zinc-300">
            <CheckCircle className="mt-0.5 size-4 shrink-0 text-cyan-300" />
            {b}
          </li>
        ))}
      </ul>
      <Button
        variant="outline"
        className="mt-7 gap-2 border-white/15 bg-white/5 text-white hover:bg-white/10 hover:text-white"
      >
        {cta} <ArrowRight className="size-4" />
      </Button>
    </motion.div>
  )
}

/* --------------------------------- Quotes --------------------------------- */

const quotes = [
  {
    name: "Sarah Johnson",
    role: "Patient · NYHA II",
    avatar: "https://api.dicebear.com/7.x/avataaars/svg?seed=Sarah&backgroundColor=222",
    body:
      "I used to dread my dose changes. TIDE explains why, in words my grandkids understand, and pings my cardiologist if anything looks off.",
  },
  {
    name: "Dr. Michael Chen",
    role: "Cardiologist",
    avatar: "https://api.dicebear.com/7.x/avataaars/svg?seed=Michael&backgroundColor=222",
    body:
      "The rule trace is the killer feature. I can defend every titration step in clinic — and reverse it in one click if the patient pushes back.",
  },
  {
    name: "Marcus Lin",
    role: "CMIO",
    avatar: "https://api.dicebear.com/7.x/avataaars/svg?seed=Marcus&backgroundColor=222",
    body:
      "Local‑first means I can pilot this on a single department's data before anything touches the EHR. Rare and refreshing.",
  },
]

export function Testimonials() {
  return (
    <section className="px-6 py-28">
      <div className="mx-auto max-w-7xl">
        <SectionHeader
          eyebrow="From the field"
          title={
            <>
              Patients and clinicians,{" "}
              <span className="text-cyan-300">on the same page.</span>
            </>
          }
        />

        <div className="mt-14 grid grid-cols-1 gap-5 md:grid-cols-3">
          {quotes.map((q, i) => (
            <motion.div
              key={q.name}
              initial={{ opacity: 0, y: 24 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-10%" }}
              transition={{ duration: 0.4, delay: i * 0.08 }}
            >
              <Card className="h-full border-white/10 bg-zinc-950/60 transition hover:border-cyan-400/30">
                <CardContent className="flex h-full flex-col gap-5 pt-6">
                  <p className="text-sm leading-relaxed text-zinc-200">
                    &ldquo;{q.body}&rdquo;
                  </p>
                  <div className="mt-auto flex items-center gap-3 border-t border-white/5 pt-4">
                    <Avatar className="size-10 border border-white/10">
                      <AvatarImage src={q.avatar} alt={q.name} />
                      <AvatarFallback className="bg-zinc-800 text-zinc-300">
                        {q.name[0]}
                      </AvatarFallback>
                    </Avatar>
                    <div>
                      <div className="text-sm font-medium text-white">{q.name}</div>
                      <div className="text-xs text-zinc-500">{q.role}</div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  )
}

/* --------------------------------- Contact -------------------------------- */

export function Contact() {
  const items = [
    { icon: Phone, label: "Phone", value: "+1 (555) 010‑0420" },
    { icon: Mail, label: "Email", value: "team@tide.health" },
    { icon: MapPin, label: "Office", value: "Boston, MA" },
    { icon: Clock, label: "Support", value: "24/7 for active pilots" },
  ]
  return (
    <section id="contact" className="border-y border-white/5 bg-zinc-950/40 px-6 py-28">
      <div className="mx-auto grid max-w-7xl grid-cols-1 gap-10 lg:grid-cols-2">
        <div>
          <Badge
            variant="outline"
            className="border-cyan-400/30 bg-cyan-400/5 text-cyan-300"
          >
            Get in touch
          </Badge>
          <h2 className="mt-5 text-3xl font-semibold tracking-tight text-white md:text-5xl">
            We&apos;re here when you&apos;re ready to pilot.
          </h2>
          <p className="mt-4 max-w-md text-zinc-400">
            Talk to us about deploying TIDE on your local environment, or join the open beta as a clinician.
          </p>
          <ul className="mt-10 space-y-4">
            {items.map((it) => (
              <li
                key={it.label}
                className="flex items-center gap-4 rounded-xl border border-white/10 bg-zinc-950/60 px-4 py-3"
              >
                <span className="inline-flex size-9 items-center justify-center rounded-lg border border-cyan-400/20 bg-cyan-400/5 text-cyan-300">
                  <it.icon className="size-4" />
                </span>
                <div className="flex flex-col">
                  <span className="text-[11px] uppercase tracking-widest text-zinc-500">
                    {it.label}
                  </span>
                  <span className="text-sm text-zinc-200">{it.value}</span>
                </div>
              </li>
            ))}
          </ul>
        </div>

        <Card className="self-start border-white/10 bg-zinc-950/60">
          <CardHeader>
            <CardTitle className="text-white">Request a demo</CardTitle>
            <CardDescription className="text-zinc-400">
              We&apos;ll get back inside one business day.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <form
              onSubmit={(e) => e.preventDefault()}
              className="grid grid-cols-1 gap-4 sm:grid-cols-2"
            >
              <Field label="Name" placeholder="Jane Doe" />
              <Field label="Role" placeholder="Cardiologist" />
              <Field label="Email" placeholder="jane@hospital.org" className="sm:col-span-2" />
              <Field label="Organization" placeholder="General Hospital" className="sm:col-span-2" />
              <div className="flex flex-col gap-1.5 sm:col-span-2">
                <label className="text-xs uppercase tracking-widest text-zinc-500">Message</label>
                <textarea
                  rows={4}
                  placeholder="Tell us about your environment and goals…"
                  className="rounded-lg border border-white/10 bg-zinc-900 px-3 py-2 text-sm text-white placeholder:text-zinc-500 focus:border-cyan-400/50 focus:outline-none focus:ring-2 focus:ring-cyan-400/20"
                />
              </div>
              <Button
                type="submit"
                className="gap-2 bg-cyan-400 text-black hover:bg-cyan-300 sm:col-span-2"
              >
                Send request <ArrowRight className="size-4" />
              </Button>
            </form>
          </CardContent>
        </Card>
      </div>
    </section>
  )
}

function Field({
  label,
  placeholder,
  className,
}: {
  label: string
  placeholder: string
  className?: string
}) {
  return (
    <div className={`flex flex-col gap-1.5 ${className ?? ""}`}>
      <label className="text-xs uppercase tracking-widest text-zinc-500">{label}</label>
      <input
        placeholder={placeholder}
        className="rounded-lg border border-white/10 bg-zinc-900 px-3 py-2 text-sm text-white placeholder:text-zinc-500 focus:border-cyan-400/50 focus:outline-none focus:ring-2 focus:ring-cyan-400/20"
      />
    </div>
  )
}

/* ----------------------------------- CTA ---------------------------------- */

export function CtaBanner() {
  return (
    <section className="px-6 pb-24 pt-20">
      <div className="relative mx-auto max-w-7xl overflow-hidden rounded-3xl border border-white/10 bg-zinc-950 p-10 md:p-16">
        <div className="pointer-events-none absolute -left-20 top-1/2 h-72 w-72 -translate-y-1/2 rounded-full bg-cyan-500/15 blur-[100px]" />
        <div className="pointer-events-none absolute -right-20 -top-20 h-72 w-72 rounded-full bg-cyan-400/10 blur-[100px]" />
        <div className="pointer-events-none absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-cyan-400/40 to-transparent" />
        <div className="relative max-w-2xl">
          <Badge
            variant="outline"
            className="border-cyan-400/30 bg-cyan-400/5 text-cyan-300"
          >
            <Sparkles className="size-3" /> Open beta
          </Badge>
          <h2 className="mt-5 text-4xl font-semibold tracking-tight text-white md:text-5xl">
            Run TIDE on your laptop
            <br />
            <span className="text-cyan-300">in under two minutes.</span>
          </h2>
          <p className="mt-4 max-w-lg text-zinc-400">
            No EHR plumbing, no MIMIC dataset, no black boxes. Pure local pipeline you can pilot today.
          </p>
          <div className="mt-8 flex flex-wrap gap-3">
            <Button size="lg" className="gap-2 bg-cyan-400 text-black hover:bg-cyan-300">
              Get started free <ArrowRight className="size-4" />
            </Button>
            <Button
              size="lg"
              variant="outline"
              className="gap-2 border-white/15 bg-white/5 text-white hover:bg-white/10 hover:text-white"
            >
              Read the docs <ChevronRight className="size-4" />
            </Button>
          </div>
        </div>
      </div>
    </section>
  )
}

/* ---------------------------------- Footer -------------------------------- */

export function Footer() {
  return (
    <footer className="border-t border-white/5 bg-black px-6 pb-10 pt-16">
      <div className="mx-auto grid max-w-7xl grid-cols-2 gap-10 md:grid-cols-5">
        <div className="col-span-2">
          <a href="#" className="flex items-center gap-2">
            <span className="inline-flex size-9 items-center justify-center rounded-lg border border-white/10 bg-white/5">
              <HeartPulse className="size-5 text-cyan-400" />
            </span>
            <div className="leading-none">
              <div className="text-base font-semibold tracking-tight text-white">TIDE</div>
              <div className="text-[10px] uppercase tracking-[0.18em] text-zinc-500">
                CHF Titration
              </div>
            </div>
          </a>
          <p className="mt-4 max-w-sm text-sm leading-relaxed text-zinc-500">
            Local‑first, guideline‑directed titration support for heart‑failure care. Built for the
            clinicians and patients managing it together.
          </p>
        </div>

        <FooterCol title="Product" links={["Platform", "Titration", "Open beta", "Changelog"]} />
        <FooterCol title="Resources" links={["Docs", "Research", "Status", "Contact"]} />
        <FooterCol title="Company" links={["About", "Privacy", "Security", "Terms"]} />
      </div>

      <div className="mx-auto mt-12 flex max-w-7xl flex-col items-start justify-between gap-3 border-t border-white/5 pt-6 text-xs text-zinc-600 md:flex-row md:items-center">
        <div>© {new Date().getFullYear()} TIDE Health. Local‑first, by design.</div>
        <div className="flex items-center gap-2 font-mono uppercase tracking-widest">
          <span className="size-1.5 animate-pulse rounded-full bg-cyan-400" />
          all systems · operational
        </div>
      </div>
    </footer>
  )
}

function FooterCol({ title, links }: { title: string; links: string[] }) {
  return (
    <div>
      <div className="mb-3 text-xs uppercase tracking-widest text-zinc-500">{title}</div>
      <ul className="space-y-2 text-sm">
        {links.map((l) => (
          <li key={l}>
            <a href="#" className="text-zinc-300 transition hover:text-cyan-300">
              {l}
            </a>
          </li>
        ))}
      </ul>
    </div>
  )
}
