"use client"

import { useEffect, useRef, useState } from "react"
import { motion, AnimatePresence } from "framer-motion"
import {
  Activity,
  ArrowRight,
  Bot,
  ClipboardList,
  MessageCircle,
  Pill,
  Send,
  ShieldCheck,
  Sparkles,
  Stethoscope,
  User,
} from "lucide-react"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Avatar, AvatarFallback } from "@/components/ui/avatar"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"

type Role = "patient" | "doctor"
type Msg = { id: number; from: "ai" | "user"; text: string }

const greetings: Record<Role, string> = {
  patient:
    "Hi — I'm TIDE. I can explain your dose changes, what side‑effects to watch for, and when to call your team. What would you like to talk about?",
  doctor:
    "Welcome. TIDE can summarize a patient's titration trajectory, surface adverse‑effect risk, and explain why the engine held or advanced a dose. What do you need?",
}

const suggestions: Record<Role, { icon: typeof Pill; label: string }[]> = {
  patient: [
    { icon: Pill, label: "View my medications" },
    { icon: Activity, label: "Check my latest readings" },
    { icon: MessageCircle, label: "What is GDMT?" },
    { icon: ShieldCheck, label: "When should I call my doctor?" },
  ],
  doctor: [
    { icon: ClipboardList, label: "Summarize patient #1042" },
    { icon: Pill, label: "Show titration protocol" },
    { icon: Activity, label: "Why did TIDE hold the ARNI?" },
    { icon: ShieldCheck, label: "Lab gating thresholds" },
  ],
}

function aiReply(role: Role, text: string): string {
  const p = text.toLowerCase()
  if (role === "patient") {
    if (p.includes("dizzy")) return "Dizziness in the first 1–2 weeks of a dose change is common. Sit down, hydrate, stand up slowly. If it lasts more than a few minutes or you faint, message your care team today."
    if (p.includes("medication") || p.includes("medications")) return "You're on bisoprolol 5 mg AM, sacubitril/valsartan 49/51 mg BID, dapagliflozin 10 mg AM, and spironolactone 25 mg AM. Your next titration check is in 7 days."
    if (p.includes("reading") || p.includes("metric")) return "Today: HR 72 bpm, BP 118/76, weight 81.2 kg (steady). All in range — no action needed."
    if (p.includes("gdmt")) return "GDMT = Guideline‑Directed Medical Therapy. The four pillars proven to reduce HF hospitalization and mortality: ARNI/ACE/ARB, beta‑blocker, MRA, and SGLT2 inhibitor."
    if (p.includes("call") || p.includes("doctor")) return "Call your team if: weight up >2 kg in 3 days, new shortness of breath at rest, fainting, swelling worse than usual, or HR persistently below 50."
    return "I'm here for medications, symptoms, daily weight, sodium, or when to seek help. What's on your mind?"
  }
  if (p.includes("hold") || p.includes("arni")) return "Most common holds: K⁺ ≥ 5.0, eGFR drop > 30% from baseline, SBP < 90, or an active adverse‑effect flag. The trajectory view shows the gating reason next to each step."
  if (p.includes("summar") || p.includes("1042")) return "Patient #1042 · 4‑week trajectory: 2 successful uptitrations (bisoprolol 2.5→5, dapagliflozin start), 1 hold (ARNI — SBP 88, recheck day +7), 0 adverse events."
  if (p.includes("threshold") || p.includes("lab")) return "Default gates — K⁺: hold ≥ 5.0, stop ≥ 5.5; SCr: hold if Δ > 30% baseline; SBP: hold < 90 (ARNI/ACE/ARB), < 100 (BB up‑titrate); eGFR: SGLT2i continue ≥ 20."
  if (p.includes("protocol") || p.includes("titration")) return "TIDE follows a 2‑week step cadence per pillar: assess → check gates → propose step → log rule trace. Override is a single click and is logged for audit."
  return "I can pull patient summaries, explain a specific titration decision, or run what‑ifs against the rule engine. Which would help?"
}

export default function TideAIWidget() {
  const [open, setOpen] = useState(false)
  const [role, setRole] = useState<Role | null>(null)
  const [messages, setMessages] = useState<Msg[]>([])
  const [input, setInput] = useState("")
  const scrollRef = useRef<HTMLDivElement>(null)
  const idRef = useRef(0)

  useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight
  }, [messages, role])

  const pickRole = (r: Role) => {
    setRole(r)
    setMessages([{ id: ++idRef.current, from: "ai", text: greetings[r] }])
  }

  const send = (text: string) => {
    const v = text.trim()
    if (!v || !role) return
    setMessages((m) => [...m, { id: ++idRef.current, from: "user", text: v }])
    setInput("")
    setTimeout(() => {
      setMessages((m) => [...m, { id: ++idRef.current, from: "ai", text: aiReply(role, v) }])
    }, 500)
  }

  const reset = () => {
    setRole(null)
    setMessages([])
  }

  return (
    <>
      {/* Floating launcher */}
      <motion.button
        aria-label="Open TIDE AI assistant"
        onClick={() => setOpen(true)}
        className="group fixed bottom-6 right-6 z-40 inline-flex items-center gap-2.5 overflow-hidden rounded-full border border-cyan-400/30 bg-zinc-950/90 px-4 py-3 text-sm font-medium text-white shadow-2xl backdrop-blur-md transition hover:border-cyan-400/60"
        initial={{ y: 80, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ delay: 1.2, type: "spring", stiffness: 200, damping: 18 }}
        whileHover={{ scale: 1.04 }}
        whileTap={{ scale: 0.96 }}
      >
        <span className="absolute inset-0 -z-10 bg-cyan-400/0 transition group-hover:bg-cyan-400/10" />
        <span className="relative inline-flex size-7 items-center justify-center rounded-full bg-cyan-400/15 text-cyan-300">
          <Bot className="size-4" />
          <span className="absolute -inset-0.5 rounded-full ring-1 ring-cyan-400/30" />
        </span>
        <span>Ask TIDE AI</span>
        <span className="size-1.5 rounded-full bg-cyan-400 shadow-[0_0_8px_rgba(34,211,238,0.9)]" />
      </motion.button>

      <Dialog
        open={open}
        onOpenChange={(v) => {
          setOpen(v)
          if (!v) {
            // keep role so user doesn't have to re-pick on re-open
          }
        }}
      >
        <DialogContent className="border-white/10 bg-zinc-950 p-0 sm:max-w-md">
          <div className="relative overflow-hidden rounded-2xl">
            {/* glow */}
            <div className="pointer-events-none absolute -top-24 left-1/2 h-48 w-72 -translate-x-1/2 rounded-full bg-cyan-500/20 blur-3xl" />

            <DialogHeader className="space-y-2 border-b border-white/5 bg-gradient-to-b from-zinc-900/80 to-zinc-950 px-6 pb-4 pt-6">
              <DialogTitle className="flex items-center gap-2.5 text-base">
                <span className="relative inline-flex size-8 items-center justify-center rounded-full bg-cyan-400/15 text-cyan-300">
                  <motion.span
                    animate={{ scale: [1, 1.15, 1] }}
                    transition={{ duration: 1.6, repeat: Infinity, ease: "easeInOut" }}
                  >
                    <Bot className="size-4" />
                  </motion.span>
                  <span className="absolute inset-0 rounded-full ring-1 ring-cyan-400/30" />
                </span>
                <span>TIDE AI</span>
                {role && (
                  <Badge
                    variant="outline"
                    className="ml-1 border-cyan-400/30 bg-cyan-400/5 text-[10px] uppercase tracking-wider text-cyan-300"
                  >
                    {role === "patient" ? "Patient" : "Clinician"}
                  </Badge>
                )}
                {role && (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={reset}
                    className="ml-auto h-7 px-2 text-xs text-zinc-400 hover:text-white"
                  >
                    Switch
                  </Button>
                )}
              </DialogTitle>
              <DialogDescription className="text-zinc-500">
                {role === null
                  ? "Heart‑failure care assistant. Pick your role to start."
                  : role === "patient"
                    ? "Plain‑language guidance for your medications and symptoms."
                    : "Trajectory summaries, rule traces, and what‑ifs."}
              </DialogDescription>
            </DialogHeader>

            {role === null ? (
              <RolePicker onPick={pickRole} />
            ) : (
              <Chat
                role={role}
                messages={messages}
                input={input}
                setInput={setInput}
                onSend={send}
                scrollRef={scrollRef}
              />
            )}

            <div className="border-t border-white/5 px-6 py-2.5 text-center text-[10px] text-zinc-600">
              TIDE AI is a demo assistant. Not a substitute for professional medical advice.
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </>
  )
}

function RolePicker({ onPick }: { onPick: (r: Role) => void }) {
  return (
    <div className="px-6 py-6">
      <p className="mb-4 text-center text-sm text-zinc-500">
        Are you a patient managing heart failure, or a clinician titrating GDMT?
      </p>
      <div className="grid grid-cols-2 gap-3">
        <RoleCard
          icon={<User className="size-5" />}
          title="I'm a patient"
          desc="Help me understand my meds and symptoms."
          onClick={() => onPick("patient")}
        />
        <RoleCard
          icon={<Stethoscope className="size-5" />}
          title="I'm a clinician"
          desc="Show trajectories, risks, and rule traces."
          onClick={() => onPick("doctor")}
        />
      </div>
      <div className="mt-4 flex items-center gap-2 rounded-lg border border-cyan-400/15 bg-cyan-400/5 px-3 py-2 text-[11px] text-cyan-200/80">
        <Sparkles className="size-3.5 shrink-0 text-cyan-400" />
        TIDE replies are scoped to your role and never include another patient's data.
      </div>
    </div>
  )
}

function RoleCard({
  icon,
  title,
  desc,
  onClick,
}: {
  icon: React.ReactNode
  title: string
  desc: string
  onClick: () => void
}) {
  return (
    <motion.button
      whileHover={{ y: -2 }}
      whileTap={{ scale: 0.98 }}
      onClick={onClick}
      className="group relative overflow-hidden rounded-xl border border-white/10 bg-zinc-900/60 p-4 text-left transition hover:border-cyan-400/40 hover:bg-zinc-900"
    >
      <span className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-cyan-400/40 to-transparent opacity-0 transition group-hover:opacity-100" />
      <span className="mb-3 inline-flex size-9 items-center justify-center rounded-lg bg-cyan-400/10 text-cyan-300">
        {icon}
      </span>
      <div className="text-sm font-medium text-white">{title}</div>
      <div className="mt-1 text-xs leading-snug text-zinc-500">{desc}</div>
      <ArrowRight className="absolute bottom-3 right-3 size-3.5 text-zinc-600 transition group-hover:translate-x-0.5 group-hover:text-cyan-300" />
    </motion.button>
  )
}

function Chat({
  role,
  messages,
  input,
  setInput,
  onSend,
  scrollRef,
}: {
  role: Role
  messages: Msg[]
  input: string
  setInput: (v: string) => void
  onSend: (v: string) => void
  scrollRef: React.RefObject<HTMLDivElement | null>
}) {
  return (
    <>
      <div
        ref={scrollRef}
        className="flex max-h-[55vh] min-h-[300px] flex-col gap-3 overflow-y-auto bg-zinc-950 px-5 py-4"
      >
        {messages.map((m) => (
          <motion.div
            key={m.id}
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            className={`flex items-start gap-2 ${m.from === "user" ? "justify-end" : ""}`}
          >
            {m.from === "ai" && (
              <Avatar className="size-7 border border-cyan-400/20 bg-cyan-400/10">
                <AvatarFallback className="bg-transparent text-cyan-300">
                  <Bot className="size-3.5" />
                </AvatarFallback>
              </Avatar>
            )}
            <div
              className={
                m.from === "user"
                  ? "max-w-[80%] rounded-2xl rounded-br-sm bg-cyan-400 px-3.5 py-2 text-sm text-black shadow-md shadow-cyan-500/20"
                  : "max-w-[85%] rounded-2xl rounded-bl-sm border border-white/5 bg-zinc-900 px-3.5 py-2 text-sm text-zinc-200"
              }
            >
              {m.text}
            </div>
          </motion.div>
        ))}

        {messages.length <= 1 && (
          <div className="pt-2">
            <div className="mb-2 text-[10px] uppercase tracking-widest text-zinc-600">Try</div>
            <div className="grid grid-cols-1 gap-1.5">
              {suggestions[role].map((s) => (
                <button
                  key={s.label}
                  onClick={() => onSend(s.label)}
                  className="group inline-flex items-center gap-2 rounded-lg border border-white/5 bg-zinc-900/60 px-3 py-2 text-left text-xs text-zinc-300 transition hover:border-cyan-400/30 hover:bg-zinc-900 hover:text-white"
                >
                  <s.icon className="size-3.5 text-cyan-300" />
                  <span>{s.label}</span>
                  <ArrowRight className="ml-auto size-3 opacity-0 transition group-hover:translate-x-0.5 group-hover:opacity-100" />
                </button>
              ))}
            </div>
          </div>
        )}
      </div>

      <form
        onSubmit={(e) => {
          e.preventDefault()
          onSend(input)
        }}
        className="flex items-center gap-2 border-t border-white/5 bg-zinc-950 px-3 py-3"
      >
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={role === "patient" ? "Ask about your meds…" : "Ask about a patient or rule…"}
          className="flex-1 rounded-full border border-white/10 bg-zinc-900 px-4 py-2 text-sm text-white placeholder:text-zinc-500 focus:border-cyan-400/50 focus:outline-none focus:ring-2 focus:ring-cyan-400/20"
        />
        <button
          type="submit"
          aria-label="Send"
          disabled={!input.trim()}
          className="inline-flex size-9 items-center justify-center rounded-full bg-cyan-400 text-black shadow-md transition hover:bg-cyan-300 disabled:opacity-40"
        >
          <Send className="size-4" />
        </button>
      </form>
    </>
  )
}
