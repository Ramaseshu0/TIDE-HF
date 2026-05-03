import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { Send, Sparkles, User, Bot, Stethoscope, UserRound, FlaskConical } from "lucide-react";
import {
  apiHealth,
  askChat,
  recallPatient,
  type ChatAudience,
  type EvaluateResponse,
} from "@/lib/tide-api";
import type { Patient, Strategy } from "@/lib/tide-engine";

type Msg = {
  id: string;
  role: "user" | "assistant";
  content: string;
  source?: "python" | "browser";
};

const STARTERS_PATIENT = [
  "What do my heart-failure medications do?",
  "Why was my dose changed this visit?",
  "What symptoms should I watch for this week?",
  "When should I call the doctor?",
];

const STARTERS_CLINICIAN = [
  "Why did the engine hold the ARNi for this patient?",
  "AHA 2022 threshold for MRA dose reduction in hyperkalemia?",
  "STRONG-HF up-titration schedule for this regimen?",
  "Compare strong_hf vs rapid_sequence for this patient.",
];

const formatLine = (line: string) => {
  const parts = line.split(/(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)/g);
  return parts.map((p, i) => {
    if (p.startsWith("**") && p.endsWith("**"))
      return <strong key={i} className="text-foreground">{p.slice(2, -2)}</strong>;
    if (p.startsWith("`") && p.endsWith("`"))
      return <code key={i} className="font-mono text-primary text-[0.85em] px-1 rounded bg-secondary/60">{p.slice(1, -1)}</code>;
    if (p.startsWith("*") && p.endsWith("*"))
      return <em key={i} className="text-primary/90">{p.slice(1, -1)}</em>;
    return <span key={i}>{p}</span>;
  });
};

const Markdown = ({ text }: { text: string }) => {
  const lines = text.split("\n");
  return (
    <div className="space-y-2 text-sm leading-relaxed">
      {lines.map((l, i) => {
        if (!l.trim()) return <div key={i} className="h-1" />;
        if (/^\d+\.\s/.test(l))
          return <div key={i} className="pl-2">{formatLine(l)}</div>;
        if (l.startsWith("- "))
          return (
            <div key={i} className="flex gap-2 pl-2">
              <span className="text-primary mt-1.5">•</span>
              <span>{formatLine(l.slice(2))}</span>
            </div>
          );
        return <p key={i}>{formatLine(l)}</p>;
      })}
    </div>
  );
};

const ChatBot = () => {
  const [messages, setMessages] = useState<Msg[]>([
    {
      id: "intro",
      role: "assistant",
      content:
        "I'm **TIDE-HF**, your guideline-directed AI for heart failure. Open the **Engine** tab, pick a patient, click **Compute**, and I'll answer questions grounded in *that patient's* labs, vitals, meds, and the engine's decisions — sourced from AHA 2022, STRONG-HF, and drug monographs.",
    },
  ]);
  const [input, setInput] = useState("");
  const [thinking, setThinking] = useState(false);
  const [audience, setAudience] = useState<ChatAudience>("patient");
  const [backend, setBackend] = useState<"python" | "browser" | null>(null);
  const [patientCtx, setPatientCtx] = useState<{
    patient: Patient;
    result: EvaluateResponse;
    strategy: Strategy;
  } | null>(null);

  const scrollerRef = useRef<HTMLDivElement>(null);
  const firstRenderRef = useRef(true);

  // Refresh "last computed patient" each time the chat section comes into view.
  useEffect(() => {
    const reload = () => setPatientCtx(recallPatient());
    reload();
    window.addEventListener("focus", reload);
    window.addEventListener("storage", reload);
    return () => {
      window.removeEventListener("focus", reload);
      window.removeEventListener("storage", reload);
    };
  }, []);

  useEffect(() => {
    apiHealth().then((h) => setBackend(h.source)).catch(() => setBackend("browser"));
  }, []);

  useEffect(() => {
    if (firstRenderRef.current) {
      firstRenderRef.current = false;
      return;
    }
    const el = scrollerRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages, thinking]);

  const send = async (text: string) => {
    const q = text.trim();
    if (!q || thinking) return;
    setMessages((m) => [...m, { id: crypto.randomUUID(), role: "user", content: q }]);
    setInput("");
    setThinking(true);
    try {
      const res = await askChat({
        question: q,
        audience,
        patient: patientCtx?.patient ?? null,
        strategy: patientCtx?.strategy ?? "strong_hf",
        result: patientCtx?.result,
      });
      setMessages((m) => [...m, {
        id: crypto.randomUUID(), role: "assistant", content: res.answer, source: res.source,
      }]);
    } catch (e) {
      setMessages((m) => [...m, {
        id: crypto.randomUUID(),
        role: "assistant",
        content: `⚠ ${(e as Error).message}`,
      }]);
    } finally {
      setThinking(false);
    }
  };

  const starters = audience === "patient" ? STARTERS_PATIENT : STARTERS_CLINICIAN;

  return (
    <section id="chat" className="relative py-28">
      <div className="mx-auto max-w-6xl px-6">
        <div className="text-center max-w-2xl mx-auto mb-12">
          <div className="text-xs font-mono uppercase tracking-[0.2em] text-primary mb-4">04 — TIDE AI</div>
          <h2 className="font-display text-4xl md:text-5xl font-semibold tracking-tight">
            Talk to the <span className="text-gradient">guideline-directed engine</span>.
          </h2>
          <p className="mt-4 text-muted-foreground">
            A clinical co-pilot — not a chatbot. Every response is grounded in the patient's engine state and your local guideline PDFs.
          </p>
        </div>

        <div className="glass-panel rounded-3xl overflow-hidden shadow-elegant glow-border">
          {/* Header */}
          <div className="flex items-center justify-between px-5 py-4 border-b border-border bg-secondary/40 gap-3">
            <div className="flex items-center gap-3">
              <div className="relative">
                <div className="w-10 h-10 rounded-xl bg-gradient-primary grid place-items-center">
                  <Stethoscope className="w-5 h-5 text-primary-foreground" />
                </div>
                <span className={`absolute -bottom-0.5 -right-0.5 w-3 h-3 rounded-full ring-2 ring-background ${backend === "python" ? "bg-emerald-400 animate-pulse" : "bg-amber-400"}`} />
              </div>
              <div>
                <div className="font-display font-semibold leading-tight">TIDE-HF Assistant</div>
                <div className="text-xs text-muted-foreground font-mono">
                  {backend === null ? "checking…" : backend === "python" ? "RAG · Mistral · ChromaDB" : "offline mode (engine summary only)"}
                </div>
              </div>
            </div>

            {/* Audience toggle */}
            <div className="flex items-center gap-1 rounded-xl border border-border bg-secondary/60 p-1">
              <button
                onClick={() => setAudience("patient")}
                className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-mono transition-colors ${
                  audience === "patient"
                    ? "bg-primary text-primary-foreground shadow-glow-soft"
                    : "text-muted-foreground hover:text-foreground"
                }`}
              >
                <UserRound className="w-3.5 h-3.5" />
                Patient
              </button>
              <button
                onClick={() => setAudience("clinician")}
                className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-mono transition-colors ${
                  audience === "clinician"
                    ? "bg-primary text-primary-foreground shadow-glow-soft"
                    : "text-muted-foreground hover:text-foreground"
                }`}
              >
                <FlaskConical className="w-3.5 h-3.5" />
                Clinician
              </button>
            </div>
          </div>

          {/* Patient context strip */}
          <div className="px-5 py-2.5 border-b border-border bg-background/40 flex flex-wrap items-center gap-2 text-xs">
            {patientCtx ? (
              <>
                <span className="text-muted-foreground font-mono">grounded on</span>
                <span className="font-mono px-2 py-0.5 rounded bg-primary/10 text-primary border border-primary/30">
                  {patientCtx.patient.name}
                </span>
                <span className="text-muted-foreground">{patientCtx.patient.age}{patientCtx.patient.gender}</span>
                {patientCtx.result.global_stop && (
                  <span className="font-mono px-2 py-0.5 rounded border border-rose-400/40 bg-rose-500/10 text-rose-300">
                    global_stop
                  </span>
                )}
                {patientCtx.result.order_labs && !patientCtx.result.global_stop && (
                  <span className="font-mono px-2 py-0.5 rounded border border-amber-400/40 bg-amber-500/10 text-amber-300">
                    order_labs
                  </span>
                )}
                <span className="font-mono text-muted-foreground/80">strategy={patientCtx.strategy}</span>
                <Link to="/engine" className="ml-auto text-primary/80 hover:text-primary font-mono">edit ↗</Link>
              </>
            ) : (
              <>
                <span className="text-muted-foreground">No patient loaded.</span>
                <Link to="/engine" className="ml-auto font-mono text-primary/80 hover:text-primary">
                  Open Engine to ground answers on a real patient ↗
                </Link>
              </>
            )}
          </div>

          {/* Messages */}
          <div ref={scrollerRef} className="h-[520px] overflow-y-auto px-5 py-6 space-y-5 bg-background/50">
            {messages.map((m) => (
              <div
                key={m.id}
                className={`flex gap-3 ${m.role === "user" ? "flex-row-reverse" : ""} animate-fade-up`}
              >
                <div
                  className={`shrink-0 w-9 h-9 rounded-xl grid place-items-center ${
                    m.role === "user" ? "bg-secondary border border-border" : "bg-gradient-primary shadow-glow-soft"
                  }`}
                >
                  {m.role === "user" ? (
                    <User className="w-4 h-4 text-foreground" />
                  ) : (
                    <Bot className="w-4 h-4 text-primary-foreground" />
                  )}
                </div>
                <div
                  className={`max-w-[80%] rounded-2xl px-4 py-3 ${
                    m.role === "user"
                      ? "bg-primary/15 border border-primary/30 text-foreground"
                      : "bg-card border border-border"
                  }`}
                >
                  <Markdown text={m.content} />
                  {m.role === "assistant" && m.source && (
                    <div className="mt-2 text-[10px] font-mono uppercase tracking-widest text-muted-foreground/70">
                      {m.source === "python" ? "via local RAG" : "offline summary"}
                    </div>
                  )}
                </div>
              </div>
            ))}

            {thinking && (
              <div className="flex gap-3 animate-fade-up">
                <div className="shrink-0 w-9 h-9 rounded-xl bg-gradient-primary grid place-items-center shadow-glow-soft">
                  <Bot className="w-4 h-4 text-primary-foreground" />
                </div>
                <div className="bg-card border border-border rounded-2xl px-4 py-3.5 flex items-center gap-1.5">
                  <span className="typing-dot w-1.5 h-1.5 rounded-full bg-primary" />
                  <span className="typing-dot w-1.5 h-1.5 rounded-full bg-primary" />
                  <span className="typing-dot w-1.5 h-1.5 rounded-full bg-primary" />
                </div>
              </div>
            )}
          </div>

          {/* Starters */}
          {messages.length <= 1 && (
            <div className="px-5 pb-3 flex flex-wrap gap-2">
              {starters.map((s) => (
                <button
                  key={s}
                  onClick={() => send(s)}
                  className="text-xs px-3 py-2 rounded-lg border border-border bg-secondary/40 hover:bg-secondary hover:border-primary/40 text-muted-foreground hover:text-foreground transition-all"
                >
                  {s}
                </button>
              ))}
            </div>
          )}

          {/* Input */}
          <form
            onSubmit={(e) => {
              e.preventDefault();
              send(input);
            }}
            className="border-t border-border bg-secondary/40 p-3 flex gap-2"
          >
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder={audience === "patient" ? "Ask about your meds, symptoms, what to watch for…" : "Ask about a titration decision, threshold, or rationale…"}
              className="flex-1 bg-input/60 border border-border rounded-xl px-4 py-3 text-sm placeholder:text-muted-foreground focus:outline-none focus:border-primary/60 focus:ring-2 focus:ring-primary/20 transition-all"
            />
            <button
              type="submit"
              disabled={!input.trim() || thinking}
              className="rounded-xl bg-gradient-primary px-5 text-primary-foreground shadow-glow-soft hover:shadow-glow disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center gap-2"
            >
              <Send className="w-4 h-4" />
            </button>
          </form>
        </div>

        <p className="mt-4 text-center text-xs text-muted-foreground font-mono flex flex-wrap items-center justify-center gap-2">
          <Sparkles className="w-3.5 h-3.5 text-primary" />
          For clinical decision support only. Always confirm with your local protocol.
        </p>
      </div>
    </section>
  );
};

export default ChatBot;
