import { useEffect, useRef, useState } from "react";
import { Send, Sparkles, User, Bot, Stethoscope } from "lucide-react";

type Msg = { id: string; role: "user" | "assistant"; content: string };

const STARTERS = [
  "Newly diagnosed HFrEF, stable — what does TIDE recommend?",
  "Suspected hyperkalemia: T-peaked, wide QRS, no labs yet",
  "Confirmed K = 6.3 — what does the engine do?",
  "Compare strong_hf vs rapid_sequence strategies",
];

const sampleReply = (q: string): string => {
  if (/k\s*=\s*6|confirmed|6\.3/i.test(q)) {
    return `**Confirmed hyperkalemia (K = 6.3) → \`global_stop\`**\n\nLab gate triggered: K > 6.0 mEq/L.\n\n**Engine actions (per class):**\n\n- **RAAS** — \`hold\` · reason: \`global_stop\`\n- **MRA** — \`hold\` · reason: \`global_stop\`\n- **Beta-blocker** — \`hold\` · reason: \`global_stop\`\n- **SGLT2i** — \`hold\` · reason: \`global_stop\`\n- **Loop diuretic** — \`maintain\` (volume status driven)\n\n**Next:** escalate, repeat BMP, treat hyperK acutely. Resume titration only after K normalizes.\n\n*Preset:* \`Confirmed hyperkalemia (K=6.3)\``;
  }
  if (/suspected hyperkalemia|t-?peaked|wide qrs/i.test(q)) {
    return `**Suspected hyperkalemia — ECG signs, no labs yet**\n\nClassifier fires: \`hyperkalemia_detected\` (from T-peaked + wide QRS features).\n\n**Engine actions:**\n\n- **RAAS** — \`hold\` + \`order_labs\`\n- **MRA** — \`hold\` + \`order_labs\`\n- **Beta-blocker** — \`hold\` + \`order_labs\`\n- **SGLT2i** — continue\n- **Loop** — continue\n\nTitration is *gated* until BMP resolves the suspicion. If K confirmed > 6.0 → \`global_stop\`.\n\n*Preset:* \`Suspected hyperkalemia\``;
  }
  if (/strong_hf|rapid_sequence|strategy|compare/i.test(q)) {
    return `**\`strong_hf\` vs \`rapid_sequence\`**\n\nBoth start *all eligible classes at once* — they differ on dose-rung speed.\n\n- **\`strong_hf\`** — double-rung up-titration each week. Fastest path to target doses; most aggressive.\n- **\`rapid_sequence\`** — single-rung steps each week (Greene 2021). All four pillars on board quickly, but each titrated more cautiously.\n\nFor reference, the other two:\n\n- **\`traditional\`** — one new class per week (RAAS → BB → MRA → SGLT2i → loop), single-rung.\n- **\`sglt_mra_first\`** — Phase 1: SGLT2i + MRA. Phase 2: add ARNi + BB.\n\n*Pick the strategy at the top of the Streamlit UI before clicking Compute.*`;
  }
  return `**Newly diagnosed HFrEF, stable**\n\nClassifier: no AE flags. Engine: no contraindications, no awaiting labs.\n\n**Recommendation (\`strong_hf\` strategy):**\n\n1. **RAAS** — \`start_medication\` · ARNi (sacubitril/valsartan) preferred\n2. **Beta-blocker** — \`start_medication\` · carvedilol / bisoprolol / metoprolol succinate\n3. **MRA** — \`start_medication\` · spironolactone (eGFR ≥ 30, K ≤ 5.0)\n4. **SGLT2i** — \`start_medication\` · dapagliflozin / empagliflozin\n5. **Loop** — driven by volume status\n\nAll four pillars initiated this week. Up-titrate per strategy next visit.\n\n*Preset:* \`Newly diagnosed, stable\``;
};

const formatLine = (line: string) => {
  // simple bold + italic markdown
  const parts = line.split(/(\*\*[^*]+\*\*|\*[^*]+\*)/g);
  return parts.map((p, i) => {
    if (p.startsWith("**") && p.endsWith("**"))
      return <strong key={i} className="text-foreground">{p.slice(2, -2)}</strong>;
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
        "I'm **TIDE-HF**, your guideline-directed AI for heart failure. Ask me anything — GDMT titration, device eligibility, advanced therapy referral, or interpret a trajectory.",
    },
  ]);
  const [input, setInput] = useState("");
  const [thinking, setThinking] = useState(false);
  const scrollerRef = useRef<HTMLDivElement>(null);
  const firstRenderRef = useRef(true);

  useEffect(() => {
    if (firstRenderRef.current) {
      firstRenderRef.current = false;
      return;
    }
    const el = scrollerRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages, thinking]);

  const send = (text: string) => {
    const q = text.trim();
    if (!q) return;
    setMessages((m) => [...m, { id: crypto.randomUUID(), role: "user", content: q }]);
    setInput("");
    setThinking(true);
    setTimeout(() => {
      setMessages((m) => [
        ...m,
        { id: crypto.randomUUID(), role: "assistant", content: sampleReply(q) },
      ]);
      setThinking(false);
    }, 1100);
  };

  return (
    <section id="chat" className="relative py-28">
      <div className="mx-auto max-w-6xl px-6">
        <div className="text-center max-w-2xl mx-auto mb-12">
          <div className="text-xs font-mono uppercase tracking-[0.2em] text-primary mb-4">
            04 — TIDE AI
          </div>
          <h2 className="font-display text-4xl md:text-5xl font-semibold tracking-tight">
            Talk to the <span className="text-gradient">guideline-directed engine</span>.
          </h2>
          <p className="mt-4 text-muted-foreground">
            A clinical co-pilot — not a chatbot. Every response is grounded, cited, and traceable.
          </p>
        </div>

        <div className="glass-panel rounded-3xl overflow-hidden shadow-elegant glow-border">
          {/* Header */}
          <div className="flex items-center justify-between px-5 py-4 border-b border-border bg-secondary/40">
            <div className="flex items-center gap-3">
              <div className="relative">
                <div className="w-10 h-10 rounded-xl bg-gradient-primary grid place-items-center">
                  <Stethoscope className="w-5 h-5 text-primary-foreground" />
                </div>
                <span className="absolute -bottom-0.5 -right-0.5 w-3 h-3 rounded-full bg-emerald-400 ring-2 ring-background animate-pulse" />
              </div>
              <div>
                <div className="font-display font-semibold leading-tight">TIDE-HF Assistant</div>
                <div className="text-xs text-muted-foreground font-mono">online · guideline v2024.2</div>
              </div>
            </div>
            <div className="hidden sm:flex items-center gap-2 text-xs font-mono text-muted-foreground">
              <Sparkles className="w-3.5 h-3.5 text-primary" />
              evidence-grounded
            </div>
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
                    m.role === "user"
                      ? "bg-secondary border border-border"
                      : "bg-gradient-primary shadow-glow-soft"
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
              {STARTERS.map((s) => (
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
              placeholder="Ask about GDMT, titration, device eligibility…"
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

        <p className="mt-4 text-center text-xs text-muted-foreground font-mono">
          For clinical decision support only. Always confirm with your local protocol.
        </p>
      </div>
    </section>
  );
};

export default ChatBot;
