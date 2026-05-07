import { Activity } from "lucide-react";
import { Link, useLocation, useNavigate } from "react-router-dom";

const Navbar = () => {
  const location = useLocation();
  const navigate = useNavigate();

  const scrollTo = (id: string) => (e: React.MouseEvent) => {
    e.preventDefault();
    const go = () => {
      if (id === "top") {
        window.scrollTo({ top: 0, behavior: "smooth" });
        return;
      }
      const el = document.getElementById(id);
      if (el) el.scrollIntoView({ behavior: "smooth", block: "start" });
    };
    if (location.pathname !== "/") {
      navigate("/");
      // wait for landing page to mount before scrolling
      setTimeout(go, 60);
    } else {
      go();
    }
  };

  return (
    <header className="fixed top-0 inset-x-0 z-50">
      <div className="mx-auto max-w-7xl px-6 mt-4">
        <nav className="glass-panel rounded-2xl flex items-center justify-between px-5 py-3">
          <a href="#" onClick={scrollTo("top")} className="flex items-center gap-2.5 group">
            <div className="relative w-9 h-9 rounded-xl bg-gradient-primary grid place-items-center shadow-glow-soft">
              <Activity className="w-5 h-5 text-primary-foreground" strokeWidth={2.5} />
            </div>
            <div className="leading-tight">
              <div className="font-display font-semibold tracking-tight text-foreground">TIDE-HF</div>
              <div className="text-[10px] font-mono uppercase tracking-[0.18em] text-muted-foreground">Guideline-Directed AI</div>
            </div>
          </a>
          <div className="hidden md:flex items-center gap-7 text-sm text-muted-foreground">
            <a href="#" onClick={scrollTo("about")} className="hover:text-foreground transition-colors">About</a>
            <a href="#" onClick={scrollTo("how")} className="hover:text-foreground transition-colors">How it works</a>
            <a href="#" onClick={scrollTo("strategies")} className="hover:text-foreground transition-colors">Strategies</a>
            <a href="#" onClick={scrollTo("chat")} className="hover:text-foreground transition-colors">Chat</a>
          </div>
          <div className="flex items-center gap-3">
            <a
              href="https://qas.ai"
              target="_blank"
              rel="noreferrer"
              className="hidden sm:flex items-center gap-2 rounded-xl border border-border bg-secondary/40 hover:bg-secondary px-2 py-1 transition-colors"
              title="Affiliated with QAS.AI"
            >
              <img
                src={`${import.meta.env.BASE_URL}qas-ai.png`}
                alt="QAS.AI"
                className="w-7 h-7 rounded-md object-contain"
              />
              <span className="text-[10px] font-mono uppercase tracking-[0.18em] text-muted-foreground leading-tight">
                Affiliated with<br />QAS.AI
              </span>
            </a>
            <Link
              to="/engine"
              className="rounded-xl bg-gradient-primary px-4 py-2 text-sm font-medium text-primary-foreground shadow-glow-soft hover:shadow-glow transition-shadow"
            >
              Open Engine →
            </Link>
          </div>
        </nav>
      </div>
    </header>
  );
};

export default Navbar;
