import { Activity } from "lucide-react";
import { Link } from "react-router-dom";

const Navbar = () => {
  return (
    <header className="fixed top-0 inset-x-0 z-50">
      <div className="mx-auto max-w-7xl px-6 mt-4">
        <nav className="glass-panel rounded-2xl flex items-center justify-between px-5 py-3">
          <a href="#top" className="flex items-center gap-2.5 group">
            <div className="relative w-9 h-9 rounded-xl bg-gradient-primary grid place-items-center shadow-glow-soft">
              <Activity className="w-5 h-5 text-primary-foreground" strokeWidth={2.5} />
            </div>
            <div className="leading-tight">
              <div className="font-display font-semibold tracking-tight text-foreground">TIDE-HF</div>
              <div className="text-[10px] font-mono uppercase tracking-[0.18em] text-muted-foreground">Guideline-Directed AI</div>
            </div>
          </a>
          <div className="hidden md:flex items-center gap-7 text-sm text-muted-foreground">
            <a href="#about" className="hover:text-foreground transition-colors">About</a>
            <a href="#how" className="hover:text-foreground transition-colors">How it works</a>
            <a href="#engine" className="hover:text-foreground transition-colors">Strategies</a>
            <a href="#chat" className="hover:text-foreground transition-colors">Chat</a>
          </div>
          <Link
            to="/engine"
            className="rounded-xl bg-gradient-primary px-4 py-2 text-sm font-medium text-primary-foreground shadow-glow-soft hover:shadow-glow transition-shadow"
          >
            Open Engine →
          </Link>
        </nav>
      </div>
    </header>
  );
};

export default Navbar;
