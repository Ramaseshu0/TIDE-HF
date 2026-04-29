import { Activity } from "lucide-react";

const Footer = () => (
  <footer className="border-t border-border py-12 mt-10">
    <div className="mx-auto max-w-7xl px-6 flex flex-col md:flex-row items-center justify-between gap-6">
      <div className="flex items-center gap-2.5">
        <div className="w-8 h-8 rounded-lg bg-gradient-primary grid place-items-center">
          <Activity className="w-4 h-4 text-primary-foreground" strokeWidth={2.5} />
        </div>
        <div>
          <div className="font-display font-semibold text-sm">TIDE-HF</div>
          <div className="text-[10px] font-mono uppercase tracking-widest text-muted-foreground">
            Trajectory · Integrated · Decision · Engine
          </div>
        </div>
      </div>
      <div className="text-xs text-muted-foreground font-mono">
        © {new Date().getFullYear()} TIDE-HF · Guideline-Directed AI for Heart Failure
      </div>
    </div>
  </footer>
);

export default Footer;
