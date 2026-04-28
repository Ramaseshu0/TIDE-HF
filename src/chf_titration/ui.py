"""TIDE-HF — Trajectory Integrated Decision Engine for Heart Failure
TIDE.AI — clinical AI assistant for GDMT titration
"""
from __future__ import annotations

import io
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from chf_titration.constants import CLASSES, RHYTHMS, LABEL_COLS
from chf_titration.synthesize import PATIENTS, synthesize_week
from chf_titration.engine import TitrationEngine, derive_contraindications, CMAP
from chf_titration.strategy import apply_strategy, STRATEGIES, TARGETS, LADDERS
from chf_titration.classifier import load_bundle, predict_flags

BUNDLE_PATH = Path("models/chf_classifier_lgbm.pkl")

_REP_DRUG = {
    "ACEi": "Lisinopril", "ARB": "Losartan",
    "ARNi": "Sacubitril-Valsartan (Entresto)", "beta_blocker": "Carvedilol",
    "MRA": "Spironolactone", "SGLT2i": "Empagliflozin (Jardiance)", "loop": "Furosemide",
}

# ══════════════════════════════════════════════════════════════════════════════
# ECG SVG  —  cyan, animated (medically informative, not decorative)
# ══════════════════════════════════════════════════════════════════════════════
_ECG_SVG = """
<div class="tide-ecg">
<svg width="360" height="52" viewBox="0 0 360 52" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
  <path stroke="#22d3ee" stroke-width="2.5" fill="none" stroke-linecap="round" stroke-linejoin="round"
        stroke-dasharray="700" stroke-dashoffset="700"
        style="animation:ecgDraw 3s linear infinite; filter:drop-shadow(0 0 6px rgba(34,211,238,0.55));"
        d="M0,26 L20,26 L28,26 L32,22 L36,26 L40,6 L46,46 L52,6 L58,26 L68,26 L72,24 L76,22 L80,26
           L100,26 L108,26 L112,22 L116,26 L120,6 L126,46 L132,6 L138,26 L148,26 L152,24 L156,22 L160,26
           L180,26 L188,26 L192,22 L196,26 L200,6 L206,46 L212,6 L218,26 L228,26 L232,24 L236,22 L240,26
           L260,26 L268,26 L272,22 L276,26 L280,6 L286,46 L292,6 L298,26 L308,26 L312,24 L316,22 L320,26
           L340,26 L350,26 L360,26"/>
</svg>
</div>
"""

# ══════════════════════════════════════════════════════════════════════════════
# Hero monitor card  —  faux clinical readout
# ══════════════════════════════════════════════════════════════════════════════
_MONITOR_SVG = """
<div class="tide-monitor" aria-hidden="true">
  <div class="tm-bar">
    <div class="tm-dots"><span></span><span></span><span></span></div>
    <div class="tm-id"><span class="tm-pulse"></span>tide://patient/01</div>
  </div>
  <div class="tm-stats">
    <div class="tm-stat"><div class="tm-l">Age</div><div class="tm-v">67<span>yrs</span></div></div>
    <div class="tm-stat"><div class="tm-l">HR</div><div class="tm-v">72<span>bpm</span></div></div>
    <div class="tm-stat"><div class="tm-l">SBP</div><div class="tm-v">118<span>mmHg</span></div></div>
    <div class="tm-stat"><div class="tm-l">K&#8314;</div><div class="tm-v">4.2<span>mEq/L</span></div></div>
  </div>
  <div class="tm-ecg">
    <svg width="100%" height="68" viewBox="0 0 360 68" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <pattern id="tmgrid-sm" width="10" height="10" patternUnits="userSpaceOnUse">
          <path d="M 10 0 L 0 0 0 10" fill="none" stroke="rgba(34,211,238,0.06)" stroke-width="0.5"/>
        </pattern>
        <pattern id="tmgrid-lg" width="40" height="40" patternUnits="userSpaceOnUse">
          <path d="M 40 0 L 0 0 0 40" fill="none" stroke="rgba(34,211,238,0.12)" stroke-width="1"/>
        </pattern>
      </defs>
      <rect width="360" height="68" fill="url(#tmgrid-sm)"/>
      <rect width="360" height="68" fill="url(#tmgrid-lg)"/>
      <path d="M0,34 L34,34 L40,34 L44,28 L48,34 L52,8 L58,60 L64,8 L70,34 L80,34 L86,32 L92,28 L96,34
               L132,34 L138,34 L144,28 L148,34 L152,8 L158,60 L164,8 L170,34 L180,34 L186,32 L192,28 L196,34
               L232,34 L238,34 L244,28 L248,34 L252,8 L258,60 L264,8 L270,34 L280,34 L286,32 L292,28 L296,34
               L332,34 L360,34"
            stroke="#22d3ee" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"
            stroke-dasharray="800" stroke-dashoffset="800"
            style="animation:ecgDraw 3.4s linear infinite; filter:drop-shadow(0 0 5px rgba(34,211,238,0.7));"/>
    </svg>
    <div class="tm-ecg-label">ECG · LEAD II</div>
    <div class="tm-ecg-meta">sinus rhythm</div>
  </div>
  <div class="tm-foot">
    <div class="tm-foot-row"><span>Bisoprolol — week 4</span><span class="tm-mono">2.5 → 5 mg</span></div>
    <div class="tm-progress"><div class="tm-progress-fill"></div></div>
  </div>
</div>
"""

# ══════════════════════════════════════════════════════════════════════════════
# CSS  —  pure black + cyan accent (matches the Next.js site)
# ══════════════════════════════════════════════════════════════════════════════
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Keyframes ────────────────────────────────────────────────────────────── */
@keyframes fadeInUp { from{opacity:0;transform:translateY(18px);} to{opacity:1;transform:translateY(0);} }
@keyframes springBounce {
  0%{opacity:0;transform:scale(.92) translateY(14px);}
  60%{opacity:1;transform:scale(1.02) translateY(-2px);}
  100%{opacity:1;transform:scale(1) translateY(0);}
}
@keyframes ecgDraw {
  0%{stroke-dashoffset:700;opacity:.2;}
  30%{opacity:1;}
  100%{stroke-dashoffset:0;opacity:.85;}
}
@keyframes ringExpand {
  0%,100%{transform:scale(1);opacity:.55;}
  50%{transform:scale(1.5);opacity:0;}
}
@keyframes pulseDot {
  0%,100%{opacity:1;box-shadow:0 0 0 0 rgba(34,211,238,.6);}
  50%{opacity:.55;box-shadow:0 0 0 8px rgba(34,211,238,0);}
}
@keyframes shimmer {
  0%{background-position:-300% center;}
  100%{background-position:300% center;}
}
@keyframes popupIn {
  0%{opacity:0;transform:scale(.85) translateY(28px);}
  100%{opacity:1;transform:scale(1) translateY(0);}
}
@keyframes popupOut {
  0%{opacity:1;transform:scale(1) translateY(0);}
  100%{opacity:0;transform:scale(.92) translateY(20px);}
}
@keyframes fabIn {
  from{transform:translateY(40px) scale(.6);opacity:0;}
  to{transform:translateY(0) scale(1);opacity:1;}
}

/* ── Base ─────────────────────────────────────────────────────────────────── */
html,body { font-family:'Inter','-apple-system',sans-serif !important; font-size:15px; }
[data-testid="stAppViewContainer"] > .main {
  background:#000 !important;
  background-image:
    radial-gradient(circle at 1px 1px, rgba(255,255,255,0.05) 1px, transparent 0),
    radial-gradient(ellipse at 50% -10%, rgba(34,211,238,.12) 0%, transparent 55%),
    radial-gradient(ellipse at 100% 90%, rgba(34,211,238,.05) 0%, transparent 60%);
  background-size:24px 24px, 100% 100%, 100% 100%;
}
[data-testid="stSidebar"] {
  background:#000 !important;
  border-right:1px solid rgba(255,255,255,.06) !important;
}
[data-testid="stHeader"] { background:transparent !important; }
#MainMenu, footer, [data-testid="stDeployButton"] { display:none !important; }
section[data-testid="stSidebar"] > div { padding-top:14px !important; }

/* ── Hero ─────────────────────────────────────────────────────────────────── */
.tide-hero {
  position:relative; overflow:hidden;
  background:#000;
  border:1px solid rgba(255,255,255,.08);
  border-radius:24px;
  padding:38px 44px 30px;
  margin-bottom:24px;
  animation:fadeInUp .55s cubic-bezier(.34,1.56,.64,1);
}
.tide-hero::before {
  content:''; position:absolute; inset:0; pointer-events:none;
  background:
    radial-gradient(ellipse at 100% 0%, rgba(34,211,238,.16) 0%, transparent 55%),
    radial-gradient(ellipse at 0% 100%, rgba(34,211,238,.07) 0%, transparent 55%);
}
.tide-hero::after {
  content:''; position:absolute; top:0; left:0; right:0; height:1px;
  background:linear-gradient(90deg,transparent,rgba(34,211,238,.55),transparent);
}
.tide-hero-grid {
  display:grid; grid-template-columns:1fr 360px; gap:38px; align-items:start; position:relative; z-index:1;
}
@media (max-width:1100px){ .tide-hero-grid{ grid-template-columns:1fr; } .tide-monitor{ max-width:520px; margin-top:6px; } }

.tide-logo-row { display:flex; align-items:center; gap:14px; margin-bottom:6px; flex-wrap:wrap; }
.tide-mark {
  width:42px; height:42px; border-radius:10px;
  background:rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.08);
  display:inline-flex; align-items:center; justify-content:center;
}
.tide-mark svg { width:22px; height:22px; }
.tide-name {
  font-size:26px; font-weight:800; letter-spacing:-.5px; line-height:1; color:#fff;
}
.tide-name-sub {
  font-size:10px; font-weight:700; letter-spacing:3px; text-transform:uppercase; color:rgba(255,255,255,.45);
  margin-top:4px;
}
.tide-badge {
  display:inline-flex; align-items:center; gap:6px;
  background:rgba(34,211,238,.06); border:1px solid rgba(34,211,238,.32);
  border-radius:999px; padding:5px 12px;
  font-size:10px; font-weight:700; letter-spacing:1.6px; text-transform:uppercase; color:#67e8f9;
}
.tide-badge .dot { width:6px; height:6px; border-radius:50%; background:#22d3ee; box-shadow:0 0 8px rgba(34,211,238,.9); }

.tide-headline {
  font-size:46px; font-weight:700; letter-spacing:-1.5px; line-height:1.05;
  color:#fff; margin:18px 0 12px;
}
.tide-headline em {
  font-style:normal;
  background:linear-gradient(135deg,#fff 0%, #67e8f9 50%, #22d3ee 100%);
  background-size:200% auto; -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
  animation:shimmer 6s linear infinite;
}
.tide-tagline { font-size:14px; color:rgba(255,255,255,.55); font-weight:400; line-height:1.65; max-width:540px; }

.tide-hero-stats { display:flex; gap:14px; margin-top:22px; flex-wrap:wrap; }
.tide-stat {
  background:rgba(255,255,255,.02); border:1px solid rgba(255,255,255,.07);
  border-radius:14px; padding:12px 18px; min-width:108px;
  animation:springBounce .55s cubic-bezier(.34,1.56,.64,1) backwards;
  transition:all .25s ease;
}
.tide-stat:nth-child(1){animation-delay:.06s}
.tide-stat:nth-child(2){animation-delay:.14s}
.tide-stat:nth-child(3){animation-delay:.22s}
.tide-stat:nth-child(4){animation-delay:.30s}
.tide-stat:hover { border-color:rgba(34,211,238,.32); background:rgba(34,211,238,.04); transform:translateY(-3px); }
.tide-stat-val { font-size:22px; font-weight:800; color:#fff; letter-spacing:-.5px; }
.tide-stat-lbl { font-size:9px; color:rgba(255,255,255,.4); letter-spacing:1.6px; text-transform:uppercase; margin-top:2px; }
.tide-ecg { margin-top:18px; }

/* Monitor card on the right side of the hero */
.tide-monitor {
  border:1px solid rgba(255,255,255,.08);
  background:rgba(10,10,10,.85); backdrop-filter:blur(8px);
  border-radius:18px; overflow:hidden;
  box-shadow:0 30px 60px rgba(0,0,0,.6), 0 0 0 1px rgba(34,211,238,.04);
  animation:fadeInUp .65s cubic-bezier(.34,1.56,.64,1) .1s backwards;
}
.tm-bar {
  display:flex; align-items:center; justify-content:space-between;
  padding:9px 14px; background:rgba(20,20,20,.85);
  border-bottom:1px solid rgba(255,255,255,.06);
}
.tm-dots { display:flex; gap:5px; }
.tm-dots span { width:9px; height:9px; border-radius:50%; background:rgba(255,255,255,.18); }
.tm-id { font-family:'JetBrains Mono',monospace; font-size:9px; letter-spacing:2px;
  text-transform:uppercase; color:rgba(255,255,255,.4); display:flex; align-items:center; gap:8px; }
.tm-pulse { width:6px; height:6px; border-radius:50%; background:#22d3ee; animation:pulseDot 1.4s infinite; }
.tm-stats { display:grid; grid-template-columns:1fr 1fr 1fr 1fr; }
.tm-stat { padding:12px 14px; background:#0a0a0a; border-right:1px solid rgba(255,255,255,.05); }
.tm-stat:last-child { border-right:none; }
.tm-l { font-size:9px; letter-spacing:2px; text-transform:uppercase; color:rgba(255,255,255,.4); }
.tm-v { font-family:'JetBrains Mono',monospace; font-size:22px; color:#fff; margin-top:3px; font-weight:600; }
.tm-v span { font-size:9px; color:rgba(255,255,255,.4); margin-left:5px; letter-spacing:1px; }
.tm-ecg {
  position:relative; height:68px; background:#0a0a0a; border-top:1px solid rgba(255,255,255,.05);
}
.tm-ecg svg { display:block; }
.tm-ecg-label, .tm-ecg-meta {
  position:absolute; top:8px; font-family:'JetBrains Mono',monospace;
  font-size:9px; letter-spacing:1.6px; text-transform:uppercase;
}
.tm-ecg-label { left:10px; color:#67e8f9; }
.tm-ecg-meta  { right:10px; color:rgba(255,255,255,.4); }
.tm-foot { padding:14px 16px; background:#0a0a0a; border-top:1px solid rgba(255,255,255,.05); }
.tm-foot-row {
  display:flex; justify-content:space-between; align-items:center;
  font-size:12px; color:rgba(255,255,255,.7); margin-bottom:8px;
}
.tm-mono { font-family:'JetBrains Mono',monospace; color:#22d3ee; }
.tm-progress { height:4px; background:rgba(255,255,255,.06); border-radius:999px; overflow:hidden; }
.tm-progress-fill {
  width:66%; height:100%;
  background:linear-gradient(90deg,#22d3ee,#67e8f9); border-radius:999px;
}
.tm-foot-note { display:flex; gap:6px; align-items:center; margin-top:9px; font-size:11px; color:rgba(255,255,255,.45); }
.tm-tick { color:#22d3ee; }

/* ── Glass card ───────────────────────────────────────────────────────────── */
.glass {
  background:rgba(10,10,10,.7);
  border:1px solid rgba(255,255,255,.08);
  border-radius:18px; padding:22px 26px; margin:8px 0;
  animation:springBounce .5s cubic-bezier(.34,1.56,.64,1) backwards;
  transition:all .25s ease;
  position:relative; overflow:hidden;
}
.glass:hover {
  border-color:rgba(34,211,238,.28);
  box-shadow:0 12px 36px rgba(0,0,0,.5);
}

/* ── Section title ────────────────────────────────────────────────────────── */
.sec-title {
  font-size:10px; font-weight:800; letter-spacing:2.5px; text-transform:uppercase;
  color:#67e8f9; margin-bottom:18px; padding-bottom:9px;
  border-bottom:1px solid rgba(255,255,255,.06);
  display:flex; align-items:center; gap:8px;
}
.sec-title::before {
  content:''; width:6px; height:6px; border-radius:50%; background:#22d3ee;
  box-shadow:0 0 8px rgba(34,211,238,.7);
}

/* ── Vital label ──────────────────────────────────────────────────────────── */
.vl {
  font-size:10px; font-weight:700; letter-spacing:1.6px; text-transform:uppercase;
  color:rgba(255,255,255,.55); margin-bottom:4px;
  display:flex; align-items:center; gap:8px;
}
.vi { font-size:14px; opacity:.85; }

/* ── Lab pills ────────────────────────────────────────────────────────────── */
.lab-row { display:flex; flex-wrap:wrap; gap:8px; margin:6px 0; }
.lp {
  background:rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.08);
  border-radius:999px; padding:5px 14px; font-size:12px; color:rgba(255,255,255,.85);
  font-weight:500; transition:all .2s ease;
  font-family:'Inter',sans-serif;
}
.lp:hover { background:rgba(34,211,238,.06); border-color:rgba(34,211,238,.32); }
.lp.ok     { background:rgba(34,197,94,.07);  border-color:rgba(34,197,94,.28);  color:#86efac; }
.lp.warn   { background:rgba(234,179,8,.08);  border-color:rgba(234,179,8,.30);  color:#fde047; }
.lp.danger { background:rgba(239,68,68,.10);  border-color:rgba(239,68,68,.55);  color:#fca5a5; }

/* ── Action badges ────────────────────────────────────────────────────────── */
.ab { display:inline-block; padding:5px 12px; border-radius:6px; font-size:10px; font-weight:800; letter-spacing:1px; }
.ab-start    { background:rgba(34,197,94,.12);  border:1px solid rgba(34,197,94,.55); color:#86efac; }
.ab-stop     { background:rgba(239,68,68,.12);  border:1px solid rgba(239,68,68,.55); color:#fca5a5; }
.ab-up       { background:rgba(34,211,238,.10); border:1px solid rgba(34,211,238,.45); color:#67e8f9; }
.ab-hold     { background:rgba(234,179,8,.12);  border:1px solid rgba(234,179,8,.55); color:#fde047; }
.ab-maintain { background:rgba(148,163,184,.08); border:1px solid rgba(148,163,184,.4); color:#cbd5e1; }
.ab-down     { background:rgba(249,115,22,.12); border:1px solid rgba(249,115,22,.55); color:#fdba74; }

/* ── Rec card ─────────────────────────────────────────────────────────────── */
.rec-card {
  background:rgba(10,10,10,.7); border:1px solid rgba(255,255,255,.08);
  border-radius:14px; padding:16px 20px; margin:8px 0;
  display:flex; align-items:flex-start; gap:18px;
  transition:all .25s ease;
  animation:springBounce .45s cubic-bezier(.34,1.56,.64,1) backwards;
}
.rec-card:hover {
  border-color:rgba(34,211,238,.34); transform:translateX(4px);
  background:rgba(15,15,15,.85);
}
.rec-drug  { font-size:14px; font-weight:700; color:#fff; }
.rec-brand { font-size:11px; color:rgba(255,255,255,.35); margin-top:2px; }
.rec-dose  { font-size:12px; color:#67e8f9; margin-top:5px; font-family:'JetBrains Mono',monospace; }
.rec-reason { font-size:12px; color:rgba(255,255,255,.5); margin-top:5px; flex:1; line-height:1.6; }

/* ── Risk gauge ───────────────────────────────────────────────────────────── */
.gauge-bg   { background:rgba(255,255,255,.05); border-radius:999px; height:8px; overflow:hidden; margin:6px 0; }
.gauge-fill { height:100%; border-radius:999px;
  background:linear-gradient(90deg,#22c55e 0%,#84cc16 30%,#eab308 55%,#f97316 75%,#ef4444 100%);
  transition:width 1.2s cubic-bezier(.34,1.56,.64,1); }
.pbar { height:6px; border-radius:999px; margin-top:3px; transition:width .9s cubic-bezier(.34,1.56,.64,1); }

/* ══════════════════════════════════════════════════════════════════════════
   FLOATING AI AGENT
   ══════════════════════════════════════════════════════════════════════════ */
#tide-fab-wrapper {
  position:fixed; bottom:24px; right:24px; z-index:9990;
  display:flex; flex-direction:column; align-items:flex-end; gap:8px;
}
.tide-fab-btn {
  display:inline-flex; align-items:center; gap:10px;
  padding:11px 18px 11px 12px;
  background:rgba(10,10,10,.92);
  border:1px solid rgba(34,211,238,.32);
  border-radius:999px; cursor:pointer;
  color:#fff; font-size:13px; font-weight:600; letter-spacing:.2px;
  box-shadow:0 16px 40px rgba(0,0,0,.6), 0 0 0 1px rgba(34,211,238,.04);
  animation:fabIn .55s cubic-bezier(.34,1.56,.64,1);
  transition:all .25s ease;
  backdrop-filter:blur(10px);
  min-height:44px;
}
.tide-fab-btn:hover {
  border-color:rgba(34,211,238,.7);
  transform:translateY(-2px);
  box-shadow:0 20px 50px rgba(34,211,238,.18);
}
.tide-fab-icon {
  position:relative;
  width:28px; height:28px; border-radius:50%;
  background:rgba(34,211,238,.14);
  display:inline-flex; align-items:center; justify-content:center;
  color:#22d3ee;
}
.tide-fab-icon::before {
  content:''; position:absolute; inset:-2px; border-radius:50%;
  border:1px solid rgba(34,211,238,.4);
  animation:ringExpand 2.4s ease-out infinite;
}
.tide-fab-icon svg { width:14px; height:14px; }
.tide-fab-pulse {
  width:6px; height:6px; border-radius:50%; background:#22d3ee;
  box-shadow:0 0 8px rgba(34,211,238,.9);
  animation:pulseDot 1.6s infinite;
}

/* ══════════════════════════════════════════════════════════════════════════
   AI POPUP MODAL
   ══════════════════════════════════════════════════════════════════════════ */
#tide-popup-overlay {
  display:none; position:fixed; inset:0; z-index:99999;
  background:rgba(0,0,0,.78); backdrop-filter:blur(10px);
  align-items:center; justify-content:center; padding:20px;
}
#tide-popup {
  background:#0a0a0a;
  border:1px solid rgba(255,255,255,.1);
  border-radius:24px;
  padding:34px 30px 26px; max-width:440px; width:100%;
  position:relative; overflow:hidden;
  box-shadow:0 40px 80px rgba(0,0,0,.7), 0 0 0 1px rgba(34,211,238,.06);
  animation:popupIn .45s cubic-bezier(.34,1.56,.64,1) forwards;
}
#tide-popup::before {
  content:''; position:absolute; top:-80px; left:50%; transform:translateX(-50%);
  width:280px; height:160px; border-radius:50%;
  background:rgba(34,211,238,.18); filter:blur(60px); pointer-events:none;
}
#tide-popup::after {
  content:''; position:absolute; top:0; left:0; right:0; height:1px;
  background:linear-gradient(90deg,transparent,rgba(34,211,238,.7),transparent);
}
.popup-close {
  position:absolute; top:14px; right:14px;
  background:rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.1);
  color:rgba(255,255,255,.55); font-size:14px;
  width:32px; height:32px; border-radius:50%; cursor:pointer;
  display:flex; align-items:center; justify-content:center;
  transition:all .22s ease; z-index:10;
}
.popup-close:hover {
  background:rgba(34,211,238,.12); border-color:rgba(34,211,238,.5);
  color:#67e8f9; transform:rotate(90deg);
}
.popup-head {
  display:flex; align-items:center; gap:12px; margin-bottom:6px;
  position:relative; z-index:2;
}
.popup-head-icon {
  width:38px; height:38px; border-radius:50%;
  background:rgba(34,211,238,.14); border:1px solid rgba(34,211,238,.32);
  display:flex; align-items:center; justify-content:center; color:#22d3ee;
}
.popup-head-icon svg { width:18px; height:18px; }
.popup-title {
  font-size:18px; font-weight:700; color:#fff; letter-spacing:-.3px; line-height:1;
}
.popup-sub {
  font-size:11px; color:rgba(255,255,255,.45); margin-top:3px; letter-spacing:.5px;
}
.popup-divider {
  border:none; height:1px;
  background:linear-gradient(90deg,transparent,rgba(255,255,255,.1),transparent);
  margin:20px 0 18px;
}
.popup-question {
  font-size:13px; color:rgba(255,255,255,.65); margin-bottom:14px; text-align:center;
  letter-spacing:.2px;
}

/* Popup role cards */
.popup-role-grid { display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-bottom:18px; }
.popup-role-card {
  position:relative; overflow:hidden;
  background:rgba(15,15,15,.85); border:1px solid rgba(255,255,255,.08);
  border-radius:14px; padding:18px 14px 16px; cursor:pointer;
  transition:all .3s cubic-bezier(.34,1.56,.64,1);
  text-align:left; min-height:44px;
  animation:springBounce .5s cubic-bezier(.34,1.56,.64,1) backwards;
}
.popup-role-card:nth-child(1){animation-delay:.08s}
.popup-role-card:nth-child(2){animation-delay:.18s}
.popup-role-card:hover {
  background:rgba(20,20,20,.95); border-color:rgba(34,211,238,.5);
  transform:translateY(-2px);
  box-shadow:0 8px 24px rgba(0,0,0,.5), 0 0 0 1px rgba(34,211,238,.18);
}
.popup-role-card::before {
  content:''; position:absolute; top:0; left:0; right:0; height:1px;
  background:linear-gradient(90deg,transparent,rgba(34,211,238,.45),transparent);
  opacity:0; transition:opacity .25s ease;
}
.popup-role-card:hover::before { opacity:1; }
.prc-icon {
  width:34px; height:34px; border-radius:9px;
  background:rgba(34,211,238,.1); color:#22d3ee;
  display:flex; align-items:center; justify-content:center;
  margin-bottom:10px;
}
.prc-icon svg { width:18px; height:18px; }
.prc-name { font-size:13px; font-weight:700; color:#fff; margin-bottom:3px; }
.prc-desc { font-size:11px; color:rgba(255,255,255,.45); line-height:1.5; }

.popup-loading {
  padding:40px 20px;
  display:flex; flex-direction:column; align-items:center; gap:14px;
}
.popup-loading-bot {
  width:46px; height:46px; border-radius:50%;
  background:rgba(34,211,238,.12); border:1px solid rgba(34,211,238,.32);
  display:flex; align-items:center; justify-content:center; color:#22d3ee;
  animation:pulseDot 1.2s infinite;
}
.popup-loading-text  { font-size:13px; color:rgba(255,255,255,.65); font-weight:500; }
.popup-foot {
  font-size:9px; color:rgba(255,255,255,.3); letter-spacing:1.8px; text-transform:uppercase;
  text-align:center; margin-top:6px;
}

/* ── Role gate (in-page) ──────────────────────────────────────────────────── */
.role-gate {
  background:#0a0a0a;
  border:1px solid rgba(255,255,255,.08); border-radius:24px;
  padding:44px 36px; text-align:center;
  position:relative; overflow:hidden;
  animation:springBounce .55s cubic-bezier(.34,1.56,.64,1);
}
.role-gate::before {
  content:''; position:absolute; top:-100px; left:50%; transform:translateX(-50%);
  width:300px; height:200px; border-radius:50%;
  background:rgba(34,211,238,.14); filter:blur(70px);
}
.role-gate::after {
  content:''; position:absolute; top:0; left:0; right:0; height:1px;
  background:linear-gradient(90deg,transparent,rgba(34,211,238,.6),transparent);
}
.role-gate h2 { font-size:28px; font-weight:700; color:#fff; margin-bottom:8px; letter-spacing:-.6px; position:relative; z-index:1; }
.role-gate p  { font-size:14px; color:rgba(255,255,255,.5); margin-bottom:0; position:relative; z-index:1; }

/* ── Chat banner ──────────────────────────────────────────────────────────── */
.chat-banner {
  background:#0a0a0a;
  border-radius:18px 18px 0 0; padding:22px 28px;
  border:1px solid rgba(255,255,255,.08); border-bottom:none;
  display:flex; align-items:center; gap:14px;
  position:relative; overflow:hidden;
}
.chat-banner::after {
  content:''; position:absolute; top:0; left:0; right:0; height:1px;
  background:linear-gradient(90deg,transparent,rgba(34,211,238,.5),transparent);
}
.cb-icon {
  width:38px; height:38px; border-radius:50%;
  background:rgba(34,211,238,.14); border:1px solid rgba(34,211,238,.32);
  display:inline-flex; align-items:center; justify-content:center; color:#22d3ee;
}
.cb-icon svg { width:18px; height:18px; }
.cb-title { font-size:18px; font-weight:700; color:#fff; letter-spacing:-.3px; }
.cb-sub   { font-size:11px; color:rgba(255,255,255,.45); letter-spacing:.5px; margin-top:3px; }

/* ── Sidebar ──────────────────────────────────────────────────────────────── */
.sb-brand  { text-align:left; padding:14px 6px 10px; display:flex; gap:10px; align-items:center; }
.sb-icon   { width:32px; height:32px; border-radius:8px;
  background:rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.08);
  display:inline-flex; align-items:center; justify-content:center; color:#22d3ee; }
.sb-icon svg { width:16px; height:16px; }
.sb-name   { font-size:16px; font-weight:700; color:#fff; letter-spacing:-.3px; line-height:1; }
.sb-ver    { font-size:9px; color:rgba(255,255,255,.4); letter-spacing:2.2px; text-transform:uppercase; margin-top:3px; }
.sb-patient {
  background:rgba(15,15,15,.85); border:1px solid rgba(255,255,255,.07);
  border-radius:14px; padding:13px 14px; margin:10px 0;
}
.sb-pname { font-size:13px; font-weight:700; color:#fff; }
.sb-pmeta { font-size:11px; color:rgba(255,255,255,.5); margin-top:4px; }

/* ── Buttons ──────────────────────────────────────────────────────────────── */
div[data-testid="stButton"] > button {
  border-radius:10px !important; font-size:13px !important; font-weight:500 !important;
  transition:all .22s ease !important;
  border:1px solid rgba(255,255,255,.1) !important;
  background:rgba(255,255,255,.03) !important; color:#fff !important;
  cursor:pointer !important; min-height:44px !important;
}
div[data-testid="stButton"] > button:hover {
  background:rgba(34,211,238,.08) !important; border-color:rgba(34,211,238,.45) !important;
  color:#67e8f9 !important;
  transform:translateY(-1px) !important;
  box-shadow:0 4px 14px rgba(0,0,0,.45) !important;
}
div[data-testid="stButton"] > button[kind="primary"] {
  background:#22d3ee !important;
  color:#000 !important; border:1px solid #22d3ee !important; font-weight:700 !important;
  font-size:14px !important; letter-spacing:.2px !important;
  box-shadow:0 6px 20px rgba(34,211,238,.32) !important;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
  background:#67e8f9 !important; border-color:#67e8f9 !important; color:#000 !important;
  transform:translateY(-2px) !important;
  box-shadow:0 10px 26px rgba(34,211,238,.45) !important;
}

/* ── Inputs ───────────────────────────────────────────────────────────────── */
div[data-testid="stNumberInput"] > div,
div[data-testid="stTextInput"]   > div {
  border-color:rgba(255,255,255,.1) !important;
  background:rgba(10,10,10,.85) !important; border-radius:10px !important;
  transition:all .2s ease !important;
}
div[data-testid="stNumberInput"]:focus-within > div,
div[data-testid="stTextInput"]:focus-within   > div {
  border-color:#22d3ee !important; box-shadow:0 0 0 3px rgba(34,211,238,.16) !important;
}
div[data-testid="stSelectbox"] > div {
  border-color:rgba(255,255,255,.1) !important; border-radius:10px !important;
  background:rgba(10,10,10,.85) !important;
}
[data-testid="stExpander"] {
  border:1px solid rgba(255,255,255,.07) !important;
  border-radius:14px !important; background:rgba(10,10,10,.55) !important;
  transition:border-color .2s ease !important;
}
[data-testid="stExpander"]:hover { border-color:rgba(34,211,238,.28) !important; }
[data-testid="stDataFrame"] {
  border:1px solid rgba(255,255,255,.07) !important; border-radius:12px !important; overflow:hidden !important;
}
[data-testid="stAlert"]    { border-radius:12px !important; }
[data-testid="stCheckbox"] label { color:#fff !important; font-size:13px !important; }
hr { border-color:rgba(255,255,255,.06) !important; }

/* Streamlit chat */
[data-testid="stChatMessage"] {
  background:rgba(10,10,10,.7) !important;
  border:1px solid rgba(255,255,255,.06) !important;
  border-radius:14px !important;
}
[data-testid="stChatMessageContent"] { color:#e5e5e5 !important; }
[data-testid="stChatInput"] {
  background:rgba(10,10,10,.85) !important;
  border:1px solid rgba(255,255,255,.1) !important; border-radius:14px !important;
}

/* Code blocks */
code, pre code {
  background:rgba(34,211,238,.06) !important;
  color:#67e8f9 !important;
  border:1px solid rgba(34,211,238,.18) !important;
  border-radius:6px; padding:2px 6px;
  font-family:'JetBrains Mono',monospace;
}

/* ── Accessibility ────────────────────────────────────────────────────────── */
:focus-visible { outline:2px solid rgba(34,211,238,.85) !important; outline-offset:2px !important; }
.popup-role-card:focus { outline:2px solid rgba(34,211,238,.85); outline-offset:2px; }
.tide-fab-btn:focus    { outline:2px solid rgba(34,211,238,.85); outline-offset:2px; }

/* ── Reduced motion ───────────────────────────────────────────────────────── */
@media (prefers-reduced-motion:reduce) {
  *,*::before,*::after {
    animation-duration:.01ms !important;
    animation-iteration-count:1 !important;
    transition-duration:.01ms !important;
  }
}

/* ── Responsive ───────────────────────────────────────────────────────────── */
@media (max-width:768px) {
  .tide-hero { padding:28px 20px 22px; }
  .tide-name  { font-size:22px; }
  .tide-headline { font-size:34px; }
  .popup-role-grid { grid-template-columns:1fr; }
  #tide-popup { padding:28px 22px 22px; }
  .tide-hero-stats { gap:8px; }
  .tide-stat { min-width:90px; padding:10px 14px; }
}
</style>

<script>
/* TIDE.AI Popup Controller */
function toggleTidePopup() {
  var ov = document.getElementById('tide-popup-overlay');
  var pp = document.getElementById('tide-popup');
  if (!ov) return;
  if (ov.style.display === 'flex') {
    closeTidePopup();
  } else {
    ov.style.display = 'flex';
    if (pp) {
      pp.style.animation = 'none';
      void pp.offsetWidth;
      pp.style.animation = 'popupIn .45s cubic-bezier(.34,1.56,.64,1) forwards';
    }
  }
}
function closeTidePopup() {
  var ov = document.getElementById('tide-popup-overlay');
  var pp = document.getElementById('tide-popup');
  if (!pp || !ov) return;
  pp.style.animation = 'popupOut .25s cubic-bezier(.4,0,.2,1) forwards';
  setTimeout(function(){ ov.style.display = 'none'; }, 240);
}
function selectRole(role) {
  var pp = document.getElementById('tide-popup');
  if (!pp) return;
  pp.innerHTML =
    '<div class="popup-loading">' +
    '<div class="popup-loading-bot">' +
      '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="22" height="22">' +
        '<path d="M12 8V4H8"/><rect width="16" height="12" x="4" y="8" rx="2"/><path d="M2 14h2"/><path d="M20 14h2"/><path d="M15 13v2"/><path d="M9 13v2"/>' +
      '</svg>' +
    '</div>' +
    '<div class="popup-loading-text">Opening ' + role + ' mode…</div>' +
    '</div>';
  setTimeout(function(){
    var p = new URLSearchParams(window.location.search);
    p.set('role', role); p.set('chat', 'open');
    window.location.href = window.location.pathname + '?' + p.toString();
  }, 600);
}
function handlePopupKey(e, fn) {
  if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fn(); }
}
</script>
"""

# ══════════════════════════════════════════════════════════════════════════════
# POPUP HTML — FAB + modal
# ══════════════════════════════════════════════════════════════════════════════
_POPUP_HTML = """
<div id="tide-fab-wrapper">
  <div class="tide-fab-btn"
       onclick="toggleTidePopup()"
       onkeydown="handlePopupKey(event, toggleTidePopup)"
       role="button" aria-label="Open TIDE.AI assistant" tabindex="0">
    <span class="tide-fab-icon">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M12 8V4H8"/><rect width="16" height="12" x="4" y="8" rx="2"/><path d="M2 14h2"/><path d="M20 14h2"/><path d="M15 13v2"/><path d="M9 13v2"/>
      </svg>
    </span>
    <span>Ask TIDE.AI</span>
    <span class="tide-fab-pulse"></span>
  </div>
</div>

<div id="tide-popup-overlay" style="display:none;"
     onclick="if(event.target===this)closeTidePopup()">
  <div id="tide-popup">
    <button class="popup-close" onclick="closeTidePopup()" aria-label="Close assistant">✕</button>

    <div class="popup-head">
      <span class="popup-head-icon">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M12 8V4H8"/><rect width="16" height="12" x="4" y="8" rx="2"/><path d="M2 14h2"/><path d="M20 14h2"/><path d="M15 13v2"/><path d="M9 13v2"/>
        </svg>
      </span>
      <div>
        <div class="popup-title">TIDE.AI</div>
        <div class="popup-sub">Clinical assistant — LightGBM · Llama-3.3 · TIDE-HF</div>
      </div>
    </div>

    <hr class="popup-divider"/>

    <div class="popup-question">Pick your role to continue.</div>

    <div class="popup-role-grid">
      <div class="popup-role-card"
           onclick="selectRole('Doctor')"
           onkeydown="handlePopupKey(event, function(){selectRole('Doctor')})"
           role="button" aria-label="I am a Doctor" tabindex="0">
        <span class="prc-icon">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M11 2v2"/><path d="M5 2v2"/><path d="M5 3H4a2 2 0 0 0-2 2v4a6 6 0 0 0 12 0V5a2 2 0 0 0-2-2h-1"/>
            <path d="M8 15a6 6 0 0 0 12 0v-3"/><circle cx="20" cy="10" r="2"/>
          </svg>
        </span>
        <div class="prc-name">I'm a Doctor</div>
        <div class="prc-desc">Clinical AI for GDMT decisions and safety monitoring.</div>
      </div>
      <div class="popup-role-card"
           onclick="selectRole('Patient')"
           onkeydown="handlePopupKey(event, function(){selectRole('Patient')})"
           role="button" aria-label="I am a Patient" tabindex="0">
        <span class="prc-icon">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>
          </svg>
        </span>
        <div class="prc-name">I'm a Patient</div>
        <div class="prc-desc">Plain-language guidance about your medications and symptoms.</div>
      </div>
    </div>

    <div class="popup-foot">Personalised, role-scoped care</div>
  </div>
</div>
"""


# ══════════════════════════════════════════════════════════════════════════════
# DATA HELPERS  — unchanged logic
# ══════════════════════════════════════════════════════════════════════════════

def _tps_to_stats(tps):
    def vals(k): return [t.get(k) for t in tps if t.get(k) is not None]
    hr, sbp, spo2, wt = vals("HR"), vals("SBP"), vals("SpO2"), vals("weight_kg")
    e_qrs = [t["ecg"].get("qrs_ms") for t in tps if t.get("ecg") and t["ecg"].get("qrs_ms") is not None]
    e_qtc = [t["ecg"].get("qtc_ms") for t in tps if t.get("ecg") and t["ecg"].get("qtc_ms") is not None]
    rhys  = [t.get("ecg", {}).get("rhythm", "NSR") for t in tps]
    dom   = max(set(rhys), key=rhys.count) if rhys else "NSR"
    return dict(
        HR_mean=round(float(np.mean(hr)), 1) if hr else 72,
        HR_min=round(float(min(hr)), 1) if hr else 60,
        SBP_mean=round(float(np.mean(sbp)), 1) if sbp else 125,
        SBP_min=round(float(min(sbp)), 1) if sbp else 110,
        SpO2_min=round(float(min(spo2)), 1) if spo2 else 96,
        weight_delta=round(float(wt[-1] - wt[0]), 2) if len(wt) >= 2 else 0.0,
        QRS_max=round(float(max(e_qrs)), 0) if e_qrs else 95,
        QTc_max=round(float(max(e_qtc)), 0) if e_qtc else 420,
        T_peaked=any(t.get("ecg", {}).get("t_peaked", False) for t in tps),
        rhythm=dom,
    )


def _stats_to_tps(s):
    fake = {"age": 65, "gender": "M", "meds": {}, "vitals": dict(s)}
    return synthesize_week(fake, seed=0)


def _egfr_stage(e: float) -> str:
    if e >= 90: return "G1 – normal"
    if e >= 60: return "G2 – mildly reduced"
    if e >= 45: return "G3a – mild-moderate CKD"
    if e >= 30: return "G3b – moderate-severe CKD"
    if e >= 15: return "G4 – severe CKD"
    return "G5 – kidney failure"


def _vl(icon: str, label: str):
    st.markdown(
        f'<div class="vl"><span class="vi">{icon}</span>{label}</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# RENDER HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _action_badge(action: str) -> tuple[str, str]:
    return {
        "start_medication": ("ab-start",    "▲ START"),
        "stop_medication":  ("ab-stop",     "✕ STOP"),
        "uptitrate":        ("ab-up",       "↑ UPTITRATE"),
        "hold_titration":   ("ab-hold",     "⏸ HOLD"),
        "maintain_dose":    ("ab-maintain", "— MAINTAIN"),
        "downtitrate":      ("ab-down",     "↓ REDUCE"),
    }.get(action, ("ab-maintain", action))


def _render_rec_cards(changes):
    html = ""
    for cls in CLASSES:
        c   = changes[cls]
        cur = c["current"] or 0
        nd  = c["new_dose"] or 0
        ac  = c["concrete_action"]
        bc, bt = _action_badge(ac)
        if ac == "start_medication":   dose_txt = f"Begin at {nd} mg"
        elif ac == "stop_medication":  dose_txt = f"{cur} mg → Discontinue"
        elif ac == "uptitrate":        dose_txt = f"{cur} mg → {nd} mg"
        elif ac == "downtitrate":      dose_txt = f"{cur} mg → {nd} mg"
        elif ac == "hold_titration":   dose_txt = f"{cur} mg (awaiting labs)"
        else:                          dose_txt = f"{cur} mg (maintain)" if cur else "Not yet started"
        html += f"""
<div class="rec-card">
  <div style="min-width:160px;">
    <div class="rec-drug">{cls}</div>
    <div class="rec-brand">{_REP_DRUG.get(cls,'')}</div>
  </div>
  <div style="flex:1;">
    <span class="ab {bc}">{bt}</span>
    <div class="rec-dose">{dose_txt}</div>
    <div class="rec-reason">{c['reason']}</div>
  </div>
</div>"""
    st.markdown(html, unsafe_allow_html=True)


def _render_flags(flags, probs):
    html = ""
    for flag in LABEL_COLS:
        pct    = round(probs[flag] * 100, 1)
        active = flags[flag]
        clr    = "#ef4444" if active else "#22d3ee"
        fw     = "700" if active else "400"
        dot    = "●" if active else "○"
        fc     = "#fca5a5" if active else "rgba(255,255,255,.55)"
        html += f"""
<div style="margin:8px 0;">
  <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
    <span style="font-size:12px;color:{fc};font-weight:{fw};">{dot} {flag}</span>
    <span style="font-size:12px;color:{clr};font-weight:600;">{pct}%</span>
  </div>
  <div class="gauge-bg"><div class="pbar" style="width:{pct}%;background:{clr};"></div></div>
</div>"""
    st.markdown(html, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# AI PIPELINE  — unchanged logic
# ══════════════════════════════════════════════════════════════════════════════

def _build_patient_context(preset: str, bundle: dict, labs_flag: bool) -> str:
    p  = PATIENTS[preset]
    bl = p["baseline"]
    tps = synthesize_week(p)
    pf  = {**p, "targets": TARGETS}
    flags, probs = predict_flags(pf, tps, bundle)
    labs   = p.get("labs") if labs_flag else None
    result = TitrationEngine().evaluate(pf, tps, flags, labs=labs, awaiting_labs=set())
    changes = apply_strategy(result, pf, strategy="strong_hf")
    contras = derive_contraindications(p)

    hrs  = [t.get("HR") or 0 for t in tps]
    sbps = [t.get("SBP") or 0 for t in tps]
    spo2 = [t.get("SpO2") or 0 for t in tps]
    wts  = [t.get("weight_kg") or 0 for t in tps]

    active_flags   = [k for k, v in flags.items() if v]
    active_contras = [k for k, v in contras.items() if v]
    meds_on = [(cls, p["meds"][cls]["dose"]) for cls in CLASSES if p["meds"][cls]["on"]]
    SEVERE = {"severe_hypotension_detected", "severe_bradycardia_detected", "any_emergency_flag"}
    weighted = sum((3 if k in SEVERE else 1) * v for k, v in probs.items())
    maxw = sum(3 if k in SEVERE else 1 for k in probs)
    risk_pct = round(weighted / maxw * 100, 1)

    lines = [
        "═" * 60,
        f"PATIENT PROFILE: {preset}",
        "═" * 60,
        f"Sex / clinical note: {'Female' if p['gender']=='F' else 'Male'} | {p.get('note','None')}",
        "",
        "── LABS ────────────────────────────────────────────────────",
        f"Baseline  K={bl['K']} | Na={bl['Na']} | Cr={bl['Cr']:.2f} | eGFR={bl['eGFR']:.0f} ({_egfr_stage(bl['eGFR'])})",
    ]
    if labs:
        cr_d = (labs["Cr"] - bl["Cr"]) / bl["Cr"] * 100 if bl.get("Cr") else 0
        k_flag = " ← GLOBAL STOP" if labs["K"] > 6.0 else (" ← HOLD RAAS/MRA" if labs["K"] > 5.5 else "")
        cr_flag = f" ← AKI (Δ={cr_d:+.0f}%)" if cr_d > 30 else f" (Δ={cr_d:+.0f}%)"
        lines.append(
            f"This cycle  K={labs['K']}{k_flag} | Na={labs['Na']} | "
            f"Cr={labs['Cr']:.2f}{cr_flag} | eGFR={labs['eGFR']:.0f} ({_egfr_stage(labs['eGFR'])})"
        )
    else:
        lines.append("This cycle: labs not available")

    lines += ["", "── MEDICATIONS ─────────────────────────────────────────────"]
    if meds_on:
        for cls, dose in meds_on:
            tgt = TARGETS.get(cls) or 1
            lines.append(f"  {cls:<14} {_REP_DRUG.get(cls,''):<30} {dose}mg ({round(dose/tgt*100)}% of target)")
    else:
        lines.append("  No GDMT started.")

    lines += [
        "", "── VITALS (7-day, 14 timepoints) ───────────────────────────",
        f"  HR:    mean {round(float(np.mean(hrs)),1)}, min {round(float(min(hrs)),1)}, max {round(float(max(hrs)),1)} bpm",
        f"  SBP:   mean {round(float(np.mean(sbps)),1)}, min {round(float(min(sbps)),1)} mmHg",
        f"  SpO2:  mean {round(float(np.mean(spo2)),1)}%, min {round(float(min(spo2)),1)}%",
        f"  Weight: {round(float(wts[0]),1)} → {round(float(wts[-1]),1)} kg (Δ {round(float(wts[-1]-wts[0]),1):+.1f} kg)",
        "", "── AI FLAGS ────────────────────────────────────────────────",
        f"  Risk score: {risk_pct}%",
        f"  Active ({len(active_flags)}/11): {', '.join(active_flags) if active_flags else 'NONE — stable'}",
    ]
    for flag in LABEL_COLS:
        marker = "  ▶ ACTIVE" if flags[flag] else "    clear "
        lines.append(f"    {marker}  {flag:<40} p={probs[flag]:.3f}")

    lines += [
        "", "── ENGINE DECISIONS ────────────────────────────────────────",
        f"  Global stop: {'YES — HOLD ALL' if result['global_stop'] else 'no'}",
        f"  Order labs:  {'YES — '+', '.join(sorted(result['labs_requested'])) if result['order_labs'] else 'no'}",
        f"  Preferred RAAS: {result['preferred_raas'] or 'none'}",
        "", "  Per-drug recommendations (strong_hf):",
    ]
    for cls in CLASSES:
        c = changes[cls]; cur = c["current"] or 0; nd = c["new_dose"] or 0
        a = c["concrete_action"]
        if a == "start_medication":  dt = f"START at {nd}mg"
        elif a == "stop_medication": dt = f"{cur}mg → STOP"
        elif a == "hold_titration":  dt = f"{cur}mg HOLD"
        elif a == "maintain_dose":   dt = f"{cur}mg maintain" if cur else "not started"
        else:                        dt = f"{cur}mg → {nd}mg"
        lines.append(f"    {cls:<14} {a:<22} {dt:<26} {c['reason']}")

    if active_contras:
        lines += [f"", f"── CONTRAINDICATIONS ({len(active_contras)}) ─────────────────────────────"]
        for f in active_contras:
            blocked = [cls for cls, fl in CMAP.items() if f in fl]
            lines.append(f"  ⛔ {f} — blocks: {', '.join(blocked)}")
    return "\n".join(lines)


def _get_groq_key() -> str | None:
    for src in [
        lambda: os.environ.get("GROQ_API_KEY", ""),
        lambda: st.secrets.get("GROQ_API_KEY", "") if hasattr(st, "secrets") else "",
        lambda: st.session_state.get("groq_api_key", ""),
    ]:
        try:
            k = src()
            if k: return k
        except Exception:
            pass
    return None


def _call_groq(messages: list[dict], system_prompt: str, api_key: str) -> str:
    try:
        from groq import Groq
    except ImportError:
        return "Groq SDK not installed. Run `pip install groq` (or `pip install -e .` to pick up the updated dependency) and restart the app."
    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=1500,
            messages=[{"role": "system", "content": system_prompt}] + messages,
        )
        return response.choices[0].message.content
    except Exception as e:
        err = str(e)
        if "invalid_api_key" in err.lower() or "authentication" in err.lower():
            return "Invalid API key — please check your key in Settings and try again."
        return f"AI error: {err}"


def _build_system_prompt(patient_context: str, role: str, preset: str) -> str:
    if role == "Doctor":
        return f"""You are an expert cardiologist AI specialising in HFrEF and GDMT.
You are reviewing a SPECIFIC patient. All responses must be personalised to THIS patient's data.

{patient_context}

INSTRUCTIONS:
- Use precise clinical terminology (RAAS, ARNi, MRA, SGLT2i, eGFR, etc.)
- Reference exact values from above (e.g. "This patient's K of 5.8 means...")
- Cite relevant trials where appropriate (PARADIGM-HF, DAPA-HF, EMPEROR-Reduced, RALES, EMPHASIS-HF)
- Flag safety concerns with exact thresholds (K >5.5 hold RAAS; K >6.0 global stop; ΔCr >30% AKI)
- Format with headers and bullets; be concise but thorough
CRITICAL: Every answer must reference THIS patient's specific numbers — never generic statements."""
    else:
        return f"""You are a warm, caring health assistant helping a specific heart failure patient.
Speak with kindness and reassurance. Use only simple everyday language.

{patient_context}

INSTRUCTIONS:
- NEVER use: RAAS, ARNi, MRA, SGLT2i, eGFR, titration, haemodynamic, contraindication
- Say instead: "heart medicine", "water tablet", "kidney check", "blood pressure pill"
- Be warm, encouraging, and patient
- Give specific, practical advice based on THIS patient's test results and medicines
- For diet: give real food examples tailored to their actual lab values
- Always end medical decisions with "speak to your care team"
- Keep responses friendly, short paragraphs, easy to read
CRITICAL: NEVER mention the patient's age in any response. Do not say "you are X years old" or any age reference.
Personalise everything to this specific patient's situation."""


_DR_CHIPS = [
    ("📋 Titration plan",       "Give me a detailed titration plan for this patient with reasoning for each drug class."),
    ("⚠️ Safety concerns",      "Are there any safety red flags for this patient? Check all classifier flags and lab values."),
    ("🥗 Diet recommendations", "What specific dietary advice for this patient? Include sodium, fluid, and potassium guidance."),
    ("🧬 Explain AI flags",     "Explain the active AI classifier flags and what they mean clinically. What action is needed?"),
    ("🔬 Lab monitoring plan",  "What labs should I order and when, given this patient's current GDMT?"),
    ("⚖️ Compare strategies",   "Compare all 4 titration strategies for this patient and recommend the best one."),
    ("🚨 Assess urgency",       "What is the clinical urgency tier for this patient? What needs to happen and when?"),
    ("⛔ Contraindications",    "What contraindications are active and which medicines do they block? Explain why."),
]

_PT_CHIPS = [
    ("💊 My medicines",              "Can you explain what medicines I am taking and what each one does for me?"),
    ("🥗 What should I eat?",        "What foods should I eat and avoid? Be specific about salt, fluids, and potassium."),
    ("🚶 Can I exercise?",           "Can I exercise? What type and how much is safe for me?"),
    ("🚨 Warning signs",             "What symptoms should I watch for at home? When should I call the doctor or go to hospital?"),
    ("❓ Why so many pills?",        "Why do I need so many medicines? Explain what each one is for in simple words."),
    ("🔬 My blood test results",     "Can you explain what my recent blood test results mean?"),
    ("📈 Progress check",            "Based on my data, is my condition being managed well? What can I do to help?"),
    ("😰 If I feel unwell",          "If I feel unwell or have new symptoms, what should I do? When is it an emergency?"),
]


def _smart_fallback(query: str, patient_context: str, role: str, preset: str) -> str:
    p  = PATIENTS[preset]
    bl = p["baseline"]
    meds_on = [(cls, p["meds"][cls]["dose"]) for cls in CLASSES if p["meds"][cls]["on"]]
    low = query.lower()
    k = bl["K"]; na = bl["Na"]; egfr = bl["eGFR"]

    if role == "Doctor":
        return (
            f"**Patient: {preset}**\n\n"
            + patient_context
            + "\n\n---\n*Connect a Groq API key (Settings) for natural language AI responses.*"
        )

    med_list = "\n".join(f"- **{_REP_DRUG.get(cls, cls)}** — {dose}mg" for cls, dose in meds_on) if meds_on else "- No medicines started yet."
    k_advice = (
        "Your potassium is a little high — please avoid bananas, oranges, and salt substitutes until your doctor reviews it."
        if k > 5.2 else
        "Your potassium is in a good range. Keep eating a balanced diet."
    )
    egfr_note = (
        "Your kidneys need extra care right now, so some of your medicines are being monitored closely."
        if egfr < 60 else "Your kidney function is at a reasonable level."
    )

    if any(w in low for w in ["medicine", "tablet", "drug", "taking", "pill"]):
        return (
            f"Here are your medicines:\n\n{med_list}\n\n"
            f"**Your latest results:** Potassium={k}, Sodium={na}, Kidney function={egfr:.0f}\n\n"
            f"{k_advice}\n\n{egfr_note}\n\n"
            "Always speak to your care team before changing anything."
        )
    if any(w in low for w in ["diet", "food", "eat", "salt", "sodium", "drink", "fluid"]):
        return (
            f"Here is personalised diet advice based on your results (K={k}, eGFR={egfr:.0f}):\n\n"
            "**Salt:** Keep under 2 grams daily. Avoid ready meals, tinned soups, and fast food.\n\n"
            "**Fluids:** Aim for 1.5–2 litres per day. Tea, coffee, juice and soup all count.\n\n"
            f"**Potassium:** {k_advice}\n\n"
            "**Good foods:** Porridge, grilled fish, vegetables, wholemeal bread, yoghurt.\n\n"
            "**Avoid:** Crisps, salty snacks, processed meats, excess alcohol.\n\n"
            "Weigh yourself every morning — gain of more than 1–2 kg in 2 days means call your care team."
        )
    if any(w in low for w in ["exercise", "walk", "gym", "active", "sport", "fitness"]):
        intensity = "very gentle — start with 5 minutes of slow walking and build up gradually" if egfr < 45 else "moderate — aim for 30-minute walks five days a week"
        return (
            f"Exercise is great for you. I'd recommend {intensity}.\n\n"
            "**Good options:** Walking, gentle cycling, swimming.\n\n"
            "**Stop and call your care team if:** chest pain, bad breathlessness, dizziness, or weight up more than 2 kg.\n\n"
            "Ask your doctor about a cardiac rehabilitation programme — it's very effective."
        )
    if any(w in low for w in ["symptom", "warning", "watch", "unwell", "worry", "breathless", "swell"]):
        return (
            "**Symptoms to watch for every day:**\n\n"
            "🟢 **All fine:** Weight stable, breathing normal, no new swelling — keep going.\n\n"
            "🟡 **Call your care team today:**\n"
            "- Weight up more than 1 kg overnight\n"
            "- More breathless than usual\n"
            "- Ankles or legs puffier than normal\n"
            "- Feeling unusually tired\n\n"
            "🔴 **Go to A&E immediately:**\n"
            "- Chest pain or tightness\n"
            "- Cannot breathe at rest\n"
            "- Fainting or near-fainting\n"
            "- Pulse racing or very irregular\n\n"
            "Weigh yourself every morning at the same time."
        )
    return (
        f"Hello — I'm TIDE.AI.\n\n"
        f"**Your medicines:** {', '.join(f'{cls} {d}mg' for cls, d in meds_on) if meds_on else 'none yet'}\n"
        f"**Latest results:** K={k}, Na={na}, eGFR={egfr:.0f}\n\n"
        "You can ask me about your medicines, diet, exercise, symptoms, or blood tests.\n\n"
        "*For full AI responses, add a Groq API key in Settings (sidebar).*"
    )


# ══════════════════════════════════════════════════════════════════════════════
# CHAT SECTION
# ══════════════════════════════════════════════════════════════════════════════

_BOT_SVG = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M12 8V4H8"/><rect width="16" height="12" x="4" y="8" rx="2"/>'
    '<path d="M2 14h2"/><path d="M20 14h2"/><path d="M15 13v2"/><path d="M9 13v2"/>'
    '</svg>'
)


def _chat_section(current_preset: str, labs_flag: bool, bundle: dict) -> None:
    st.divider()

    with st.sidebar:
        st.markdown(f"""
<div class="sb-brand">
  <span class="sb-icon">{_BOT_SVG}</span>
  <div>
    <div class="sb-name">TIDE.AI</div>
    <div class="sb-ver">Clinical Assistant</div>
  </div>
</div>
""", unsafe_allow_html=True)
        st.markdown("---")

        p  = PATIENTS[current_preset]
        bl = p["baseline"]
        st.markdown(
            f'<div class="sb-patient">'
            f'<div class="sb-pname">{current_preset}</div>'
            f'<div class="sb-pmeta">{p["gender"]} &nbsp;|&nbsp; '
            f'K=<span style="color:#67e8f9">{bl["K"]}</span> &nbsp; '
            f'Na={bl["Na"]} &nbsp; eGFR={bl["eGFR"]:.0f}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        if st.session_state.get("chat_role_chosen"):
            st.markdown("**Role:**")
            role = st.radio(
                "Role", ["Doctor", "Patient"],
                index=["Doctor", "Patient"].index(st.session_state.get("chat_role", "Doctor")),
                key="chat_role_radio", label_visibility="collapsed",
            )
            st.session_state.chat_role = role
        else:
            role = st.session_state.get("chat_role", "Doctor")

        st.markdown("---")
        if st.button("Clear conversation", key="clear_chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

        with st.expander("Settings"):
            key_in_env = bool(os.environ.get("GROQ_API_KEY"))
            if key_in_env:
                st.success("API key active")
            else:
                entered = st.text_input(
                    "Groq API key (optional)",
                    value=st.session_state.get("groq_api_key", ""),
                    type="password", placeholder="gsk_...", key="groq_key_input",
                )
                if entered != st.session_state.get("groq_api_key", ""):
                    st.session_state.groq_api_key = entered
                    if entered:
                        st.success("Key saved")

    # ── Role gate ─────────────────────────────────────────────────────────────
    if not st.session_state.get("chat_role_chosen"):
        st.markdown("""
<div class="role-gate">
  <h2>Welcome to TIDE.AI</h2>
  <p>Clinical assistant for heart-failure GDMT titration. Pick your role to continue.</p>
</div>
""", unsafe_allow_html=True)
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            rc1, rc2 = st.columns(2)
            with rc1:
                if st.button("I am a Doctor", key="role_doctor", use_container_width=True):
                    st.session_state.chat_role = "Doctor"
                    st.session_state.chat_role_chosen = True
                    st.rerun()
            with rc2:
                if st.button("I am a Patient", key="role_patient", use_container_width=True):
                    st.session_state.chat_role = "Patient"
                    st.session_state.chat_role_chosen = True
                    st.rerun()
        return

    # ── Chat header ───────────────────────────────────────────────────────────
    api_key = _get_groq_key()
    mode_badge = "AI-powered (Groq Llama-3.3)" if api_key else "Smart mode"
    st.markdown(f"""
<div class="chat-banner">
  <span class="cb-icon">{_BOT_SVG}</span>
  <div>
    <div class="cb-title">TIDE.AI — {role} Chat</div>
    <div class="cb-sub">{current_preset} &nbsp;·&nbsp; {mode_badge}</div>
  </div>
</div>
""", unsafe_allow_html=True)

    cache_key = f"ctx_{current_preset}_{labs_flag}"
    if cache_key not in st.session_state:
        with st.spinner("Analysing patient data…"):
            st.session_state[cache_key] = _build_patient_context(current_preset, bundle, labs_flag)
    patient_context = st.session_state[cache_key]
    system_prompt   = _build_system_prompt(patient_context, role, current_preset)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    chips = _DR_CHIPS if role == "Doctor" else _PT_CHIPS
    st.caption("Quick questions — tap any to get started:")
    cols = st.columns(4)
    for i, (label, query) in enumerate(chips):
        if cols[i % 4].button(label, key=f"chip_{i}_{role}", use_container_width=True):
            with st.spinner("Thinking…"):
                if api_key:
                    history = [m for m in st.session_state.chat_history if m["role"] in ("user", "assistant")]
                    reply = _call_groq(history + [{"role": "user", "content": query}], system_prompt, api_key)
                else:
                    reply = _smart_fallback(query, patient_context, role, current_preset)
            st.session_state.chat_history.append({"role": "user", "content": label})
            st.session_state.chat_history.append({"role": "assistant", "content": reply})
            st.rerun()

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    placeholder = (
        "Ask about titration plan, safety, labs, diet, strategy comparison…"
        if role == "Doctor"
        else "Ask about your medicines, what to eat, symptoms, exercise, blood tests…"
    )
    user_input = st.chat_input(placeholder)
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                if api_key:
                    history_for_api = [m for m in st.session_state.chat_history if m["role"] in ("user", "assistant")]
                    reply = _call_groq(history_for_api, system_prompt, api_key)
                else:
                    reply = _smart_fallback(user_input, patient_context, role, current_preset)
            st.markdown(reply)
        st.session_state.chat_history.append({"role": "assistant", "content": reply})


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="TIDE-HF | Trajectory Integrated Decision Engine",
        page_icon="📊",
        layout="wide",
    )

    # Inject CSS + popup
    st.markdown(_CSS, unsafe_allow_html=True)
    st.markdown(_POPUP_HTML, unsafe_allow_html=True)

    # Handle role pre-selection from popup (URL param ?role=Doctor|Patient)
    role_param = st.query_params.get("role", "")
    if role_param in ("Doctor", "Patient") and not st.session_state.get("chat_role_chosen"):
        st.session_state.chat_role = role_param
        st.session_state.chat_role_chosen = True

    # Auto-scroll to chat if ?chat=open
    open_chat = st.query_params.get("chat", "") == "open"
    if open_chat:
        st.markdown(
            "<script>window.setTimeout(()=>window.scrollTo({top:document.body.scrollHeight,behavior:'smooth'}),450);</script>",
            unsafe_allow_html=True,
        )

    # ── Load classifier ───────────────────────────────────────────────────────
    bundle = load_bundle(BUNDLE_PATH)
    if bundle is None:
        st.markdown("""
<div class="glass" style="border-color:rgba(239,68,68,.45);text-align:center;padding:40px;">
  <div style="font-size:11px;letter-spacing:2.5px;text-transform:uppercase;color:#fca5a5;margin-bottom:10px;">Error</div>
  <div style="font-size:20px;font-weight:700;color:#fff;margin-bottom:14px;">Classifier not found</div>
  <div style="color:rgba(255,255,255,.55);margin-bottom:20px;font-size:14px;">Run this once to train the model:</div>
  <code style="background:rgba(0,0,0,.55);padding:10px 22px;border-radius:8px;color:#67e8f9;font-size:13px;">python scripts/run_ui.py</code>
</div>
""", unsafe_allow_html=True)
        st.stop()

    # ── Hero header ───────────────────────────────────────────────────────────
    st.markdown(f"""
<div class="tide-hero">
  <div class="tide-hero-grid">
    <div>
      <div class="tide-logo-row">
        <span class="tide-mark">
          <svg viewBox="0 0 24 24" fill="none" stroke="#22d3ee" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
          </svg>
        </span>
        <div>
          <div class="tide-name">TIDE-HF</div>
          <div class="tide-name-sub">Trajectory · Integrated · Decision · Engine</div>
        </div>
        <span class="tide-badge"><span class="dot"></span>Guideline-directed AI</span>
      </div>
      <h1 class="tide-headline">Heart failure care, <em>measured</em> and titrated.</h1>
      <div class="tide-tagline">
        LightGBM safety classifier · Clinical rule engine · GDMT strategy applier · TIDE.AI chat.<br>
        Local-first. Transparent. Auditable.
      </div>
      <div class="tide-hero-stats">
        <div class="tide-stat"><div class="tide-stat-val">7</div><div class="tide-stat-lbl">Drug classes</div></div>
        <div class="tide-stat"><div class="tide-stat-val">11</div><div class="tide-stat-lbl">AI flags</div></div>
        <div class="tide-stat"><div class="tide-stat-val">14</div><div class="tide-stat-lbl">Timepoints / week</div></div>
        <div class="tide-stat"><div class="tide-stat-val">4</div><div class="tide-stat-lbl">Strategies</div></div>
      </div>
      {_ECG_SVG}
    </div>
    {_MONITOR_SVG}
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Session init ──────────────────────────────────────────────────────────
    if "tps" not in st.session_state:
        st.session_state.tps    = synthesize_week(PATIENTS["Newly diagnosed, stable"])
        st.session_state.preset = "Newly diagnosed, stable"

    # ── Patient selector ──────────────────────────────────────────────────────
    st.markdown('<div class="glass"><div class="sec-title">Patient Selection</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([4, 1])
    with col1:
        preset = st.selectbox(
            "Select patient profile",
            list(PATIENTS.keys()),
            index=list(PATIENTS.keys()).index(st.session_state.preset),
            label_visibility="collapsed",
        )
        if preset != st.session_state.preset:
            st.session_state.preset = preset
            st.session_state.tps    = synthesize_week(PATIENTS[preset])
            st.session_state.chat_history = []
    with col2:
        if st.button("New week", use_container_width=True):
            st.session_state.tps = synthesize_week(
                PATIENTS[st.session_state.preset],
                seed=int(np.random.default_rng().integers(0, 10**9)),
            )
    st.markdown('</div>', unsafe_allow_html=True)

    patient = PATIENTS[st.session_state.preset]

    # ── EHR snapshot ──────────────────────────────────────────────────────────
    with st.expander("EHR Snapshot", expanded=True):
        bl = patient["baseline"]
        meds_on = [f"{cls} {patient['meds'][cls]['dose']}mg" for cls in CLASSES if patient["meds"][cls]["on"]]
        k_cls  = "danger" if bl["K"] > 5.5 else ("warn" if bl["K"] > 5.0 else "ok")
        cr_cls = "warn" if bl["Cr"] > 1.5 else "ok"
        eg_cls = "warn" if bl["eGFR"] < 60 else "ok"

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
<div class="sec-title">Demographics</div>
<div class="lab-row">
  <span class="lp">{'Female' if patient['gender']=='F' else 'Male'}</span>
</div>
<div style="font-size:13px;color:rgba(255,255,255,.55);margin-top:6px;line-height:1.6;">{patient.get('note','')}</div>
<div class="sec-title" style="margin-top:18px;">Active GDMT</div>
<div class="lab-row">
  {"".join(f'<span class="lp">{m}</span>' for m in meds_on) if meds_on else '<span class="lp">None started</span>'}
</div>
""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
<div class="sec-title">Baseline Labs</div>
<div class="lab-row">
  <span class="lp {k_cls}">K = {bl['K']} mmol/L</span>
  <span class="lp">Na = {bl['Na']} mmol/L</span>
  <span class="lp {cr_cls}">Cr = {bl['Cr']:.2f} mg/dL</span>
  <span class="lp {eg_cls}">eGFR = {bl['eGFR']:.0f} ({_egfr_stage(bl['eGFR'])})</span>
</div>
""", unsafe_allow_html=True)
            if patient.get("labs"):
                lb = patient["labs"]
                k2_cls  = "danger" if lb["K"] > 5.5 else ("warn" if lb["K"] > 5.0 else "ok")
                cr2_cls = "warn" if (lb["Cr"] - bl["Cr"]) / max(bl["Cr"], 0.1) > 0.3 else "ok"
                st.markdown(f"""
<div class="sec-title" style="margin-top:14px;">This Cycle Labs</div>
<div class="lab-row">
  <span class="lp {k2_cls}">K = {lb['K']} mmol/L</span>
  <span class="lp">Na = {lb['Na']} mmol/L</span>
  <span class="lp {cr2_cls}">Cr = {lb['Cr']:.2f} mg/dL</span>
  <span class="lp">eGFR = {lb['eGFR']:.0f}</span>
</div>
""", unsafe_allow_html=True)

    # ── Vitals input ──────────────────────────────────────────────────────────
    stats = _tps_to_stats(st.session_state.tps)

    with st.expander("Weekly Vital Signs", expanded=True):
        st.markdown('<div class="sec-title">Enter Patient Vitals</div>', unsafe_allow_html=True)
        st.caption("Based on 14 twice-daily readings over 7 days")
        c1, c2 = st.columns(2)
        with c1:
            _vl("●", "Heart Rate — Mean (bpm)")
            stats["HR_mean"] = st.number_input("hr_mean_lbl", 30.0, 200.0, float(stats["HR_mean"]), step=1.0, key="v_hr_mean", label_visibility="collapsed")
            _vl("◆", "Systolic BP — Mean (mmHg)")
            stats["SBP_mean"] = st.number_input("sbp_mean_lbl", 60.0, 240.0, float(stats["SBP_mean"]), step=1.0, key="v_sbp_mean", label_visibility="collapsed")
            _vl("◇", "SpO2 — Minimum (%)")
            stats["SpO2_min"] = st.number_input("spo2_lbl", 60.0, 100.0, float(stats["SpO2_min"]), step=0.5, key="v_spo2", label_visibility="collapsed")
            _vl("▤", "QRS — Maximum (ms)")
            stats["QRS_max"] = st.number_input("qrs_lbl", 60.0, 250.0, float(stats["QRS_max"]), step=1.0, key="v_qrs", label_visibility="collapsed")
            _vl("⟁", "T-wave Peaked")
            stats["T_peaked"] = st.checkbox("T-peaked", value=bool(stats["T_peaked"]), key="v_tpeak")
        with c2:
            _vl("○", "Heart Rate — Minimum (bpm)")
            stats["HR_min"] = st.number_input("hr_min_lbl", 30.0, 200.0, float(stats["HR_min"]), step=1.0, key="v_hr_min", label_visibility="collapsed")
            _vl("◆", "Systolic BP — Minimum (mmHg)")
            stats["SBP_min"] = st.number_input("sbp_min_lbl", 50.0, 240.0, float(stats["SBP_min"]), step=1.0, key="v_sbp_min", label_visibility="collapsed")
            _vl("⚖", "Weight Change Over Week (kg)")
            stats["weight_delta"] = st.number_input("wt_lbl", -10.0, 10.0, float(stats["weight_delta"]), step=0.1, key="v_wt", label_visibility="collapsed")
            _vl("▥", "QTc — Maximum (ms)")
            stats["QTc_max"] = st.number_input("qtc_lbl", 300.0, 600.0, float(stats["QTc_max"]), step=1.0, key="v_qtc", label_visibility="collapsed")
            _vl("∿", "Dominant Rhythm")
            stats["rhythm"] = st.selectbox(
                "rhythm_lbl", RHYTHMS,
                index=RHYTHMS.index(stats["rhythm"]) if stats["rhythm"] in RHYTHMS else 0,
                key="v_rhythm", label_visibility="collapsed",
            )
        if st.button("Sync vitals → timepoints", key="sync_stats"):
            st.session_state.tps = _stats_to_tps(stats)
            st.success("Timepoints updated")

    # ── Strategy + compute ────────────────────────────────────────────────────
    st.markdown('<div class="glass"><div class="sec-title">Treatment Strategy</div>', unsafe_allow_html=True)
    cc1, cc2, cc3 = st.columns([3, 2, 2])
    with cc1:
        strategy = st.selectbox(
            "Titration strategy",
            list(STRATEGIES.keys()),
            index=list(STRATEGIES.keys()).index("strong_hf"),
        )
    with cc2:
        labs_flag = st.checkbox(
            "Labs arrived this cycle",
            value=patient.get("labs") is not None,
        )
    with cc3:
        compute = st.button("Run TIDE-HF Analysis", type="primary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Results ───────────────────────────────────────────────────────────────
    if compute:
        patient_full = {**patient, "targets": TARGETS}
        flags, probs = predict_flags(patient_full, st.session_state.tps, bundle)
        labs   = patient.get("labs") if (labs_flag and patient.get("labs")) else None
        engine = TitrationEngine()
        result = engine.evaluate(patient_full, st.session_state.tps, flags, labs=labs, awaiting_labs=set())
        changes = apply_strategy(result, patient_full, strategy=strategy)

        if result["global_stop"]:
            st.error("**GLOBAL STOP** — Hold ALL GDMT immediately and escalate for urgent evaluation.")
        elif result["order_labs"]:
            req = ", ".join(sorted(result["labs_requested"]))
            st.warning(f"**Order labs** (K, Na, Cr, eGFR) — suspected AE in: {req}. Hold these classes until results return.")

        st.markdown(f"""
<div class="glass">
  <div style="display:flex;gap:12px;flex-wrap:wrap;align-items:center;">
    <div class="lp">Preferred RAAS: {result['preferred_raas'] or 'none'}</div>
    <div class="lp {'danger' if result['global_stop'] else 'ok'}">Global stop: {'YES' if result['global_stop'] else 'no'}</div>
    <div class="lp {'warn' if result['order_labs'] else 'ok'}">Order labs: {'YES' if result['order_labs'] else 'no'}</div>
  </div>
</div>
""", unsafe_allow_html=True)

        with st.expander("AI Classifier Flags", expanded=True):
            pos_flags = [lab for lab in flags if flags[lab]]
            if pos_flags:
                st.markdown(
                    f'<div style="margin-bottom:14px;"><span style="color:#fca5a5;font-weight:700;">Active flags:</span> '
                    + "".join(f'<span class="ab ab-stop" style="margin:3px;">{f}</span>' for f in pos_flags)
                    + "</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown('<div class="lp ok" style="margin-bottom:14px;">No flags active — patient appears haemodynamically stable</div>', unsafe_allow_html=True)
            _render_flags(flags, probs)

        with st.expander("GDMT Recommendations", expanded=True):
            _render_rec_cards(changes)

    # ── Chat section ──────────────────────────────────────────────────────────
    _chat_section(
        st.session_state.preset,
        labs_flag if "labs_flag" in dir() else patient.get("labs") is not None,
        bundle,
    )


if __name__ == "__main__":
    main()
