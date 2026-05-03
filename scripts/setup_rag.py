#!/usr/bin/env python
"""
scripts/setup_rag.py
====================
One-time RAG setup: index all PDFs in rag_docs/ into ChromaDB.

Run ONCE from the repo root after adding your PDFs:
    python scripts/setup_rag.py

Options
-------
    --docs   PATH    Docs folder (default: rag_docs/)
    --force          Delete existing DB and re-index from scratch
    --model  NAME    Ollama model name (default: mistral)
    --smoke          Run a quick smoke test after indexing
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the package importable without pip install
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Index PDFs for TIDE-HF RAG")
    ap.add_argument("--docs",  default="rag_docs", help="Path to PDF docs folder")
    ap.add_argument("--force", action="store_true", help="Re-index from scratch")
    ap.add_argument("--model", default="mistral",  help="Ollama model name")
    ap.add_argument("--smoke", action="store_true", help="Run smoke test after indexing")
    args = ap.parse_args()

    _banner("TIDE-HF RAG — one-time setup")

    # ── dependency check ──────────────────────────────────────────────────────
    missing = []
    for pkg in ("chromadb", "sentence_transformers", "pypdf", "requests"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg.replace("_", "-"))
    if missing:
        _fail(
            f"Missing packages: {', '.join(missing)}\n\n"
            "Install with:\n"
            "    pip install -e '.[rag]'"
        )

    # ── import engine ─────────────────────────────────────────────────────────
    try:
        from chf_titration.rag import TideRAG
    except ImportError as e:
        _fail(f"Cannot import chf_titration.rag: {e}\n\nRun: pip install -e '.[rag]'")

    # ── docs folder check ─────────────────────────────────────────────────────
    docs_dir = Path(args.docs)
    if not docs_dir.exists():
        print(f"\n✗  Docs folder not found: {docs_dir.resolve()}")
        print("\nCreate the layout and add your PDFs:")
        print(f"    mkdir -p {docs_dir}/clinical {docs_dir}/patient")
        print()
        _print_pdf_guide(docs_dir)
        sys.exit(1)

    all_pdfs = list(docs_dir.rglob("*.pdf"))
    if not all_pdfs:
        print(f"\n✗  No PDF files found inside {docs_dir.resolve()}")
        _print_pdf_guide(docs_dir)
        sys.exit(1)

    print(f"\n  Found {len(all_pdfs)} PDF(s) in {docs_dir.resolve()}")

    # ── index ─────────────────────────────────────────────────────────────────
    rag = TideRAG()
    print()
    n = rag.setup(docs_dir=docs_dir, force=args.force)

    _banner("Setup complete")
    if n > 0:
        print(f"  ✓  {n:,} chunks indexed and stored in rag_db/")
    else:
        print("  ✓  Already indexed (use --force to re-index)")

    # ── smoke test ────────────────────────────────────────────────────────────
    if args.smoke:
        print("\nRunning smoke test…")
        ctx    = (
            "Patient: 70y M | hyperkalemia_detected | K+: 5.9 mEq/L\n"
            "Titration decisions:\n"
            "  MRA   decrease_dose   25mg → 12.5mg  [immediate_ae]"
        )
        answer = rag.ask(
            "Why was my spironolactone reduced?",
            ctx,
            audience="patient",
        )
        print("\n── Sample patient response ────────────────────────────")
        print(answer[:600] + ("…" if len(answer) > 600 else ""))
        print("───────────────────────────────────────────────────────")

    # ── next steps ────────────────────────────────────────────────────────────
    print()
    print("Next steps:")
    print("  1.  Make sure Ollama is running:   ollama serve")
    print(f"  2.  Make sure model is pulled:     ollama pull {args.model}")
    print("  3.  Launch the app:                python scripts/run_ui.py")
    print("  4.  Open the '💬 Ask AI' tab in the Streamlit UI")
    print()


# ── helpers ───────────────────────────────────────────────────────────────────

def _banner(text: str) -> None:
    w = max(60, len(text) + 4)
    print("\n" + "=" * w)
    print(f"  {text}")
    print("=" * w)


def _fail(msg: str) -> None:
    print(f"\n✗  {msg}\n")
    sys.exit(1)


def _print_pdf_guide(docs_dir: Path) -> None:
    print(
        f"\nExpected layout:\n"
        f"  {docs_dir}/\n"
        f"    clinical/   ← AHA guidelines, drug monographs\n"
        f"    patient/    ← patient-friendly guides\n"
        f"    shared/     ← shown to both (optional)\n"
        f"\nSuggested PDFs for clinical/:\n"
        f"  aha_hf_guidelines_2022.pdf     (ahajournals.org)\n"
        f"  strong_hf_protocol.pdf         (nejm.org/doi/10.1056/NEJMoa2204512)\n"
        f"  sacubitril_valsartan_pi.pdf    (drugs.com)\n"
        f"  spironolactone_pi.pdf          (drugs.com)\n"
        f"  carvedilol_pi.pdf              (drugs.com)\n"
        f"  empagliflozin_pi.pdf           (drugs.com)\n"
        f"  furosemide_pi.pdf              (drugs.com)\n"
        f"\nSuggested PDFs for patient/:\n"
        f"  entresto_patient_guide.pdf     (novartis.com)\n"
        f"  chf_lifestyle_guide.pdf        (heart.org)\n"
        f"  fluid_restriction_guide.pdf    (heart.org)\n"
    )


if __name__ == "__main__":
    main()
