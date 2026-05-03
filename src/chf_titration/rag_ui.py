"""
src/chf_titration/rag_ui.py
===========================
Drop-in RAG tab for the TIDE-HF Streamlit UI  (ui.py).

How to add it to ui.py
-----------------------
1.  Add the import near the top of ui.py:

        try:
            from chf_titration.rag_ui import render_rag_tab
            _RAG_OK = True
        except ImportError:
            _RAG_OK = False

2.  Add "💬 Ask AI" to your st.tabs() call, e.g.:

        tab_summary, tab_recs, tab_rag = st.tabs(
            ["Summary", "Recommendations", "💬 Ask AI"]
        )

3.  Add a block after your existing tab blocks:

        with tab_rag:
            if _RAG_OK:
                render_rag_tab(result, changes, patient, flags, labs)
            else:
                st.info("Run  pip install -e '.[rag]'  to enable the AI assistant.")
"""

from __future__ import annotations

from typing import Optional
import streamlit as st


# ── public entry point ────────────────────────────────────────────────────────

def render_rag_tab(
    result:  dict,
    changes: dict,
    patient: dict,
    flags:   dict,
    labs:    Optional[dict] = None,
) -> None:
    """Render the full Ask-AI tab. Pass the same variables ui.py already has."""

    st.markdown("### 💬  Clinical Assistant — Ask AI")
    st.caption(
        "Answers are grounded in your local guideline PDFs "
        "(AHA 2022 · STRONG-HF · drug monographs). "
        "**No internet. No API key. All local.**"
    )

    # ── lazy-load deps ────────────────────────────────────────────────────────
    missing = _missing_deps()
    if missing:
        st.error(
            f"**Missing RAG packages:** {', '.join(missing)}\n\n"
            "```bash\npip install -e '.[rag]'\n```"
        )
        return

    # ── check Ollama ──────────────────────────────────────────────────────────
    ollama_ok = _ping_ollama()
    if not ollama_ok:
        st.warning(
            "**Ollama is not running.** Start it in a terminal:\n\n"
            "```bash\nollama serve\n```\n\n"
            "If you haven't pulled a model yet:\n\n"
            "```bash\nollama pull mistral\n```"
        )

    # ── load RAG (cached) ─────────────────────────────────────────────────────
    rag = _get_rag()
    if rag is None:
        return

    chunk_count = rag.chunk_count
    if chunk_count == 0:
        st.warning(
            "**Knowledge base is empty.** Index your PDFs first:\n\n"
            "```bash\npython scripts/setup_rag.py\n```"
        )
        _show_doc_instructions()
        return

    st.success(f"✓  Knowledge base ready — **{chunk_count:,}** chunks indexed")

    # ── audience toggle ───────────────────────────────────────────────────────
    col_l, col_r = st.columns([2, 3])
    with col_l:
        audience = st.radio(
            "Response for:",
            ["patient", "clinician"],
            format_func=lambda x: "🧑 Patient" if x == "patient" else "👨‍⚕️ Clinician",
            horizontal=True,
            key="rag_audience",
        )
    with col_r:
        if audience == "patient":
            st.caption("Plain English · lifestyle advice · when to call doctor")
        else:
            st.caption("Clinical rationale · AHA thresholds · lab citations")

    st.markdown("---")

    # ── context-aware suggested questions ─────────────────────────────────────
    suggestions = _suggest(flags, changes, audience)
    if suggestions:
        st.markdown("**Suggested questions:**")
        cols = st.columns(len(suggestions))
        for col, q in zip(cols, suggestions):
            with col:
                if st.button(q, key=f"sq_{hash(q)}", use_container_width=True):
                    st.session_state["rag_q"] = q

    # ── question input ────────────────────────────────────────────────────────
    question = st.text_area(
        "Ask a question about this patient's treatment plan:",
        value=st.session_state.get("rag_q", ""),
        height=80,
        placeholder=(
            "e.g. 'Why was my spironolactone reduced?'  ·  "
            "'What side effects should I watch for with Entresto?'  ·  "
            "'What is the AHA threshold for MRA dose reduction?'"
        ),
        key="rag_q_input",
    )

    c1, c2 = st.columns([3, 1])
    with c1:
        ask = st.button(
            "🔍  Get answer",
            use_container_width=True,
            type="primary",
            disabled=(not question.strip() or not ollama_ok),
        )
    with c2:
        if st.button("Clear", use_container_width=True):
            for k in ("rag_q", "rag_answer", "rag_q_last", "rag_audience_last"):
                st.session_state.pop(k, None)
            st.rerun()

    # ── run RAG query ─────────────────────────────────────────────────────────
    if ask and question.strip():
        from chf_titration.rag import build_engine_context
        ctx = build_engine_context(result, changes, patient, flags, labs)

        spinner_msg = (
            "Retrieving guidelines · generating patient response…"
            if audience == "patient"
            else "Retrieving guidelines · generating clinical rationale…"
        )
        with st.spinner(spinner_msg):
            answer = rag.ask(question.strip(), ctx, audience=audience)

        st.session_state["rag_answer"]       = answer
        st.session_state["rag_q_last"]       = question.strip()
        st.session_state["rag_audience_last"] = audience
        st.session_state["rag_ctx_last"]      = ctx

    # ── display answer ────────────────────────────────────────────────────────
    if st.session_state.get("rag_answer"):
        st.markdown("---")
        label = (
            "🧑 Patient response"
            if st.session_state.get("rag_audience_last") == "patient"
            else "👨‍⚕️ Clinician response"
        )
        st.markdown(f"**{label}**")
        st.markdown(st.session_state["rag_answer"])

        with st.expander("📋 Patient context sent to model", expanded=False):
            st.code(st.session_state.get("rag_ctx_last", ""), language=None)

        st.download_button(
            "⬇  Download response as .txt",
            data=st.session_state["rag_answer"],
            file_name="tide_hf_response.txt",
            mime="text/plain",
        )

    # ── history ───────────────────────────────────────────────────────────────
    if "rag_hist" not in st.session_state:
        st.session_state["rag_hist"] = []

    if ask and st.session_state.get("rag_answer"):
        st.session_state["rag_hist"].insert(0, {
            "Q":        question.strip(),
            "A":        st.session_state["rag_answer"][:150] + "…",
            "audience": audience,
        })

    if st.session_state["rag_hist"]:
        with st.expander(
            f"📜  History ({len(st.session_state['rag_hist'])} questions)",
            expanded=False,
        ):
            for i, h in enumerate(st.session_state["rag_hist"]):
                st.markdown(
                    f"**{i+1}.** [{h['audience']}]  *{h['Q']}*\n\n{h['A']}"
                )
                st.markdown("---")
            if st.button("Clear history", key="rag_clear_hist"):
                st.session_state["rag_hist"] = []
                st.rerun()


# ── helpers ───────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading RAG engine…")
def _get_rag():
    """Load TideRAG once per Streamlit session."""
    try:
        from chf_titration.rag import TideRAG
        return TideRAG()
    except Exception as exc:
        st.error(f"RAG engine failed to load: {exc}")
        return None


def _missing_deps() -> list[str]:
    out = []
    for pkg in ("chromadb", "sentence_transformers", "pypdf", "requests"):
        try:
            __import__(pkg)
        except ImportError:
            out.append(pkg.replace("_", "-"))
    return out


def _ping_ollama() -> bool:
    import os
    url = os.environ.get("TIDE_OLLAMA_URL", "http://localhost:11434")
    try:
        import requests
        r = requests.get(f"{url}/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def _suggest(flags: dict, changes: dict, audience: str) -> list[str]:
    """Return 2-3 context-aware suggested questions."""
    active = [f for f, v in flags.items() if v]
    stopped = [cls for cls, c in changes.items() if c.get("concrete_action") == "stop_medication"]
    started = [cls for cls, c in changes.items() if c.get("concrete_action") == "start_medication"]
    decreased = [cls for cls, c in changes.items() if c.get("concrete_action") == "decrease_dose"]

    if audience == "patient":
        q = []
        if "hyperkalemia_detected" in active:
            q.append("Why was my potassium too high?")
        if "bradycardia_detected" in active:
            q.append("Why was my heart rate too slow?")
        if "volume_depletion_detected" in active:
            q.append("Why was my water pill reduced?")
        if "worsening_HF_detected" in active:
            q.append("My heart failure is getting worse — what should I do?")
        if stopped:
            q.append(f"Why was my {stopped[0].replace('_',' ')} stopped?")
        if started:
            q.append(f"What is {started[0].replace('_',' ')} and why am I starting it?")
        if not q:
            q = [
                "What do my medications do for my heart?",
                "What symptoms should I watch for this week?",
            ]
        return q[:3]
    else:
        q = []
        if "hyperkalemia_detected" in active:
            q.append("AHA threshold for MRA reduction in hyperkalemia?")
        if "renal_dysfunction_detected" in active:
            q.append("STRONG-HF protocol for creatinine rise > 30%?")
        if "worsening_HF_detected" in active:
            q.append("When to hold beta-blocker in worsening HF?")
        if decreased:
            q.append(f"Rationale for {decreased[0].replace('_',' ')} dose reduction?")
        if not q:
            q = [
                "AHA 2022 GDMT target doses for HFrEF?",
                "STRONG-HF up-titration schedule?",
            ]
        return q[:3]


def _show_doc_instructions() -> None:
    st.info(
        "**Create the rag_docs/ folder and add PDFs:**\n\n"
        "```bash\n"
        "mkdir -p rag_docs/clinical rag_docs/patient\n"
        "```\n\n"
        "**clinical/** — AHA 2022 HF guidelines · STRONG-HF protocol · "
        "drug prescribing information (sacubitril/valsartan, spironolactone, "
        "carvedilol, empagliflozin, furosemide)\n\n"
        "**patient/** — Patient-friendly guides from heart.org or novartis.com\n\n"
        "Free sources:\n"
        "- ahajournals.org  \n"
        "- drugs.com (free prescribing info PDFs)  \n"
        "- heart.org (Heart Failure Society resources)"
    )
