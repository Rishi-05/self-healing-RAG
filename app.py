import html
import re
import sys
import tempfile
import time
import uuid
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).parent / "ingestion"))
sys.path.append(str(Path(__file__).parent / "graph"))

from ingest import ingest_pdf                         # noqa: E402
from retrieve_generate import invalidate_bm25_cache   # noqa: E402
from self_healing_graph import ask                    # noqa: E402
from store import list_sources                        # noqa: E402

# True  -> every browser session gets its OWN private collection (visitors can't see
#          each other's uploads; documents are gone when the session ends).
# False -> one shared collection "documents" (fine for local use, NOT for a public demo).
ISOLATE_SESSIONS = True
SCOPE_ALL = "All documents"

CONFIDENCE_DOT_COLOR = {
    "high": "#1F9D8A",
    "medium": "#C9861A",
    "low": "#C24C43",
    "none": "#9AA0AC",
}

st.set_page_config(page_title="Self-Healing RAG", layout="wide")

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif !important;
}

#console-header {
    display: flex; align-items: center; gap: 10px;
    padding: 14px 18px; margin-bottom: 12px;
    background: #F4F5F7;
    border: 1px solid #E1E4E9;
    border-radius: 10px;
}
#console-header .status-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: #1F9D8A;
}
#console-header .title {
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600; font-size: 14px; letter-spacing: 0.06em;
    color: #000000 !important;
}
#console-header .subtitle {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px; color: #000000 !important; margin-left: auto;
}

.diag-text, .diag-text * {
    color: #000000 !important;
    font-size: 13px;
}
.diag-text code {
    font-family: 'IBM Plex Mono', monospace !important;
    background: #EEF0F3 !important;
    color: #000000 !important;
    padding: 1px 5px; border-radius: 4px; font-size: 12px;
}
.status-line, .status-line * {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 13px !important;
    color: #000000 !important;
}

/* --- Chat layout --- */
.sidebar-title {
    font-family: 'IBM Plex Mono', monospace; font-weight: 700; font-size: 15px;
    letter-spacing: .04em; color: #000000; margin-bottom: 18px;
}
.msg-row { display: flex; margin-bottom: 14px; }
.msg-row.user { justify-content: flex-end; }
.msg-row.assistant { justify-content: flex-start; align-items: flex-start; gap: 8px; }
.avatar {
    width: 26px; height: 26px; border-radius: 50%;
    background: #1F9D8A; color: #ffffff; display: flex;
    align-items: center; justify-content: center;
    font-size: 11px; font-family: 'IBM Plex Mono', monospace;
    flex-shrink: 0; margin-top: 2px;
}
.bubble {
    max-width: 78%; padding: 10px 14px; border-radius: 14px;
    font-size: 13.5px; line-height: 1.55;
}
.bubble.user, .bubble.user * {
    background: #1F9D8A; color: #ffffff !important;
    border-bottom-right-radius: 4px;
}
.bubble.assistant {
    background: #F4F5F7; border: 1px solid #E1E4E9; color: #000000;
    border-bottom-left-radius: 4px;
}
.citation-badges { margin-top: 8px; }
.citation-chip {
    display: inline-flex; align-items: center; justify-content: center;
    width: 19px; height: 19px; border-radius: 50%;
    background: #DFF3EF; color: #1F9D8A; font-size: 10px; font-weight: 700;
    margin-right: 5px;
}
.citations-panel-title {
    font-family: 'IBM Plex Mono', monospace; font-weight: 600;
    font-size: 12px; letter-spacing: .04em; color: #000000; margin-bottom: 10px;
}
.citation-card { border-bottom: 1px solid #E1E4E9; padding: 10px 0; }
.citation-card .src-name { font-weight: 600; font-size: 12.5px; color: #000000; }
.citation-card .snippet {
    font-size: 11.5px; color: #4B4F58; font-style: italic; margin-top: 4px;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

HEADER_HTML = """
<div id="console-header">
    <span class="status-dot"></span>
    <span class="title">SELF-HEALING RAG</span>
    <span class="subtitle">hybrid search &middot; cross-encoder rerank &middot; critic-agent retry loop</span>
</div>
"""
st.markdown(HEADER_HTML, unsafe_allow_html=True)

st.markdown(
    "<div class='diag-text'>Upload a PDF, then ask questions. Unlike plain RAG, this pipeline "
    "runs a <b>critic agent</b> that checks whether the answer is actually "
    "grounded in the retrieved chunks — if not, it <b>reformulates the "
    "query and retries</b> (up to 2x) before ever admitting it doesn't know. "
    "Citations under each answer link to the panel on the right; expand "
    "<b>Diagnostics</b> below to watch the self-healing loop live.</div>",
    unsafe_allow_html=True,
)
st.write("")

# --- Session state ---
if "history" not in st.session_state:
    st.session_state.history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "last_elapsed" not in st.session_state:
    st.session_state.last_elapsed = 0.0
if "collection_name" not in st.session_state:
    st.session_state.collection_name = (
        f"docs_{uuid.uuid4().hex[:12]}" if ISOLATE_SESSIONS else "documents"
    )
if "sources" not in st.session_state:
    st.session_state.sources = [] if ISOLATE_SESSIONS else list_sources("documents")

COLLECTION_NAME = st.session_state.collection_name


def esc(text) -> str:
    """Everything that came from a user or an LLM is escaped before going into HTML."""
    return html.escape(str(text))


def render_text(text: str) -> str:
    """Escape, then re-apply the little formatting we want: bold, bullets, line breaks,
    and [[Source n]] markers shown as small [n] superscripts."""
    out = html.escape(text)
    out = re.sub(r"\[\[Source\s*(\d+)\]\]", r"<sup>[\1]</sup>", out)
    out = re.sub(r"(?m)^\s*[-*]\s+", "&bull; ", out)
    out = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", out)
    return out.replace("\n", "<br>")


def extract_citation_numbers(text: str) -> list:
    """Unique [[Source n]] markers in the answer (the exact format the generator is told to use)."""
    return sorted({int(n) for n in re.findall(r"\[\[Source\s*(\d+)\]\]", text)})


# --- Upload section ---
col1, col2 = st.columns([3, 1])
with col1:
    pdf_file = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed")
with col2:
    ingest_clicked = st.button("Ingest", type="primary", use_container_width=True)

status_placeholder = st.empty()


def show_status(message: str) -> None:
    status_placeholder.markdown(
        f"<span class='status-line'>{esc(message)}</span>", unsafe_allow_html=True
    )


if ingest_clicked:
    if pdf_file is None:
        show_status("NO FILE — select a PDF before ingesting")
    else:
        show_status("Processing... ingesting document")
        tmp_path = None
        try:
            # Random temp name: the uploaded filename is never used as a filesystem path.
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(pdf_file.getvalue())
                tmp_path = tmp.name

            start = time.time()
            info = ingest_pdf(
                tmp_path,
                collection_name=COLLECTION_NAME,
                display_name=Path(pdf_file.name).stem,
            )
            invalidate_bm25_cache(COLLECTION_NAME)
            elapsed = time.time() - start

            if info["source"] not in st.session_state.sources:
                st.session_state.sources.append(info["source"])
            st.session_state.scope = info["source"]   # "this pdf" = the one just uploaded

            if info["skipped"]:
                show_status(f"ALREADY INGESTED — '{info['source']}' ({info['chunks']} chunks) is ready for questions")
            else:
                note = "" if info["summary_ok"] else " · summary will be generated on your first overview question"
                show_status(
                    f"INGESTED '{info['source']}' in {elapsed:.1f}s · "
                    f"{info['chunks']} chunks indexed · ready for questions{note}"
                )
        except Exception as e:
            show_status(f"INGEST FAILED — {e}")
        finally:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)

st.markdown("---")

# --- Three-pane chat layout: sidebar / chat / active citations ---
left_col, main_col, right_col = st.columns([1, 3, 1.3], gap="large")

active_source = None
with left_col:
    st.markdown('<div class="sidebar-title">Self-Healing RAG</div>', unsafe_allow_html=True)
    if st.button("+ New Chat", use_container_width=True):
        st.session_state.history = []
        st.session_state.last_result = None
        st.rerun()

    if st.session_state.sources:
        scope_options = list(reversed(st.session_state.sources)) + [SCOPE_ALL]
        if st.session_state.get("scope") not in scope_options:
            st.session_state.scope = scope_options[0]
        st.selectbox("Ask about", scope_options, key="scope")
        active_source = None if st.session_state.scope == SCOPE_ALL else st.session_state.scope

with main_col:
    if not st.session_state.history:
        st.markdown(
            "<div class='diag-text'><i>Try: &ldquo;summarize this pdf&rdquo; or ask a specific "
            "question about the document.</i></div>",
            unsafe_allow_html=True,
        )
    for idx, msg in enumerate(st.session_state.history):
        if msg["role"] == "user":
            st.markdown(
                f'<div class="msg-row user"><div class="bubble user">{esc(msg["content"])}</div></div>',
                unsafe_allow_html=True,
            )
        else:
            cite_nums = msg.get("citation_numbers", [])
            badges = "".join(f'<span class="citation-chip">{n}</span>' for n in cite_nums)
            badges_html = f'<div class="citation-badges">{badges}</div>' if badges else ""
            st.markdown(
                f'<div class="msg-row assistant"><div class="avatar">AI</div>'
                f'<div class="bubble assistant">{render_text(msg["content"])}{badges_html}</div></div>',
                unsafe_allow_html=True,
            )
            fcol1, fcol2, _ = st.columns([0.07, 0.07, 0.86])
            with fcol1:
                if st.button("👍", key=f"up_{idx}"):
                    msg["feedback"] = "up"
                    st.rerun()
            with fcol2:
                if st.button("👎", key=f"down_{idx}"):
                    msg["feedback"] = "down"
                    st.rerun()

with right_col:
    st.markdown('<div class="citations-panel-title">ACTIVE CITATIONS</div>', unsafe_allow_html=True)
    last_result = st.session_state.last_result
    if not last_result or not last_result["chunks"]:
        st.markdown(
            "<div class='diag-text'><i>Ask a question to see citations here.</i></div>",
            unsafe_allow_html=True,
        )
    else:
        last_assistant = next(
            (m for m in reversed(st.session_state.history) if m["role"] == "assistant"), None
        )
        cited = last_assistant.get("citation_numbers", []) if last_assistant else []
        chunks_to_show = (
            [last_result["chunks"][n - 1] for n in cited if 0 < n <= len(last_result["chunks"])]
            if cited else last_result["chunks"]
        )
        for c in chunks_to_show:
            snippet = c["text"][:160] + ("..." if len(c["text"]) > 160 else "")
            st.markdown(
                f'<div class="citation-card"><div class="src-name">{esc(c["source"])} &middot; p.{esc(c["page"])}</div>'
                f'<div class="snippet">"{esc(snippet)}"</div></div>',
                unsafe_allow_html=True,
            )

query = st.chat_input("Ask about the document, or say 'summarize this pdf'...")

if query:
    st.session_state.history.append({"role": "user", "content": query})
    if not st.session_state.sources:
        st.session_state.history.append({
            "role": "assistant",
            "content": "Upload a PDF and click Ingest first, then ask me about it.",
            "citation_numbers": [],
        })
        st.session_state.last_result = None
    else:
        try:
            with st.spinner("Thinking..."):
                start = time.time()
                result = ask(query, collection_name=COLLECTION_NAME, source=active_source)
                elapsed = time.time() - start
            answer = result["final_output"]
            st.session_state.history.append(
                {"role": "assistant", "content": answer, "citation_numbers": extract_citation_numbers(answer)}
            )
            st.session_state.last_result = result
            st.session_state.last_elapsed = elapsed
        except Exception as e:
            st.session_state.history.append(
                {"role": "assistant", "content": f"Error: {e}", "citation_numbers": []}
            )
            st.session_state.last_result = None
    st.rerun()

# --- Diagnostics for the most recent answer ---
if st.session_state.last_result:
    result = st.session_state.last_result
    elapsed = st.session_state.last_elapsed

    with st.expander("Diagnostics"):
        conf = result["confidence"]
        dot_color = CONFIDENCE_DOT_COLOR.get(conf["label"], "#9AA0AC")
        retry_dots = "".join(
            f'<span style="display:inline-block;width:8px;height:8px;border-radius:50%;'
            f'background:{"#C9861A" if i < result["retry_count"] else "#D8DBE0"};margin-right:4px;"></span>'
            for i in range(2)
        )
        sub_queries_html = "<br>".join(f"&nbsp;&nbsp;- {esc(sq)}" for sq in result["sub_queries"])

        st.markdown(
            f"""<div class='diag-text'>
            <span style="display:inline-block;width:9px;height:9px;border-radius:50%;
            background:{dot_color};margin-right:6px;"></span>
            <b>CONFIDENCE</b> <code>{esc(conf['label'].upper())}</code>
            &middot; best rerank score <code>{esc(conf['best_rerank_score'])}</code><br><br>
            <b>VERDICT</b> <code>{esc((result['verdict'] or '—').upper())}</code><br><br>
            <b>RETRIES</b> {retry_dots} {result['retry_count']} / 2<br><br>
            <b>CRITIC NOTE</b> {esc(result['critique_reason'] or '—')}<br><br>
            <b>LATENCY</b> <code>{elapsed:.2f}s</code><br><br>
            <b>SEARCHED IN</b> <code>{esc(result.get('source') or SCOPE_ALL)}</code><br><br>
            <b>SUB-QUERIES</b><br>{sub_queries_html}<br><br>
            <b>FINAL QUERY</b> <code>{esc(result['query'])}</code>
            </div>""",
            unsafe_allow_html=True,
        )
