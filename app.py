import sys
import time
from pathlib import Path

import streamlit as st
import tempfile
sys.path.append(str(Path(__file__).parent / "ingestion"))
sys.path.append(str(Path(__file__).parent / "graph"))

from ingest import ingest_pdf                        # noqa: E402
from self_healing_graph import ask                    # noqa: E402
from retrieve_generate import invalidate_bm25_cache    # noqa: E402

COLLECTION_NAME = "documents"

CONFIDENCE_DOT_COLOR = {
    "high": "#1F9D8A",
    "medium": "#C9861A",
    "low": "#C24C43",
    "none": "#9AA0AC",
}

st.set_page_config(page_title="Self-Healing RAG", layout="centered")

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

/* Every diagnostic/status text block below forces black explicitly,
   overriding any inherited theme color so nothing can go invisible
   against a light or dark background. */
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
    "Expand <b>Diagnostics</b> below each answer to watch this happen live.</div>",
    unsafe_allow_html=True,
)
st.write("")

if "history" not in st.session_state:
    st.session_state.history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "last_elapsed" not in st.session_state:
    st.session_state.last_elapsed = 0.0

# --- Upload section ---
col1, col2 = st.columns([3, 1])
with col1:
    pdf_file = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed")
with col2:
    ingest_clicked = st.button("Ingest", type="primary", use_container_width=True)

status_placeholder = st.empty()

if ingest_clicked:
    if pdf_file is None:
        status_placeholder.markdown(
            "<span class='status-line'>NO FILE — select a PDF before ingesting</span>",
            unsafe_allow_html=True,
        )
    else:
        status_placeholder.markdown(
            "<span class='status-line'>Processing... ingesting document</span>",
            unsafe_allow_html=True,
        )
        try:
            tmp_path = Path(tempfile.gettempdir()) / pdf_file.name
            tmp_path.write_bytes(pdf_file.getvalue())
            start = time.time()
            collection = ingest_pdf(str(tmp_path), collection_name=COLLECTION_NAME)
            invalidate_bm25_cache(COLLECTION_NAME)
            elapsed = time.time() - start
            status_placeholder.markdown(
                f"<span class='status-line'>INGESTED in {elapsed:.1f}s &middot; "
                f"{collection.count()} chunks indexed &middot; ready for questions</span>",
                unsafe_allow_html=True,
            )
        except Exception as e:
            status_placeholder.markdown(
                f"<span class='status-line'>INGEST FAILED — {e}</span>",
                unsafe_allow_html=True,
            )

st.markdown("---")

# --- Chat history ---
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(f"<div class='diag-text'>{msg['content']}</div>", unsafe_allow_html=True)

query = st.chat_input("What does this document say about...?")

if query:
    st.session_state.history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(f"<div class='diag-text'>{query}</div>", unsafe_allow_html=True)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown(
            "<span class='status-line'>Processing... running self-healing retrieval</span>",
            unsafe_allow_html=True,
        )
        try:
            start = time.time()
            result = ask(query, collection_name=COLLECTION_NAME)
            elapsed = time.time() - start
            answer = result["final_output"]
            placeholder.markdown(f"<div class='diag-text'>{answer}</div>", unsafe_allow_html=True)
            st.session_state.history.append({"role": "assistant", "content": answer})
            st.session_state.last_result = result
            st.session_state.last_elapsed = elapsed
        except Exception as e:
            error_msg = f"Error: {e}"
            placeholder.markdown(f"<div class='diag-text'>{error_msg}</div>", unsafe_allow_html=True)
            st.session_state.history.append({"role": "assistant", "content": error_msg})
            st.session_state.last_result = None

# --- Sources & Diagnostics for the most recent answer ---
if st.session_state.last_result:
    result = st.session_state.last_result
    elapsed = st.session_state.last_elapsed

    with st.expander("Sources used"):
        if not result["chunks"]:
            st.markdown("<div class='diag-text'><i>no chunks retrieved</i></div>", unsafe_allow_html=True)
        for i, c in enumerate(result["chunks"], start=1):
            snippet = c["text"][:250] + ("..." if len(c["text"]) > 250 else "")
            st.markdown(
                f"<div class='diag-text'><b>[{i}]</b> <code>{c['source']}</code> "
                f"page {c['page']} &middot; rerank score <code>{c['rerank_score']:.3f}</code>"
                f"<br><i>{snippet}</i></div><br>",
                unsafe_allow_html=True,
            )

    with st.expander("Diagnostics"):
        conf = result["confidence"]
        dot_color = CONFIDENCE_DOT_COLOR.get(conf["label"], "#9AA0AC")
        retry_dots = "".join(
            f'<span style="display:inline-block;width:8px;height:8px;border-radius:50%;'
            f'background:{"#C9861A" if i < result["retry_count"] else "#D8DBE0"};margin-right:4px;"></span>'
            for i in range(2)
        )
        sub_queries_html = "<br>".join(f"&nbsp;&nbsp;- {sq}" for sq in result["sub_queries"])

        st.markdown(
            f"""<div class='diag-text'>
            <span style="display:inline-block;width:9px;height:9px;border-radius:50%;
            background:{dot_color};margin-right:6px;"></span>
            <b>CONFIDENCE</b> <code>{conf['label'].upper()}</code>
            &middot; best rerank score <code>{conf['best_rerank_score']}</code><br><br>
            <b>VERDICT</b> <code>{(result['verdict'] or '—').upper()}</code><br><br>
            <b>RETRIES</b> {retry_dots} {result['retry_count']} / 2<br><br>
            <b>CRITIC NOTE</b> {result['critique_reason'] or '—'}<br><br>
            <b>LATENCY</b> <code>{elapsed:.2f}s</code><br><br>
            <b>SUB-QUERIES</b><br>{sub_queries_html}<br><br>
            <b>FINAL QUERY</b> <code>{result['query']}</code>
            </div>""",
            unsafe_allow_html=True,
        )
