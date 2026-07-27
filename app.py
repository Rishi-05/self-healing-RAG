import os
import sys
import time
import traceback
from pathlib import Path

import gradio as gr

sys.path.append(str(Path(__file__).parent / "ingestion"))
sys.path.append(str(Path(__file__).parent / "graph"))

from ingest import ingest_pdf                      # noqa: E402
from self_healing_graph import ask                  # noqa: E402
from retrieve_generate import invalidate_bm25_cache  # noqa: E402

COLLECTION_NAME = "documents"

CONFIDENCE_DOT_COLOR = {
    "high": "#4FD1C5",
    "medium": "#F2B84B",
    "low": "#E8615B",
    "none": "#4A5164",
}


def render_retry_dots(retries: int, max_retries: int = 2) -> str:
    dots = []
    for i in range(max_retries):
        filled = i < retries
        color = "#F2B84B" if filled else "#2A3040"
        dots.append(f'<span style="display:inline-block;width:8px;height:8px;'
                    f'border-radius:50%;background:{color};margin-right:4px;"></span>')
    return "".join(dots)


def render_confidence_dot(label: str) -> str:
    color = CONFIDENCE_DOT_COLOR.get(label, "#4A5164")
    return (f'<span style="display:inline-block;width:9px;height:9px;'
            f'border-radius:50%;background:{color};margin-right:6px;'
            f'box-shadow:0 0 6px {color}66;"></span>')


def handle_upload(pdf_file):
    if pdf_file is None:
        return "⚠ NO FILE — select a PDF before ingesting"
    try:
        start = time.time()
        collection = ingest_pdf(pdf_file.name, collection_name=COLLECTION_NAME)
        invalidate_bm25_cache(COLLECTION_NAME)
        elapsed = time.time() - start
        return (
            f'<span style="color:#4FD1C5;">●</span> INGESTED in {elapsed:.1f}s &nbsp;·&nbsp; '
            f'{collection.count()} chunks indexed &nbsp;·&nbsp; ready for questions'
        )
    except Exception as e:
        return f'<span style="color:#E8615B;">●</span> INGEST FAILED — {e}'


def handle_question(query, history):
    if not query.strip():
        return history, "", "", ""

    try:
        start = time.time()
        result = ask(query, collection_name=COLLECTION_NAME)
        elapsed = time.time() - start

        answer = result["final_output"]
        history = history + [
            {"role": "user", "content": query},
            {"role": "assistant", "content": answer},
        ]

        sources_md = "\n\n".join(
            f"**[{i+1}]** `{c['source']}` — page {c['page']} "
            f"&nbsp;·&nbsp; rerank score `{c['rerank_score']:.3f}`\n\n"
            f"> {c['text'][:250]}{'...' if len(c['text']) > 250 else ''}"
            for i, c in enumerate(result["chunks"])
        ) or "_no chunks retrieved_"

        conf = result["confidence"]
        debug_md = (
            f"{render_confidence_dot(conf['label'])}"
            f"**CONFIDENCE** &nbsp;`{conf['label'].upper()}`"
            f" &nbsp;·&nbsp; best rerank score `{conf['best_rerank_score']}`\n\n"
            f"**VERDICT** &nbsp;`{(result['verdict'] or '—').upper()}`\n\n"
            f"**RETRIES** &nbsp;{render_retry_dots(result['retry_count'])}"
            f"&nbsp;{result['retry_count']} / 2\n\n"
            f"**CRITIC NOTE** &nbsp;{result['critique_reason'] or '—'}\n\n"
            f"**LATENCY** &nbsp;`{elapsed:.2f}s`\n\n"
            f"**FINAL QUERY** &nbsp;`{result['query']}`"
        )

        return history, "", sources_md, debug_md

    except Exception as e:
        error_msg = f"❌ Error: {e}"
        history = history + [
            {"role": "user", "content": query},
            {"role": "assistant", "content": error_msg},
        ]
        return history, "", "", traceback.format_exc()


CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
    --bg: #1A1E26;
    --panel: #262B35;
    --panel-border: #363D4A;
    --teal: #4FD1C5;
    --amber: #F2B84B;
    --coral: #E8615B;
    --text: #EDEFF3;
    --text-muted: #8B92A3;
}

.gradio-container {
    background: var(--bg) !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    color: var(--text) !important;
    max-width: 1100px !important;
    width: 100% !important;
    margin: 0 auto !important;
}

html, body, gradio-app, .app {
    background: var(--bg) !important;
}

#console-header {
    display: flex; align-items: center; gap: 10px;
    padding: 14px 18px; margin-bottom: 4px;
    background: var(--panel);
    border: 1px solid var(--panel-border);
    border-radius: 10px;
}
#console-header .status-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--teal);
    box-shadow: 0 0 8px var(--teal);
    animation: pulse 2.4s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}
#console-header .title {
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600; font-size: 14px; letter-spacing: 0.06em;
    color: var(--text);
}
#console-header .subtitle {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px; color: var(--text-muted); margin-left: auto;
}

.gr-block, .gr-box, .block {
    background: var(--panel) !important;
    border-color: var(--panel-border) !important;
    border-radius: 10px !important;
}

#upload-status {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important;
    color: var(--text-muted) !important;
    padding: 4px 2px !important;
}

button.primary {
    background: var(--teal) !important;
    color: #0B1013 !important;
    border: none !important;
    font-weight: 600 !important;
}
button.primary:hover {
    background: #6BE0D5 !important;
}

#chatbot {
    border: 1px solid var(--panel-border) !important;
    border-radius: 10px !important;
}
#chatbot .message.bot {
    border-left: 2px solid var(--teal) !important;
}

#chatbot .message-row.user-row .message-bubble-border {
    background: #2E5F58 !important;
    border-radius: 12px 12px 2px 12px !important;
    border: none !important;
}
#chatbot .message-row.bot-row .message-bubble-border {
    background: #31394A !important;
    border-radius: 12px 12px 12px 2px !important;
    border: none !important;
}

textarea, input[type="text"] {
    background: #151922 !important;
    color: var(--text) !important;
    border-color: var(--panel-border) !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
}

.label-wrap span, label span {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
    color: var(--text-muted) !important;
}

#sources-panel, #debug-panel {
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 13px !important;
}
#sources-panel code, #debug-panel code {
    font-family: 'IBM Plex Mono', monospace !important;
    background: #0F1319 !important;
    color: var(--teal) !important;
    padding: 1px 5px !important;
    border-radius: 4px !important;
    font-size: 12px !important;
}
#debug-panel strong {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 0.03em !important;
    color: var(--text-muted) !important;
}

#sources-panel li, #debug-panel li {
    color: var(--text) !important;
}
#sources-panel blockquote {
    border-left: 2px solid var(--panel-border) !important;
    padding-left: 10px !important;
    color: var(--text-muted) !important;
}
#sources-panel strong, #debug-panel strong {
    color: var(--teal) !important;
}

#intro-text p {
    color: var(--text) !important;
    font-size: 13px !important;
    line-height: 1.6 !important;
}

#intro-text strong {
    color: var(--teal) !important;
    font-weight: 600 !important;
}
"""

HEADER_HTML = """
<div id="console-header">
    <span class="status-dot"></span>
    <span class="title">SELF-HEALING RAG</span>
    <span class="subtitle">hybrid search &nbsp;·&nbsp; cross-encoder rerank &nbsp;·&nbsp; critic-agent retry loop</span>
</div>
"""

with gr.Blocks(title="Self-Healing RAG") as demo:
    gr.HTML(HEADER_HTML)

    gr.Markdown(
        "Upload a PDF, then ask questions. Unlike plain RAG, this pipeline "
        "runs a **critic agent** that checks whether the answer is actually "
        "grounded in the retrieved chunks — if not, it **reformulates the "
        "query and retries** (up to 2x) before ever admitting it doesn't know. "
        "Expand **Diagnostics** below each answer to watch this happen live.",
        elem_id="intro-text",
    )

    with gr.Row():
        pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
        upload_btn = gr.Button("Ingest", variant="primary", scale=0)
    upload_status = gr.HTML(elem_id="upload-status")

    upload_btn.click(handle_upload, inputs=pdf_input, outputs=upload_status)

    chatbot = gr.Chatbot(label="Chat", elem_id="chatbot", height=380)
    query_input = gr.Textbox(
        label="Question", placeholder="What does this document say about...?"
    )
    ask_btn = gr.Button("Ask", variant="primary")

    with gr.Accordion("Sources used", open=False):
        sources_panel = gr.Markdown(elem_id="sources-panel")

    with gr.Accordion("Diagnostics", open=False):
        debug_panel = gr.Markdown(elem_id="debug-panel")

    ask_btn.click(
        handle_question,
        inputs=[query_input, chatbot],
        outputs=[chatbot, query_input, sources_panel, debug_panel],
    )
    query_input.submit(
        handle_question,
        inputs=[query_input, chatbot],
        outputs=[chatbot, query_input, sources_panel, debug_panel],
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Base(), css=CUSTOM_CSS)
