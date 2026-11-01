import json
import os
import threading
from pathlib import Path
 
from llm_client import chat_text
from store import get_collection, list_sources
 
SUMMARY_STORE_PATH = Path(__file__).parent.parent / "data" / "summaries.json"
MAX_SAMPLE_CHUNKS = 14
MAX_SAMPLE_CHARS = 14000
 
_lock = threading.Lock()
def _key(collection_name: str, source: str) -> str:
    return f"{collection_name}::{source}"
 
 
def _load() -> dict:
    try:
        return json.loads(SUMMARY_STORE_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
 
 
def _save(data: dict) -> None:
    SUMMARY_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = SUMMARY_STORE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, SUMMARY_STORE_PATH)   # atomic: never leaves a half-written file
 
 
def _sample_evenly(items: list, k: int) -> list:
    """k items spread across the whole list, in order (always includes first and last)."""
    n = len(items)
    if n <= k:
        return items
    idx = sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})
    return [items[i] for i in idx]
 
 
def _chunk_texts(collection_name: str, source: str) -> list[str]:
    data = get_collection(collection_name).get(
        where={"source": source}, include=["documents", "metadatas"]
    )
    order = sorted(range(len(data["ids"])), key=lambda i: ((data["metadatas"][i] or {}).get("page", 0), i))
    return [data["documents"][i] for i in order]
 
 
def generate_summary(texts: list[str], source: str) -> str:
    sample = "\n\n".join(_sample_evenly(texts, MAX_SAMPLE_CHUNKS))[:MAX_SAMPLE_CHARS]
    system = (
        "You summarize a document from excerpts sampled in order across it, so "
        "some parts may be skipped. Reply in exactly this format:\n"
        "**What it is:** <one line: document type and subject>\n"
        "**Overview:** <3-5 sentences>\n"
        "**Key points:**\n- <point>\n- <point>\n(4-6 bullets)\n"
        "Use only what the excerpts support. Do not invent sections, names or numbers."
    )
    return chat_text(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"Document: {source}\n\nExcerpts:\n\n{sample}"},
        ],
        temperature=0.2,
    )
 
 
def get_or_create_summary(collection_name: str, source: str) -> str:
    key = _key(collection_name, source)
    with _lock:
        cached = _load().get(key)
    if cached:
        return cached
 
    texts = _chunk_texts(collection_name, source)
    if not texts:
        return ""
    summary = generate_summary(texts, source)
    if summary:
        with _lock:
            data = _load()
            data[key] = summary
            _save(data)
    return summary
 
 
def drop_summary(collection_name: str, source: str) -> None:
    with _lock:
        data = _load()
        if data.pop(_key(collection_name, source), None) is not None:
            _save(data)
 
 
def get_summaries(collection_name: str, source: str | None = None) -> dict:
    """{source: summary} for one document, or for every document in the collection."""
    sources = [source] if source else list_sources(collection_name)
    out = {}
    for s in sources:
        text = get_or_create_summary(collection_name, s)
        if text:
            out[s] = text
    return out
