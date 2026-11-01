import argparse
import hashlib
from pathlib import Path

import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter

from store import get_collection
from summaries import drop_summary, get_or_create_summary

CHUNK_SIZE = 800          # characters per chunk
CHUNK_OVERLAP = 150       # overlap so context isn't cut mid-idea
UPSERT_BATCH = 256        # keeps big PDFs under Chroma's max batch size


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """
    Returns a list of {page_number, text} dicts — one per PDF page.
    Uses pdfplumber (layout-aware) instead of pypdf, since pdfplumber
    correctly reconstructs reading order for multi-column layouts
    (e.g. resumes, invoices) that pypdf tends to jumble.
    """
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append({"page_number": i + 1, "text": text})
    return pages


def chunk_pages(pages: list[dict], source_name: str, doc_hash: str = "") -> list[dict]:
    """
    Splits each page's text into overlapping chunks and attaches the metadata
    (source file, page number, file hash) needed for citations and scoping.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for page in pages:
        for idx, chunk_text in enumerate(splitter.split_text(page["text"])):
            # Deterministic ID so re-ingesting the same PDF doesn't duplicate.
            # The index is part of the hash: identical text repeated on one page
            # (headers, boilerplate) used to produce duplicate IDs and crash upsert.
            chunk_id = hashlib.md5(
                f"{source_name}-{page['page_number']}-{idx}-{chunk_text}".encode()
            ).hexdigest()
            chunks.append({
                "id": chunk_id,
                "text": chunk_text,
                "metadata": {
                    "source": source_name,
                    "page": page["page_number"],
                    "doc_hash": doc_hash,
                },
            })
    return chunks


def _resolve_source(collection, stem: str, doc_hash: str) -> tuple[str, bool]:
    """
    Decides the stored name for this file and whether it's already ingested.
      - same name + same content (or legacy chunks with no hash) -> already ingested
      - same name but DIFFERENT content -> stored as '<name>-<hash6>' so it never
        gets silently skipped or overwritten.
    """
    for candidate in (stem, f"{stem}-{doc_hash[:6]}"):
        got = collection.get(where={"source": candidate}, include=["metadatas"], limit=1)
        if not got["ids"]:
            return candidate, False
        stored = (got["metadatas"][0] or {}).get("doc_hash")
        if stored is None or stored == doc_hash:
            return candidate, True
    return f"{stem}-{doc_hash[:6]}", True


def ingest_pdf(pdf_path: str, collection_name: str = "documents",
               force: bool = False, display_name: str | None = None) -> dict:
    """
    Returns {"collection", "source", "chunks", "total_chunks", "skipped", "summary_ok"}.
    `display_name` lets the web app pass the user's original filename while the
    PDF itself sits in a temp file with a random name.
    """
    stem = display_name or Path(pdf_path).stem
    doc_hash = file_sha256(pdf_path)
    collection = get_collection(collection_name)

    source, exists = _resolve_source(collection, stem, doc_hash)

    if exists and not force:
        n = len(collection.get(where={"source": source}, include=[])["ids"])
        return {"collection": collection_name, "source": source, "chunks": n,
                "total_chunks": collection.count(), "skipped": True, "summary_ok": True}

    if exists and force:
        collection.delete(where={"source": source})
        drop_summary(collection_name, source)

    print(f"[1/4] Extracting text from {pdf_path} ...")
    pages = extract_text_from_pdf(pdf_path)
    if not pages:
        raise ValueError(
            "No extractable text found. This looks like a scanned/image-only PDF — "
            "it needs OCR before it can be indexed."
        )
    print(f"      -> {len(pages)} pages with extractable text")

    print(f"[2/4] Chunking (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}) ...")
    chunks = chunk_pages(pages, source, doc_hash)
    print(f"      -> {len(chunks)} chunks")

    print(f"[3/4] Embedding + storing in collection '{collection_name}' ...")
    for i in range(0, len(chunks), UPSERT_BATCH):
        batch = chunks[i:i + UPSERT_BATCH]
        collection.upsert(
            ids=[c["id"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[c["metadata"] for c in batch],
        )

    # The summary is a bonus at this point: if the LLM is rate-limited, ingestion
    # still succeeds and the summary is generated on the first overview question.
    print("[4/4] Generating document summary ...")
    summary_ok = True
    try:
        get_or_create_summary(collection_name, source)
    except Exception as e:
        summary_ok = False
        print(f"      -> summary skipped for now ({e})")

    return {"collection": collection_name, "source": source, "chunks": len(chunks),
            "total_chunks": collection.count(), "skipped": False, "summary_ok": summary_ok}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest a PDF into the RAG vector store.")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("--collection", default="documents", help="ChromaDB collection name")
    parser.add_argument("--force", action="store_true", help="Re-ingest even if source already exists")
    args = parser.parse_args()

    info = ingest_pdf(args.pdf_path, args.collection, force=args.force)
    if info["skipped"]:
        print(f"'{info['source']}' is already in '{info['collection']}'. Use --force to re-ingest.")
    else:
        print(f"Done: {info['chunks']} chunks from '{info['source']}' "
              f"({info['total_chunks']} total in '{info['collection']}').")
