import argparse
import hashlib
import os
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter

EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # 80MB, runs on CPU, no cost
CHROMA_PERSIST_DIR = str(Path(__file__).parent.parent / "data" / "chroma_store")
CHUNK_SIZE = 800          # characters per chunk
CHUNK_OVERLAP = 150       # overlap so context isn't cut mid-idea


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


def chunk_pages(pages: list[dict], source_name: str) -> list[dict]:
    """
    Splits each page's text into overlapping chunks and attaches
    metadata (source file, page number) needed later for citations.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for page in pages:
        page_chunks = splitter.split_text(page["text"])
        for chunk_text in page_chunks:
            # Deterministic ID so re-ingesting the same PDF doesn't duplicate.
            # Hash the FULL chunk text (not just a prefix) to avoid collisions
            # between different chunks that happen to start identically.
            chunk_id = hashlib.md5(
                f"{source_name}-{page['page_number']}-{chunk_text}".encode()
            ).hexdigest()
            chunks.append({
                "id": chunk_id,
                "text": chunk_text,
                "metadata": {
                    "source": source_name,
                    "page": page["page_number"],
                },
            })
    return chunks


def source_already_ingested(collection, source_name: str) -> bool:
    """Checks if any chunks from this source filename already exist."""
    existing = collection.get(where={"source": source_name}, limit=1)
    return len(existing["ids"]) > 0


def ingest_pdf(pdf_path: str, collection_name: str = "documents", force: bool = False):
    source_name = Path(pdf_path).stem

    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    collection = client.get_or_create_collection(
        name=collection_name, embedding_function=embed_fn, metadata={"hnsw:space": "cosine"}
    )

    if not force and source_already_ingested(collection, source_name):
        print(f"⚠️  '{source_name}' already has chunks in '{collection_name}'. "
              f"Skipping re-ingestion. Use --force to re-ingest anyway.")
        return collection

    print(f"[1/3] Extracting text from {pdf_path} ...")
    pages = extract_text_from_pdf(pdf_path)
    print(f"      -> {len(pages)} pages with extractable text")

    print(f"[2/3] Chunking (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}) ...")
    chunks = chunk_pages(pages, source_name)
    print(f"      -> {len(chunks)} chunks")

    print(f"[3/3] Embedding + storing in ChromaDB collection '{collection_name}' ...")
    collection.upsert(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
    )
    print(f"      -> Done. Collection now has {collection.count()} total chunks.")

    return collection


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest a PDF into the RAG vector store.")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("--collection", default="documents", help="ChromaDB collection name")
    parser.add_argument("--force", action="store_true", help="Re-ingest even if source already exists")
    args = parser.parse_args()

    ingest_pdf(args.pdf_path, args.collection, force=args.force)
