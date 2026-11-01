"""
Single place that owns the Chroma client and the embedding model, so ingest.py
and retrieve_generate.py share ONE client and ONE loaded model per process
(before, each ingest call reloaded the sentence-transformer from scratch).
"""
import threading
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # ~80MB, CPU, free
CHROMA_PERSIST_DIR = str(Path(__file__).parent.parent / "data" / "chroma_store")

_lock = threading.RLock()
_client = None
_embed_fn = None
_collections: dict = {}


def get_client():
    global _client
    with _lock:
        if _client is None:
            _client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        return _client


def get_embed_fn():
    global _embed_fn
    with _lock:
        if _embed_fn is None:
            _embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=EMBEDDING_MODEL
            )
        return _embed_fn


def get_collection(collection_name: str = "documents"):
    with _lock:
        if collection_name not in _collections:
            _collections[collection_name] = get_client().get_or_create_collection(
                name=collection_name,
                embedding_function=get_embed_fn(),
                metadata={"hnsw:space": "cosine"},
            )
        return _collections[collection_name]


def list_sources(collection_name: str = "documents") -> list[str]:
    """Distinct document names currently stored in a collection."""
    data = get_collection(collection_name).get(include=["metadatas"])
    return sorted({m["source"] for m in data["metadatas"] if m and "source" in m})
