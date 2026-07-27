import os
import re
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from llm_client import chat_completion

load_dotenv()

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CHROMA_PERSIST_DIR = str(Path(__file__).parent.parent / "data" / "chroma_store")

# Reranker score threshold — the cross-encoder outputs an unbounded relevance
# logit (roughly: >2 = strong match, 0 to 2 = plausible, <0 = weak/irrelevant).
# Chunks scoring below this after reranking are dropped rather than passed
# to the LLM. Tune against your own golden-dataset results.
RERANK_THRESHOLD = 0.0

# How many candidates each retriever (vector + BM25) pulls BEFORE fusion and
# reranking narrow it down to top_k. Wider net = better recall going into
# the rerank step, at the cost of a slightly larger cross-encoder batch.
CANDIDATE_POOL_SIZE = 15

# Reciprocal Rank Fusion constant — standard default, dampens the influence
# of any single retriever's exact rank position.
RRF_K = 60

# Cache these once at import time instead of recreating on every retrieve()
# call — loading the embedding model and reconnecting the DB client repeatedly
# was slow and unnecessary (Issues 10/11).
_chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
_embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
_collection_cache: dict = {}


def get_collection(collection_name: str = "documents"):
    if collection_name not in _collection_cache:
        _collection_cache[collection_name] = _chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=_embed_fn,
        )
    return _collection_cache[collection_name]


_bm25_cache: dict = {}   # collection_name -> (BM25Okapi, [chunk_dict, ...])
_cross_encoder = None    # lazy-loaded singleton, ~80MB model


def _get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder(RERANK_MODEL)
    return _cross_encoder


def _tokenize(text: str) -> list[str]:
    """
    Regex word-extraction instead of naive .lower().split() — strips
    punctuation so 'sandstone.' and 'sandstone' tokenize identically,
    which matters for BM25 matching on chunks ending mid-fact.
    """
    return re.findall(r"\w+", text.lower())


def _build_bm25_index(collection_name: str):
    """
    Pulls the full corpus for a collection out of ChromaDB and builds a
    BM25 keyword index in memory. Cached per collection — call
    invalidate_bm25_cache() after ingesting new documents into a
    collection that's already been queried, or the keyword index will
    be stale for the lifetime of a long-running process (e.g. app.py).
    """
    collection = get_collection(collection_name)
    data = collection.get(include=["documents", "metadatas"])

    corpus_chunks = []
    for i in range(len(data["ids"])):
        corpus_chunks.append({
            "id": data["ids"][i],
            "text": data["documents"][i],
            "source": data["metadatas"][i]["source"],
            "page": data["metadatas"][i]["page"],
        })

    tokenized_corpus = [_tokenize(c["text"]) for c in corpus_chunks]
    bm25 = BM25Okapi(tokenized_corpus) if tokenized_corpus else None

    _bm25_cache[collection_name] = (bm25, corpus_chunks)
    return bm25, corpus_chunks


def invalidate_bm25_cache(collection_name: str):
    """Call this after ingesting new docs into a collection mid-process
    (e.g. from app.py after a user uploads a PDF) so the next retrieve()
    rebuilds the keyword index instead of using a stale one."""
    _bm25_cache.pop(collection_name, None)


def _vector_search(query: str, collection_name: str, pool_size: int) -> list[dict]:
    collection = get_collection(collection_name)
    results = collection.query(query_texts=[query], n_results=pool_size)
    candidates = []
    for i in range(len(results["documents"][0])):
        candidates.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "page": results["metadatas"][0][i]["page"],
        })
    return candidates


def _bm25_search(query: str, collection_name: str, pool_size: int) -> list[dict]:
    if collection_name not in _bm25_cache:
        _build_bm25_index(collection_name)
    bm25, corpus_chunks = _bm25_cache[collection_name]

    if bm25 is None or not corpus_chunks:
        return []

    scores = bm25.get_scores(_tokenize(query))
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [corpus_chunks[i] for i in ranked_indices[:pool_size]]


def _reciprocal_rank_fusion(vector_results: list[dict], bm25_results: list[dict]) -> list[dict]:
    """
    Merges two ranked lists into one, using Reciprocal Rank Fusion: each
    chunk's fused score = sum of 1/(RRF_K + rank) across whichever
    retriever(s) it appeared in. Chunks found by BOTH retrievers naturally
    rank higher than chunks found by only one.
    """
    scores: dict = {}
    chunk_lookup: dict = {}

    for rank, chunk in enumerate(vector_results):
        scores[chunk["id"]] = scores.get(chunk["id"], 0) + 1 / (RRF_K + rank)
        chunk_lookup[chunk["id"]] = chunk

    for rank, chunk in enumerate(bm25_results):
        scores[chunk["id"]] = scores.get(chunk["id"], 0) + 1 / (RRF_K + rank)
        chunk_lookup.setdefault(chunk["id"], chunk)

    fused_ids = sorted(scores, key=scores.get, reverse=True)
    return [chunk_lookup[cid] for cid in fused_ids]


def retrieve(query: str, collection_name: str = "documents", top_k: int = 4,
             rerank_threshold: float = RERANK_THRESHOLD) -> list[dict]:
    """
    Hybrid retrieval: runs vector search AND BM25 keyword search in
    parallel, fuses the two ranked lists (Reciprocal Rank Fusion), then
    reranks the fused candidate pool with a cross-encoder for actual
    query-chunk relevance before returning the final top_k.

    Returns chunks as [{text, source, page, score}, ...] — score is the
    cross-encoder relevance score (HIGHER is better, unlike the old
    cosine distance where lower was better). Chunks scoring below
    rerank_threshold are dropped rather than passed to the LLM as noise.
    """
    vector_candidates = _vector_search(query, collection_name, CANDIDATE_POOL_SIZE)
    bm25_candidates = _bm25_search(query, collection_name, CANDIDATE_POOL_SIZE)

    if not vector_candidates and not bm25_candidates:
        return []

    fused = _reciprocal_rank_fusion(vector_candidates, bm25_candidates)

    cross_encoder = _get_cross_encoder()
    pairs = [(query, c["text"]) for c in fused]
    ce_scores = cross_encoder.predict(pairs)

    reranked = sorted(zip(fused, ce_scores), key=lambda x: x[1], reverse=True)

    chunks = []
    for chunk, score in reranked[:top_k]:
        if score < rerank_threshold:
            continue
        chunks.append({
            "text": chunk["text"],
            "source": chunk["source"],
            "page": chunk["page"],
            "rerank_score": round(float(score), 3),
        })
    return chunks


def compute_confidence(chunks: list[dict]) -> dict:
    """
    Cheap, no-extra-API-call confidence signal based on the best (highest)
    reranked relevance score. This is a proxy, not a calibrated
    probability — thresholds here are heuristic for ms-marco-MiniLM-L-6-v2
    and should be tuned against your own golden-dataset results.
    """
    if not chunks:
        return {"label": "none", "score": 0.0, "best_rerank_score": None}

    best_score = max(c["rerank_score"] for c in chunks)

    if best_score >= 2.0:
        label = "high"
    elif best_score >= 0.0:
        label = "medium"
    else:
        label = "low"

    return {"label": label, "score": round(best_score, 3), "best_rerank_score": round(best_score, 3)}


def build_context_block(chunks: list[dict]) -> str:
    """Formats chunks with [1], [2]... markers so the LLM can cite them."""
    lines = []
    for i, c in enumerate(chunks, start=1):
        lines.append(f"[{i}] (source: {c['source']}, page {c['page']})\n{c['text']}")
    return "\n\n".join(lines)


def generate_answer(query: str, chunks: list[dict]) -> str:
    """
    Calls the LLM with a strict grounding instruction: answer ONLY from
    the provided chunks, cite using [n], and explicitly say if the
    context doesn't contain the answer.
    """
    context_block = build_context_block(chunks)

    system_prompt = (
        "You are a precise research assistant. Answer the user's question "
        "using ONLY the numbered context chunks below. "
        "Cite every claim with its chunk number like [1] or [2]. "
        "If the context does not contain enough information to answer, "
        "say exactly: 'The provided context does not contain enough "
        "information to answer this question.' Do not use outside knowledge."
    )

    user_prompt = f"Context:\n{context_block}\n\nQuestion: {query}\n\nAnswer:"

    response, model_used = chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,   # low temperature -> less improvisation, more grounding
    )

    return response.choices[0].message.content


def answer_question(query: str, collection_name: str = "documents", top_k: int = 4) -> dict:
    """End-to-end: retrieve + generate. Returns answer + chunks + confidence."""
    chunks = retrieve(query, collection_name, top_k)
    confidence = compute_confidence(chunks)

    if not chunks:
        return {
            "query": query,
            "answer": "No relevant content was found in the collection for this question.",
            "chunks": [],
            "confidence": confidence,
        }

    answer = generate_answer(query, chunks)
    return {"query": query, "answer": answer, "chunks": chunks, "confidence": confidence}


if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "What is this document about?"
    collection = sys.argv[2] if len(sys.argv) > 2 else "documents"
    result = answer_question(q, collection_name=collection)
    print("\n--- ANSWER ---")
    print(result["answer"])
    print(f"\n--- CONFIDENCE: {result['confidence']['label']} "
          f"(best rerank score {result['confidence']['best_rerank_score']}) ---")
    print("\n--- SOURCES USED ---")
    for i, c in enumerate(result["chunks"], start=1):
        print(f"[{i}] {c['source']} (page {c['page']}) — rerank score {c['rerank_score']:.3f}")
