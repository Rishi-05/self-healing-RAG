# Self-Healing RAG

A retrieval-augmented question answering system that checks its own answers before returning them. Instead of trusting the first retrieval attempt, the pipeline runs a critic model against every generated answer, and if the answer is not fully grounded in the retrieved context, it rewrites the query and tries again, up to a bounded number of retries, before ever admitting it does not know.

The project is built around one core principle: keep the deterministic parts deterministic. Retrieval, reranking, and quality thresholds are handled with local, non-LLM components wherever possible. LLM calls are reserved for the steps that genuinely require language understanding: generation, critique, query reformulation, and query decomposition.

## How it works

The pipeline is implemented as a LangGraph state machine with the following flow:

```
decompose -> retrieve -> generate -> critique -> grounded -> done
                              |            |
                              |            +-> not grounded -> reformulate -> retrieve (retry)
                              |
                              +-> refusal (empty context) -> reformulate -> retrieve (retry)

After MAX_RETRIES is reached without a grounded answer -> fallback response
```

**Decompose.** The original question is checked for whether it asks for multiple distinct facts or requires comparing information across sections. If so, it is split into 2-3 focused sub-questions before retrieval. Simple, single-fact questions pass through unchanged.

**Retrieve.** Each query (the sub-questions on the first pass, or the reformulated query on a retry) is run through hybrid search:
- Vector search over locally computed embeddings (ChromaDB, cosine similarity)
- BM25 keyword search (pure algorithm, no model, no LLM call) over the same collection
- Results from both are merged and deduplicated
- A local cross-encoder reranks the merged candidate pool by actual query-relevance
- The top-scoring chunks are kept and a confidence label (high / medium / low / none) is computed from the best rerank score

**Generate.** The LLM is given only the retrieved chunks and instructed to answer strictly from them, citing each claim with a bracketed chunk number, and to say explicitly if the context is insufficient rather than guessing.

**Critique.** A second LLM call, acting as a strict fact-checker, is given the context and the generated answer and asked to verdict whether every claim in the answer is actually traceable to the retrieved chunks. It responds in structured JSON so the graph can route on it reliably.

**Reformulate.** If the critic marks the answer as not grounded, or if generation itself refused to answer (no chunks retrieved, or the generator reported the context was insufficient), the query is rewritten by the LLM using terminology drawn from the chunks that were actually retrieved, so the next retrieval pass is more likely to land on the right section of the document. This is bounded to a small number of retries; after that the pipeline returns an explicit fallback message rather than looping indefinitely or guessing.

## Tech stack

- **Orchestration:** LangGraph (state machine for the retrieve/generate/critique/retry loop)
- **LLM inference:** Groq API
- **Vector store:** ChromaDB (persistent, local)
- **Embeddings:** sentence-transformers, `all-MiniLM-L6-v2` (local, CPU, no API cost)
- **Keyword search:** BM25 via `rank-bm25` (local, no API cost)
- **Reranking:** cross-encoder, `ms-marco-MiniLM-L-6-v2` (local, CPU, no API cost)
- **PDF parsing:** pdfplumber
- **Chunking:** langchain-text-splitters (recursive character splitter)
- **UI:** Gradio

The only components that make network calls to a paid or rate-limited API are generation, critique, reformulation, and decomposition. Retrieval, reranking, and confidence scoring never call an LLM.

## Setup

**1. Clone and install dependencies**

```
git clone <repository-url>
cd self-healing-rag
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

**2. Configure your API key**

Copy `.env.example` to `.env` and add a free Groq API key:

```
GROQ_API_KEY=your_key_here
```

If you run into environment variable issues (particularly on Windows, where saving `.env` in some editors introduces a byte-order-mark that breaks `dotenv` parsing silently), run:

```
python check_env.py
```

**3. Run the app**

```
python app.py
```

This starts a local Gradio server. Upload a PDF, wait for ingestion to complete, then ask questions.

## Usage outside the UI

Ingest a document directly from the command line:

```
python ingestion/ingest.py path/to/file.pdf --collection my_docs
```

Ask a question through the plain (non-self-healing) retrieve-and-generate path:

```
python ingestion/retrieve_generate.py "your question" --collection my_docs
```

Ask a question through the full self-healing graph:

```
python graph/self_healing_graph.py "your question" my_docs
```

Debug a single question with verbose sub-query, retrieval, and grounding output:

```
python eval/debug_question.py "your question" my_docs
```

Inspect or clean up the vector store:

```
python ingestion/manage_store.py list
python ingestion/manage_store.py sources --collection my_docs
python ingestion/manage_store.py delete-source --collection my_docs --source some_file
python ingestion/manage_store.py delete-collection --collection my_docs
python ingestion/manage_store.py wipe-all
```

## Design decisions

**Why hybrid search instead of pure vector search.** Embedding-based retrieval is strong at semantic similarity but weaker on exact terms, numbers, and proper nouns. BM25 is the structural complement: a deterministic keyword-frequency algorithm that is strong exactly where vector search is weak. Running both independently and merging the candidate pools covers more retrieval failure modes than either alone.

**Why rerank instead of just concatenating both result sets.** Vector distance and BM25 score are not on a comparable scale, so there is no principled way to decide which candidate from which method is actually more relevant without a common yardstick. A cross-encoder scores the query and each candidate chunk jointly, producing one relevance score the merged pool can be sorted and thresholded on.

**Why the quality gate thresholds on rerank score, not raw distance.** Cosine distance is bounded and has a fixed, interpretable range. Cross-encoder scores are unbounded logits with no fixed range, so any threshold has to be calibrated against real data rather than assumed from a formula.

**Why the critic runs after generation instead of trusting the generator's own refusal.** The generator is instructed to say when it lacks sufficient context, and that refusal is treated as a legitimate outcome and routed to retry without ever being sent to the critic — an honest "I don't know" should never be mislabeled as ungrounded by a second model. Everything else the generator does produce is independently checked, since a generator can be confidently wrong in ways it will not admit to itself.

**Why retries are bounded.** Unbounded LLM loops are the most reliable way to burn tokens without improving output. The retry count is capped, and after the cap is reached, the pipeline returns an explicit statement that it could not find a confident answer, along with how many reformulation attempts were made and the critic's last note, rather than either looping forever or fabricating a confident-sounding response.

## Known limitations

- ChromaDB storage is local and persistent by default; if deployed on ephemeral hosting (for example, a free-tier hosted Space), the vector store does not survive a restart and documents need to be re-ingested.
- The current deployment uses a single shared collection rather than per-session isolation, so in a multi-user deployment, documents ingested by one user are searchable by all users of the same running instance.
- Re-ingesting a PDF with the same filename but changed content is skipped by default (deduplication is filename-based, not content-hash-based) unless `--force` is passed.
- Rerank score thresholds and confidence-label boundaries are configured from general defaults and should be recalibrated against your own documents and query patterns for production use.
