---
title: Self-Healing RAG
emoji: 🔁
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# 🔁 Self-Healing RAG

A Retrieval-Augmented Generation system that doesn't just retrieve-and-generate —
it **critiques its own output** and retries when the answer isn't grounded in the
source document, instead of hallucinating.

## How it works

1. **Retrieve** — embed the query locally (sentence-transformers, free) and pull
   top-k chunks from a ChromaDB vector store
2. **Generate** — an LLM (Groq, free tier) drafts an answer using only those chunks
3. **Critique** — a second LLM call fact-checks the answer against the retrieved
   chunks and returns a grounded / not-grounded verdict
4. **Self-heal** — if not grounded, the query is reformulated and the loop retries
   (max 2x) before falling back to an honest "I don't have enough information"
   instead of making something up

Built with **LangGraph** as a cyclical state machine (not a linear chain) — see
`graph/self_healing_graph.py`.

## Stack (100% free tier)

| Component | Tool |
|---|---|
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`), runs locally |
| Vector DB | ChromaDB, self-hosted |
| LLM (generation + critic) | Groq API free tier (Llama 3.3 70B) |
| Orchestration | LangGraph |
| UI + hosting | Gradio on Hugging Face Spaces |

## Running locally

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Get a free key at https://console.groq.com
echo "GROQ_API_KEY=your_key_here" > .env

python app.py
```

## Deploying to Hugging Face Spaces (free)

1. Create a new Space at https://huggingface.co/new-space
   - SDK: **Gradio**
   - Hardware: **CPU basic** (free)
2. Push this folder's contents to the Space's git repo (or upload via the web UI)
3. In the Space settings → **Repository secrets**, add `GROQ_API_KEY` with your
   free Groq key — never commit `.env` to the repo
4. The Space will build automatically and give you a public URL

## Known limitation

HF Spaces free tier storage is **ephemeral** — if the Space restarts, the
ChromaDB store is wiped. This is fine for a demo (users re-upload a PDF per
session) but would need persistent storage or an external vector DB (e.g. a
free Qdrant Cloud cluster) for a "real" deployment.

## Project structure

```
self-healing-rag/
├── app.py                       # Gradio UI + entry point (HF Spaces reads this)
├── requirements.txt
├── ingestion/
│   ├── ingest.py                 # PDF -> chunks -> embeddings -> ChromaDB
│   └── retrieve_generate.py      # plain retrieve + generate (building block)
├── graph/
│   └── self_healing_graph.py     # LangGraph: retrieve -> generate -> critique -> retry
└── data/
    └── chroma_store/              # persisted vector DB (gitignored)
```
