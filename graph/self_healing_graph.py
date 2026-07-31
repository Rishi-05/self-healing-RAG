import json
import os
import sys
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langgraph import graph
from langgraph.graph import StateGraph, END

sys.path.append(str(Path(__file__).parent.parent / "ingestion"))
from retrieve_generate import retrieve, generate_answer, build_context_block, compute_confidence  # noqa: E402
from llm_client import chat_completion  # noqa: E402

load_dotenv()

MAX_RETRIES = 2


# ---------------------------------------------------------------------
# Graph state — this dict is passed between every node
# ---------------------------------------------------------------------
class RAGState(TypedDict):
    original_query: str
    query: str                
    sub_queries: list           # decomposed sub-questions for compound queries
    collection_name: str
    chunks: list
    answer: str
    verdict: str                # "grounded" | "not_grounded"
    critique_reason: str
    retry_count: int
    confidence: dict
    final_output: str


#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------

from retrieve_generate import retrieve, generate_answer, build_context_block, get_all_summaries  # noqa: E402

META_QUERY_PHRASES = (
    "what is this document about",
    "what's this document about",
    "tell me about this document",
    "tell me about this pdf",
    "summarize this document",
    "summarize this pdf",
    "give me an overview",
    "what is this pdf about",
    "what does this document cover",
)


def is_meta_query(query: str) -> bool:
    q = query.lower().strip()
    return any(phrase in q for phrase in META_QUERY_PHRASES)


def check_meta_node(state: RAGState) -> RAGState:
    return state  # pass-through, exists purely as a routing point


def summary_node(state: RAGState) -> RAGState:
    summaries = get_all_summaries(state["collection_name"])
    if not summaries:
        state["final_output"] = "No documents have been ingested into this collection yet."
    elif len(summaries) == 1:
        state["final_output"] = next(iter(summaries.values()))
    else:
        state["final_output"] = "\n\n".join(f"{src}: {summ}" for src, summ in summaries.items())

    state["chunks"] = []
    state["sub_queries"] = []
    state["query"] = state["original_query"]
    state["verdict"] = "summary"
    state["confidence"] = {"label": "n/a", "score": None, "best_rerank_score": None}
    state["critique_reason"] = (
        "Answered from a summary generated at ingestion time; the retrieval "
        "and critic loop is skipped for whole-document questions."
    )
    return state


def route_entry(state: RAGState) -> str:
    return "summary" if is_meta_query(state["original_query"]) else "decompose"

#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------

def decompose_node(state: RAGState) -> RAGState:
    decompose_prompt = (
        "Read the following question. If it asks for MULTIPLE distinct "
        "facts, or requires comparing/synthesizing information across "
        "different topics or sections, split it into 2-3 focused "
        "sub-questions that together would answer it.\n"
        "IMPORTANT: if the question COMPARES or DIFFERENTIATES BETWEEN two or "
        "more named things (e.g. 'compare X and Y', 'differentiate between A "
        "and B'), split by ENTITY, not by dimension — one sub-question per "
        "thing being compared, each asking for ALL relevant details about "
        "that one thing. Do NOT create sub-questions that still mention both "
        "things together, since a single document section rarely discusses "
        "two compared items in the same place.\n"
        "Separately: if the question asks about MULTIPLE ATTRIBUTES of a SINGLE "
        "named thing (e.g. 'What is the structure, mobility, and survival "
        "strategy of conidia?' — one entity, three attributes), do NOT split by "
        "attribute. Output it UNCHANGED as one question — a document section "
        "usually covers one entity's details together in one place, and "
        "splitting by attribute often uses wording that doesn't match how the "
        "source text is actually phrased."
        "When splitting by entity for a comparison, keep each sub-question broad "
        "(e.g. 'What are zoospores?') rather than repeating all the comparison "
        "attributes in each one — this matches short, self-contained document "
        "sections better than an attribute-heavy phrasing does."
        "If it is already a single, focused question, output it UNCHANGED as "
        "the only line. Output ONLY the sub-question(s), one per line, no "
        f"numbering, no explanation.\n\nQuestion: {state['original_query']}"
    )
    response, model_used = chat_completion(
        messages=[{"role": "user", "content": decompose_prompt}],
        temperature=0.0,
    )
    sub_queries = [
        line.strip() for line in response.choices[0].message.content.strip().split("\n")
        if line.strip()
    ]
    state["sub_queries"] = sub_queries or [state["original_query"]]
    state["query"] = state["original_query"]
    return state

def retrieve_node(state: RAGState) -> RAGState:
    queries_to_run = state["sub_queries"] if state["retry_count"] == 0 else [state["query"]]
    merged: dict = {}
    for q in queries_to_run:
        for chunk in retrieve(q, state["collection_name"], top_k=6):
            key = (chunk["source"], chunk["page"], chunk["text"])
            if key not in merged or chunk["rerank_score"] > merged[key]["rerank_score"]:
                merged[key] = chunk
    chunks = sorted(merged.values(), key=lambda c: c["rerank_score"], reverse=True)[:8]
    state["chunks"] = chunks
    state["confidence"] = compute_confidence(chunks)
    return state


def generate_node(state: RAGState) -> RAGState:
    if not state["chunks"]:
        # Issue 6/9: nothing was retrieved — don't even call the LLM,
        # there's nothing to hallucinate from but also nothing to ground in.
        state["answer"] = "No relevant content was found in the collection for this question."
        state["critique_reason"] = "No chunks were retrieved for this query."
        return state
    answer = generate_answer(state["query"], state["chunks"])
    state["answer"] = answer
    if is_refusal(answer):
        state["critique_reason"] = "Generator itself reported the retrieved chunks were insufficient."
    return state


REFUSAL_PHRASES = (
    "does not contain enough information",
    "no relevant content was found",
)


def is_refusal(answer: str) -> bool:
    return any(phrase in answer.lower() for phrase in REFUSAL_PHRASES)


def critique_node(state: RAGState) -> RAGState:
    """
    A second LLM call acting as a strict fact-checker: does every claim
    in the answer actually trace back to the provided chunks?
    Returns structured JSON so the graph can branch on it reliably.
    """
    context_block = build_context_block(state["chunks"])

    critic_system_prompt = (
        "You are a strict fact-checking critic. You will be given the "
        "user's ORIGINAL question, a set of numbered context chunks, and an "
        "answer generated from those chunks. THREE separate checks must ALL "
        "pass for a 'grounded' verdict:\n"
        "1. RELEVANCE: the context chunks must actually be ABOUT the same "
        "topic as the original question.\n"
        "2. FAITHFULNESS: every claim in the answer must trace directly and "
        "exactly back to the chunks — no partial claims, no inferred "
        "details, no numbers that don't exactly match.\n"
        "3. SPECIFIC-FACT PRESENCE (most important — check this carefully): "
        "if the ORIGINAL question asks for a specific number, date, name, "
        "duration, or fact, that EXACT fact must be LITERALLY stated in the "
        "chunks. Being on the same general topic is NOT enough. Do not "
        "accept an answer that discusses a related but different fact (for "
        "example, the question asks about a 'false-start penalty' and the "
        "chunk only discusses 'foot faults' — these are NOT the same thing, "
        "even though both are volleyball violations). If the exact fact "
        "asked for is not literally present in the chunks, mark "
        "'not_grounded' even if the chunks are topically relevant and the "
        "answer sounds plausible.\n"
        "4. PARAPHRASE IS ACCEPTABLE: a fact counts as present if the same "
        "underlying number, name, or specific value appears in the source, "
        "even if the surrounding wording is paraphrased or rephrased (for "
        "example, a chunk saying 'X percent when grouped by event' DOES "
        "support an answer describing that as 'event-based accuracy of X "
        "percent' — this is a faithful paraphrase, not a fabrication). Only "
        "mark 'not_grounded' for a specific fact if the exact number, name, "
        "or value itself is absent, different, or contradicted by the "
        "source — not merely because the wording differs from the source's "
        "phrasing."
        "Mark 'not_grounded' if ANY of the three checks fail. "
        "Respond ONLY with valid JSON in this exact format: "
        '{"verdict": "grounded" or "not_grounded", "reason": "<one sentence>"}'
    )

    critic_user_prompt = (
        f"Original question: {state['original_query']}\n\n"
        f"Context chunks retrieved:\n{context_block}\n\n"
        f"Generated answer: {state['answer']}\n\n"
        "First check: are these chunks actually about the same topic as "
        "the ORIGINAL question above? Then check: is the answer fully "
        "faithful to the chunks? Respond with JSON only."
    )

    response, model_used = chat_completion(
        messages=[
            {"role": "system", "content": critic_system_prompt},
            {"role": "user", "content": critic_user_prompt},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )

    try:
        result = json.loads(response.choices[0].message.content)
        state["verdict"] = result.get("verdict", "not_grounded")
        state["critique_reason"] = result.get("reason", "")
    except (json.JSONDecodeError, AttributeError):
        # If the critic itself fails to return clean JSON, fail safe
        state["verdict"] = "not_grounded"
        state["critique_reason"] = "Critic response could not be parsed."

    return state


def reformulate_node(state: RAGState) -> RAGState:
    """
    Rewrites the query to try to pull back better chunks next attempt.
    Now sees the chunks that WERE retrieved (Issue 8 fix) so it can steer
    toward terminology that actually appears in the document, instead of
    guessing blind and potentially drifting further from a good match.
    """
    context_block = build_context_block(state["chunks"]) if state["chunks"] else "(no chunks retrieved)"

    reformulate_prompt = (
        f"The following question failed to get a grounded answer.\n"
        f"Reason: {state['critique_reason']}\n\n"
        f"Original question: {state['original_query']}\n\n"
        f"Chunks that WERE retrieved (but didn't fully answer the question):\n"
        f"{context_block}\n\n"
        "Rewrite the question as a single alternative search query, using "
        "terminology and phrasing that appears in the chunks above where "
        "relevant, so the next retrieval attempt is more likely to find the "
        "right section. Respond with ONLY the rewritten query, no explanation."
    )

    response, model_used = chat_completion(
        messages=[{"role": "user", "content": reformulate_prompt}],
        temperature=0.3,
    )

    state["query"] = response.choices[0].message.content.strip()
    state["retry_count"] += 1
    return state


def finalize_grounded_node(state: RAGState) -> RAGState:
    state["final_output"] = state["answer"]
    return state


def finalize_fallback_node(state: RAGState) -> RAGState:
    state["final_output"] = (
        "I don't have enough information in the provided document(s) to "
        "answer this question confidently. "
        f"(Attempted {state['retry_count']} query reformulation(s); "
        f"last critic note: {state['critique_reason']})"
    )
    return state


# ---------------------------------------------------------------------
# Conditional routing after generate — catches refusals before the critic
# ever sees them, so an honest "I don't know" never gets labeled "grounded"
# ---------------------------------------------------------------------
def route_after_generate(state: RAGState) -> str:
    if is_refusal(state["answer"]):
        if state["retry_count"] >= MAX_RETRIES:
            return "give_up"
        return "retry"
    return "critique"


# ---------------------------------------------------------------------
# Conditional routing after the critic node
# ---------------------------------------------------------------------
def route_after_critique(state: RAGState) -> str:
    if state["verdict"] == "grounded":
        return "grounded"
    if state["retry_count"] >= MAX_RETRIES:
        return "give_up"
    return "retry"


# ---------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------
def build_graph():
    graph = StateGraph(RAGState)
    
    graph.add_node("check_meta", check_meta_node)
    graph.add_node("summary", summary_node)
    
    graph.add_node("decompose", decompose_node)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("critique", critique_node)
    graph.add_node("reformulate", reformulate_node)
    graph.add_node("finalize_grounded", finalize_grounded_node)
    graph.add_node("finalize_fallback", finalize_fallback_node)

    graph.set_entry_point("check_meta")
    graph.add_conditional_edges(
        "check_meta", 
        route_entry, 
        {
            "summary": "summary", 
            "decompose": "decompose",
        },
    )
    graph.add_edge("decompose", "retrieve")
    graph.add_edge("retrieve", "generate")

    graph.add_conditional_edges(
        "generate",
        route_after_generate,
        {
            "critique": "critique",
            "retry": "reformulate",
            "give_up": "finalize_fallback",
        },
    )

    graph.add_conditional_edges(
        "critique",
        route_after_critique,
        {
            "grounded": "finalize_grounded",
            "retry": "reformulate",
            "give_up": "finalize_fallback",
        },
    )

    graph.add_edge("reformulate", "retrieve")   # the self-healing loop
    graph.add_edge("finalize_grounded", END)
    graph.add_edge("finalize_fallback", END)
    graph.add_edge("summary", END)

    return graph.compile()


rag_app = build_graph()


def ask(query: str, collection_name: str = "documents") -> dict:
    """Public entry point: run the full self-healing RAG graph on a query."""
    initial_state: RAGState = {
        "sub_queries": [],
        "original_query": query,
        "query": query,
        "collection_name": collection_name,
        "chunks": [],
        "answer": "",
        "verdict": "",
        "critique_reason": "",
        "retry_count": 0,
        "confidence": {"label": "none", "score": 0.0, "best_rerank_score": None},
        "final_output": "",
    }

    final_state = rag_app.invoke(initial_state)
    return final_state


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "What is this document about?"
    collection = sys.argv[2] if len(sys.argv) > 2 else "documents"
    result = ask(q, collection_name=collection)
    print("\n--- SUB-QUERIES ---")
    for sq in result["sub_queries"]:
        print(f"  - {sq}")
    print("\n--- FINAL OUTPUT ---")
    print(result["final_output"])
    print(f"\n--- retries used: {result['retry_count']} | verdict: {result['verdict']} "
          f"| confidence: {result['confidence']['label']} "
          f"(best rerank score {result['confidence']['best_rerank_score']}) ---")
