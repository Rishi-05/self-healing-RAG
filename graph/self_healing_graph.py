import json
import re
import sys
from pathlib import Path
from typing import Optional, TypedDict

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

sys.path.append(str(Path(__file__).parent.parent / "ingestion"))
from retrieve_generate import retrieve, generate_answer, build_context_block, compute_confidence  # noqa: E402
from summaries import get_summaries  # noqa: E402
from llm_client import chat_completion  # noqa: E402

load_dotenv()
MAX_RETRIES = 2
WHOLE_DOC_MARKER = "__WHOLE_DOCUMENT__"

# ---------------------------------------------------------------------
# Graph state — this dict is passed between every node
# ---------------------------------------------------------------------
class RAGState(TypedDict):
    original_query: str
    query: str
    sub_queries: list           # decomposed sub-questions for compound queries
    collection_name: str
    source: Optional[str]       # restrict to ONE document ("this pdf"); None = all documents
    chunks: list
    answer: str
    verdict: str                # "grounded" | "not_grounded" | "summary"
    critique_reason: str
    retry_count: int
    confidence: dict
    final_output: str


_DOC = (r"(?:doc|docs|document|documents|pdf|pdfs|file|files|paper|papers|report|resume|cv|"
        r"article|thesis|book|manual|upload|uploads|uploaded file|attachment|text|notes)")
_THIS = r"(?:this|the|my|that|these|those|uploaded|attached|current)"
_THIS_DOC = rf"{_THIS}(?: (?:{_THIS}|whole|entire|full|complete))? {_DOC}"
_TAIL = (r"(?: (?:in|into|as|using|with|within) (?:\d+|one|two|three|four|five|six|a few|few|a|an|bullet|short|brief|"
         r"simple|plain|easy)(?: [a-z]+){0,4})?")

_LEAD_FILLER = re.compile(
    r"^(?:hey|hi|hello|ok|okay|so|please|pls|kindly|can you|could you|would you|will you|can u|could u|"
    r"i want you to|i need you to|i would like you to|i want to|lets|let us|just)\s+")
_TRAIL_FILLER = re.compile(
    r"\s+(?:please|pls|thanks|thank you|for me|to me|briefly|quickly|in detail|in short|in brief|"
    r"in simple terms|in simple words|in simple language|in easy words|simply|clearly|properly|"
    r"completely|fully|thoroughly|now|again)$")

_VERBS = (r"(?:summari[sz]e|explain|describe|analy[sz]e|review|interpret|break down|walk through|"
          r"go through|go over|run through|read|understand|make sense of|walk me through|take me through|"
          r"help me understand|help me make sense of|help me with)")

_META_PATTERNS = [re.compile(p) for p in (
    # "summarize this pdf", "explain the document", "analyze my resume in 5 bullets"
    rf"^{_VERBS} {_THIS_DOC}{_TAIL}$",
    # bare commands: "summarize", "summarize this", "explain this", "tldr", "sum it up"
    r"^(?:summari[sz]e|summari[sz]e (?:this|it)|explain|explain (?:this|everything)|summary|overview|recap|"
    r"tldr|tl dr|gist|sum up|sum it up|sum this up|in a nutshell)$",
    # "give me a summary", "a quick overview of this pdf"
    rf"^(?:give|show|write|provide|create|make|get)(?: me)? (?:a |an |the |some )?(?:quick |short |brief |detailed |"
    rf"full |one line |one paragraph |high level |rough )?(?:summary|overview|synopsis|recap|abstract|gist|rundown|"
    rf"outline)(?: of (?:{_THIS_DOC}|it|this))?{_TAIL}$",
    rf"^(?:summary|overview|synopsis|recap|abstract|gist|rundown|outline) of {_THIS_DOC}$",
    # "what is this pdf about", "what's in this file", "what does this document say"
    rf"^what(?: is|s) {_THIS_DOC}(?: all)?(?: about)?$",
    r"^what(?: is|s) (?:this|it) about$",
    r"^what(?: is|s) (?:this|it) all about$",
    rf"^what (?:does|did|do) {_THIS_DOC} (?:say|contain|cover|discuss|talk about|include|mean)$",
    rf"^what(?: is|s) in {_THIS_DOC}$",
    rf"^what kind of {_DOC} is (?:this|it)$",
    rf"^what type of {_DOC} is (?:this|it)$",
    r"^what did i (?:just )?upload(?:ed)?$",
    rf"^what(?: is|s) the (?:main (?:idea|point|topic|theme|purpose|subject)|gist|topic|purpose|subject)(?: of {_THIS_DOC})?$",
    rf"^(?:key|main) (?:points|takeaways|ideas|highlights|topics)(?: of {_THIS_DOC})?$",
    rf"^what are the (?:key|main) (?:points|takeaways|ideas|highlights|topics)(?: of {_THIS_DOC})?$",
    rf"^(?:tell|teach|brief|fill) me (?:about|on|in on) {_THIS_DOC}$",
    rf"^i (?:dont|do not) understand {_THIS_DOC}$",
    rf"^(?:what )?(?:should|do) i (?:need to )?know (?:about|from) {_THIS_DOC}$",
    rf"^{_THIS_DOC}(?: summary| overview)$",
)]


def _normalise(query: str) -> str:
    q = query.lower().replace("\u2019", "'")
    q = re.sub(r"tl\s*;\s*dr", "tldr", q)
    q = q.replace("'", "")
    q = re.sub(r"[^a-z0-9 ]+", " ", q)
    q = " ".join(q.split())
    prev = None
    while prev != q:                       # peel off stacked politeness: "hey can you ... please"
        prev = q
        q = _LEAD_FILLER.sub("", q)
        q = _TRAIL_FILLER.sub("", q)
    return q


def is_meta_query(query: str) -> bool:
    q = _normalise(query)
    return bool(q) and any(p.match(q) for p in _META_PATTERNS)


def check_meta_node(state: RAGState) -> RAGState:
    return state  # pass-through, exists purely as a routing point


def summary_node(state: RAGState) -> RAGState:
    try:
        summaries = get_summaries(state["collection_name"], state.get("source"))
        error = None
    except Exception as e:  # LLM down / rate-limited on every model
        summaries, error = {}, str(e)

    if error:
        state["final_output"] = f"I couldn't generate the summary right now ({error}). Please try again in a moment."
    elif not summaries:
        state["final_output"] = "No documents have been ingested yet — upload a PDF and click Ingest first."
    elif len(summaries) == 1:
        state["final_output"] = next(iter(summaries.values()))
    else:
        state["final_output"] = "\n\n".join(f"**{src}**\n{summ}" for src, summ in summaries.items())

    state["chunks"] = []
    state["sub_queries"] = []
    state["query"] = state["original_query"]
    state["verdict"] = "summary"
    state["confidence"] = {"label": "n/a", "score": None, "best_rerank_score": None}
    state["critique_reason"] = (
        "Answered from a document summary; the retrieval and critic loop is "
        "skipped for whole-document questions."
    )
    return state


def route_entry(state: RAGState) -> str:
    return "summary" if is_meta_query(state["original_query"]) else "decompose"


def route_after_decompose(state: RAGState) -> str:
    return "summary" if state["sub_queries"] == [WHOLE_DOC_MARKER] else "retrieve"


# ---------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------

def decompose_node(state: RAGState) -> RAGState:
    decompose_prompt = (
        "FIRST CHECK: if the question asks for a general summary, overview or "
        "explanation of the WHOLE document (for example 'what is this about', "
        "'explain this pdf', 'give me the gist') and names NO specific topic, "
        "section, fact or entity, output exactly "
        f"{WHOLE_DOC_MARKER} and nothing else.\n"
        "Otherwise, read the question. If it asks for MULTIPLE distinct "
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
        "source text is actually phrased.\n"
        "When splitting by entity for a comparison, keep each sub-question broad "
        "(e.g. 'What are zoospores?') rather than repeating all the comparison "
        "attributes in each one — this matches short, self-contained document "
        "sections better than an attribute-heavy phrasing does.\n"
        "If it is already a single, focused question, output it UNCHANGED as "
        "the only line. Output ONLY the sub-question(s), one per line, no "
        f"numbering, no explanation.\n\nQuestion: {state['original_query']}"
    )
    response, model_used = chat_completion(
        messages=[{"role": "user", "content": decompose_prompt}],
        temperature=0.0,
    )
    raw = re.sub(r"<think>.*?</think>", "", response.choices[0].message.content or "", flags=re.DOTALL)
    sub_queries = [line.strip() for line in raw.strip().split("\n") if line.strip()]
    if any(WHOLE_DOC_MARKER in sq for sq in sub_queries):
        sub_queries = [WHOLE_DOC_MARKER]
    state["sub_queries"] = sub_queries or [state["original_query"]]
    state["query"] = state["original_query"]
    return state


def retrieve_node(state: RAGState) -> RAGState:
    queries_to_run = state["sub_queries"] if state["retry_count"] == 0 else [state["query"]]
    merged: dict = {}
    for q in queries_to_run:
        for chunk in retrieve(q, state["collection_name"], top_k=6, source=state.get("source")):
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
        "answer this question confidently. If you wanted a general overview "
        "instead, try asking: \"summarize this document\". "
        f"(Attempted {state['retry_count']} query reformulation(s); "
        f"last critic note: {state['critique_reason']})"
    )
    return state


# ---------------------------------------------------------------------
# conditional routing after generate — catches refusals before the critic
# ever sees them, so an honest "I don't know" never gets labeled "grounded"
# ---------------------------------------------------------------------
def route_after_generate(state: RAGState) -> str:
    if is_refusal(state["answer"]):
        if state["retry_count"] >= MAX_RETRIES:
            return "give_up"
        return "retry"
    return "critique"


# ---------------------------------------------------------------------
# after critic node conditional routing 
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
        "check_meta", route_entry, {"summary": "summary", "decompose": "decompose"},
    )
    graph.add_conditional_edges(
        "decompose", route_after_decompose, {"summary": "summary", "retrieve": "retrieve"},
    )
    graph.add_edge("retrieve", "generate")

    graph.add_conditional_edges(
        "generate",
        route_after_generate,
        {"critique": "critique", "retry": "reformulate", "give_up": "finalize_fallback"},
    )
    graph.add_conditional_edges(
        "critique",
        route_after_critique,
        {"grounded": "finalize_grounded", "retry": "reformulate", "give_up": "finalize_fallback"},
    )

    graph.add_edge("reformulate", "retrieve")   # the self-healing loop
    graph.add_edge("finalize_grounded", END)
    graph.add_edge("finalize_fallback", END)
    graph.add_edge("summary", END)

    return graph.compile()


rag_app = build_graph()


def ask(query: str, collection_name: str = "documents", source: Optional[str] = None) -> dict:
    """Public entry point: run the full self-healing RAG graph on a query.
    `source` limits the search to one document; None searches the whole collection."""
    initial_state: RAGState = {
        "sub_queries": [],
        "original_query": query,
        "query": query,
        "collection_name": collection_name,
        "source": source,
        "chunks": [],
        "answer": "",
        "verdict": "",
        "critique_reason": "",
        "retry_count": 0,
        "confidence": {"label": "none", "score": 0.0, "best_rerank_score": None},
        "final_output": "",
    }
    return rag_app.invoke(initial_state)


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
