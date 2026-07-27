import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "graph"))
from self_healing_graph import ask  # noqa: E402


def debug(question: str, collection: str):
    print(f"QUESTION: {question}")
    print(f"COLLECTION: {collection}")
    print("=" * 70)

    result = ask(question, collection_name=collection)

    print(f"\nSUB-QUERIES USED FOR RETRIEVAL:")
    for sq in result["sub_queries"]:
        print(f"  - {sq}")

    print(f"\nFINAL QUERY USED: {result['query']}")
    print(f"(original was: {result['original_query']})\n")

    print(f"CHUNKS RETRIEVED ({len(result['chunks'])}):")
    if not result["chunks"]:
        print("  (none)")
    for i, c in enumerate(result["chunks"], start=1):
        print(f"  [{i}] {c['source']} (page {c['page']}) — rerank score {c['rerank_score']:.3f}")
        print(f"      \"{c['text'][:200]}{'...' if len(c['text']) > 200 else ''}\"")

    print(f"\nLAST GENERATED ANSWER:\n  {result['answer']}")

    print(f"\nCRITIC VERDICT: {result['verdict']}")
    print(f"CRITIC REASON: {result['critique_reason']}")

    print(f"\nRETRIES USED: {result['retry_count']}")
    print(f"CONFIDENCE: {result['confidence']}")

    print(f"\nFINAL OUTPUT SHOWN TO USER:\n  {result['final_output']}")
    print("=" * 70)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python eval/debug_question.py \"question\" [collection]")
        sys.exit(1)

    question = sys.argv[1]
    collection = sys.argv[2] if len(sys.argv) > 2 else "documents"
    debug(question, collection)
