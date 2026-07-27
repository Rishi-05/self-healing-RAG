import argparse
import json
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "ingestion"))
sys.path.append(str(Path(__file__).parent.parent / "graph"))
sys.path.append(str(Path(__file__).parent))

from retrieve_generate import answer_question                 # noqa: E402
from self_healing_graph import ask                              # noqa: E402
from scoring_utils import keywords_present, print_comparison    # noqa: E402

REFUSAL_PHRASES = (
    "does not contain enough information",
    "no relevant content was found",
    "i don't have enough information",
)


def is_refusal(answer: str) -> bool:
    return any(p in answer.lower() for p in REFUSAL_PHRASES)

def score_plain(item: dict, collection: str) -> dict:
    start = time.time()
    result = answer_question(item["question"], collection_name=collection)
    latency = time.time() - start

    sources_hit = {c["source"] for c in result["chunks"]}
    retrieval_hit = (item["expected_source"] in sources_hit) if item["expected_answerable"] else None
    refused = is_refusal(result["answer"])

    return {
        "answer_text": result["answer"],
        "latency": round(latency, 2),
        "refused": refused,
        "retrieval_hit": retrieval_hit,
        "keywords_correct": keywords_present(result["answer"], item["expected_keywords"]),
        "confidence": result["confidence"]["label"],
    }


def score_self_healing(item: dict, collection: str) -> dict:
    start = time.time()
    result = ask(item["question"], collection_name=collection)
    latency = time.time() - start

    sources_hit = {c["source"] for c in result["chunks"]}
    retrieval_hit = (item["expected_source"] in sources_hit) if item["expected_answerable"] else None
    refused = is_refusal(result["final_output"])

    return {
        "answer_text": result["final_output"],
        "critique_reason": result["critique_reason"],
        "latency": round(latency, 2),
        "refused": refused,
        "retrieval_hit": retrieval_hit,
        "keywords_correct": keywords_present(result["final_output"], item["expected_keywords"]),
        "confidence": result["confidence"]["label"],
        "retries": result["retry_count"],
    }


def aggregate(results: list[dict], dataset: list[dict], has_retries: bool) -> dict:
    answerable = [r for r, d in zip(results, dataset) if d["expected_answerable"]]
    unanswerable = [r for r, d in zip(results, dataset) if not d["expected_answerable"]]

    metrics = {
        "n_questions": len(results),
        "avg_latency_sec": round(sum(r["latency"] for r in results) / len(results), 2),

        # On answerable questions: did it actually answer, and correctly?
        "false_refusal_rate": round(
            sum(1 for r in answerable if r["refused"]) / len(answerable), 2
        ) if answerable else None,
        "retrieval_accuracy": round(
            sum(1 for r in answerable if r["retrieval_hit"]) / len(answerable), 2
        ) if answerable else None,
        "correct_answer_rate": round(
            sum(1 for r in answerable if not r["refused"] and r["keywords_correct"]) / len(answerable), 2
        ) if answerable else None,

        # On unanswerable questions: did it correctly refuse, or hallucinate?
        "hallucination_rate": round(
            sum(1 for r in unanswerable if not r["refused"]) / len(unanswerable), 2
        ) if unanswerable else None,
        "correct_refusal_rate": round(
            sum(1 for r in unanswerable if r["refused"]) / len(unanswerable), 2
        ) if unanswerable else None,
    }

    if has_retries:
        metrics["avg_retries"] = round(sum(r["retries"] for r in results) / len(results), 2)
        metrics["retry_rate"] = round(sum(1 for r in results if r["retries"] > 0) / len(results), 2)

    return metrics

def run(collection: str, dataset_path: str):
    with open(dataset_path) as f:
        dataset = json.load(f)

    print(f"Running {len(dataset)} questions through PLAIN RAG...")
    plain_results = []
    for item in dataset:
        r = score_plain(item, collection)
        plain_results.append(r)
        print(f"  [{item['id']}] refused={r['refused']} retrieval_hit={r['retrieval_hit']} "
              f"correct={r['keywords_correct']} ({r['latency']}s)")

    print(f"\nRunning {len(dataset)} questions through SELF-HEALING RAG...")
    healing_results = []
    for item in dataset:
        r = score_self_healing(item, collection)
        healing_results.append(r)
        print(f"  [{item['id']}] refused={r['refused']} retrieval_hit={r['retrieval_hit']} "
              f"correct={r['keywords_correct']} retries={r['retries']} ({r['latency']}s)")

    plain_metrics = aggregate(plain_results, dataset, has_retries=False)
    healing_metrics = aggregate(healing_results, dataset, has_retries=True)

    print_comparison(plain_metrics, healing_metrics)

    output = {
        "plain_rag": plain_metrics,
        "self_healing_rag": healing_metrics,
        "raw_plain_results": [{**dataset[i], **plain_results[i]} for i in range(len(dataset))],
        "raw_healing_results": [{**dataset[i], **healing_results[i]} for i in range(len(dataset))],
    }
    out_path = Path(__file__).parent / "eval_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the golden dataset eval.")
    parser.add_argument("--collection", default="test_docs", help="ChromaDB collection to query")
    parser.add_argument("--dataset", default=str(Path(__file__).parent / "golden_dataset.json"))
    args = parser.parse_args()

    run(args.collection, args.dataset)
