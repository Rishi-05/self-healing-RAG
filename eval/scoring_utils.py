import re

NUMBER_WORDS = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}

def normalize_text(text: str) -> str:
    text = re.sub(r"[\u00A0\u202F\u2007\u2060]", " ", text.lower())
    for word, digit in NUMBER_WORDS.items():
        text = re.sub(rf"\b{word}\b", digit, text)
    return text

def keywords_present(answer: str, keywords: list[str]) -> bool:
    if not keywords:
        return True
    answer_norm = normalize_text(answer)
    return all(normalize_text(kw) in answer_norm for kw in keywords)

def print_comparison(plain_metrics: dict, healing_metrics: dict):
    print("\n" + "=" * 70)
    print(f"{'METRIC':35} {'PLAIN RAG':>15} {'SELF-HEALING':>15}")
    print("=" * 70)
    rows = [
        ("Avg latency (sec)", "avg_latency_sec"),
        ("Retrieval accuracy", "retrieval_accuracy"),
        ("Correct answer rate", "correct_answer_rate"),
        ("False refusal rate", "false_refusal_rate"),
        ("Hallucination rate  <-- key metric", "hallucination_rate"),
        ("Correct refusal rate", "correct_refusal_rate"),
    ]
    for label, key in rows:
        p = plain_metrics.get(key, "-")
        h = healing_metrics.get(key, "-")
        print(f"{label:35} {str(p):>15} {str(h):>15}")
    print(f"{'Avg retries (self-healing only)':35} {'-':>15} {str(healing_metrics.get('avg_retries', '-')):>15}")
    print(f"{'Retry rate (self-healing only)':35} {'-':>15} {str(healing_metrics.get('retry_rate', '-')):>15}")
    print("=" * 70)