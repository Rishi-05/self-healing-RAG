import os
import re
import threading

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# Order = preference. If a model is rate-limited or unavailable, the next one is tried.
FALLBACK_MODELS = [
    "openai/gpt-oss-120b",
    "qwen/qwen3-32b",
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-20b",
]

_client = None
_client_lock = threading.Lock()


def get_client() -> Groq:
    """Created lazily so a missing key doesn't crash the whole app at import time."""
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                key = os.environ.get("GROQ_API_KEY")
                if not key:
                    raise RuntimeError(
                        "GROQ_API_KEY is not set. Put it in .env locally, or in "
                        "Streamlit Cloud -> App settings -> Secrets when deployed."
                    )
                _client = Groq(api_key=key)
    return _client


_FALLBACK_HINTS = (
    "rate_limit", "rate limit", "429", "503", "overloaded", "over capacity",
    "model_not_found", "decommissioned", "does not exist",
)


def should_fallback(e: Exception) -> bool:
    msg = str(e).lower()
    return any(h in msg for h in _FALLBACK_HINTS)


is_rate_limit_error = should_fallback  # backwards-compatible alias


def chat_completion(messages: list[dict], temperature: float = 0.1,
                    response_format: dict = None, models: list[str] = None):
    client = get_client()
    models_to_try = models or FALLBACK_MODELS
    last_error = None

    for model in models_to_try:
        try:
            kwargs = {"model": model, "messages": messages, "temperature": temperature}
            if response_format:
                kwargs["response_format"] = response_format
            response = client.chat.completions.create(**kwargs)
            return response, model
        except Exception as e:
            if should_fallback(e):
                print(f"  [model '{model}' unavailable ({type(e).__name__}) — trying next...]")
                last_error = e
                continue
            raise  # auth errors, bad requests etc. should surface, not be swallowed

    raise RuntimeError(
        f"All fallback models exhausted ({models_to_try}). Last error: {last_error}"
    )


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def clean_output(text: str) -> str:
    """Strips <think>...</think> blocks (qwen3 emits them) and whitespace."""
    return _THINK_RE.sub("", text or "").strip()


def chat_text(messages: list[dict], temperature: float = 0.1, models: list[str] = None) -> str:
    """Convenience wrapper: returns just the cleaned text of the reply."""
    response, _ = chat_completion(messages, temperature=temperature, models=models)
    return clean_output(response.choices[0].message.content)
