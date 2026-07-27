import os

from groq import Groq
from dotenv import load_dotenv

load_dotenv()

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# add/remove based on what's currently rate-limited for you.
FALLBACK_MODELS = [
    "openai/gpt-oss-120b",
    "qwen/qwen3-32b",
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-20b",
]


def is_rate_limit_error(e: Exception) -> bool:
    msg = str(e).lower()
    return "rate_limit" in msg or "429" in msg


def chat_completion(messages: list[dict], temperature: float = 0.1,
                     response_format: dict = None, models: list[str] = None):
    models_to_try = models or FALLBACK_MODELS
    last_error = None

    for model in models_to_try:
        try:
            kwargs = {"model": model, "messages": messages, "temperature": temperature}
            if response_format:
                kwargs["response_format"] = response_format
            response = groq_client.chat.completions.create(**kwargs)
            return response, model
        except Exception as e:
            if is_rate_limit_error(e):
                print(f"  [rate limit hit on '{model}' — falling back to next model...]")
                last_error = e
                continue
            raise  # don't swallow non-rate-limit errors (auth, bad request, etc.)

    raise RuntimeError(
        f"All fallback models exhausted ({models_to_try}). Last error: {last_error}"
    )
