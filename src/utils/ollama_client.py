import ollama

from src.config import OLLAMA_HOST, OLLAMA_MODEL

_client = ollama.Client(host=OLLAMA_HOST)


def generate(prompt: str, system: str | None = None, model: str = OLLAMA_MODEL) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = _client.chat(model=model, messages=messages)
    return response["message"]["content"]


def is_reachable() -> bool:
    try:
        _client.list()
        return True
    except Exception:
        return False
