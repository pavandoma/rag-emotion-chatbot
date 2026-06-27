"""Non-retrieval baseline: direct Ollama call with no retrieved context.

Used purely as the A/B comparison point for the relevance-improvement eval --
same model, same prompt structure, just without the RAG context block.
"""

from src.utils.ollama_client import generate

SYSTEM_PROMPT = "You are a helpful assistant. Answer the user's question as best you can."


def answer(query: str) -> str:
    return generate(query, system=SYSTEM_PROMPT)
