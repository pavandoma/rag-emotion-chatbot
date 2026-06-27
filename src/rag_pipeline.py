"""Retrieval-augmented answer generation: embed query -> retrieve top-k
chunks from Chroma -> build a grounded prompt -> generate via Ollama ->
gate the response through the safety filter.
"""

from dataclasses import dataclass, field

import chromadb
from sentence_transformers import SentenceTransformer

from src.config import CHROMA_COLLECTION, CHROMA_PERSIST_DIR, EMBEDDING_MODEL, RETRIEVAL_TOP_K
from src.safety.bias_filter import check_safety
from src.utils.ollama_client import generate

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question using ONLY the "
    "context provided below. If the context does not contain the answer, say "
    "you don't know rather than making something up."
)

_embedder: SentenceTransformer | None = None
_collection = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        _collection = client.get_collection(CHROMA_COLLECTION)
    return _collection


@dataclass
class RagResult:
    answer: str
    retrieved_chunks: list[dict] = field(default_factory=list)
    is_safe: bool = True
    safety_score: float = 0.0


def retrieve(query: str, top_k: int = RETRIEVAL_TOP_K) -> list[dict]:
    embedder = _get_embedder()
    collection = _get_collection()
    query_embedding = embedder.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)

    chunks = []
    for doc, dist, doc_id in zip(
        results["documents"][0], results["distances"][0], results["ids"][0]
    ):
        chunks.append({"id": doc_id, "text": doc, "distance": dist})
    return chunks


def build_prompt(query: str, chunks: list[dict]) -> str:
    context = "\n\n".join(f"[{c['id']}] {c['text']}" for c in chunks)
    return f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"


def answer(query: str, system_prefix: str | None = None, apply_safety_filter: bool = True) -> RagResult:
    chunks = retrieve(query)
    prompt = build_prompt(query, chunks)
    system = SYSTEM_PROMPT if system_prefix is None else f"{SYSTEM_PROMPT}\n\n{system_prefix}"

    raw_answer = generate(prompt, system=system)

    if apply_safety_filter:
        is_safe, score, safe_answer = check_safety(raw_answer)
        return RagResult(answer=safe_answer, retrieved_chunks=chunks, is_safe=is_safe, safety_score=score)

    return RagResult(answer=raw_answer, retrieved_chunks=chunks)
