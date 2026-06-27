"""Startup-time warmup of model singletons so the first request isn't slow.

The actual singletons live as lazy-loaded module globals in rag_pipeline,
emotion_classifier, and bias_filter -- this just forces them to load once
at app startup instead of on the first incoming request.
"""

import chromadb

from src.config import CHROMA_COLLECTION, CHROMA_PERSIST_DIR
from src.emotion.emotion_classifier import predict_emotion
from src.rag_pipeline import _get_collection, _get_embedder
from src.safety.bias_filter import _get_classifier as _get_toxicity_classifier
from src.utils.ollama_client import is_reachable


def warm_up() -> None:
    _get_embedder()
    _get_collection()
    _get_toxicity_classifier()
    predict_emotion("warmup")


def get_chroma_collection_count() -> int:
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    try:
        return client.get_collection(CHROMA_COLLECTION).count()
    except Exception:
        return 0


def ollama_reachable() -> bool:
    return is_reachable()
