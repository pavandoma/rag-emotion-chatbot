from contextlib import asynccontextmanager

import chromadb
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer

from src.api.deps import get_chroma_collection_count, ollama_reachable, warm_up
from src.api.schemas import (
    ChatRequest,
    ChatResponse,
    ClassifyEmotionRequest,
    ClassifyEmotionResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    RetrievedChunk,
)
from src.baseline_pipeline import answer as baseline_answer
from src.config import CHROMA_COLLECTION, CHROMA_PERSIST_DIR, EMBEDDING_MODEL
from src.emotion.emotion_classifier import predict_emotion
from src.emotion.prompt_conditioning import build_emotion_aware_system_prefix
from src.rag_pipeline import answer as rag_answer
from src.utils.chunking import chunk_text


@asynccontextmanager
async def lifespan(app: FastAPI):
    warm_up()
    yield


app = FastAPI(title="RAG Emotion-Aware Conversational Assistant", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        ollama_reachable=ollama_reachable(),
        chroma_collection_count=get_chroma_collection_count(),
    )


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    system_prefix = None
    detected_emotion = None
    emotion_confidence = None

    if request.emotion_aware:
        system_prefix, detected_emotion, emotion_confidence = build_emotion_aware_system_prefix(
            request.query
        )

    if request.use_rag:
        result = rag_answer(request.query, system_prefix=system_prefix)
        return ChatResponse(
            answer=result.answer,
            retrieved_chunks=[RetrievedChunk(**c) for c in result.retrieved_chunks],
            detected_emotion=detected_emotion,
            emotion_confidence=emotion_confidence,
            is_safe=result.is_safe,
            safety_score=result.safety_score,
        )

    text = baseline_answer(request.query)
    return ChatResponse(
        answer=text,
        retrieved_chunks=[],
        detected_emotion=detected_emotion,
        emotion_confidence=emotion_confidence,
    )


@app.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest) -> IngestResponse:
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    collection = client.get_or_create_collection(CHROMA_COLLECTION)
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    chunks: list[str] = []
    for text in request.texts:
        chunks.extend(chunk_text(text))

    if not chunks:
        return IngestResponse(chunks_added=0)

    embeddings = embedder.encode(chunks).tolist()
    existing_count = collection.count()
    ids = [f"ingested-{existing_count + i}" for i in range(len(chunks))]
    collection.add(ids=ids, embeddings=embeddings, documents=chunks)

    return IngestResponse(chunks_added=len(chunks))


@app.get("/collections/stats")
def collection_stats() -> dict:
    return {"collection": CHROMA_COLLECTION, "count": get_chroma_collection_count()}


@app.post("/classify_emotion", response_model=ClassifyEmotionResponse)
def classify_emotion(request: ClassifyEmotionRequest) -> ClassifyEmotionResponse:
    emotion, confidence = predict_emotion(request.text)
    return ClassifyEmotionResponse(emotion=emotion, confidence=confidence)
