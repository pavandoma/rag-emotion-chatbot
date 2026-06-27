from pydantic import BaseModel


class ChatRequest(BaseModel):
    query: str
    use_rag: bool = True
    emotion_aware: bool = True


class RetrievedChunk(BaseModel):
    id: str
    text: str
    distance: float


class ChatResponse(BaseModel):
    answer: str
    retrieved_chunks: list[RetrievedChunk] = []
    detected_emotion: str | None = None
    emotion_confidence: float | None = None
    is_safe: bool = True
    safety_score: float = 0.0


class IngestRequest(BaseModel):
    texts: list[str]


class IngestResponse(BaseModel):
    chunks_added: int


class ClassifyEmotionRequest(BaseModel):
    text: str


class ClassifyEmotionResponse(BaseModel):
    emotion: str
    confidence: float


class HealthResponse(BaseModel):
    status: str
    ollama_reachable: bool
    chroma_collection_count: int
