from unittest.mock import patch

from fastapi.testclient import TestClient


def _build_client() -> TestClient:
    with patch("src.api.main.warm_up"):
        from src.api.main import app

        return TestClient(app)


@patch("src.api.deps.ollama_reachable", return_value=True)
@patch("src.api.main.get_chroma_collection_count", return_value=42)
def test_health_endpoint(mock_count, mock_reachable):
    client = _build_client()
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["chroma_collection_count"] == 42


@patch("src.api.main.build_emotion_aware_system_prefix", return_value=("", "joy", 0.9))
@patch("src.api.main.rag_answer")
def test_chat_endpoint_with_rag(mock_rag_answer, mock_emotion):
    from src.rag_pipeline import RagResult

    mock_rag_answer.return_value = RagResult(
        answer="Paris.",
        retrieved_chunks=[{"id": "chunk-0", "text": "Paris is the capital.", "distance": 0.1}],
        is_safe=True,
        safety_score=0.01,
    )

    client = _build_client()
    response = client.post("/chat", json={"query": "What is the capital of France?"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Paris."
    assert body["detected_emotion"] == "joy"
    assert len(body["retrieved_chunks"]) == 1
