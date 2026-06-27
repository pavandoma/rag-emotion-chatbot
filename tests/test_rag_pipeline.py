from unittest.mock import patch

from src.rag_pipeline import build_prompt


def test_build_prompt_includes_context_and_question():
    chunks = [{"id": "chunk-0", "text": "Paris is the capital of France."}]
    prompt = build_prompt("What is the capital of France?", chunks)
    assert "Paris is the capital of France." in prompt
    assert "What is the capital of France?" in prompt
    assert "[chunk-0]" in prompt


@patch("src.rag_pipeline.generate", return_value="Paris.")
@patch("src.rag_pipeline.check_safety", return_value=(True, 0.01, "Paris."))
@patch(
    "src.rag_pipeline.retrieve",
    return_value=[{"id": "chunk-0", "text": "Paris is the capital of France.", "distance": 0.1}],
)
def test_answer_returns_retrieved_chunks_and_safe_answer(mock_retrieve, mock_safety, mock_generate):
    from src.rag_pipeline import answer

    result = answer("What is the capital of France?")
    assert result.answer == "Paris."
    assert result.is_safe is True
    assert len(result.retrieved_chunks) == 1
