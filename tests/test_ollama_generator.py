"""Unit tests for Ollama Local LLM generator integration and fallback."""
from __future__ import annotations

from unittest.mock import MagicMock, patch
from app.question_generation.generator import generate_questions
from app.question_generation.ollama_generator import check_ollama_status, generate_with_ollama
from app.services.pipeline import analyze_document


def test_check_ollama_status_offline():
    status = check_ollama_status("http://localhost:9999")
    assert not status["available"]
    assert status["models"] == []


@patch("requests.post")
def test_generate_with_ollama_success(mock_post):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "response": '{"prompt": "Explain TF-IDF calculation according to Page 1.", "reference_answer": "TF-IDF is calculated by multiplying term frequency and inverse document frequency.", "expected_concepts": ["TF-IDF", "Term Frequency"]}'
    }
    mock_post.return_value = mock_resp

    res = generate_with_ollama("TF-IDF", "TF-IDF is calculated by multiplying term frequency...", page_number=1)
    assert res is not None
    prompt, ref, concepts = res
    assert "TF-IDF" in prompt
    assert "term frequency" in ref.lower()
    assert "TF-IDF" in concepts


@patch("requests.post")
def test_generate_questions_with_ollama_fallback_on_failure(mock_post):
    # Simulate Ollama connection failure
    mock_post.side_effect = Exception("Connection refused")

    text = (
        "Stemming and lemmatization are fundamental text normalization techniques. "
        "Stemming is a crude heuristic process that cuts off word ends, whereas lemmatization "
        "uses vocabulary and morphological analysis to return the base dictionary form."
    )
    doc = analyze_document(text)
    questions = generate_questions(
        doc["topics"], text, count=1, chunks=doc["chunks"], use_ollama=True, ollama_model="llama3"
    )

    # Should fall back to local rule engine and return a valid question
    assert len(questions) >= 1
    assert questions[0]["quality_score"] >= 80.0
