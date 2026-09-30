from app.utils.config import settings
from app.evaluation.scoring import evaluate_answer
from app.nlp.preprocessing import analyze_text
from app.services.pipeline import analyze_document


def test_settings_load():
    assert settings.app_name


def test_preprocessing_exposes_nlp_features():
    analysis = analyze_text("TF-IDF weights important terms. Important terms improve retrieval.")
    assert "tf-idf" in analysis.tokens
    assert analysis.bow["important"] == 2
    assert analysis.ngrams["2-grams"]
    assert analysis.tfidf_terms


def test_document_pipeline_generates_topics_and_summary():
    result = analyze_document("Natural language processing uses TF-IDF and embeddings. TF-IDF weights terms for document retrieval.")
    assert result["topics"]
    assert result["summary"]


def test_explainable_score_is_bounded_and_weighted():
    result = evaluate_answer("TF-IDF weights terms using inverse document frequency.", "TF-IDF weights terms using inverse document frequency.", ["TF-IDF", "inverse document frequency"], ["TF-IDF", "weights", "terms"])
    assert 0 <= result.overall_score <= 100
    assert result.semantic_similarity >= 0
    assert result.missing_concepts == []



def test_topic_extraction_filters_generic_single_words():
    result = analyze_document("NLP uses machine learning to extract features from text. Machine learning improves language models.")
    topics = {item["topic"] for item in result["topics"]}
    assert "word" not in topics
    assert "text" not in topics
    assert "feature" not in topics
    assert "machine" not in topics
