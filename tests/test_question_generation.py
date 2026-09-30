from app.pdf.extraction import chunk_document_pages
from app.question_generation.generator import generate_questions, synthesize_prompt
from app.question_generation.validator import QuestionValidator
from app.evaluation.scoring import evaluate_answer
from app.services.pipeline import analyze_document



def test_chunk_document_pages_creates_logical_passages_with_page_numbers():
    pages = [
        {"page_number": 1, "text": "Stemming is a crude heuristic process that chops off the ends of words. Stemming reduces inflected words to their root form.\n\nLemmatization uses a vocabulary and morphological analysis of words to remove inflectional endings."},
        {"page_number": 2, "text": "TF-IDF stands for Term Frequency Inverse Document Frequency. TF-IDF evaluates how important a word is to a document in a collection or corpus."},
    ]
    chunks = chunk_document_pages(pages)
    assert len(chunks) >= 2
    assert chunks[0]["page_number"] == 1
    assert chunks[1]["page_number"] == 2
    assert "Stemming" in chunks[0]["text"]
    assert "TF-IDF" in chunks[1]["text"]


def test_generate_questions_creates_comparison_question_for_stemming_and_lemmatization():
    source_text = (
        "Stemming and lemmatization are fundamental text normalization techniques in NLP. "
        "Stemming is a crude heuristic process that cuts off word suffixes, whereas lemmatization "
        "uses vocabulary and morphological analysis to return the base dictionary form."
    )
    topics = [{"topic": "Stemming", "source_page": 1}]
    questions = generate_questions(topics, source_text, count=1)
    assert len(questions) == 1
    q = questions[0]
    assert q["source_page"] == 1
    assert "difference between Stemming and Lemmatization" in q["prompt"] or "Stemming" in q["prompt"]
    assert "Lemmatization" in q["expected_concepts"]
    assert q["reference_answer"]


def test_question_validation_filters_generic_and_short_references():
    validator = QuestionValidator(min_quality_threshold=0.80)
    v1 = validator.validate("What is text?", "text is words", "text", "text is words")
    assert not v1.accepted
    v2 = validator.validate("Explain the difference between stemming and lemmatization on Page 1.", "Stemming cuts off word ends whereas lemmatization uses vocabulary and morphological analysis.", "Stemming", "Stemming cuts off word ends whereas lemmatization uses vocabulary and morphological analysis.")
    assert v2.accepted


def test_evaluate_answer_includes_page_reference_in_feedback():
    reference = "TF-IDF evaluates how important a word is to a document compared to Bag of Words."
    answer = "TF-IDF measures word frequency in a document."
    concepts = ["TF-IDF", "Bag of Words"]

    result = evaluate_answer(answer, reference, concepts, source_page=3)
    assert "TF-IDF" in result.covered_concepts
    assert "Bag of Words" in result.missing_concepts
    assert "Page 3" in result.feedback
    assert "Bag of Words" in result.feedback


def test_document_pipeline_end_to_end_grounding():
    doc_text = (
        "Page 1 topic: Natural language processing models analyze human language text. "
        "TF-IDF weighs term importance by multiplying Term Frequency with Inverse Document Frequency. "
        "Bag of Words simply counts word frequencies without considering document frequencies."
    )
    result = analyze_document(doc_text)
    assert result["chunks"]
    questions = generate_questions(result["topics"], doc_text, count=2, chunks=result["chunks"])
    assert len(questions) >= 1
    assert questions[0]["source_page"] >= 1
    assert questions[0]["source_passage"]
