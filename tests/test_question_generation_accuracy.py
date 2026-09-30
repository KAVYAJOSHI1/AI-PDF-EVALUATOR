from app.question_generation.generator import generate_questions, Question
from app.question_generation.validator import QuestionValidator
from app.services.pipeline import analyze_document


def test_case1_definition_passage_generates_definition_question():
    text = "Term Frequency Inverse Document Frequency (TF-IDF) is defined as a numerical statistic intended to reflect how important a word is to a document in a collection or corpus."
    doc = analyze_document(text)
    questions = generate_questions(doc["topics"], text, count=1, chunks=doc["chunks"])

    assert len(questions) == 1
    q = questions[0]
    assert "TF-IDF" in q["topic"] or "Term Frequency" in q["topic"]
    assert any(word in q["prompt"].lower() for word in ["defined", "concept", "explain", "role"])
    assert q["quality_score"] >= 80.0
    assert q["validation_info"]["checklist"]["source_supported"]


def test_case2_comparison_passage_generates_comparison_question():
    text = (
        "Stemming and lemmatization are fundamental text normalization techniques. "
        "Stemming is a crude heuristic process that cuts off word ends, whereas lemmatization "
        "uses vocabulary and morphological analysis to return the base dictionary form."
    )
    doc = analyze_document(text)
    questions = generate_questions(doc["topics"], text, count=1, chunks=doc["chunks"])

    assert len(questions) >= 1
    q = questions[0]
    assert "difference between" in q["prompt"] or "Stemming" in q["prompt"]
    assert "Lemmatization" in q["expected_concepts"]
    assert q["validation_info"]["checklist"]["source_supported"]


def test_case3_process_steps_passage_generates_process_question():
    text = (
        "The TF-IDF score is calculated by multiplying two metrics: the term frequency of a word in a document, "
        "and the inverse document frequency of the word across the whole corpus."
    )
    doc = analyze_document(text)
    questions = generate_questions(doc["topics"], text, count=1, chunks=doc["chunks"])

    assert len(questions) == 1
    q = questions[0]
    assert "Describe how" in q["prompt"] or "calculated" in q["prompt"] or "Explain" in q["prompt"]
    assert "multiplying" in q["reference_answer"] or "term frequency" in q["reference_answer"].lower()


def test_case4_isolated_topic_without_explanation_is_not_generated():
    # 'Word2Vec' appears as a lone mention without explanation in text
    text = (
        "Page 1 overview: NLP preprocessing uses tokenization and stop-word removal to prepare text data. "
        "Also Word2Vec."
    )
    doc = analyze_document(text)
    # Filter topics for Word2Vec
    w2v_topics = [t for t in doc["topics"] if t["topic"].lower() == "word2vec"]
    if w2v_topics:
        questions = generate_questions(w2v_topics, text, count=1, chunks=doc["chunks"])
        # If reference answer is too short (< 8 words), validator rejects it
        assert len(questions) == 0


def test_case5_duplicate_topics_and_prompts_are_rejected():
    validator = QuestionValidator(min_quality_threshold=0.80)
    existing = ["Explain the concept of TF-IDF as presented in the study material (Page 1)."]
    prompt = "Explain the concept of TF-IDF as presented in the study material (Page 1)."
    ref = "TF-IDF evaluates word importance across a corpus."

    v_res = validator.validate(
        prompt=prompt,
        reference_answer=ref,
        topic="TF-IDF",
        source_passage="TF-IDF evaluates word importance across a corpus.",
        existing_prompts=existing,
    )
    assert not v_res.accepted
    assert not v_res.checklist["unique"]


def test_case6_question_with_unsupported_facts_is_rejected():
    validator = QuestionValidator(min_quality_threshold=0.80)
    passage = "TF-IDF evaluates word importance by calculating term frequency."
    # Prompt asks to compare computational complexity of TF-IDF and Word2Vec, but Word2Vec is absent from passage
    prompt = "Explain the difference between TF-IDF and Word2Vec as described on Page 1."
    ref = "TF-IDF evaluates word importance by calculating term frequency."

    v_res = validator.validate(
        prompt=prompt,
        reference_answer=ref,
        topic="TF-IDF",
        source_passage=passage,
        existing_prompts=[],
    )
    assert not v_res.accepted
    assert not v_res.checklist["source_supported"]
