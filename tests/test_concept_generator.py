from app.question_generation.concept_generator import generate_concept_questions, mine_concepts
from app.question_generation.generator import generate_questions

SLIDES = [
    {"page_number": 1, "text": "Stemming removes prefixes or suffixes to obtain the root form of a word.\nThe result may not be a valid word.\nExample\nprint(stem('studies'))"},
    {"page_number": 2, "text": "Lemmatization converts words to their dictionary (base) form, considering grammar and\nvocabulary.\nIt returns valid words.\nStop words are common words that usually carry little meaning."},
    {"page_number": 3, "text": "Bag of Words (BoW) is the simplest text representation technique.\nIt counts the frequency of each word in a document.\nBoW converts text into numerical vectors by counting the frequency of each word.\nTF-IDF assigns weights to words.\n- Cross Entropy\nMeasures how close predicted probabilities are to actual probabilities.\nIt is very small.\nHeight and weight are strongly related."},
]


def _chunks():
    from app.pdf.extraction import chunk_document_pages
    return chunk_document_pages(SLIDES)


def test_mines_real_definitions_and_ignores_noise():
    terms = {c.short.lower() for c in mine_concepts(_chunks())}
    assert {"stemming", "lemmatization", "stop words", "bag of words", "cross entropy"} <= terms
    assert not any("height" in t or "print" in t or t == "example" for t in terms)


def test_wrapped_lines_are_rejoined():
    lemma = next(c for c in mine_concepts(_chunks()) if c.short == "Lemmatization")
    assert "grammar and vocabulary" in lemma.definition


def test_difficulty_changes_question_type():
    chunks = _chunks()
    easy = generate_concept_questions(chunks, 2, "Easy")
    hard = generate_concept_questions(chunks, 3, "Hard")
    assert all(q["prompt"].startswith(("Define", "What is", "In your own words")) for q in easy)
    assert any("Compare and contrast" in q["prompt"] or "in depth" in q["prompt"] for q in hard)


def test_questions_are_grounded_and_unique():
    qs = generate_questions([], "", count=5, difficulty="Medium", chunks=_chunks())
    assert len({q["topic"] for q in qs}) == len(qs) >= 3
    for q in qs:
        assert q["reference_answer"] and q["source_page"] in (1, 2, 3)
        assert q["expected_concepts"][0].split(" / ")[0].lower() in q["reference_answer"].lower()


def test_alias_counts_as_concept_in_scoring():
    from app.evaluation.scoring import evaluate_answer
    result = evaluate_answer("BoW counts words.", "Bag of Words counts the frequency of each word.", ["Bag of Words / BoW"])
    assert result.covered_concepts == ["Bag of Words / BoW"]
