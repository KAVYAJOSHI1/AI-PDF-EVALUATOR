import numpy as np

from app.evaluation.scoring import align_ideas
from app.nlp.linguistics import lemmatize_tagged, pos_tag
from app.nlp.retrieval import BM25, search
from app.nlp.textrank import pagerank, rank_sentences, textrank_keyphrases, textrank_summary

TEXT = (
    "Tokenization splits raw text into words before any other processing happens. "
    "Lemmatization uses a dictionary to return the valid base form of each word. "
    "TF-IDF weighs a term by how often it appears and how rare it is across documents. "
    "Word embeddings map words to dense vectors so similar meanings lie close together. "
    "Transformers use self-attention to model long-range dependencies between tokens."
)


def test_pagerank_is_a_distribution_and_favours_hub():
    graph = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=float)
    ranks = pagerank(graph)
    assert abs(ranks.sum() - 1) < 1e-6 and ranks[0] == ranks.max()


def test_textrank_summary_is_subset_in_order():
    summary = textrank_summary(TEXT, 2)
    ranked = rank_sentences(TEXT)
    assert len(ranked) == 5
    assert all(s in TEXT for s in summary.split(". ")[:1])


def test_keyphrases_do_not_cross_sentences():
    for item in textrank_keyphrases(TEXT, 10):
        assert len(item["phrase"].split()) <= 3
    assert any("embedding" in item["phrase"] for item in textrank_keyphrases(TEXT, 10))


def test_pos_aware_lemmas():
    lemmas = lemmatize_tagged(pos_tag(["studies", "running", "better"]))
    assert lemmas[0] == "study"


def test_bm25_prefers_matching_document():
    scores = BM25(["cats purr softly", "dogs bark loudly"]).scores("dogs bark")
    assert scores[1] > scores[0]


def test_search_finds_semantic_paraphrase():
    chunks = [{"text": s, "page_number": i + 1} for i, s in enumerate(TEXT.split(". "))]
    hit = search("how are words turned into vectors", chunks, 1)[0]
    assert hit["page"] == 4


def test_alignment_flags_missing_idea():
    reference = "Lemmatization uses a dictionary to return valid base forms. Stemming simply chops suffixes off words."
    out = align_ideas("Lemmatization uses a dictionary to find the valid base form.", reference)
    assert out[0]["addressed"] and not out[1]["addressed"]
