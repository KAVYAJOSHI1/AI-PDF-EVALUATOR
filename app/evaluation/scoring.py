"""Explainable, local answer scoring; no external LLM is used here."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from app.nlp.embeddings import semantic_similarity
from app.nlp.preprocessing import STOP_WORDS, simple_lemma, tokenize

WEIGHTS = {"semantic_similarity": 0.50, "concept_coverage": 0.30, "keyword_coverage": 0.20}


@dataclass
class EvaluationResult:
    overall_score: float
    semantic_similarity: float
    concept_coverage: float
    keyword_coverage: float
    covered_concepts: list[str]
    missing_concepts: list[str]
    missing_keywords: list[str]
    feedback: str
    baseline_tfidf_similarity: float

    def to_dict(self) -> dict:
        return asdict(self)


def _lexical_similarity(a: str, b: str) -> float:
    a_tokens = set(simple_lemma(t) for t in tokenize(a) if t not in STOP_WORDS)
    b_tokens = set(simple_lemma(t) for t in tokenize(b) if t not in STOP_WORDS)
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def evaluate_answer(
    answer: str,
    reference_answer: str,
    concepts: list[str] | None = None,
    keywords: list[str] | None = None,
    source_page: int | None = None,
    source_passage: str | None = None,
) -> EvaluationResult:
    concepts = concepts or []
    keywords = keywords or [t for t in tokenize(reference_answer) if t.lower() not in STOP_WORDS and len(t) > 3]

    answer_lower = answer.lower()
    answer_terms = set(simple_lemma(t) for t in tokenize(answer) if t.lower() not in STOP_WORDS)

    covered = []
    missing = []
    for concept in concepts:
        concept_lower = concept.lower()
        concept_terms = [simple_lemma(t) for t in tokenize(concept) if t.lower() not in STOP_WORDS]

        if not concept_terms:
            if concept_lower in answer_lower:
                covered.append(concept)
            else:
                missing.append(concept)
            continue

        # Exact phrase match or matching all key concept terms
        if concept_lower in answer_lower:
            covered.append(concept)
        elif len(concept_terms) == 1:
            if concept_terms[0] in answer_terms:
                covered.append(concept)
            else:
                missing.append(concept)
        else:
            matched_count = sum(1 for term in concept_terms if term in answer_terms)
            if matched_count == len(concept_terms):
                covered.append(concept)
            else:
                missing.append(concept)


    covered_keywords = [word for word in keywords if simple_lemma(word.lower()) in answer_terms or word.lower() in answer_lower]
    concept_score = len(covered) / len(concepts) if concepts else _lexical_similarity(answer, reference_answer)
    keyword_score = len(covered_keywords) / len(keywords) if keywords else 0.0
    semantic_score = semantic_similarity(answer, reference_answer)

    overall = 100 * (
        WEIGHTS["semantic_similarity"] * semantic_score
        + WEIGHTS["concept_coverage"] * concept_score
        + WEIGHTS["keyword_coverage"] * keyword_score
    )

    covered_keyword_terms = {simple_lemma(word.lower()) for word in covered_keywords}
    missing_keywords = [word for word in keywords if simple_lemma(word.lower()) not in covered_keyword_terms][:15]

    page_ref = f" (Page {source_page})" if source_page else ""
    if not covered and missing:
        feedback = f"Your answer does not mention key concepts from the study material{page_ref}. Be sure to address: " + ", ".join(missing[:3]) + "."
    elif missing:
        feedback = f"Your answer covers the main idea correctly, but it misses key details{page_ref}: " + ", ".join(missing[:3]) + "."
    elif semantic_score < 0.35:
        feedback = f"Your answer mentions the main concepts, but needs more detail and terminology as described in the study material{page_ref}."
    else:
        feedback = f"Great job! Your answer covers the central reference concepts from the study material{page_ref}."

    return EvaluationResult(
        round(overall, 2),
        round(semantic_score * 100, 2),
        round(concept_score * 100, 2),
        round(keyword_score * 100, 2),
        covered,
        missing,
        missing_keywords,
        feedback,
        round(semantic_score * 100, 2),
    )

