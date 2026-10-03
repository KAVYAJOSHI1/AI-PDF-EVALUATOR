"""Explainable, local answer scoring; no external LLM is used here."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from app.nlp.embeddings import embed, lexical_similarity, semantic_similarity
from app.nlp.nli import nli_probs
from app.nlp.textrank import split_sentences
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
    alignment: list[dict] = field(default_factory=list)
    contradictions: list[dict] = field(default_factory=list)
    contradiction_penalty: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def _lexical_similarity(a: str, b: str) -> float:
    a_tokens = set(simple_lemma(t) for t in tokenize(a) if t not in STOP_WORDS)
    b_tokens = set(simple_lemma(t) for t in tokenize(b) if t not in STOP_WORDS)
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def _mentions(concept: str, answer_lower: str, answer_terms: set[str]) -> bool:
    """Phrase match, or every content term of the concept appears (lemmatised) in the answer."""
    if concept.lower() in answer_lower:
        return True
    terms = [simple_lemma(t) for t in tokenize(concept) if t not in STOP_WORDS]
    return bool(terms) and all(term in answer_terms for term in terms)


def align_ideas(answer: str, reference: str, threshold: float = 0.55) -> list[dict]:
    """Match each reference sentence ("idea") to the answer sentence that best expresses it.

    This makes the semantic score inspectable: the student sees which ideas were
    expressed (and by which of their sentences) and which were never addressed.
    """
    ref_sents = split_sentences(reference) or [reference]
    ans_sents = split_sentences(answer) or ([answer] if answer.strip() else [])
    if not ans_sents:
        return [{"idea": r, "matched_sentence": None, "similarity": 0.0, "addressed": False, "contradiction": 0.0} for r in ref_sents]
    vectors = embed(ref_sents + ans_sents)
    out = []
    for i, ref in enumerate(ref_sents):
        if vectors is not None:
            sims = vectors[len(ref_sents):] @ vectors[i]
            thr = threshold
        else:
            sims = [lexical_similarity(ref, a) for a in ans_sents]
            thr = 0.25
        j = int(max(range(len(ans_sents)), key=lambda k: sims[k]))
        item = {"idea": ref, "matched_sentence": ans_sents[j], "similarity": round(max(0.0, float(sims[j])), 3), "addressed": bool(sims[j] >= thr), "contradiction": 0.0}
        # Related-looking sentences are checked for logical conflict (NLI).
        if sims[j] >= (0.35 if vectors is not None else 0.15):
            probs = nli_probs(ref, ans_sents[j])
            if probs:
                item["contradiction"] = probs["contradiction"]
                item["entailment"] = probs["entailment"]
                if probs["contradiction"] > 0.6:
                    item["addressed"] = False
        out.append(item)
    return out


def evaluate_answer(
    answer: str,
    reference_answer: str,
    concepts: list[str] | None = None,
    keywords: list[str] | None = None,
    source_page: int | None = None,
    source_passage: str | None = None,
) -> EvaluationResult:
    concepts = concepts or []
    if not answer.strip():
        return EvaluationResult(0.0, 0.0, 0.0, 0.0, [], list(concepts), list(keywords or []), "No answer was written.", 0.0, align_ideas("", reference_answer))
    keywords = keywords or [t for t in tokenize(reference_answer) if t.lower() not in STOP_WORDS and len(t) > 3]

    answer_lower = answer.lower()
    answer_terms = set(simple_lemma(t) for t in tokenize(answer) if t.lower() not in STOP_WORDS)

    covered = []
    missing = []
    for concept in concepts:
        # "Natural Language Processing / NLP": any listed name counts.
        if any(_mentions(alt.strip(), answer_lower, answer_terms) for alt in concept.split(" / ")):
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

    alignment = align_ideas(answer, reference_answer)
    contradictions = [a for a in alignment if a.get("contradiction", 0.0) > 0.6]
    penalty = 0.6 * len(contradictions) / len(alignment) if alignment else 0.0
    overall *= 1 - penalty

    covered_keyword_terms = {simple_lemma(word.lower()) for word in covered_keywords}
    missing_keywords = [word for word in keywords if simple_lemma(word.lower()) not in covered_keyword_terms][:15]

    page_ref = f" (Page {source_page})" if source_page else ""
    if contradictions:
        feedback = (
            f"Your answer contradicts the study material{page_ref}: you wrote \"{contradictions[0]['matched_sentence']}\" "
            f"but the material says \"{contradictions[0]['idea']}\""
        )
    elif not covered and missing:
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
        alignment,
        contradictions,
        round(penalty * 100, 1),
    )

