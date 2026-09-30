"""Validation engine for source-grounded question quality metrics."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from app.nlp.preprocessing import STOP_WORDS, tokenize


@dataclass
class ValidationResult:
    accepted: bool
    quality_score: float
    source_support_score: float
    topic_relevance_score: float
    answerability_score: float
    uniqueness_score: float
    checklist: dict[str, bool]
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


PROMPT_FRAMING_WORDS = {
    "explain", "describe", "discuss", "compare", "contrast", "difference", "differ",
    "study", "material", "according", "presented", "text", "passage", "primary",
    "purpose", "defined", "represents", "role", "key", "characteristics", "function",
    "detail", "mechanism", "achieves", "involving", "based", "material", "detail",
}


class QuestionValidator:
    """Validator enforcing source support, topic relevance, answerability, uniqueness, and overall quality threshold."""

    def __init__(self, min_quality_threshold: float = 0.80) -> None:
        self.min_quality_threshold = min_quality_threshold

    def calculate_source_support(self, prompt: str, reference_answer: str, source_passage: str) -> float:
        """Check how strongly the question prompt and reference answer are backed by the source passage."""
        if not source_passage or not reference_answer:
            return 0.0

        passage_lower = source_passage.lower()
        ref_lower = reference_answer.lower()

        # Check if reference answer sentences are contained in or derived from passage
        ref_words = [w for w in tokenize(ref_lower) if w not in STOP_WORDS]
        if not ref_words:
            return 0.0

        matched_ref_words = sum(1 for w in ref_words if w in passage_lower)
        ref_coverage = matched_ref_words / len(ref_words)

        # Check if key domain content terms in prompt are present in passage
        prompt_clean = re.sub(r"\(?page\s+[0-9]+\)?", "", prompt, flags=re.IGNORECASE)
        prompt_words = [
            w for w in tokenize(prompt_clean.lower())
            if w not in STOP_WORDS and w not in PROMPT_FRAMING_WORDS and len(w) > 3
        ]
        if prompt_words:
            matched_prompt_words = sum(1 for w in prompt_words if w in passage_lower)
            prompt_coverage = matched_prompt_words / len(prompt_words)
        else:
            prompt_coverage = 1.0

        # Check for ungrounded external assumptions
        unsupported_penalty = 0.0
        comparison_match = re.search(r"difference between ([A-Za-z0-9\s-]+?) and ([A-Za-z0-9\s-]+?)(?:\s+as|\s+on|\s+in|\.|\?|$)", prompt, re.IGNORECASE)
        if comparison_match:
            c1 = comparison_match.group(1).strip().lower()
            c2 = comparison_match.group(2).strip().lower()
            # Clean page/section badges from c1/c2
            c1 = re.sub(r"\b(page|section)\b.*", "", c1).strip()
            c2 = re.sub(r"\b(page|section)\b.*", "", c2).strip()
            if c1 and c1 not in passage_lower:
                unsupported_penalty += 0.35
            if c2 and c2 not in passage_lower:
                unsupported_penalty += 0.35

        score = (0.60 * ref_coverage + 0.40 * prompt_coverage) - unsupported_penalty
        return max(0.0, min(1.0, score))



    def calculate_topic_relevance(
        self, prompt: str, topic: str, expected_concepts: list[str] | None = None
    ) -> float:
        """Check if the prompt directly tests the selected topic or expected concepts."""
        if not topic or not prompt:
            return 0.0
        topic_lower = topic.lower()
        prompt_lower = prompt.lower()

        if topic_lower in prompt_lower:
            return 1.0

        # Filter out generic/framing/structural terms from topic words
        structural_terms = {
            "fundamental", "technique", "techniques", "method", "methods",
            "process", "processes", "system", "systems", "approach", "concept",
            "concepts", "task", "tasks", "analysis", "return", "procedure",
        }
        topic_words = [
            w for w in tokenize(topic_lower)
            if w not in STOP_WORDS
            and w not in PROMPT_FRAMING_WORDS
            and w not in structural_terms
        ]
        if not topic_words:
            topic_words = [w for w in tokenize(topic_lower) if w not in STOP_WORDS]
        if not topic_words:
            return 0.50

        matched = sum(1 for w in topic_words if w in prompt_lower)
        score = matched / len(topic_words)

        if expected_concepts:
            for concept in expected_concepts:
                c_words = [w for w in tokenize(concept.lower()) if w not in STOP_WORDS]
                if c_words:
                    c_matched = sum(1 for w in c_words if w in prompt_lower)
                    c_score = c_matched / len(c_words)
                    if c_score > score:
                        score = c_score

        return score

    def calculate_answerability(self, reference_answer: str, prompt: str) -> float:
        """Check if the reference answer provides sufficient, meaningful context to answer the question."""
        if not reference_answer:
            return 0.0
        words = reference_answer.split()
        if len(words) < 8:
            return 0.20
        if len(words) < 15:
            return 0.65

        # Check if reference answer contains explanatory verbs/structure
        ref_lower = reference_answer.lower()
        has_explanatory = any(w in ref_lower for w in ["is", "are", "by", "uses", "because", "means", "used", "calculates", "represents"])
        return 1.0 if has_explanatory else 0.85

    def calculate_uniqueness(self, prompt: str, existing_prompts: list[str]) -> float:
        """Check if prompt is distinct from already generated questions in the exam."""
        if not existing_prompts:
            return 1.0

        prompt_tokens = set(tokenize(prompt.lower())) - STOP_WORDS
        if not prompt_tokens:
            return 1.0

        max_similarity = 0.0
        for existing in existing_prompts:
            existing_tokens = set(tokenize(existing.lower())) - STOP_WORDS
            if not existing_tokens:
                continue
            jaccard = len(prompt_tokens & existing_tokens) / len(prompt_tokens | existing_tokens)
            if jaccard > max_similarity:
                max_similarity = jaccard

        return max(0.0, 1.0 - max_similarity)

    def validate(
        self,
        prompt: str,
        reference_answer: str,
        topic: str,
        source_passage: str,
        existing_prompts: list[str] | None = None,
        expected_concepts: list[str] | None = None,
    ) -> ValidationResult:
        """Validate candidate question and calculate 5-metric Quality Score."""
        existing_prompts = existing_prompts or []

        support = self.calculate_source_support(prompt, reference_answer, source_passage)
        relevance = self.calculate_topic_relevance(prompt, topic, expected_concepts)
        answerability = self.calculate_answerability(reference_answer, prompt)
        uniqueness = self.calculate_uniqueness(prompt, existing_prompts)

        overall_quality = 100 * (0.35 * support + 0.30 * relevance + 0.20 * answerability + 0.15 * uniqueness)
        quality_round = round(overall_quality, 2)

        checklist = {
            "source_supported": support >= 0.70,
            "relevant": relevance >= 0.70,
            "answerable": answerability >= 0.65,
            "unique": uniqueness >= 0.60,
        }

        accepted = (quality_round / 100.0) >= self.min_quality_threshold and all(checklist.values())

        if not accepted:
            failed_criteria = [k for k, v in checklist.items() if not v]
            if quality_round / 100.0 < self.min_quality_threshold:
                failed_criteria.append(f"below threshold ({quality_round}% < {self.min_quality_threshold*100}%)")
            reason = "Rejected: " + ", ".join(failed_criteria)
        else:
            reason = "Accepted: Passed quality threshold and all validation checks."

        return ValidationResult(
            accepted=accepted,
            quality_score=quality_round,
            source_support_score=round(support * 100, 2),
            topic_relevance_score=round(relevance * 100, 2),
            answerability_score=round(answerability * 100, 2),
            uniqueness_score=round(uniqueness * 100, 2),
            checklist=checklist,
            reason=reason,
        )
