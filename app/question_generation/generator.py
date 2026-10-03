from __future__ import annotations

from dataclasses import asdict, dataclass
import logging
import re
from app.nlp.preprocessing import STOP_WORDS, simple_lemma, tokenize
from app.question_generation.validator import QuestionValidator

logger = logging.getLogger(__name__)


GENERIC_UNSUPPORTED = {
    "text", "word", "words", "sentence", "sentences", "material", "study", "thing", "things",
    "example", "examples", "information", "chapter", "chapters", "section", "sections",
    "figure", "figures", "table", "tables", "overview", "introduction", "conclusion",
    "abstract", "author", "authors", "page", "pages", "note", "notes", "summary",
    "definition", "definitions", "exercise", "exercises",
}



@dataclass
class Question:
    id: int
    topic: str
    prompt: str
    difficulty: str
    reference_answer: str
    source_page: int = 1
    source_section: str = "General Overview"
    source_chunk_id: int = 1
    source_passage: str = ""
    expected_concepts: list[str] = None  # type: ignore[assignment]
    expected_keywords: list[str] = None  # type: ignore[assignment]
    quality_score: float = 0.0
    validation_info: dict = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.expected_concepts is None:
            self.expected_concepts = [self.topic]
        if self.expected_keywords is None:
            self.expected_keywords = []
        if self.validation_info is None:
            self.validation_info = {}

    def to_dict(self) -> dict:
        return asdict(self)


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.replace("\n", " ")) if s.strip()]


def _extract_secondary_concept(passage: str, topic: str) -> str | None:
    """Check if the passage contrasts or compares topic with another key concept in the text."""
    topic_lower = topic.lower()
    patterns = [
        rf"\b{re.escape(topic_lower)}\s+(?:and|versus|vs\.?|or|compared to)\s+([a-z0-9-]+(?:\s+[a-z0-9-]+)?)\b",
        rf"\b([a-z0-9-]+(?:\s+[a-z0-9-]+)?)\s+(?:and|versus|vs\.?|or|compared to)\s+{re.escape(topic_lower)}\b",
        rf"\b{re.escape(topic_lower)}\s+differs\s+from\s+([a-z0-9-]+(?:\s+[a-z0-9-]+)?)\b",
        rf"unlike\s+([a-z0-9-]+(?:\s+[a-z0-9-]+)?)[^,.]*{re.escape(topic_lower)}",
    ]
    for pattern in patterns:
        match = re.search(pattern, passage.lower())
        if match:
            candidate = match.group(1).strip()
            candidate_words = [w for w in candidate.split() if w not in STOP_WORDS and w not in GENERIC_UNSUPPORTED]
            if candidate_words:
                clean_candidate = " ".join(candidate_words)
                if len(clean_candidate) > 3:
                    return clean_candidate.title() if clean_candidate.islower() else clean_candidate
    return None


def transform_sentence_to_question(sentence: str, topic: str, page_number: int = 1) -> tuple[str, str, list[str]] | None:
    """Transform a key factual sentence into a natural, highly relevant subjective question prompt."""
    if len(sentence.split()) < 7:
        return None

    sentence_lower = sentence.lower()
    topic_clean = topic.strip().title() if topic.islower() else topic.strip()

    # 1. Comparison / Dual Concept Pattern ("X and Y are...", "Stemming and lemmatization...")
    dual_match = re.search(r"\b([a-z0-9-]+)\s+(?:and|versus|vs\.?|or)\s+([a-z0-9-]+)\s+are\b", sentence_lower)
    if dual_match:
        c1, c2 = dual_match.group(1).title(), dual_match.group(2).title()
        if c1.lower() not in STOP_WORDS and c2.lower() not in STOP_WORDS and c1.lower() not in GENERIC_UNSUPPORTED and c2.lower() not in GENERIC_UNSUPPORTED:
            prompt = f"Explain the difference between {c1} and {c2} as described in the study material (Page {page_number})."
            return prompt, sentence, [c1, c2]

    # 2. Contrast Pattern ("differs from", "unlike", "whereas", "in contrast")
    contrast_match = re.search(r"\b(?:differs from|unlike|whereas|in contrast|compared to)\s+([a-z0-9\s,-]+)", sentence_lower)
    if contrast_match:
        other_concept = contrast_match.group(1).strip().split()[0].title()
        if other_concept.lower() not in STOP_WORDS and len(other_concept) > 3:
            prompt = f"Explain the difference between {topic_clean} and {other_concept} as described on Page {page_number} of the text."
            return prompt, sentence, [topic_clean, other_concept]

    # 3. Cause / Purpose Pattern
    purpose_match = re.search(r"\b(?:used to|used for|in order to|designed to|helps to|thereby|aims to)\s+([a-z0-9\s,-]+)", sentence_lower)
    if purpose_match:
        action_phrase = purpose_match.group(1).strip().rstrip(".,;")
        if len(action_phrase.split()) >= 3:
            prompt = f"Why is {topic_clean} used according to the study material (Page {page_number})? Explain how it achieves '{action_phrase[:70]}'."
            return prompt, sentence, [topic_clean]

    # 4. Mechanism / Calculation Pattern
    calc_match = re.search(r"\b(?:calculated by|computed by|works by|consists of|operates by|multiplies|computes)\s+([a-z0-9\s,-]+)", sentence_lower)
    if calc_match:
        mech_phrase = calc_match.group(1).strip().rstrip(".,;")
        prompt = f"Describe how {topic_clean} works according to Page {page_number}. Detail the mechanism involving {mech_phrase[:70]}."
        return prompt, sentence, [topic_clean]

    # 5. Definition / Representation Pattern
    def_match = re.search(r"\b(?:is defined as|refers to|represents|is a technique|is a method|is an algorithm)\s+([a-z0-9\s,-]+)", sentence_lower)
    if def_match:
        def_phrase = def_match.group(1).strip().rstrip(".,;")
        prompt = f"Based on the study material (Page {page_number}), how is {topic_clean} defined, and what does it represent?"
        return prompt, sentence, [topic_clean]

    # 6. Fact-based Open Synthesizer
    prompt = f"According to Page {page_number} of the study material, explain the role and key characteristics of {topic_clean}."
    return prompt, sentence, [topic_clean]



def synthesize_prompt(topic: str, passage: str, difficulty: str = "Medium", page_number: int = 1) -> tuple[str, str, list[str]]:
    """Analyze passage structure and produce a context-aware question prompt, reference answer, and expected concepts."""
    sentences = _split_sentences(passage)
    topic_lower = topic.lower()
    topic_words = set(topic_lower.split())

    matching_sentences = [
        s for s in sentences
        if topic_lower in s.lower() or any(w in s.lower() for w in topic_words if w not in STOP_WORDS)
    ]
    if not matching_sentences:
        matching_sentences = sentences

    secondary_concept = _extract_secondary_concept(passage, topic)
    expected_concepts = [topic]
    if secondary_concept:
        expected_concepts.append(secondary_concept)

    best_prompt = None
    best_ref = None
    detected_concepts = None

    for sentence in matching_sentences:
        result = transform_sentence_to_question(sentence, topic, page_number)
        if result:
            best_prompt, best_ref, extra_concepts = result
            if extra_concepts:
                detected_concepts = extra_concepts
            if any(marker in best_prompt for marker in ["Why is", "Describe how", "defined", "difference"]):
                break

    if detected_concepts:
        expected_concepts = list(dict.fromkeys(detected_concepts))

    if not best_prompt or not best_ref:
        reference_answer = " ".join(matching_sentences[:3]) if matching_sentences else passage[:400]
        if secondary_concept:
            best_prompt = f"Explain the difference between {topic} and {secondary_concept} as described in the study material (Page {page_number})."
        else:
            best_prompt = f"Explain the concept of {topic} as presented in the study material (Page {page_number})."
        best_ref = reference_answer
    else:
        ref_idx = sentences.index(best_ref) if best_ref in sentences else 0
        context_sentences = sentences[max(0, ref_idx - 1): min(len(sentences), ref_idx + 2)]
        best_ref = " ".join(context_sentences)

    return best_prompt, best_ref, expected_concepts


def generate_questions(
    topics: list[dict] | list[str],
    source_text: str,
    count: int = 5,
    difficulty: str = "Medium",
    chunks: list[dict] | None = None,
    min_quality_threshold: float = 0.80,
    use_ollama: bool = False,
    ollama_model: str = "llama3",
    ollama_host: str = "http://localhost:11434",
) -> list[dict]:
    """Generate context-aware subjective questions strictly grounded in document passage chunks with 5-metric quality validation."""
    from app.pdf.extraction import chunk_document_pages
    from app.question_generation.ollama_generator import generate_with_ollama

    if not chunks:
        chunks = chunk_document_pages([{"page_number": 1, "text": source_text}])

    validator = QuestionValidator(min_quality_threshold=min_quality_threshold)
    questions: list[dict] = []
    existing_prompts: list[str] = []
    seen_topics: set[str] = set()

    # Lecture notes/slides that define several concepts: ask about those definitions
    # (the generic topic engine below is built for flowing prose and asks junk on slides).
    from app.question_generation.concept_generator import generate_concept_questions
    concept_questions = generate_concept_questions(chunks, count, difficulty, source_text)
    if concept_questions:
        logger.info("Concept-mined generator produced %d questions", len(concept_questions))
        return concept_questions

    for item in topics:
        if len(questions) >= count:
            break

        topic = item.get("topic", "") if isinstance(item, dict) else str(item)
        topic = topic.strip()
        if not topic or topic.lower() in seen_topics or topic.lower() in GENERIC_UNSUPPORTED:
            continue
        if any(w in GENERIC_UNSUPPORTED for w in topic.lower().split()):
            continue

        # Extract source passage context
        source_passage = item.get("source_passage", "") if isinstance(item, dict) else ""
        source_page = item.get("source_page", 1) if isinstance(item, dict) else 1
        source_section = item.get("source_section", "General Overview") if isinstance(item, dict) else "General Overview"
        source_chunk_id = item.get("source_chunk_id", 1) if isinstance(item, dict) else 1

        if not source_passage and chunks:
            topic_lower = topic.lower()
            topic_words = set(topic_lower.split())
            best_chunk = None
            best_score = -1
            for chunk in chunks:
                ctext = chunk["text"].lower()
                if topic_lower in ctext:
                    score = 100 + ctext.count(topic_lower)
                else:
                    score = sum(1 for w in topic_words if w in ctext)
                if score > best_score:
                    best_score = score
                    best_chunk = chunk
            if best_chunk and best_score > 0:
                source_passage = best_chunk["text"]
                source_page = best_chunk.get("page_number", 1)
                source_section = best_chunk.get("section", "General Overview")
                source_chunk_id = best_chunk.get("chunk_id", 1)

        if not source_passage:
            source_passage = source_text

        prompt = None
        reference_answer = None
        expected_concepts = None
        generator_source = "Rule-based Engine"

        # 1. Try Ollama Local LLM if requested
        if use_ollama:
            ollama_res = generate_with_ollama(
                topic=topic,
                passage=source_passage,
                page_number=source_page,
                model=ollama_model,
                host=ollama_host,
            )
            if ollama_res:
                o_prompt, o_ref, o_concepts = ollama_res
                # Validate Ollama candidate question through 5-metric validator
                o_v_res = validator.validate(
                    prompt=o_prompt,
                    reference_answer=o_ref,
                    topic=topic,
                    source_passage=source_passage,
                    existing_prompts=existing_prompts,
                    expected_concepts=o_concepts,
                )
                if o_v_res.accepted:
                    prompt, reference_answer, expected_concepts = o_prompt, o_ref, o_concepts
                    generator_source = f"Ollama LLM ({ollama_model})"

        # 2. Fallback to Local Rule Engine if Ollama disabled, offline, or rejected
        if not prompt or not reference_answer or not expected_concepts:
            prompt, reference_answer, expected_concepts = synthesize_prompt(topic, source_passage, difficulty, source_page)
            generator_source = "Local Rule Engine (Fallback)" if use_ollama else "Local Rule Engine"

        # Run 5-metric validation engine
        v_res = validator.validate(
            prompt=prompt,
            reference_answer=reference_answer,
            topic=topic,
            source_passage=source_passage,
            existing_prompts=existing_prompts,
            expected_concepts=expected_concepts,
        )

        logger.info(
            "Question Generation Decision [Source: %s | Topic: %s | Page: %d | Quality: %.2f%% | Status: %s] | Prompt: %s | Reason: %s",
            generator_source,
            topic,
            source_page,
            v_res.quality_score,
            "ACCEPTED" if v_res.accepted else "REJECTED",
            prompt,
            v_res.reason,
        )

        if not v_res.accepted:
            continue

        ref_tokens = set(t.lower() for t in tokenize(reference_answer) if t.lower() not in STOP_WORDS and len(t) > 3)
        expected_keywords = sorted(list(ref_tokens))[:12]

        q = Question(
            id=len(questions) + 1,
            topic=topic,
            prompt=prompt,
            difficulty=difficulty,
            reference_answer=reference_answer,
            source_page=source_page,
            source_section=source_section,
            source_chunk_id=source_chunk_id,
            source_passage=source_passage,
            expected_concepts=expected_concepts,
            expected_keywords=expected_keywords,
            quality_score=v_res.quality_score,
            validation_info=v_res.to_dict(),
        )
        questions.append(q.to_dict())
        existing_prompts.append(prompt)
        seen_topics.add(topic.lower())

    return questions



