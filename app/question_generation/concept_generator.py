"""Concept-driven question generation for lecture notes and slide decks.

Slides are fragmented (titles, code, tables), so instead of asking about the
"most frequent n-grams" this module *mines the concepts the author actually
defines* ("Stemming removes prefixes or suffixes...", "- Cross Entropy / Measures
how close...") and builds questions from those definitions:

    Easy   -> define a concept
    Medium -> explain a concept (definition + supporting details)
    Hard   -> compare two related concepts (related = similar definitions)
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from app.nlp.embeddings import embed, lexical_similarity
from app.nlp.linguistics import lemmatize_tagged, pos_tag
from app.nlp.preprocessing import STOP_WORDS, tokenize

VERBS = (
    "is", "are", "means", "refers to", "stands for", "measures", "removes", "converts", "assigns", "represents",
    "reduces", "learns", "uses", "counts", "transforms", "captures", "finds", "identifies", "maps", "splits",
    "combines", "produces", "helps", "enables", "allows", "divides", "calculates", "predicts", "describes",
)
_VERB_RE = "|".join(sorted((re.escape(v) for v in VERBS), key=len, reverse=True))
DEFINITION_RE = re.compile(rf"^(?P<subject>[A-Z][A-Za-z0-9\-/' ]*?(?:\s\([A-Za-z0-9\- ]{{2,12}}\))?)\s+(?P<verb>{_VERB_RE})\s+(?P<rest>.+)$")
HEADING_RE = re.compile(r"^[\-\*•\d\.\)\s]*(?P<title>[A-Z][A-Za-z0-9\-/' ]{2,38})$")
BAD_SUBJECT_STARTS = {
    "it", "this", "that", "these", "those", "there", "every", "since", "now", "document", "answer", "example", "examples",
    "which", "what", "why", "how", "the", "if", "when", "so", "because", "here", "each", "notice", "suppose", "case",
    "step", "output", "input", "total", "then", "also", "we", "you", "i", "they", "he", "she", "let", "consider",
    "similarly", "therefore", "thus", "but", "and", "or", "unlike", "after", "before", "word", "words", "sentence",
}
BAD_PREDICATE_STARTS = {"very", "almost", "completely", "not", "strongly", "equal", "small", "large", "the same"}
BAD_TITLES = {"definition", "formula", "example", "examples", "output", "input", "steps", "code", "python", "answer", "question", "questions", "summary", "overview", "introduction", "remember", "note", "key idea", "real life example"}
DEICTIC = re.compile(r"\b(this|these|those|above|below|following)\b", re.I)
GENERIC_WORDS = {"branch", "ability", "technique", "process", "type", "form", "set", "sequence", "series", "field", "number", "kind", "unit", "way", "part", "method", "collection", "step", "result", "word", "text"}
CONNECTORS = {"of", "for", "in", "to", "the"}
SECTION_HEADS = {"advantages", "disadvantages", "limitations", "applications", "challenges", "uses", "benefits"}
CODE_HINT = re.compile(r"(=|\[|\]|\{|\}|\bprint\b|\bimport\b|\bdef\b|->|→|↓|▼|\|\||pip install|\.fit|\.transform)")


@dataclass
class Concept:
    term: str
    definition: str
    page: int
    chunk: dict
    details: list[str] = field(default_factory=list)
    aliases: set[str] = field(default_factory=set)
    importance: float = 0.0

    @property
    def short(self) -> str:
        return re.sub(r"\s*\(.*?\)", "", self.term).strip() or self.term


def clean(text: str) -> str:
    text = re.sub(r"[^\x00-ɏ‐-―‘-‟•]", " ", text)  # emoji, arrows, Devanagari...
    return re.sub(r"\s+", " ", text).strip()


def _is_noise(line: str) -> bool:
    letters = sum(c.isalpha() for c in line)
    return not line or CODE_HINT.search(line) is not None or letters < 0.6 * len(line)


def merge_wrapped(lines: list[str]) -> list[str]:
    """Re-join sentences that the slide wrapped onto a lowercase-starting next line."""
    merged: list[str] = []
    for line in (clean(ln) for ln in lines):
        if not line:
            continue
        if merged and line[0].islower() and not merged[-1].endswith((".", "?", "!", ":")):
            merged[-1] += " " + line
        else:
            merged.append(line)
    return merged


def _subject_ok(subject: str) -> bool:
    base = re.sub(r"\s*\(.*?\)", "", subject)
    words = base.split()
    if not words or len(words) > 6 or re.search(r"\b(and|or)\b", base):
        return False
    if words[0].lower() in BAD_SUBJECT_STARTS - {"word", "words"}:
        return False
    # Real terms are short noun phrases; "Calculate how samples" is a clause, not a term.
    if len(words) > 2 and any(w[0].islower() and w.lower() not in CONNECTORS for w in words[1:]):
        return False
    if len(words) == 1 and words[0].lower() in BAD_SUBJECT_STARTS:
        return False
    return len(base) >= 2


def _parse_definition(line: str) -> tuple[str, str] | None:
    """('Term', 'full definition sentence') if the line defines something."""
    stripped = re.sub(r"^[\-\*•\d\.\)\s]+", "", line)
    stripped = re.sub(r"^An?\s+(?=\w)", "", stripped)
    stripped = stripped[:1].upper() + stripped[1:]
    if stripped.endswith("?") or len(stripped.split()) < 5 or len(stripped.split()) > 60 or _is_noise(stripped):
        return None
    match = DEFINITION_RE.match(stripped)
    if not match or not _subject_ok(match["subject"]):
        return None
    rest, verb = match["rest"], match["verb"]
    predicate_words = rest.split()
    if rest.endswith(":") and verb in {"is", "are"}:
        return None
    if DEICTIC.search(rest) or predicate_words[0].lower() in BAD_PREDICATE_STARTS or len(predicate_words) < (4 if verb in {"is", "are"} else 3):
        return None
    sentence = stripped if stripped.endswith((".", "!")) else stripped + "."
    return match["subject"].strip(), sentence


def _heading_pairs(lines: list[str]) -> list[tuple[str, str]]:
    """('Cross Entropy', 'Cross Entropy measures how close...') from a title line + verb-led line."""
    out = []
    for title_line, next_line in zip(lines, lines[1:]):
        m = HEADING_RE.match(title_line)
        if not m or _is_noise(next_line):
            continue
        title = m["title"].strip()
        words = title.split()
        first = next_line.split()[0].lower() if next_line.split() else ""
        if len(words) > 4 or title.lower() in SECTION_HEADS | BAD_TITLES or words[0].lower() in BAD_SUBJECT_STARTS:
            continue
        if first in VERBS and len(next_line.split()) >= 5 and next_line[0].isupper() and not next_line.endswith("?"):
            out.append((title, f"{title} {next_line[0].lower() + next_line[1:]}"))
    return out


def _following_details(lines: list[str], idx: int) -> list[str]:
    details = []
    for nxt in lines[idx + 1: idx + 4]:
        if re.match(r"^(It|They|Its|Each)\b", nxt) and len(nxt.split()) >= 5 and not _is_noise(nxt) and not nxt.endswith("?"):
            details.append(nxt if nxt.endswith((".", "!")) else nxt + ".")
        else:
            break
    return details


def _initials(term: str) -> str:
    return "".join(w[0] for w in re.sub(r"\s*\(.*?\)", "", term).replace("-", " ").split()).lower()


def _merge_acronyms(concepts: dict[str, Concept]) -> None:
    """BoW <-> Bag of Words, LDA <-> Linear Discriminant Analysis: one concept, several names."""
    for key in list(concepts):
        c = concepts.get(key)
        if c is None or len(key) > 6 or " " in key:
            continue
        for other_key, other in list(concepts.items()):
            if other is not c and " " in other_key and _initials(other.short) == key.replace("-", ""):
                other.aliases |= c.aliases
                other.details = [c.definition] + c.details + other.details
                if "(" not in other.term:
                    other.term = f"{other.short} ({c.short})"
                del concepts[key]
                break


def _dedupe_details(c: Concept) -> list[str]:
    kept: list[str] = []
    for d in c.details:
        if lexical_similarity(d, c.definition) < 0.45 and all(lexical_similarity(d, k) < 0.45 for k in kept):
            kept.append(d)
    return kept


def mine_concepts(chunks: list[dict], full_text: str = "") -> list[Concept]:
    """Mine defined concepts from page/chunk text, ranked by how central they are."""
    concepts: dict[str, Concept] = {}
    full_lower = (full_text or " ".join(c["text"] for c in chunks)).lower()
    for chunk in chunks:
        lines = merge_wrapped(chunk["text"].split("\n"))
        found = [(i, *_parse_definition(ln)) for i, ln in enumerate(lines) if _parse_definition(ln)]
        found += [(-1, t, d) for t, d in _heading_pairs(lines)]
        for idx, term, definition in found:
            acronym = re.search(r"\(([A-Za-z0-9\- ]{2,12})\)", term)
            key = re.sub(r"\s*\(.*?\)", "", term).lower().strip()
            existing = concepts.get(key) or next((c for c in concepts.values() if key in c.aliases), None)
            details = _following_details(lines, idx) if idx >= 0 else []
            if existing:
                if idx >= 0 and len(existing.details) < 3 and definition != existing.definition:
                    existing.details.append(definition)
                existing.details.extend(d for d in details if d not in existing.details)
                if acronym:
                    existing.aliases.add(acronym.group(1).lower())
                    if "(" not in existing.term:
                        existing.term = f"{existing.short} ({acronym.group(1)})"
                continue
            c = Concept(term=term.strip(), definition=definition, page=chunk.get("page_number", 1), chunk=chunk, details=details, aliases={key})
            if acronym:
                c.aliases.add(acronym.group(1).lower())
            concepts[key] = c
    _merge_acronyms(concepts)
    for c in concepts.values():
        c.details = _dedupe_details(c)
        mentions = sum(full_lower.count(a) for a in c.aliases)
        c.importance = math.log1p(mentions) + 0.5 * min(len(c.details), 3) + (0.5 if len(c.definition.split()) >= 8 else 0)
    return sorted(concepts.values(), key=lambda c: c.importance, reverse=True)


def _key_words(text: str, exclude: str, limit: int, nouns_only: bool = False) -> list[str]:
    tagged = pos_tag([t for t in tokenize(text) if t not in STOP_WORDS and len(t) > 3])
    skip = set(tokenize(exclude))
    allowed = ("N",) if nouns_only else ("N", "J", "V")
    lemmas = lemmatize_tagged([(t, g) for t, g in tagged if g[:1] in allowed and t not in skip])
    verbish = {w for v in VERBS for w in v.split()} | {"stand", "stands", "reduce", "measure", "convert", "assign"}
    lemmas = [l for l in lemmas if l not in skip and (not nouns_only or (l not in GENERIC_WORDS and l not in verbish and len(l) > 4))]
    return list(dict.fromkeys(lemmas))[:limit]


def _answer(concept: Concept, extra: int) -> str:
    return " ".join([concept.definition] + concept.details[:extra])


def _related_pairs(concepts: list[Concept]) -> list[tuple[Concept, Concept]]:
    """Concept pairs worth comparing: definitions that are similar but not near-duplicates."""
    if len(concepts) < 2:
        return []
    vectors = embed([c.definition for c in concepts])
    pairs = []
    for i, a in enumerate(concepts):
        for j in range(i + 1, len(concepts)):
            b = concepts[j]
            if a.aliases & b.aliases or a.short.lower() in b.short.lower() or b.short.lower() in a.short.lower():
                continue
            sim = float(vectors[i] @ vectors[j]) if vectors is not None else lexical_similarity(a.definition, b.definition) * 2
            mentions_other = a.short.lower() in b.definition.lower() or b.short.lower() in a.definition.lower() or any(x in b.definition.lower() for x in a.aliases if len(x) > 2) or any(x in a.definition.lower() for x in b.aliases if len(x) > 2)
            head_in_other = any(len(x.short.split()[-1]) > 4 and x.short.split()[-1].lower() in y.definition.lower() for x, y in ((a, b), (b, a)))
            near = abs(a.page - b.page) <= 1  # sibling concepts sit on the same/adjacent slides
            score = sim + (0.25 if near else 0.0)
            if sim >= 0.3 and 0.55 <= score and sim <= 0.92 and not mentions_other and not head_in_other:
                pairs.append((score, a, b))
    pairs.sort(key=lambda p: p[0] + 0.1 * (p[1].importance + p[2].importance), reverse=True)
    return [(a, b) for _, a, b in pairs]


EASY_PROMPTS = ["Define {t}.", "What is meant by {t}?", "In your own words, what is {t}?"]
MEDIUM_PROMPTS = ["Explain {t}. What does it do, and why is it useful in NLP?", "Describe how {t} works and what problem it addresses.", "Explain the idea behind {t} and where it is used."]


def _make(qid: int, concepts: list[Concept], prompt: str, reference: str, difficulty: str, kind: str) -> dict:
    lead = concepts[0]
    words = len(reference.split())
    passage = lead.chunk["text"]
    support = 100.0 if lead.definition.split()[0].lower() in passage.lower() else 60.0
    answerability = 100.0 if 6 <= words <= 110 else 70.0
    quality = round((support + 100 + answerability + 100) / 4, 2)
    expected = []
    for c in concepts:
        acr = re.search(r"\(([A-Za-z0-9\- ]{2,12})\)", c.term)
        expected.append(" / ".join([c.short] + ([acr.group(1)] if acr else [])))
        expected += _key_words(_answer(c, 1), c.short, 1, nouns_only=True)
    return {
        "id": qid,
        "topic": " vs ".join(c.short for c in concepts),
        "prompt": prompt,
        "difficulty": difficulty,
        "reference_answer": reference,
        "source_page": lead.page,
        "source_section": lead.chunk.get("section", "General Overview"),
        "source_chunk_id": lead.chunk.get("chunk_id", 1),
        "source_passage": passage,
        "expected_concepts": list(dict.fromkeys(expected)),
        "expected_keywords": _key_words(reference, " ".join(c.short for c in concepts), 10),
        "quality_score": quality,
        "validation_info": {
            "quality_score": quality, "source_support_score": support, "topic_relevance_score": 100.0,
            "answerability_score": answerability, "uniqueness_score": 100.0,
            "checklist": {"defined_in_source": True, "answerable_from_source": words >= 6, "unique_concept": True},
            "reason": f"Concept-mined {kind} question (defined on page {lead.page}).",
        },
    }


def generate_concept_questions(chunks: list[dict], count: int = 5, difficulty: str = "Medium", full_text: str = "", concepts: list[Concept] | None = None) -> list[dict]:
    """Questions from mined concepts. Returns [] when the text defines too few concepts."""
    concepts = concepts if concepts is not None else mine_concepts(chunks, full_text)
    if len(concepts) < 3:
        return []
    out: list[dict] = []
    used: set[str] = set()

    def take(c: Concept) -> bool:
        return c.short.lower() not in used

    if difficulty == "Hard":
        for a, b in _related_pairs(concepts[:30]):
            if len(out) >= count or not (take(a) and take(b)):
                continue
            used |= {a.short.lower(), b.short.lower()}
            out.append(_make(len(out) + 1, [a, b], f"Compare and contrast {a.short} and {b.short}. How do they differ in purpose and behaviour?", f"{_answer(a, 1)} {_answer(b, 1)}", "Hard", "comparison"))
    ordered = sorted(concepts, key=lambda c: -min(len(c.details), 1)) if difficulty in ("Medium", "Hard") else concepts
    for c in ordered:
        if len(out) >= count:
            break
        if not take(c):
            continue
        used.add(c.short.lower())
        i = len(out)
        if difficulty == "Easy":
            out.append(_make(i + 1, [c], EASY_PROMPTS[i % 3].format(t=c.short), c.definition, "Easy", "definition"))
        elif difficulty == "Hard":
            out.append(_make(i + 1, [c], f"Discuss {c.short} in depth: define it, explain how it works, and describe where it is applied.", _answer(c, 3), "Hard", "deep-dive"))
        else:
            out.append(_make(i + 1, [c], MEDIUM_PROMPTS[i % 3].format(t=c.short), _answer(c, 2), "Medium", "explanation"))
    return out
