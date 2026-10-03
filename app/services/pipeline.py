"""End-to-end orchestration for analysis, questions, evaluation, and summary."""
from __future__ import annotations

import re
from app.nlp.preprocessing import NLPAnalysis, analyze_text

GENERIC_TOPIC_TERMS = {
    "word", "words", "text", "texts", "sentence", "sentences", "example",
    "examples", "thing", "things", "information", "important", "using",
    "based", "base", "make", "makes", "use", "uses", "used", "study", "material",
    "section", "sections", "chapter", "chapters", "page", "pages",
    "figure", "figures", "table", "tables", "overview", "introduction", "conclusion",
    "abstract", "author", "authors", "appendix", "reference", "references", "note", "notes",
    "definition", "definitions", "summary", "exercise", "exercises", "question", "questions",
    "contents", "index", "topic", "topics", "lecture", "slide", "slides", "part", "parts",
    "unit", "units", "book", "books", "paper", "papers", "reading", "level", "levels",
    "type", "types", "case", "cases", "way", "ways", "number", "numbers", "set", "sets",
    "show", "shows", "shown", "see", "given", "following", "main", "major", "key",
    "different", "similar", "various", "general", "specific", "however", "therefore",
    "moreover", "first", "second", "third", "thus", "data", "method", "methods",
    "process", "result", "results", "learns", "learn", "identifies", "identify",
    "people", "places", "organizations", "surrounding", "context", "system", "dictionary",
}


GENERIC_PHRASE_TERMS = {"improves", "improve", "extract", "extracts", "uses", "learns", "learn"}


def extract_topics(analysis: NLPAnalysis, source_text: str = "", limit: int = 10, chunks: list[dict] | None = None) -> list[dict]:
    phrases = {term: count for key in ("2-grams", "3-grams") for term, count in analysis.ngrams.get(key, []) if len(term.split()) > 1}
    ranked = []
    for item in analysis.tfidf_terms:
        term, score = item["term"], item["score"]
        words = term.split()
        meaningful_words = [word for word in words if word.lower() not in GENERIC_TOPIC_TERMS]
        is_academic_token = "-" in term or term.isupper() or any(char.isdigit() for char in term)
        # Standalone short words are usually document vocabulary, not concepts.
        if (len(words) == 1 and len(term) < 5 and not is_academic_token) or len(words) > 6 or len(term) < 4 or term.isnumeric() or not meaningful_words or (len(words) > 1 and len(meaningful_words) != len(words)) or any(word.lower() in GENERIC_PHRASE_TERMS for word in words):
            continue
        phrase_boost = 1.35 if len(words) > 1 else 1.0
        frequency_boost = 1.0 + min(0.25, phrases.get(term, 0) * 0.05)
        ranked.append({"topic": term, "importance_score": round(min(1.0, score * phrase_boost * frequency_boost), 4), "evidence": "TF-IDF with n-gram/frequency signal"})

    # Extract multi-word compound domain concepts and domain capitalized terms (e.g. "Stemming", "Lemmatization", "TF-IDF", "Natural Language Processing")
    explicit_terms = []
    matches = re.findall(r"\b(?:[A-Z0-9-]{2,}|[A-Z][A-Za-z0-9-]+(?:\s+[A-Z][A-Za-z0-9-]+)*)\b", source_text)
    for term in set(matches):
        normalized = term.lower()
        words = normalized.split()
        if (
            len(term) >= 4
            and normalized not in GENERIC_TOPIC_TERMS
            and not any(w in GENERIC_TOPIC_TERMS for w in words)
            and not any(item["topic"].lower() == normalized for item in ranked)
        ):
            explicit_terms.append({"topic": term, "importance_score": 1.0, "evidence": "Domain-style concept in source text"})

    ranked = explicit_terms + ranked

    # Deduplicate: if a topic is a sub-phrase of a longer multi-word topic in the list, drop the shorter partial phrase
    all_topics = [item["topic"].lower() for item in ranked]
    filtered_ranked = []
    for item in ranked:
        t_low = item["topic"].lower()
        # Drop if this topic is a proper substring of a longer topic in all_topics
        if any(t_low != other and t_low in other for other in all_topics):
            continue
        filtered_ranked.append(item)

    ranked = filtered_ranked
    ranked.sort(key=lambda item: (item["evidence"].startswith("Domain"), len(item["topic"].split()) > 1, item["importance_score"]), reverse=True)





    # Attach best matching passage chunk info if chunks provided
    top_topics = ranked[:limit]
    if chunks:
        for item in top_topics:
            topic_lower = item["topic"].lower()
            topic_words = set(topic_lower.split())
            best_chunk = None
            best_score = -1
            for chunk in chunks:
                chunk_text_lower = chunk["text"].lower()
                if topic_lower in chunk_text_lower:
                    score = 100 + chunk_text_lower.count(topic_lower)
                else:
                    matched = sum(1 for word in topic_words if word in chunk_text_lower)
                    score = matched if matched > 0 else 0
                if score > best_score:
                    best_score = score
                    best_chunk = chunk
            if best_chunk and best_score > 0:
                item["source_page"] = best_chunk.get("page_number", 1)
                item["source_section"] = best_chunk.get("section", "General")
                item["source_chunk_id"] = best_chunk.get("chunk_id", 1)
                item["source_passage"] = best_chunk.get("text", "")
                item["source_id"] = best_chunk.get("source_id", f"chunk_{best_chunk.get('chunk_id', 1)}")

    return top_topics



def summarize_text(text: str, max_sentences: int = 5) -> str:
    """TextRank extractive summary (graph-based); falls back to term-frequency ranking."""
    try:
        from app.nlp.textrank import textrank_summary
        summary = textrank_summary(text, max_sentences)
        if summary:
            return summary
    except Exception:
        pass
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.split()) > 5]
    if len(sentences) <= max_sentences:
        return " ".join(sentences)
    analysis = analyze_text(text)
    terms = {item["term"].split()[0] for item in analysis.tfidf_terms[:15]}
    ranked = sorted(sentences, key=lambda sentence: sum(word.lower() in terms for word in sentence.split()), reverse=True)
    return " ".join(ranked[:max_sentences])


def analyze_document(text: str, filename: str = "document.pdf", pages: list[dict] | None = None, chunks: list[dict] | None = None) -> dict:
    from app.pdf.extraction import chunk_document_pages
    pages = pages or [{"page_number": 1, "text": text}]
    chunks = chunks or chunk_document_pages(pages)
    analysis = analyze_text(text)
    topics = extract_topics(analysis, text, limit=10, chunks=chunks)
    return {
        "filename": filename,
        "word_count": len(text.split()),
        "analysis": analysis,
        "topics": topics,
        "summary": summarize_text(text),
        "pages": pages,
        "chunks": chunks,
    }


def serializable_analysis(result: dict) -> dict:
    analysis = result["analysis"]
    return {
        **{key: value for key, value in result.items() if key != "analysis"},
        "analysis": {
            "tokens": analysis.tokens,
            "cleaned_tokens": analysis.cleaned_tokens,
            "lemmas": analysis.lemmas,
            "bow": analysis.bow,
            "tfidf_terms": analysis.tfidf_terms,
            "ngrams": analysis.ngrams,
            "pos_tags": analysis.pos_tags,
            "named_entities": analysis.named_entities,
            "cleaned_text": analysis.cleaned_text,
        },
    }

