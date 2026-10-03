"""Transparent, lightweight NLP preprocessing and lexical analysis."""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

STOP_WORDS = set("a an the and or but if then than is are was were be been being to of in on for with by from as at into through about this that these those it its their his her our your we they he she you i not no do does did can could should would will may might have has had".split())
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:[-'][A-Za-z0-9]+)*")


@dataclass
class NLPAnalysis:
    tokens: list[str]
    cleaned_tokens: list[str]
    lemmas: list[str]
    bow: dict[str, int]
    tfidf_terms: list[dict[str, float]]
    ngrams: dict[str, list[tuple[str, int]]]
    pos_tags: list[tuple[str, str]]
    named_entities: list[dict[str, str]]
    cleaned_text: str


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def simple_lemma(token: str) -> str:
    token = token.lower()
    if len(token) > 5 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def make_ngrams(tokens: list[str], n: int) -> list[tuple[str, int]]:
    return Counter(" ".join(tokens[i:i+n]) for i in range(max(0, len(tokens)-n+1))).most_common(20)


def _tfidf_terms(text: str) -> list[dict[str, float]]:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(stop_words="english", token_pattern=r"(?u)\b[\w-]{2,}\b", ngram_range=(1, 3), max_features=100)
        documents = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
        matrix = vectorizer.fit_transform(documents or [text])
        scores = matrix.toarray().mean(axis=0)
        terms = vectorizer.get_feature_names_out()
        return [{"term": term, "score": round(float(score), 4)} for term, score in sorted(zip(terms, scores), key=lambda item: item[1], reverse=True) if score > 0][:20]
    except (ModuleNotFoundError, ValueError):
        counts = Counter(text.split())
        return [{"term": term, "score": round(count / max(counts.values()), 4)} for term, count in counts.most_common(20)]


def analyze_text(text: str) -> NLPAnalysis:
    tokens = tokenize(text)
    cleaned_tokens = [token for token in tokens if token not in STOP_WORDS]
    from app.nlp.linguistics import coarse_pos, lemmatize_tagged, named_entities, pos_tag
    tagged_all = pos_tag(tokens[:5000])
    tagged_clean = pos_tag(cleaned_tokens)
    lemmas = lemmatize_tagged(tagged_clean)
    cleaned_text = " ".join(lemmas)
    tfidf_sentences = [
        " ".join(token for token in tokenize(sentence) if token not in STOP_WORDS)
        for sentence in re.split(r"(?<=[.!?])\s+", text)
    ]
    tfidf_text = ". ".join(sentence for sentence in tfidf_sentences if sentence)
    entities = named_entities(text)
    pos_tags = [(token, coarse_pos(tag)) for token, tag in tagged_all[:100]]
    return NLPAnalysis(tokens, cleaned_tokens, lemmas, dict(Counter(lemmas)), _tfidf_terms(tfidf_text), {f"{n}-grams": make_ngrams(lemmas, n) for n in (1, 2, 3)}, pos_tags, entities, cleaned_text)
