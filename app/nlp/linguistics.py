"""Real linguistic analysis: POS tagging, WordNet lemmatisation and NER.

Each function degrades gracefully (NLTK data or models may be missing) so the
rest of the pipeline never hard-fails on a fresh laptop.
"""
from __future__ import annotations

import re
from functools import lru_cache

from app.nlp.preprocessing import simple_lemma

_WORDNET_POS = {"N": "n", "V": "v", "J": "a", "R": "r"}
NER_MODEL = "dslim/bert-base-NER"
ENTITY_LABELS = {"PER": "PERSON", "ORG": "ORGANIZATION", "LOC": "LOCATION", "MISC": "MISC"}


def pos_tag(tokens: list[str]) -> list[tuple[str, str]]:
    """Penn Treebank tags from NLTK's averaged perceptron; ('tok', 'X') if unavailable."""
    try:
        import nltk
        return nltk.pos_tag(tokens)
    except Exception:
        return [(t, "X") for t in tokens]


@lru_cache(maxsize=1)
def _lemmatizer():
    try:
        from nltk.stem import WordNetLemmatizer
        lemmatizer = WordNetLemmatizer()
        lemmatizer.lemmatize("tests")  # raises LookupError if the corpus is missing
        return lemmatizer
    except Exception:
        return None


def lemmatize_tagged(tagged: list[tuple[str, str]]) -> list[str]:
    """POS-aware WordNet lemmas ('studies'->'study', 'running'(VBG)->'run')."""
    lemmatizer = _lemmatizer()
    if lemmatizer is None:
        return [simple_lemma(token) for token, _ in tagged]
    return [lemmatizer.lemmatize(token.lower(), _WORDNET_POS.get(tag[:1], "n")) for token, tag in tagged]


def coarse_pos(tag: str) -> str:
    """Map a Penn tag to a readable universal-style label."""
    return {"N": "NOUN", "V": "VERB", "J": "ADJ", "R": "ADV"}.get(tag[:1], "OTHER")


def heuristic_entities(text: str) -> list[dict]:
    phrases = re.findall(r"\b(?:[A-Z][a-z]+\s+){0,2}[A-Z][a-z]+\b", text)
    return [{"text": p, "label": "PROPER_NOUN"} for p in dict.fromkeys(phrases)][:20]


@lru_cache(maxsize=1)
def _ner_pipeline():
    try:
        from transformers import pipeline
        return pipeline("token-classification", model=NER_MODEL, aggregation_strategy="simple")
    except Exception:
        return None


def named_entities(text: str, use_model: bool = False) -> list[dict]:
    """Entities with labels. ``use_model`` runs a local BERT NER (slower, much better)."""
    ner = _ner_pipeline() if use_model else None
    if ner is None:
        return heuristic_entities(text)
    found: dict[str, dict] = {}
    for start in range(0, len(text), 1500):  # stay inside the 512-token window
        for ent in ner(text[start:start + 1500]):
            name = ent["word"].strip()
            if len(name) > 2 and name not in found:
                found[name] = {"text": name, "label": ENTITY_LABELS.get(ent["entity_group"], ent["entity_group"]), "confidence": round(float(ent["score"]), 3)}
    return list(found.values())[:30]
