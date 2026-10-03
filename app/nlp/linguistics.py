"""Real linguistic analysis: POS tagging, WordNet lemmatisation and NER.

Each function degrades gracefully (NLTK data or models may be missing) so the
rest of the pipeline never hard-fails on a fresh laptop.
"""
from __future__ import annotations

import re
from functools import lru_cache

from app.nlp.preprocessing import STOP_WORDS, simple_lemma

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
    """Capitalised phrases. A lone capitalised word only counts if it never appears in lowercase
    (so sentence-initial 'How' or 'Because' are not mistaken for names)."""
    lower_words = set(re.findall(r"\b[a-z][a-z'-]*\b", text))
    found = []
    for phrase in dict.fromkeys(re.findall(r"\b(?:[A-Z][a-z]+\s+){0,2}[A-Z][a-z]+\b", text)):
        words = phrase.split()
        if len(words) == 1 and (phrase.lower() in lower_words or phrase.lower() in STOP_WORDS):
            continue
        if words[0].lower() in STOP_WORDS | {"how", "what", "why", "which", "because", "examples", "example", "module"}:
            words = words[1:]
            if not words:
                continue
            phrase = " ".join(words)
            if len(words) == 1 and phrase.lower() in lower_words:
                continue
        found.append({"text": phrase, "label": "PROPER_NOUN"})
    return found[:20]


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
            if len(name) > 2 and "##" not in name and ent["score"] >= 0.6 and name not in found:
                found[name] = {"text": name, "label": ENTITY_LABELS.get(ent["entity_group"], ent["entity_group"]), "confidence": round(float(ent["score"]), 3)}
    names = list(found)
    # drop word-piece fragments such as 'Art' (from 'Artificial') that are prefixes of a longer entity
    return [v for k, v in found.items() if not any(o != k and o.startswith(k) and not o.startswith(k + " ") for o in names)][:30]
