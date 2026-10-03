"""Local sentence embeddings (cached) with a dependency-free lexical fallback."""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from app.nlp.preprocessing import STOP_WORDS, simple_lemma, tokenize

MODEL_NAME = "all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=1)
def _load_model():
    """Load the sentence-transformer once per process; None when unavailable."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(MODEL_NAME)
    except Exception:
        return None


@lru_cache(maxsize=1)
def _load_reranker():
    try:
        from sentence_transformers import CrossEncoder
        return CrossEncoder(RERANK_MODEL)
    except Exception:
        return None


def rerank_scores(query: str, passages: list[str]) -> list[float] | None:
    """Cross-encoder relevance logits (query and passage read together); None if unavailable."""
    model = _load_reranker()
    if model is None or not passages:
        return None
    return [float(x) for x in model.predict([(query, p) for p in passages], show_progress_bar=False)]


def embeddings_available() -> bool:
    return _load_model() is not None


def embed(texts: list[str]) -> np.ndarray | None:
    """Return L2-normalised embeddings (n, d), or None if no model is installed."""
    model = _load_model()
    if model is None or not texts:
        return None
    return np.asarray(model.encode(texts, normalize_embeddings=True, show_progress_bar=False))


def lexical_similarity(text_a: str, text_b: str) -> float:
    a = {simple_lemma(t) for t in tokenize(text_a) if t not in STOP_WORDS}
    b = {simple_lemma(t) for t in tokenize(text_b) if t not in STOP_WORDS}
    return len(a & b) / len(a | b) if a and b else 0.0


def semantic_similarity(text_a: str, text_b: str) -> float:
    """Return a 0..1 cosine score; transformers never leave the machine."""
    vectors = embed([text_a, text_b])
    if vectors is None:
        return lexical_similarity(text_a, text_b)
    return max(0.0, min(1.0, float(vectors[0] @ vectors[1])))
