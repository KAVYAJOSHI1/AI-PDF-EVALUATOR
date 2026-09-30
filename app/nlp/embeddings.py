"""Optional local sentence embeddings with a dependency-free fallback."""
from __future__ import annotations

from app.nlp.preprocessing import STOP_WORDS, simple_lemma, tokenize


def semantic_similarity(text_a: str, text_b: str) -> float:
    """Return a 0..1 cosine-like score; transformers never leave the machine."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        vectors = model.encode([text_a, text_b], normalize_embeddings=True)
        return max(0.0, min(1.0, float(vectors[0] @ vectors[1])))
    except Exception:
        a = set(simple_lemma(t) for t in tokenize(text_a) if t not in STOP_WORDS)
        b = set(simple_lemma(t) for t in tokenize(text_b) if t not in STOP_WORDS)
        return len(a & b) / len(a | b) if a and b else 0.0
