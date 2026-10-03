"""Hybrid retrieval over document chunks: BM25 + dense embeddings, fused with RRF.

Lexical search finds exact terminology; dense search finds paraphrases.
Reciprocal Rank Fusion combines them without needing comparable score scales.
"""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

from app.nlp.embeddings import embed
from app.nlp.preprocessing import STOP_WORDS, simple_lemma, tokenize
from app.nlp.textrank import split_sentences


def _terms(text: str) -> list[str]:
    return [simple_lemma(t) for t in tokenize(text) if t not in STOP_WORDS]


class BM25:
    """Okapi BM25 implemented from scratch."""

    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.docs = [Counter(_terms(d)) for d in documents]
        self.lengths = [sum(c.values()) for c in self.docs]
        self.avg_len = (sum(self.lengths) / len(self.lengths)) if self.docs else 0.0
        df: Counter = Counter()
        for doc in self.docs:
            df.update(doc.keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(len(self.docs))
        for term in set(_terms(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, doc in enumerate(self.docs):
                tf = doc.get(term, 0)
                if tf:
                    norm = 1 - self.b + self.b * self.lengths[i] / (self.avg_len or 1)
                    out[i] += idf * tf * (self.k1 + 1) / (tf + self.k1 * norm)
        return out


def _ranks(scores: np.ndarray) -> dict[int, int]:
    order = np.argsort(-scores, kind="stable")
    return {int(idx): rank for rank, idx in enumerate(order) if scores[idx] > 0}


def best_sentence(passage: str, query: str) -> str:
    """The sentence in a passage that best answers the query (BM25, dense tie-break)."""
    sentences = split_sentences(passage) or [passage]
    scores = BM25(sentences).scores(query)
    vectors = embed([query] + sentences)
    if vectors is not None:
        scores = scores / (scores.max() or 1) + vectors[1:] @ vectors[0]
    return sentences[int(np.argmax(scores))]


def search(query: str, chunks: list[dict], top_k: int = 3, rrf_k: int = 60) -> list[dict]:
    """Rank chunks for a natural-language question; returns evidence-annotated hits."""
    if not chunks or not query.strip():
        return []
    texts = [c["text"] for c in chunks]
    lexical = BM25(texts).scores(query)
    vectors = embed([query] + texts)
    dense = vectors[1:] @ vectors[0] if vectors is not None else np.zeros(len(texts))

    lex_rank, dense_rank = _ranks(lexical), _ranks(np.clip(dense, 0, None))
    fused = {i: sum(1 / (rrf_k + r[i]) for r in (lex_rank, dense_rank) if i in r) for i in set(lex_rank) | set(dense_rank)}
    hits = []
    for i, score in sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_k]:
        chunk = chunks[i]
        hits.append({
            "score": round(score * (rrf_k + 1) / 2, 4),  # 1.0 == ranked first by both retrievers
            "bm25": round(float(lexical[i]), 3),
            "semantic": round(float(dense[i]), 3),
            "page": chunk.get("page_number", 1),
            "section": chunk.get("section", "General"),
            "passage": chunk["text"],
            "answer_sentence": best_sentence(chunk["text"], query),
            "matched_terms": sorted(set(_terms(query)) & set(_terms(chunk["text"]))),
        })
    return hits


def highlight(text: str, terms: list[str]) -> str:
    """Markdown-bold every word whose lemma is in ``terms``."""
    wanted = set(terms)
    return re.sub(r"[A-Za-z][A-Za-z0-9'-]*", lambda m: f"**{m.group(0)}**" if simple_lemma(m.group(0).lower()) in wanted else m.group(0), text)
