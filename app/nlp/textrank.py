"""Graph-based TextRank: extractive summarisation and keyphrase extraction.

Both are PageRank over a graph built from the text itself (no training):
sentences are nodes linked by TF-IDF cosine similarity; words are nodes linked
by co-occurrence inside a sliding window.
"""
from __future__ import annotations

import re
from collections import defaultdict

import numpy as np

from app.nlp.linguistics import pos_tag
from app.nlp.preprocessing import STOP_WORDS, tokenize


def pagerank(matrix: np.ndarray, damping: float = 0.85, iterations: int = 100, tol: float = 1e-6) -> np.ndarray:
    """Power-iteration PageRank on a weighted adjacency matrix."""
    n = matrix.shape[0]
    if n == 0:
        return np.array([])
    row_sums = matrix.sum(axis=1, keepdims=True)
    transition = np.divide(matrix, row_sums, out=np.full_like(matrix, 1.0 / n), where=row_sums > 0)
    rank = np.full(n, 1.0 / n)
    for _ in range(iterations):
        new = (1 - damping) / n + damping * transition.T @ rank
        if np.abs(new - rank).sum() < tol:
            return new
        rank = new
    return rank


def split_sentences(text: str) -> list[str]:
    try:
        from nltk.tokenize import sent_tokenize
        sentences = sent_tokenize(text)
    except Exception:
        sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.split()) >= 6]


def rank_sentences(text: str) -> list[dict]:
    """Every sentence with its TextRank score and original position."""
    sentences = split_sentences(text)
    if len(sentences) < 2:
        return [{"sentence": s, "score": 1.0, "position": i} for i, s in enumerate(sentences)]
    from sklearn.feature_extraction.text import TfidfVectorizer
    try:
        vectors = TfidfVectorizer(stop_words="english").fit_transform(sentences)
    except ValueError:
        return [{"sentence": s, "score": 0.0, "position": i} for i, s in enumerate(sentences)]
    graph = (vectors @ vectors.T).toarray()
    np.fill_diagonal(graph, 0.0)
    scores = pagerank(graph)
    return [{"sentence": s, "score": round(float(sc), 5), "position": i} for i, (s, sc) in enumerate(zip(sentences, scores))]


def textrank_summary(text: str, max_sentences: int = 5) -> str:
    """Top sentences by TextRank, re-ordered to follow the document's flow."""
    ranked = rank_sentences(text)
    top = sorted(sorted(ranked, key=lambda r: r["score"], reverse=True)[:max_sentences], key=lambda r: r["position"])
    return " ".join(r["sentence"] for r in top)


def textrank_keyphrases(text: str, top_n: int = 10, window: int = 4, max_words: int = 3) -> list[dict]:
    """Keyphrases: PageRank over noun/adjective co-occurrence, merged into phrases.

    Sentences are processed separately so phrases never straddle a full stop.
    """
    sentences = [pos_tag(tokenize(s)) for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
    flagged = [[(t, tag[:1] in ("N", "J") and t not in STOP_WORDS and len(t) > 2) for t, tag in sent] for sent in sentences]
    words = sorted({t for sent in flagged for t, ok in sent if ok})
    if len(words) < 2:
        return []
    index = {w: i for i, w in enumerate(words)}
    graph = np.zeros((len(words), len(words)))
    for sent in flagged:
        kept = [t for t, ok in sent if ok]
        for i, a in enumerate(kept):
            for b in kept[i + 1:i + window]:
                if a != b:
                    graph[index[a], index[b]] += 1
                    graph[index[b], index[a]] += 1
    scores = dict(zip(words, pagerank(graph)))
    phrases: dict[str, float] = defaultdict(float)
    for sent in flagged:
        run: list[str] = []
        for token, ok in sent + [("", False)]:
            if ok and len(run) < max_words:
                run.append(token)
                continue
            if run:
                phrases[" ".join(run)] += sum(scores[w] for w in run)
            run = [token] if ok else []
    ranked = sorted(phrases.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    best = float(ranked[0][1]) if ranked else 1.0
    return [{"phrase": p, "score": round(float(s) / best, 4)} for p, s in ranked]
