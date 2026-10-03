"""Hybrid retrieval over document chunks: BM25 + dense embeddings, fused with RRF.

Lexical search finds exact terminology; dense search finds paraphrases.
Reciprocal Rank Fusion combines them without needing comparable score scales.
"""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

from app.nlp.embeddings import embed, rerank_scores
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


def _body(text: str) -> str:
    """Drop heading-like lines (short, no end punctuation) so they don't fuse with sentences."""
    lines = [ln for ln in text.split("\n") if ln.strip() and not (len(ln.split()) < 8 and not ln.rstrip().endswith((".", "?", "!")))]
    return " ".join(lines) or text


def search(query: str, chunks: list[dict], top_k: int = 3, rrf_k: int = 60) -> list[dict]:
    """Rank sentences for a natural-language question; returns evidence-annotated hits.

    Sentences are the retrieval unit (long chunks dilute both BM25 and embeddings);
    each hit keeps its parent passage for context. One hit per passage.
    """
    if not chunks or not query.strip():
        return []
    units = [(i, re.sub(r"\s+", " ", sent)) for i, c in enumerate(chunks) for sent in (split_sentences(_body(c["text"])) or [c["text"]])]
    texts = [u[1] for u in units]
    lexical = BM25(texts).scores(query)
    vectors = embed([query] + texts)
    dense = vectors[1:] @ vectors[0] if vectors is not None else np.zeros(len(texts))

    lex_rank, dense_rank = _ranks(lexical), _ranks(np.clip(dense, 0, None))
    fused = {i: sum(1 / (rrf_k + r[i]) for r in (lex_rank, dense_rank) if i in r) for i in set(lex_rank) | set(dense_rank)}
    # Stage 2: a cross-encoder re-reads (query, sentence) pairs for the best candidates.
    candidates = [i for i, _ in sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:20]]
    rerank = rerank_scores(query, [texts[i] for i in candidates])
    reranked = dict(zip(candidates, rerank)) if rerank else {}
    order = sorted(candidates, key=lambda i: reranked[i], reverse=True) if rerank else candidates
    hits, seen = [], set()
    for i in order:
        score = fused[i]
        chunk_idx, sentence = units[i]
        if chunk_idx in seen:
            continue
        seen.add(chunk_idx)
        chunk = chunks[chunk_idx]
        hits.append({
            "score": round(score * (rrf_k + 1) / 2, 4),  # 1.0 == ranked first by both retrievers
            "bm25": round(float(lexical[i]), 3),
            "semantic": round(float(dense[i]), 3),
            "rerank": round(reranked[i], 3) if rerank else None,
            "page": chunk.get("page_number", 1),
            "section": chunk.get("section", "General"),
            "passage": chunk["text"],
            "answer_sentence": sentence,
            "matched_terms": sorted(set(_terms(query)) & set(_terms(sentence))),
        })
        if len(hits) == top_k:
            break
    return hits


def highlight(text: str, terms: list[str]) -> str:
    """Markdown-bold every word whose lemma is in ``terms``."""
    wanted = set(terms)
    return re.sub(r"[A-Za-z][A-Za-z0-9'-]*", lambda m: f"**{m.group(0)}**" if simple_lemma(m.group(0).lower()) in wanted else m.group(0), text)
