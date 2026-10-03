"""Data for the NLP Lab view: every stage of the pipeline, as JSON for the web UI."""
from __future__ import annotations

import re

from app.nlp.linguistics import named_entities
from app.nlp.preprocessing import analyze_text
from app.nlp.textrank import rank_sentences, textrank_keyphrases
from app.question_generation.concept_generator import _is_noise, merge_wrapped, mine_concepts
from app.services.pipeline import GENERIC_TOPIC_TERMS


def concept_graph(chunks: list[dict], text: str, max_nodes: int = 14) -> dict:
    """Nodes = central concepts; an edge links two concepts mentioned in the same passage."""
    concepts = mine_concepts(chunks, text)[:max_nodes]
    if len(concepts) < 3:  # prose rather than slides: use TextRank keyphrases instead
        names = [k["phrase"] for k in textrank_keyphrases(text, max_nodes)]
        nodes = [{"id": n, "label": n, "weight": 1.0} for n in names]
        units = [c["text"].lower() for c in chunks] or [s.lower() for s in re.split(r"(?<=[.!?])\s+", text)]
        names_aliases = {n: {n} for n in names}
    else:
        nodes = [{"id": c.short, "label": c.short, "weight": round(c.importance, 2), "page": c.page} for c in concepts]
        units = [c["text"].lower() for c in chunks]
        names_aliases = {c.short: c.aliases | {c.short.lower()} for c in concepts}
    edges = []
    ids = [n["id"] for n in nodes]
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            shared = sum(any(x in u for x in names_aliases[a]) and any(x in u for x in names_aliases[b]) for u in units)
            if shared:
                edges.append({"source": a, "target": b, "weight": shared})
    edges.sort(key=lambda e: e["weight"], reverse=True)
    return {"nodes": nodes, "edges": edges[: int(2.2 * len(nodes))]}  # keep the strongest links so the map stays readable


LAB_GENERIC = GENERIC_TOPIC_TERMS | {"love", "example", "examples", "output", "input", "step", "total", "word", "words", "text"}


def prose_text(text: str) -> str:
    """Slide text -> sentences: drop code/table/emoji lines and end every bullet with a full stop."""
    lines = [ln for ln in merge_wrapped(text.split("\n")) if len(ln.split()) >= 4 and not _is_noise(ln)]
    return " ".join(ln if ln.endswith((".", "!", "?")) else ln + "." for ln in lines) or text


def _informative(term: str) -> bool:
    return not all(w in LAB_GENERIC for w in term.lower().split())


def build_lab(text: str, chunks: list[dict], use_ner: bool = False) -> dict:
    full_text, text = text, prose_text(text)
    analysis = analyze_text(text)
    ranked = rank_sentences(text)
    seen, lemma_rows = set(), []
    for token, lemma in zip(analysis.cleaned_tokens[:800], analysis.lemmas[:800]):
        if token != lemma and token not in seen:
            seen.add(token)
            lemma_rows.append({"token": token, "lemma": lemma})
    top_positions = {r["position"] for r in sorted(ranked, key=lambda r: r["score"], reverse=True)[:6]}
    return {
        "pos_tags": analysis.pos_tags[:120],
        "lemmas": lemma_rows[:14],
        "tfidf": [t for t in analysis.tfidf_terms if _informative(t["term"])][:12],
        "keyphrases": [k for k in textrank_keyphrases(text, 30) if _informative(k["phrase"])][:12],
        "sentences": sorted(({**r, "top": True} for r in ranked if r["position"] in top_positions), key=lambda r: r["position"]),
        "entities": named_entities(text[:6000], use_model=use_ner),
        "bigrams": analysis.ngrams["2-grams"][:10],
        "trigrams": analysis.ngrams["3-grams"][:10],
        "graph": concept_graph(chunks, full_text),
    }
