"""Natural language inference: does one sentence entail or contradict another?

Embedding similarity cannot tell "TF-IDF uses term frequency" from "TF-IDF
ignores term frequency" (they are near-identical vectors). An NLI model can.
"""
from __future__ import annotations

from functools import lru_cache

NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"


@lru_cache(maxsize=1)
def _load():
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL).eval()
        labels = {int(i): label.lower() for i, label in model.config.id2label.items()}
        return tokenizer, model, labels
    except Exception:
        return None


def nli_available() -> bool:
    return _load() is not None


def nli_probs(premise: str, hypothesis: str) -> dict[str, float] | None:
    """{'entailment','neutral','contradiction'} probabilities, or None without the model."""
    loaded = _load()
    if loaded is None:
        return None
    import torch
    tokenizer, model, labels = loaded
    inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        probs = model(**inputs).logits.softmax(-1)[0]
    return {labels[i]: round(float(p), 4) for i, p in enumerate(probs)}
