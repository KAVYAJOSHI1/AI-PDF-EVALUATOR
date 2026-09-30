"""Ollama Local LLM Generator Integration with strict source grounding and JSON parsing."""
from __future__ import annotations

import json
import logging
import requests
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llama3"


def check_ollama_status(host: str = DEFAULT_OLLAMA_HOST) -> dict[str, Any]:
    """Check if Ollama service is running and retrieve list of installed local models."""
    try:
        response = requests.get(f"{host.rstrip('/')}/api/tags", timeout=3)
        if response.status_code == 200:
            data = response.json()
            models = [m.get("name") for m in data.get("models", [])]
            return {"available": True, "models": models, "message": f"Ollama online with {len(models)} model(s)."}
        return {"available": False, "models": [], "message": f"Ollama returned HTTP status {response.status_code}."}
    except Exception as err:
        return {"available": False, "models": [], "message": f"Ollama connection failed: {err}"}


def generate_with_ollama(
    topic: str,
    passage: str,
    page_number: int = 1,
    model: str = DEFAULT_OLLAMA_MODEL,
    host: str = DEFAULT_OLLAMA_HOST,
    timeout: float = 12.0,
) -> tuple[str, str, list[str]] | None:
    """Generate a source-grounded subjective question using local Ollama model.
    
    Returns (prompt, reference_answer, expected_concepts) or None if Ollama fails/times out.
    """
    system_prompt = (
        "You are an expert academic exam creator. Your task is to generate ONE high-quality, "
        "source-grounded subjective exam question strictly derived from the provided passage chunk. "
        "DO NOT use outside knowledge or introduce concepts not present in the passage. "
        "Output ONLY valid JSON with keys 'prompt', 'reference_answer', and 'expected_concepts' (array of strings)."
    )

    user_prompt = (
        f"TARGET TOPIC: {topic}\n"
        f"SOURCE PASSAGE (Page {page_number}):\n\"\"\"{passage}\"\"\"\n\n"
        "Generate a subjective question that tests understanding of the topic based ONLY on the passage above. "
        "Include Page reference in prompt (e.g., '...according to Page 1').\n"
        "Respond in JSON format like:\n"
        "{\n"
        "  \"prompt\": \"Explain how... as described on Page 1.\",\n"
        "  \"reference_answer\": \"According to the text...\",\n"
        "  \"expected_concepts\": [\"concept1\", \"concept2\"]\n"
        "}"
    )

    try:
        url = f"{host.rstrip('/')}/api/generate"
        payload = {
            "model": model,
            "prompt": f"{system_prompt}\n\n{user_prompt}",
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.2, "top_p": 0.9},
        }

        response = requests.post(url, json=payload, timeout=timeout)
        if response.status_code != 200:
            logger.warning("Ollama HTTP %d: %s", response.status_code, response.text[:200])
            return None

        result = response.json()
        raw_text = result.get("response", "").strip()
        if not raw_text:
            return None

        data = json.loads(raw_text)
        prompt = data.get("prompt", "").strip()
        ref_answer = data.get("reference_answer", "").strip()
        concepts = data.get("expected_concepts", [])

        if isinstance(concepts, str):
            concepts = [concepts]
        concepts = [str(c).strip() for c in concepts if str(c).strip()]
        if not concepts:
            concepts = [topic]

        if prompt and ref_answer and len(prompt.split()) >= 6:
            return prompt, ref_answer, concepts

        return None

    except Exception as err:
        logger.warning("Ollama generation failed or timed out: %s", err)
        return None
