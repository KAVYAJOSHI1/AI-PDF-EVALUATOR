"""Post-exam study report: weak spots mapped back to the passages that teach them."""
from __future__ import annotations

from datetime import date

from app.nlp.retrieval import search


def revision_plan(answers: list[dict], chunks: list[dict], max_items: int = 6) -> list[dict]:
    """For each missed concept, retrieve the passage that covers it (hybrid search)."""
    plan, seen = [], set()
    for ans in answers:
        for concept in ans.get("missing_concepts", []) + [c["idea"] for c in ans.get("contradictions", [])]:
            if concept.lower() in seen:
                continue
            seen.add(concept.lower())
            hits = search(concept, chunks, top_k=1)
            if hits:
                plan.append({"concept": concept, "page": hits[0]["page"], "section": hits[0]["section"], "sentence": hits[0]["answer_sentence"]})
            if len(plan) >= max_items:
                return plan
    return plan


def build_markdown_report(filename: str, answers: list[dict], plan: list[dict]) -> str:
    scores = [a["overall_score"] for a in answers]
    lines = [f"# Study Report: {filename}", f"_{date.today().isoformat()}_", "",
             f"**Overall evaluation score: {sum(scores) / len(scores):.1f}%** across {len(scores)} question(s).", "", "## Question breakdown"]
    for i, a in enumerate(answers, 1):
        lines += [f"### {i}. {a.get('question_prompt', '')}", f"- Score: {a['overall_score']:.1f}% (page {a.get('source_page', 1)})", f"- Feedback: {a['feedback']}"]
        if a.get("covered_concepts"):
            lines.append("- Covered: " + ", ".join(a["covered_concepts"]))
        if a.get("missing_concepts"):
            lines.append("- Missing: " + ", ".join(a["missing_concepts"]))
        lines.append("")
    if plan:
        lines.append("## Revise these")
        lines += [f"- **{p['concept']}** (page {p['page']}, {p['section']}): {p['sentence']}" for p in plan]
    return "\n".join(lines) + "\n"
