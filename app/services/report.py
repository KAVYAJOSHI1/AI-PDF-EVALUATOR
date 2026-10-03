"""Post-exam study report: weak spots mapped back to the passages that teach them."""
from __future__ import annotations

from datetime import date

from app.nlp.retrieval import search


def revision_plan(answers: list[dict], chunks: list[dict], max_items: int = 6) -> list[dict]:
    """Missed concepts -> the passage that teaches them (hybrid search); contradicted ideas -> the correct statement."""
    plan, seen = [], set()
    for ans in answers:
        for item in ans.get("contradictions", []):
            label = ans.get("topic") or item["idea"][:60]
            if label.lower() not in seen:
                seen.add(label.lower())
                plan.append({"concept": label, "page": ans.get("source_page", 1), "section": "", "sentence": item["idea"], "reason": "contradicted"})
        for concept in ans.get("missing_concepts", []):
            if concept.lower() in seen:
                continue
            seen.add(concept.lower())
            hits = search(concept.split(" / ")[0], chunks, top_k=1)
            if hits:
                plan.append({"concept": concept, "page": hits[0]["page"], "section": hits[0]["section"], "sentence": hits[0]["answer_sentence"], "reason": "missing"})
    return plan[:max_items]


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
        lines += [f"- **{p['concept']}** (page {p['page']}{', you contradicted this' if p.get('reason') == 'contradicted' else ''}): {p['sentence']}" for p in plan]
    return "\n".join(lines) + "\n"
