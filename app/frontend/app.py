"""Streamlit educational dashboard for the local NLP pipeline."""

from __future__ import annotations

import streamlit as st

from app.evaluation.scoring import evaluate_answer
from app.pdf.extraction import PDFExtractionError, extract_pdf
from app.question_generation.generator import generate_questions
from app.services.pipeline import analyze_document, serializable_analysis
from app.utils.config import settings


POS_COLORS = {"NOUN": "#2563eb", "VERB": "#16a34a", "ADJ": "#d97706", "ADV": "#9333ea", "OTHER": "#6b7280"}


def render_nlp_lab(doc: dict, analysis: dict) -> None:
    """Visual tour of each NLP stage, computed on the uploaded document."""
    from app.nlp.linguistics import named_entities
    from app.nlp.textrank import rank_sentences, textrank_keyphrases

    st.subheader("1. POS tagging")
    st.markdown(" ".join(f"<span style='color:{POS_COLORS.get(tag, '#6b7280')};font-weight:600' title='{tag}'>{tok}</span>" for tok, tag in analysis["pos_tags"][:80]), unsafe_allow_html=True)
    st.caption(" · ".join(f"<span style='color:{c}'>■</span> {t}" for t, c in POS_COLORS.items()), unsafe_allow_html=True)

    st.subheader("2. Tokens → lemmas (WordNet, POS-aware)")
    st.dataframe({"cleaned token": analysis["cleaned_tokens"][:40], "lemma": analysis["lemmas"][:40]}, use_container_width=True, height=200)

    st.subheader("3. TF-IDF vs. TextRank keyphrases")
    left, right = st.columns(2)
    left.caption("TF-IDF (statistical: frequent here, rare elsewhere)")
    left.dataframe(analysis["tfidf_terms"][:10], use_container_width=True)
    right.caption("TextRank (graph: central in the word co-occurrence network)")
    right.dataframe(textrank_keyphrases(doc["text"], 10), use_container_width=True)

    st.subheader("4. Concept map")
    phrases = [p["phrase"] for p in textrank_keyphrases(doc["text"], 12)]
    sentences = [x.lower() for x in doc["text"].replace("\n", " ").split(".")]
    edges = {(a, b): sum(a in x and b in x for x in sentences) for i, a in enumerate(phrases) for b in phrases[i + 1:]}
    dot = "graph G {layout=neato; overlap=false; node [shape=ellipse, style=filled, fillcolor=\"#dbeafe\", fontsize=11];" + "".join(f'"{a}" -- "{b}" [penwidth={min(5, w)}];' for (a, b), w in edges.items() if w) + "}"
    st.graphviz_chart(dot, use_container_width=True)
    st.caption("Concepts are linked when they co-occur in the same sentence; thicker edges mean more shared sentences.")

    st.subheader("5. Sentence importance (TextRank summary)")
    ranked = rank_sentences(doc["text"])
    top = {r["sentence"] for r in sorted(ranked, key=lambda r: r["score"], reverse=True)[:5]}
    for r in ranked[:12]:
        st.markdown(("🟡 **" + r["sentence"] + "**" if r["sentence"] in top else "· " + r["sentence"]) + f"  `{r['score']:.3f}`")

    st.subheader("6. Named entities")
    use_model = st.toggle("Use local BERT NER model (slower, accurate)", value=False)
    st.dataframe(named_entities(doc["text"][:6000], use_model=use_model) if use_model else analysis["named_entities"], use_container_width=True)
    st.json({"bigrams": analysis["ngrams"]["2-grams"][:10], "trigrams": analysis["ngrams"]["3-grams"][:10]}, expanded=False)


def render_search(doc: dict | None) -> None:
    """Question answering over the document: hybrid BM25 + dense retrieval."""
    from app.nlp.retrieval import highlight, search

    if not doc:
        st.warning("Analyze a PDF first.")
        return
    st.write("Ask anything about your document. Results blend **BM25** (exact terms) with **sentence embeddings** (meaning) via Reciprocal Rank Fusion, then a **cross-encoder** re-reads the best candidates.")
    query = st.text_input("Your question", placeholder="e.g. How do computers represent the meaning of words?")
    if query:
        for rank, hit in enumerate(search(query, doc.get("chunks", []), top_k=3), 1):
            st.markdown(f"#### {rank}. Page {hit['page']} · {hit['section']}")
            st.success(highlight(hit["answer_sentence"], hit["matched_terms"]))
            st.caption(f"Fused score {hit['score']:.2f} · BM25 {hit['bm25']:.2f} · semantic {hit['semantic']:.2f}{' · rerank ' + format(hit['rerank'], '.2f') if hit['rerank'] is not None else ''} · matched terms: {', '.join(hit['matched_terms']) or 'none (pure semantic match)'}")
            with st.expander("Full passage"):
                st.write(hit["passage"])


def main() -> None:
    st.set_page_config(page_title=settings.app_name, layout="wide")
    st.title("AI PDF-Based Subjective Exam Evaluator")
    st.caption("Upload study material, inspect the NLP pipeline, and take an explainable subjective exam.")

    with st.sidebar:
        st.header("⚙ Engine Settings")
        gen_engine = st.selectbox(
            "Question Generator Engine",
            ["⚡ Local NLP Engine (Fast & Offline)", "🦙 Ollama Local LLM (Llama 3 / Mistral / DeepSeek)"],
            index=0,
        )
        use_ollama = "Ollama" in gen_engine
        ollama_model = "llama3"
        ollama_host = "http://localhost:11434"

        if use_ollama:
            ollama_model = st.text_input("Ollama Model Name", value="llama3", help="e.g., llama3, mistral, gemma:2b, deepseek-r1:8b")
            ollama_host = st.text_input("Ollama Host URL", value="http://localhost:11434")
            if st.button("🔌 Test Ollama Connection"):
                from app.question_generation.ollama_generator import check_ollama_status
                res = check_ollama_status(ollama_host)
                if res["available"]:
                    st.success(res["message"])
                    if res["models"]:
                        st.caption(f"Installed models: {', '.join(res['models'])}")
                else:
                    st.error(res["message"])

    if "document" not in st.session_state:
        st.session_state.document = None
    if "questions" not in st.session_state:
        st.session_state.questions = []
    if "answers" not in st.session_state:
        st.session_state.answers = []

    tabs = st.tabs(["Home", "Upload & Analyze", "NLP Analysis", "Ask the Document", "Questions", "Exam", "Results"])
    with tabs[0]:
        st.info("The Evaluation Score estimates alignment with the extracted reference concepts; it is not absolute truth.")
        st.write("Local scoring uses TF-IDF-style lexical similarity, concept coverage, and keyword coverage. Optional transformer models are not required.")
    with tabs[1]:
        uploaded = st.file_uploader("Upload a study PDF", type=["pdf"])
        if uploaded and st.button("Analyze PDF"):
            try:
                document = extract_pdf(uploaded.getvalue(), uploaded.name, settings.max_upload_mb)
                st.session_state.document = serializable_analysis(
                    analyze_document(document.text, document.filename, document.pages, document.chunks)
                )
                st.session_state.document.update({
                    "page_count": document.page_count,
                    "warnings": document.warnings,
                    "text": document.text,
                    "pages": document.pages,
                    "chunks": document.chunks,
                })
                st.session_state.questions = []
                st.session_state.answers = []
                st.success("PDF extracted and analyzed with passage chunking.")
            except PDFExtractionError as exc:
                st.error(str(exc))
        if st.session_state.document:
            doc = st.session_state.document
            cols = st.columns(4)
            cols[0].metric("Pages", doc.get("page_count", "-"))
            cols[1].metric("Words", doc["word_count"])
            cols[2].metric("Passage Chunks", len(doc.get("chunks", [])))
            cols[3].metric("Topics", len(doc["topics"]))
            st.subheader("Summary")
            st.write(doc["summary"])
            st.subheader("Important Topics Grounded in Passages")
            st.dataframe(doc["topics"], use_container_width=True)
    with tabs[2]:
        doc = st.session_state.document
        if not doc:
            st.warning("Analyze a PDF first.")
        else:
            analysis = doc["analysis"]
            render_nlp_lab(doc, analysis)
    with tabs[3]:
        render_search(st.session_state.document)
    with tabs[4]:
        doc = st.session_state.document
        if doc:
            q_cols = st.columns(3)
            count = q_cols[0].number_input("Number of questions", min_value=1, max_value=10, value=5)
            difficulty = q_cols[1].selectbox("Difficulty", ["Easy", "Medium", "Hard"], index=1)
            debug_mode = q_cols[2].toggle("🛠 Developer / Debug Mode", value=False)

            if st.button("Generate Questions"):
                st.session_state.questions = generate_questions(
                    topics=doc["topics"],
                    source_text=doc["text"],
                    count=int(count),
                    difficulty=difficulty,
                    chunks=doc.get("chunks"),
                    min_quality_threshold=0.80,
                    use_ollama=use_ollama,
                    ollama_model=ollama_model,
                    ollama_host=ollama_host,
                )
                st.session_state.answers = []
                st.success(f"Generated {len(st.session_state.questions)} context-aware, source-grounded questions.")

        if st.session_state.questions:
            for question in st.session_state.questions:
                page_badge = f"📍 Page {question.get('source_page', 1)}"
                section_badge = f"📂 Section: {question.get('source_section', 'General Overview')}"
                quality_badge = f"⭐ Quality: {question.get('quality_score', 0):.1f}%"

                st.markdown(f"### Question {question['id']}: {question['prompt']}")
                st.caption(f"{page_badge} | {section_badge} | Topic: `{question['topic']}` | Difficulty: `{question['difficulty']}` | {quality_badge}")

                with st.expander("🔍 View Source Passage Context"):
                    st.write(question.get("source_passage", question.get("reference_answer", "")))

                v_info = question.get("validation_info", {})
                if debug_mode and v_info:
                    with st.expander("🛠 Developer / Debug Traceability & Validation Metrics", expanded=True):
                        st.markdown(f"**TOPIC:** `{question['topic']}`")
                        st.markdown(f"**SOURCE LOCATION:** Page {question.get('source_page', 1)} | Section: `{question.get('source_section', 'General')}`")
                        st.markdown(f"**GENERATED QUESTION:** {question['prompt']}")
                        st.markdown(f"**REFERENCE ANSWER:** {question['reference_answer']}")

                        st.subheader("Validation Scores Breakdown")
                        m_cols = st.columns(5)
                        m_cols[0].metric("Overall Quality", f"{v_info.get('quality_score', 0):.1f}%")
                        m_cols[1].metric("Source Support", f"{v_info.get('source_support_score', 0):.1f}%")
                        m_cols[2].metric("Topic Relevance", f"{v_info.get('topic_relevance_score', 0):.1f}%")
                        m_cols[3].metric("Answerability", f"{v_info.get('answerability_score', 0):.1f}%")
                        m_cols[4].metric("Uniqueness", f"{v_info.get('uniqueness_score', 0):.1f}%")

                        checklist = v_info.get("checklist", {})
                        st.write("Validation Checklist:", " | ".join(f"{'✓' if val else '✗'} {key.replace('_', ' ').title()}" for key, val in checklist.items()))
                        st.caption(f"Status: {v_info.get('reason', '')}")
                st.divider()
        elif not doc:
            st.warning("Analyze a PDF first.")

    with tabs[5]:
        questions = st.session_state.questions
        if not questions:
            st.warning("Generate questions first.")
        else:
            index = st.number_input("Question", 1, len(questions), 1) - 1
            question = questions[index]
            st.progress((index + 1) / len(questions))
            st.caption(f"📍 Grounded in Page {question.get('source_page', 1)} | Topic: {question['topic']}")
            st.subheader(question["prompt"])

            with st.expander("💡 Expected Reference Concepts"):
                st.write(", ".join(f"`{c}`" for c in question.get("expected_concepts", [question["topic"]])))

            answer = st.text_area("Your answer", key=f"answer_{index}", height=180)
            if st.button("Evaluate Answer"):
                result = evaluate_answer(
                    answer=answer,
                    reference_answer=question["reference_answer"],
                    concepts=question.get("expected_concepts", [question["topic"]]),
                    keywords=question.get("expected_keywords"),
                    source_page=question.get("source_page"),
                    source_passage=question.get("source_passage"),
                )
                while len(st.session_state.answers) <= index:
                    st.session_state.answers.append(None)
                st.session_state.answers[index] = {**result.to_dict(), "question_prompt": question["prompt"], "source_page": question.get("source_page", 1)}

                st.metric("Evaluation Score", f"{result.overall_score:.1f}%", delta=f"-{result.contradiction_penalty:.0f}% contradiction penalty" if result.contradiction_penalty else None)
                st.write(f"Semantic similarity: {result.semantic_similarity:.1f}% | Concept coverage: {result.concept_coverage:.1f}% | Keyword coverage: {result.keyword_coverage:.1f}%")

                if result.covered_concepts:
                    st.markdown("**Covered Concepts:** " + " ".join(f"✓ `{c}`" for c in result.covered_concepts))
                if result.missing_concepts:
                    st.markdown("**Missing Concepts:** " + " ".join(f"✗ `{c}`" for c in result.missing_concepts))

                st.info(result.feedback)
                if result.alignment:
                    st.markdown("**Idea-by-idea check** (each reference idea matched to your closest sentence)")
                    for item in result.alignment:
                        icon = "⚠️ contradicts" if item.get("contradiction", 0) > 0.6 else ("✅" if item["addressed"] else "❌")
                        st.markdown(f"{icon} *{item['idea']}*  \n&nbsp;&nbsp;&nbsp;↳ {item['matched_sentence'] or 'nothing written'} `{item['similarity']:.2f}`")
    with tabs[6]:
        answers = [item for item in st.session_state.answers if item]
        if answers:
            from app.services.report import build_markdown_report, revision_plan
            scores = [item["overall_score"] for item in answers]
            m1, m2, m3 = st.columns(3)
            m1.metric("Overall Exam Evaluation Score", f"{sum(scores) / len(scores):.1f}%")
            m2.metric("Best answer", f"{max(scores):.1f}%")
            m3.metric("Contradictions flagged", sum(len(a.get("contradictions", [])) for a in answers))
            st.bar_chart({f"Q{i + 1}": a["overall_score"] for i, a in enumerate(answers)})
            st.subheader("Score components per question")
            st.dataframe([{"Question": f"Q{i + 1}", "Semantic": a["semantic_similarity"], "Concepts": a["concept_coverage"], "Keywords": a["keyword_coverage"], "Penalty %": a.get("contradiction_penalty", 0), "Overall": a["overall_score"]} for i, a in enumerate(answers)], use_container_width=True)
            plan = revision_plan(answers, (st.session_state.document or {}).get("chunks", []))
            if plan:
                st.subheader("📚 Revise these (retrieved from your PDF)")
                for item in plan:
                    st.markdown(f"- **{item['concept']}** · page {item['page']} · {item['section']}  \n  > {item['sentence']}")
            st.download_button("⬇ Download study report (Markdown)", build_markdown_report((st.session_state.document or {}).get("filename", "document"), answers, plan), file_name="study_report.md")
            st.subheader("Detailed Question Feedback")
            for i, ans in enumerate(answers):
                with st.expander(f"Question {i+1} (Page {ans.get('source_page', 1)}): Score {ans['overall_score']:.1f}%"):
                    st.write(f"**Prompt:** {ans.get('question_prompt', '')}")
                    st.write(f"**Feedback:** {ans['feedback']}")
                    if ans.get("covered_concepts"):
                        st.write("✓ Covered:", ", ".join(ans["covered_concepts"]))
                    if ans.get("missing_concepts"):
                        st.write("✗ Missing:", ", ".join(ans["missing_concepts"]))
        else:
            st.info("Complete at least one question evaluation in the Exam tab to see results.")


if __name__ == "__main__":
    main()
