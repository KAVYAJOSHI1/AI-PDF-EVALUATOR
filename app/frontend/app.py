"""Streamlit educational dashboard for the local NLP pipeline."""

from __future__ import annotations

import streamlit as st

from app.evaluation.scoring import evaluate_answer
from app.pdf.extraction import PDFExtractionError, extract_pdf
from app.question_generation.generator import generate_questions
from app.services.pipeline import analyze_document, serializable_analysis
from app.utils.config import settings


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

    tabs = st.tabs(["Home", "Upload & Analyze", "NLP Analysis", "Questions", "Exam", "Results"])
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
            st.write("Tokens", analysis["tokens"][:100])
            st.write("Lemmas", analysis["lemmas"][:100])
            st.dataframe(analysis["tfidf_terms"], use_container_width=True)
            st.json({"bigrams": analysis["ngrams"]["2-grams"][:10], "trigrams": analysis["ngrams"]["3-grams"][:10], "named_entities": analysis["named_entities"]})
    with tabs[3]:
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

    with tabs[4]:
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

                st.metric("Evaluation Score", f"{result.overall_score:.1f}%")
                st.write(f"Semantic similarity: {result.semantic_similarity:.1f}% | Concept coverage: {result.concept_coverage:.1f}% | Keyword coverage: {result.keyword_coverage:.1f}%")

                if result.covered_concepts:
                    st.markdown("**Covered Concepts:** " + " ".join(f"✓ `{c}`" for c in result.covered_concepts))
                if result.missing_concepts:
                    st.markdown("**Missing Concepts:** " + " ".join(f"✗ `{c}`" for c in result.missing_concepts))

                st.info(result.feedback)
    with tabs[5]:
        answers = [item for item in st.session_state.answers if item]
        if answers:
            scores = [item["overall_score"] for item in answers]
            st.metric("Overall Exam Evaluation Score", f"{sum(scores) / len(scores):.1f}%")
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
