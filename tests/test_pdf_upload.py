import fitz
import pytest

from app.pdf.extraction import PDFExtractionError, extract_pdf


def _pdf(*pages: str) -> bytes:
    doc = fitz.open()
    for text in pages:
        doc.new_page().insert_textbox(fitz.Rect(50, 50, 550, 780), text, fontsize=11)
    return doc.tobytes()


def test_extract_pdf_returns_pages_and_chunks():
    result = extract_pdf(_pdf("Tokenization splits text into words. " * 5, "Embeddings map words to vectors. " * 5), "a.pdf")
    assert result.page_count == 2 and result.chunks and result.word_count > 20


def test_extract_pdf_rejects_bad_input():
    with pytest.raises(PDFExtractionError):
        extract_pdf(b"", "a.pdf")
    with pytest.raises(PDFExtractionError):
        extract_pdf(b"junk", "a.txt")
    with pytest.raises(PDFExtractionError):
        extract_pdf(b"junk", "a.pdf")


def test_blank_answer_scores_without_crashing():
    from app.evaluation.scoring import evaluate_answer
    result = evaluate_answer("", "Stemming removes suffixes to get the root form of a word.", ["Stemming"])
    assert result.overall_score == 0 and result.missing_concepts == ["Stemming"]


def test_lab_and_report_endpoints():
    from fastapi.testclient import TestClient

    from app.api.main import app
    from app.pdf.extraction import chunk_document_pages

    text = "Stemming removes suffixes to obtain the root form of a word. Lemmatization converts words to their dictionary base form. Stop words are common words that carry little meaning. Tokenization is the process of dividing text into smaller units called tokens."
    chunks = chunk_document_pages([{"page_number": 1, "text": text}])
    client = TestClient(app)
    lab = client.post("/api/lab", json={"text": text, "chunks": chunks}).json()
    assert lab["keyphrases"] and lab["graph"]["nodes"] and lab["pos_tags"]
    answers = [{"overall_score": 40, "feedback": "x", "missing_concepts": ["Stemming"], "covered_concepts": [], "question_prompt": "q", "source_page": 1}]
    rep = client.post("/api/report", json={"answers": answers, "chunks": chunks}).json()
    assert rep["plan"][0]["concept"] == "Stemming" and "Study Report" in rep["markdown"]


def test_web_ui_is_served():
    from fastapi.testclient import TestClient

    from app.api.main import app

    client = TestClient(app)
    page = client.get("/")
    assert page.status_code == 200 and "ExamLens" in page.text
    assert client.get("/static/app.js").status_code == 200 and client.get("/static/app.css").status_code == 200
