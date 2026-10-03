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
