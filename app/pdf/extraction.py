"""PDF validation and text extraction using PyMuPDF when available."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class PDFExtractionError(ValueError):
    """Raised when a PDF cannot be processed into useful text."""


import re


@dataclass
class PassageChunk:
    chunk_id: int
    page_number: int
    section: str
    text: str
    word_count: int
    start_char: int
    end_char: int
    source_id: str = ""

    def __post_init__(self) -> None:
        if not self.source_id:
            self.source_id = f"page_{self.page_number}_chunk_{self.chunk_id}"

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "page_number": self.page_number,
            "section": self.section,
            "text": self.text,
            "word_count": self.word_count,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "source_id": self.source_id,
        }


@dataclass
class ExtractedDocument:
    filename: str
    page_count: int
    text: str
    word_count: int
    warnings: list[str]
    pages: list[dict]
    chunks: list[dict]


def _is_section_header(line: str) -> str | None:
    """Detect section header lines like '1. Text Vectorization' or 'Overview'."""
    line_clean = line.strip()
    if not line_clean or len(line_clean) > 60:
        return None
    # Section header patterns
    match = re.match(r"^(?:[0-9]+(?:\.[0-9]+)*\s+)?([A-Z][A-Za-z0-9\s-]{2,50})$", line_clean)
    if match and not line_clean.endswith((".", "?", "!", ":", ";", ",")):
        return match.group(0).strip()
    return None


def chunk_document_pages(pages: list[dict]) -> list[dict]:
    """Split page texts into coherent passage chunks (100–350 words) preserving page numbers and section headers."""
    chunks: list[dict] = []
    chunk_id = 1
    running_offset = 0
    current_section = "General Overview"

    for page in pages:
        page_num = page.get("page_number", 1)
        page_text = page.get("text", "").strip()
        if not page_text:
            continue

        paragraphs = [p.strip() for p in page_text.split("\n\n") if p.strip()]
        if not paragraphs:
            paragraphs = [p.strip() for p in page_text.split("\n") if p.strip()]

        current_chunk_words: list[str] = []
        current_chunk_text_parts: list[str] = []

        for p in paragraphs:
            # Check if paragraph is a section header
            lines = p.split("\n")
            first_line_header = _is_section_header(lines[0])
            if first_line_header:
                current_section = first_line_header

            words = p.split()
            if not words:
                continue

            # Save current chunk if combining exceeds 350 words or hits new major section boundary
            if len(current_chunk_words) + len(words) > 350 and len(current_chunk_words) >= 80:
                chunk_str = " ".join(current_chunk_text_parts)
                chunks.append(
                    PassageChunk(
                        chunk_id=chunk_id,
                        page_number=page_num,
                        section=current_section,
                        text=chunk_str,
                        word_count=len(current_chunk_words),
                        start_char=running_offset,
                        end_char=running_offset + len(chunk_str),
                    ).to_dict()
                )
                running_offset += len(chunk_str) + 1
                chunk_id += 1
                current_chunk_words = list(words)
                current_chunk_text_parts = [p]
            else:
                current_chunk_words.extend(words)
                current_chunk_text_parts.append(p)

        if current_chunk_words:
            chunk_str = " ".join(current_chunk_text_parts)
            chunks.append(
                PassageChunk(
                    chunk_id=chunk_id,
                    page_number=page_num,
                    section=current_section,
                    text=chunk_str,
                    word_count=len(current_chunk_words),
                    start_char=running_offset,
                    end_char=running_offset + len(chunk_str),
                ).to_dict()
            )
            running_offset += len(chunk_str) + 1
            chunk_id += 1

    return chunks



def extract_pdf(data: bytes, filename: str = "document.pdf", max_upload_mb: int = 25) -> ExtractedDocument:
    if not filename.lower().endswith(".pdf"):
        raise PDFExtractionError("Only PDF files are accepted.")
    if not data:
        raise PDFExtractionError("The uploaded PDF is empty.")
    if len(data) > max_upload_mb * 1024 * 1024:
        raise PDFExtractionError(f"The PDF exceeds the {max_upload_mb} MB upload limit.")
    try:
        import fitz
    except ModuleNotFoundError as exc:
        raise PDFExtractionError("PyMuPDF is required for PDF extraction. Install requirements.txt.") from exc
    try:
        document = fitz.open(stream=data, filetype="pdf")
        pages = []
        page_texts = []
        for i, page in enumerate(document):
            page_text = page.get_text("text").strip()
            pages.append({"page_number": i + 1, "text": page_text})
            if page_text:
                page_texts.append(page_text)
        text = "\n\n".join(page_texts).strip()
        page_count = len(document)
        document.close()
    except Exception as exc:
        raise PDFExtractionError(f"The PDF could not be read: {exc}") from exc
    if not text:
        raise PDFExtractionError("The PDF contains no extractable text. It may be scanned; OCR is not enabled yet.")
    warnings = []
    if len(text.split()) < 80:
        warnings.append("Very little text was extracted; topic and question quality may be limited.")

    chunks = chunk_document_pages(pages)
    return ExtractedDocument(filename, page_count, text, len(text.split()), warnings, pages, chunks)


def extract_pdf_file(path: str | Path, max_upload_mb: int = 25) -> ExtractedDocument:
    path = Path(path)
    return extract_pdf(path.read_bytes(), path.name, max_upload_mb)

