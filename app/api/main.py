"""FastAPI layer over the local, explainable NLP services."""
from __future__ import annotations

from app.utils.config import settings
from app.utils.logging_config import configure_logging

configure_logging(settings.log_level)

try:
    from fastapi import FastAPI, File, HTTPException, UploadFile
    from pydantic import BaseModel, Field
    from app.evaluation.scoring import evaluate_answer
    from app.pdf.extraction import PDFExtractionError, extract_pdf
    from app.question_generation.generator import generate_questions
    from app.services.pipeline import analyze_document, serializable_analysis
except ModuleNotFoundError:
    FastAPI = None  # type: ignore[assignment]


if FastAPI is not None:
    class TextRequest(BaseModel):
        text: str = Field(min_length=1)
        filename: str = "document.pdf"

    class QuestionRequest(BaseModel):
        text: str = Field(min_length=1)
        topics: list[dict] = []
        chunks: list[dict] = []
        count: int = Field(default=5, ge=1, le=20)
        difficulty: str = "Medium"
        min_quality_threshold: float = Field(default=0.80, ge=0.0, le=1.0)
        debug_mode: bool = False
        use_ollama: bool = False
        ollama_model: str = "llama3"
        ollama_host: str = "http://localhost:11434"

    class EvaluationRequest(BaseModel):
        answer: str
        reference_answer: str
        concepts: list[str] = []
        keywords: list[str] = []
        source_page: int | None = None
        source_passage: str | None = None

    app = FastAPI(title=settings.app_name, version="1.0.0")

    @app.get("/api/health")
    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": settings.app_name}

    @app.get("/api/ollama/status")
    def ollama_status(host: str = "http://localhost:11434") -> dict:
        from app.question_generation.ollama_generator import check_ollama_status
        return check_ollama_status(host)

    @app.post("/api/upload")
    async def upload(file: UploadFile = File(...)) -> dict:
        try:
            document = extract_pdf(await file.read(), file.filename or "document.pdf", settings.max_upload_mb)
            result = analyze_document(document.text, document.filename, document.pages, document.chunks)
            payload = serializable_analysis(result)
            payload.update({"page_count": document.page_count, "warnings": document.warnings})
            return payload
        except PDFExtractionError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/analyze")
    def analyze(request: TextRequest) -> dict:
        return serializable_analysis(analyze_document(request.text, request.filename))

    @app.post("/api/questions")
    def questions(request: QuestionRequest) -> dict:
        result = analyze_document(request.text)
        qs = generate_questions(
            topics=request.topics or result["topics"],
            source_text=request.text,
            count=request.count,
            difficulty=request.difficulty,
            chunks=request.chunks or result.get("chunks"),
            min_quality_threshold=request.min_quality_threshold,
            use_ollama=request.use_ollama,
            ollama_model=request.ollama_model,
            ollama_host=request.ollama_host,
        )
        if not request.debug_mode:
            # Clean debug metadata if debug_mode is off
            for q in qs:
                q.pop("validation_info", None)
        return {"questions": qs}


    @app.post("/api/evaluate")
    def evaluate(request: EvaluationRequest) -> dict:
        return evaluate_answer(
            answer=request.answer,
            reference_answer=request.reference_answer,
            concepts=request.concepts,
            keywords=request.keywords,
            source_page=request.source_page,
            source_passage=request.source_passage,
        ).to_dict()

    @app.post("/api/summarize")
    def summarize(request: TextRequest) -> dict:
        return {"summary": analyze_document(request.text, request.filename)["summary"]}

else:
    app = {"title": settings.app_name}
