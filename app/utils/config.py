"""Application configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "AI PDF-Based Subjective Exam Evaluator")
    api_host: str = os.getenv("API_HOST", "127.0.0.1")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "25"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    project_root: Path = Path(__file__).resolve().parents[2]
    data_dir: Path = project_root / "data"
    models_dir: Path = project_root / "models"


settings = Settings()

