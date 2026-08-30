import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent

# Support Vercel serverless environment (/tmp is the only writable directory)
IS_VERCEL = bool(os.getenv("VERCEL") or os.getenv("NOW_REGION"))
if IS_VERCEL:
    PDF_DIR = Path("/tmp/generated_reports")
else:
    PDF_DIR = BASE_DIR / "generated_reports"

try:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    PDF_DIR = Path("/tmp/generated_reports")
    PDF_DIR.mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    APP_NAME: str = "Agentic YouTube Content Extractor & Intelligence Suite"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True

    # OpenAI Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"

    # Redis Configuration
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    REDIS_CACHE_TTL_SECONDS: int = 86400  # 24 hours caching for transcripts & summaries

    # Celery Configuration
    CELERY_BROKER_URL: str = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    CELERY_RESULT_BACKEND: str = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")

    # Rate Limiting & Concurrency (10,000 Concurrent Users Target)
    RATE_LIMIT_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "600"))
    MAX_CONCURRENT_TASKS: int = int(os.getenv("MAX_CONCURRENT_TASKS", "10000"))

    # Grounding & Safety Constraints
    MIN_SUMMARY_LINES: int = 6
    MAX_SUMMARY_LINES: int = 10
    STRICT_GROUNDING: bool = True
    AUTO_HITL_ON_SAFETY_FLAG: bool = True

    # Storage
    PDF_OUTPUT_DIR: str = str(PDF_DIR)


settings = Settings()
