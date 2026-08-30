import os
import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from app.models.schemas import ModelExecutionLog

IS_VERCEL = bool(os.getenv("VERCEL") or os.getenv("NOW_REGION"))
if IS_VERCEL:
    LOGS_DIR = Path("/tmp/logs")
else:
    LOGS_DIR = Path(__file__).resolve().parent.parent.parent / "logs"

try:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    LOGS_DIR = Path("/tmp/logs")
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

AUDIT_LOG_FILE = LOGS_DIR / "model_audit.log"

# Dedicated model audit logger
audit_logger = logging.getLogger("model_audit")
audit_logger.setLevel(logging.INFO)

# File handler for model audit log
if not audit_logger.handlers:
    try:
        file_handler = logging.FileHandler(str(AUDIT_LOG_FILE), encoding="utf-8")
        formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "event": %(message)s}'
        )
        file_handler.setFormatter(formatter)
        audit_logger.addHandler(file_handler)
    except Exception as e:
        # Fallback to stream handler if file cannot be created
        stream_handler = logging.StreamHandler()
        audit_logger.addHandler(stream_handler)


class AuditLoggerService:
    """
    Centralized Model Execution & Audit Logging Engine.
    Records comprehensive metrics and traces for every model invocation.
    """

    @classmethod
    def record_model_execution(
        cls,
        agent_name: str,
        model_name: str,
        input_text: str,
        latency_ms: float,
        status: str,
        output_metrics: str,
        grounding_guaranteed: bool = True,
        output_sample: Optional[str] = None,
        error_detail: Optional[str] = None
    ) -> ModelExecutionLog:
        input_chars = len(input_text) if input_text else 0
        est_tokens = max(1, input_chars // 4)

        log_entry = ModelExecutionLog(
            agent_name=agent_name,
            model_name=model_name,
            input_characters=input_chars,
            estimated_input_tokens=est_tokens,
            latency_ms=round(latency_ms, 2),
            status=status,
            grounding_guaranteed=grounding_guaranteed,
            output_metrics=output_metrics,
            output_sample=output_sample[:300] if output_sample else None,
            error_detail=error_detail
        )

        # Write structured JSON entry to audit log file
        try:
            audit_logger.info(json.dumps(log_entry.model_dump()))
        except Exception as e:
            logging.getLogger("agentic_pipeline").warning(f"Failed writing model audit log: {e}")

        return log_entry
