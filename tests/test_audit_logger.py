import os
import pytest
from pathlib import Path
from app.services.audit_logger import AuditLoggerService, AUDIT_LOG_FILE


def test_audit_logger_records_model_metrics():
    log_entry = AuditLoggerService.record_model_execution(
        agent_name="Test Model Agent",
        model_name="gpt-4o-mini",
        input_text="Sample video transcript test prompt.",
        latency_ms=142.5,
        status="SUCCESS",
        output_metrics="Grounded 8 lines generated",
        grounding_guaranteed=True,
        output_sample="Line 1 preview"
    )

    assert log_entry.agent_name == "Test Model Agent"
    assert log_entry.model_name == "gpt-4o-mini"
    assert log_entry.latency_ms == 142.5
    assert log_entry.estimated_input_tokens > 0
    assert log_entry.grounding_guaranteed is True
    assert AUDIT_LOG_FILE.exists()
