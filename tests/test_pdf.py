import os
from pathlib import Path
from app.services.pdf_generator import PDFGeneratorService
from app.models.schemas import (
    VideoMetadata, SafetyAuditReport, GroundedSummary, ActionItem, HITLState
)


def test_pdf_generation():
    metadata = VideoMetadata(
        video_id="dQw4w9WgXcQ",
        title="Distributed Agentic AI Architecture Overview",
        channel="AI Engineering Hub",
        duration_formatted="12:45",
        thumbnail_url="https://img.youtube.com/vi/dQw4w9WgXcQ/maxresdefault.jpg",
        detected_language="en"
    )

    safety = SafetyAuditReport(
        is_safe=True,
        harm_flags_detected=[],
        summary_assessment="Content verified 100% harm-free and safe for enterprise publication.",
        badge_status="SAFE"
    )

    summary = GroundedSummary(
        lines=[
            "Introduction to scalable agentic pipelines and event-driven architecture.",
            "Demonstration of Redis token-bucket rate limiting for 10,000 concurrent clients.",
            "Multilingual transcript parsing across auto-generated and manual captions.",
            "OpenAI GPT-4o-mini integration with temperature=0 for zero-hallucination grounding.",
            "Safety guardrails detecting violence, harassment, and sexual abuse triggers.",
            "Implementation of Celery asynchronous task queues and Redis result backend.",
            "Human-in-the-Loop review gating for flagged content and supervisor oversight.",
            "Automated high-fidelity vector PDF generation with ReportLab."
        ],
        line_count=8,
        grounding_verified=True,
        source_video_id="dQw4w9WgXcQ"
    )

    actions = [
        ActionItem(id=1, task="Deploy Redis and Celery worker cluster.", category="Infrastructure", priority="High"),
        ActionItem(id=2, task="Configure OpenAI API key in .env file.", category="Security", priority="High"),
        ActionItem(id=3, task="Test multilingual transcript extraction on non-English videos.", category="Testing", priority="Medium"),
    ]

    hitl = HITLState(
        status="APPROVED",
        reviewed_by="Senior Safety Auditor",
        review_comments="Content verified safe and grounded."
    )

    pdf_path = PDFGeneratorService.generate_pdf(
        job_id="test_job_9999",
        metadata=metadata,
        safety_report=safety,
        summary=summary,
        action_items=actions,
        hitl_state=hitl
    )

    assert os.path.exists(pdf_path), f"PDF was not generated at {pdf_path}"
    assert os.path.getsize(pdf_path) > 1000, "PDF file is suspiciously small"
