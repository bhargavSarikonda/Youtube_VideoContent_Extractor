from datetime import datetime, timezone
from typing import List, Optional
from pydantic import BaseModel, Field, HttpUrl


class VideoExtractionRequest(BaseModel):
    url: str = Field(..., description="YouTube URL (must be a valid youtube.com or youtu.be link)")
    target_language: Optional[str] = Field("en", description="Target language code for summary (default: en)")
    require_human_review: bool = Field(False, description="Explicitly route to HITL review regardless of safety score")
    custom_focus: Optional[str] = Field(None, description="Optional custom focus or domain instructions")


class VideoMetadata(BaseModel):
    video_id: str
    title: str
    channel: str
    duration_seconds: Optional[int] = None
    duration_formatted: Optional[str] = None
    thumbnail_url: Optional[str] = None
    view_count: Optional[int] = None
    detected_language: str = "en"
    available_languages: List[str] = []
    is_auto_generated: bool = False


class SafetyCategoryScore(BaseModel):
    category: str
    flagged: bool
    score: float
    description: Optional[str] = None


class SafetyAuditReport(BaseModel):
    is_safe: bool = True
    harm_flags_detected: List[str] = []
    categories: List[SafetyCategoryScore] = []
    summary_assessment: str = "Safe: No violence, sexual abuse, self-harm, or harassment detected."
    badge_status: str = "SAFE"  # SAFE, FLAGGED, WARNING, HITL_REQUIRED
    requires_human_intervention: bool = False


class ActionItem(BaseModel):
    id: int
    task: str
    category: str = "General"  # e.g., Implementation, Resource, Follow-up, Research
    priority: str = "Medium"  # High, Medium, Low
    timestamp_reference: Optional[str] = None


class GroundedSummary(BaseModel):
    lines: List[str]
    line_count: int
    grounding_verified: bool = True
    source_video_id: str
    language: str = "en"


class HITLState(BaseModel):
    status: str = "NONE"  # NONE, PENDING, APPROVED, EDITED, REJECTED
    triggered_reason: Optional[str] = None
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[datetime] = None
    review_comments: Optional[str] = None
    edited_summary: Optional[List[str]] = None
    edited_action_items: Optional[List[ActionItem]] = None


class ModelExecutionLog(BaseModel):
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    agent_name: str
    model_name: str
    input_characters: int
    estimated_input_tokens: int
    latency_ms: float
    status: str
    grounding_guaranteed: bool = True
    output_metrics: str
    output_sample: Optional[str] = None
    error_detail: Optional[str] = None


class PipelineJobResult(BaseModel):
    job_id: str
    url: str
    video_id: Optional[str] = None
    status: str = "QUEUED"  # QUEUED, INGESTING, AUDITING_SAFETY, SUMMARIZING, EXTRACTING_ACTIONS, WAITING_HUMAN_REVIEW, COMPLETED, FAILED
    progress_percent: int = 0
    current_stage: str = "Job initialized in Redis queue"
    error_message: Optional[str] = None
    metadata: Optional[VideoMetadata] = None
    transcript_sample: Optional[str] = None
    transcript_word_count: Optional[int] = 0
    safety_report: Optional[SafetyAuditReport] = None
    summary: Optional[GroundedSummary] = None
    action_items: Optional[List[ActionItem]] = []
    hitl_state: Optional[HITLState] = None
    pdf_filename: Optional[str] = None
    pdf_download_url: Optional[str] = None
    cached: bool = False
    model_logs: List[ModelExecutionLog] = []
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None


class HITLReviewActionRequest(BaseModel):
    action: str = Field(..., description="Action to take: 'approve', 'edit', or 'reject'")
    reviewer_name: Optional[str] = Field("Human Supervisor", description="Name/ID of reviewer")
    comments: Optional[str] = Field(None, description="Review remarks")
    edited_summary: Optional[List[str]] = Field(None, description="Modified 6-10 line summary")
    edited_action_items: Optional[List[ActionItem]] = Field(None, description="Modified action items")
