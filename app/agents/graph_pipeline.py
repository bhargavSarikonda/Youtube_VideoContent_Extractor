import asyncio
import logging
import operator
from typing import Dict, Any, List, Optional, TypedDict, Annotated
from datetime import datetime, timezone

from langgraph.graph import StateGraph, START, END

from app.models.schemas import (
    VideoMetadata,
    SafetyAuditReport,
    GroundedSummary,
    ActionItem,
    HITLState,
    ModelExecutionLog,
    PipelineJobResult
)
from app.agents.url_agent import YouTubeURLAgent
from app.agents.transcript_agent import TranscriptAgent
from app.agents.hitl_manager import HITLManager
from app.services.llm_service import LLMService
from app.services.pdf_generator import PDFGeneratorService
from app.services.redis_service import redis_service

logger = logging.getLogger("agentic_pipeline.langgraph")


class PipelineGraphState(TypedDict):
    """
    Typed State Graph for LangGraph Agentic Pipeline with Reducers.
    """
    job_id: str
    url: str
    target_language: str
    custom_focus: Optional[str]
    require_human_review: bool
    video_id: Optional[str]
    canonical_url: Optional[str]
    metadata: Optional[VideoMetadata]
    transcript_text: Optional[str]
    detected_language: Optional[str]
    safety_report: Optional[SafetyAuditReport]
    summary: Optional[GroundedSummary]
    action_items: Optional[List[ActionItem]]
    hitl_state: Optional[HITLState]
    pdf_filename: Optional[str]
    pdf_download_url: Optional[str]
    status: str
    progress_percent: int
    current_stage: str
    error_message: Optional[str]
    cached: bool
    # Parallel fan-out reducer for model audit logging
    model_logs: Annotated[List[ModelExecutionLog], operator.add]


# Initialize LLM Service instance for graph nodes
llm_service = LLMService()


# =============================================================================
# LangGraph Nodes
# =============================================================================

async def node_validate_url(state: PipelineGraphState) -> Dict[str, Any]:
    """Node 1: Validates YouTube URL domain and extracts video ID."""
    url = state["url"]
    is_valid, video_id, canonical_url, err = YouTubeURLAgent.validate_and_extract(url)
    
    if not is_valid:
        return {
            "status": "FAILED",
            "error_message": err or "Invalid YouTube URL format.",
            "progress_percent": 100,
            "current_stage": "URL Validation Failed"
        }

    return {
        "video_id": video_id,
        "canonical_url": canonical_url,
        "url": canonical_url,
        "progress_percent": 25,
        "current_stage": f"URL validated. Extracting captions for video {video_id}..."
    }


async def node_ingest_transcript(state: PipelineGraphState) -> Dict[str, Any]:
    """Node 2: Extracts multilingual captions and video metadata."""
    if state.get("status") == "FAILED":
        return {}

    video_id = state["video_id"]
    target_lang = state.get("target_language", "en")

    try:
        transcript_text, metadata, detected_lang = await TranscriptAgent.extract_transcript_and_metadata(
            video_id=video_id,
            target_language=target_lang
        )
        word_count = len(transcript_text.split())
        return {
            "transcript_text": transcript_text,
            "metadata": metadata,
            "detected_language": detected_lang,
            "progress_percent": 50,
            "current_stage": f"Ingested {word_count} words. Running parallel LangGraph agents..."
        }
    except Exception as e:
        return {
            "status": "FAILED",
            "error_message": f"Transcript extraction failed: {str(e)}",
            "progress_percent": 100,
            "current_stage": "Caption Ingestion Error"
        }


async def node_guardrail_safety(state: PipelineGraphState) -> Dict[str, Any]:
    """Node 3A: Safety & Harm Guardrail Auditor with Model Logging."""
    if state.get("status") == "FAILED":
        return {}

    transcript_text = state.get("transcript_text", "")
    report, log = await llm_service.audit_safety(transcript_text)
    
    return {
        "safety_report": report,
        "model_logs": [log]
    }


async def node_grounded_summarizer(state: PipelineGraphState) -> Dict[str, Any]:
    """Node 3B: Strictly Grounded 6-10 Line Summarizer with Model Logging."""
    if state.get("status") == "FAILED":
        return {}

    transcript_text = state.get("transcript_text", "")
    video_id = state.get("video_id", "")
    target_lang = state.get("target_language", "en")
    custom_focus = state.get("custom_focus")

    summary, log = await llm_service.generate_grounded_summary(
        transcript_text=transcript_text,
        video_id=video_id,
        target_language=target_lang,
        custom_focus=custom_focus
    )

    return {
        "summary": summary,
        "model_logs": [log]
    }


async def node_action_items(state: PipelineGraphState) -> Dict[str, Any]:
    """Node 3C: Action Items & Takeaways Generator with Model Logging."""
    if state.get("status") == "FAILED":
        return {}

    transcript_text = state.get("transcript_text", "")
    video_id = state.get("video_id", "")

    actions, log = await llm_service.extract_action_items(
        transcript_text=transcript_text,
        video_id=video_id
    )

    return {
        "action_items": actions,
        "model_logs": [log]
    }


async def node_translator(state: PipelineGraphState) -> Dict[str, Any]:
    """Node 3D: Multilingual Translation & Localization Specialist Agent."""
    if state.get("status") in ("FAILED", "WAITING_HUMAN_REVIEW"):
        return {}

    target_lang = state.get("target_language", "en")
    summary = state.get("summary")
    action_items = state.get("action_items")

    # Translate if target language is specified and not default English
    if target_lang and target_lang.lower() not in ("en", "english"):
        new_summary, new_actions, log = await llm_service.translate_and_localize(
            summary=summary,
            action_items=action_items,
            target_language=target_lang
        )
        return {
            "summary": new_summary,
            "action_items": new_actions,
            "model_logs": [log]
        }

    return {}


async def node_hitl_gate(state: PipelineGraphState) -> Dict[str, Any]:
    """Node 4: Human-in-the-Loop Decision & Synchronization Gate."""
    if state.get("status") == "FAILED":
        return {}

    safety_report = state.get("safety_report")
    require_human = state.get("require_human_review", False)

    hitl_state = HITLManager.evaluate_initial_state(
        safety_report=safety_report,
        require_human_review=require_human
    )

    if hitl_state.status == "PENDING":
        return {
            "hitl_state": hitl_state,
            "status": "WAITING_HUMAN_REVIEW",
            "progress_percent": 85,
            "current_stage": f"Paused for Supervisor Oversight: {hitl_state.triggered_reason}"
        }

    return {
        "hitl_state": hitl_state,
        "progress_percent": 90,
        "current_stage": "Passed all safety guardrails. Compiling PDF report..."
    }


async def node_pdf_compiler(state: PipelineGraphState) -> Dict[str, Any]:
    """Node 5: High-Fidelity Vector PDF Compiler."""
    if state.get("status") in ("FAILED", "WAITING_HUMAN_REVIEW"):
        return {}

    job_id = state["job_id"]
    metadata = state["metadata"]
    safety_report = state["safety_report"]
    summary = state["summary"]
    action_items = state.get("action_items") or []
    hitl_state = state.get("hitl_state")

    pdf_path = PDFGeneratorService.generate_pdf(
        job_id=job_id,
        metadata=metadata,
        safety_report=safety_report,
        summary=summary,
        action_items=action_items,
        hitl_state=hitl_state
    )
    pdf_filename = f"report_{metadata.video_id}_{job_id[:8]}.pdf"

    return {
        "status": "COMPLETED",
        "progress_percent": 100,
        "current_stage": "Complete: Grounded Summary, Safety Seal & PDF compiled.",
        "pdf_filename": pdf_filename,
        "pdf_download_url": f"/api/pdf/{job_id}",
        "cached": False
    }


# =============================================================================
# Conditional Routing Functions
# =============================================================================

def route_after_validation(state: PipelineGraphState) -> str:
    if state.get("status") == "FAILED":
        return END
    return "node_ingest_transcript"


def route_after_hitl_gate(state: PipelineGraphState) -> str:
    if state.get("status") in ("FAILED", "WAITING_HUMAN_REVIEW"):
        return END
    return "node_pdf_compiler"


# =============================================================================
# Build & Compile LangGraph StateGraph
# =============================================================================

def build_langgraph_pipeline():
    workflow = StateGraph(PipelineGraphState)

    # Register Nodes
    workflow.add_node("node_validate_url", node_validate_url)
    workflow.add_node("node_ingest_transcript", node_ingest_transcript)
    workflow.add_node("node_guardrail_safety", node_guardrail_safety)
    workflow.add_node("node_grounded_summarizer", node_grounded_summarizer)
    workflow.add_node("node_action_items", node_action_items)
    workflow.add_node("node_translator", node_translator)
    workflow.add_node("node_hitl_gate", node_hitl_gate)
    workflow.add_node("node_pdf_compiler", node_pdf_compiler)

    # Edge Definitions
    workflow.add_edge(START, "node_validate_url")
    workflow.add_conditional_edges(
        "node_validate_url",
        route_after_validation,
        {
            "node_ingest_transcript": "node_ingest_transcript",
            END: END
        }
    )

    # Concurrent Parallel Fan-Out from Transcript Ingestion to Agents
    workflow.add_edge("node_ingest_transcript", "node_guardrail_safety")
    workflow.add_edge("node_ingest_transcript", "node_grounded_summarizer")
    workflow.add_edge("node_ingest_transcript", "node_action_items")

    # Fan-In to Translation Agent
    workflow.add_edge("node_guardrail_safety", "node_translator")
    workflow.add_edge("node_grounded_summarizer", "node_translator")
    workflow.add_edge("node_action_items", "node_translator")

    # Translation to HITL Gate
    workflow.add_edge("node_translator", "node_hitl_gate")

    # Conditional Routing from HITL Gate
    workflow.add_conditional_edges(
        "node_hitl_gate",
        route_after_hitl_gate,
        {
            "node_pdf_compiler": "node_pdf_compiler",
            END: END
        }
    )

    workflow.add_edge("node_pdf_compiler", END)
    return workflow.compile()


# Compiled LangGraph instance
langgraph_app = build_langgraph_pipeline()


# =============================================================================
# LangGraph Execution Driver with Realtime Redis & SSE Streaming
# =============================================================================

async def execute_langgraph_pipeline(
    job_id: str,
    url: str,
    target_language: str = "en",
    require_human_review: bool = False,
    custom_focus: Optional[str] = None
) -> PipelineJobResult:
    """
    Executes the compiled LangGraph pipeline while streaming live state to Redis & SSE.
    """
    initial_state: PipelineGraphState = {
        "job_id": job_id,
        "url": url,
        "target_language": target_language,
        "custom_focus": custom_focus,
        "require_human_review": require_human_review,
        "video_id": None,
        "canonical_url": None,
        "metadata": None,
        "transcript_text": None,
        "detected_language": None,
        "safety_report": None,
        "summary": None,
        "action_items": [],
        "hitl_state": None,
        "pdf_filename": None,
        "pdf_download_url": None,
        "status": "QUEUED",
        "progress_percent": 5,
        "current_stage": "LangGraph pipeline initialized",
        "error_message": None,
        "cached": False,
        "model_logs": []
    }

    # Stream state transitions through LangGraph
    current_state = initial_state
    
    try:
        async for output in langgraph_app.astream(initial_state):
            if isinstance(output, dict):
                for node_name, node_state in output.items():
                    logger.info(f"LangGraph executed node: {node_name}")
                    if node_state and isinstance(node_state, dict):
                        # Safely merge model_logs list
                        if "model_logs" in node_state and isinstance(node_state["model_logs"], list):
                            existing_logs = current_state.get("model_logs") or []
                            for new_log in node_state["model_logs"]:
                                if new_log not in existing_logs:
                                    existing_logs.append(new_log)
                            current_state["model_logs"] = existing_logs
                            del node_state["model_logs"]

                        current_state.update(node_state)
                    
                    # Snapshot to PipelineJobResult and save to Redis
                    job_snapshot = PipelineJobResult(
                        job_id=job_id,
                        url=current_state.get("url") or url,
                        video_id=current_state.get("video_id"),
                        status=current_state.get("status", "PROCESSING"),
                        progress_percent=current_state.get("progress_percent", 50),
                        current_stage=current_state.get("current_stage", "Running LangGraph agent..."),
                        error_message=current_state.get("error_message"),
                        metadata=current_state.get("metadata"),
                        transcript_sample=current_state.get("transcript_text", "")[:500] if current_state.get("transcript_text") else None,
                        transcript_word_count=len((current_state.get("transcript_text") or "").split()),
                        safety_report=current_state.get("safety_report"),
                        summary=current_state.get("summary"),
                        action_items=current_state.get("action_items") or [],
                        hitl_state=current_state.get("hitl_state"),
                        pdf_filename=current_state.get("pdf_filename"),
                        pdf_download_url=current_state.get("pdf_download_url"),
                        cached=current_state.get("cached", False),
                        model_logs=current_state.get("model_logs") or []
                    )
                    await redis_service.save_job_state(job_id, job_snapshot.model_dump())

        # Final snapshot
        final_job = PipelineJobResult(
            job_id=job_id,
            url=current_state.get("url") or url,
            video_id=current_state.get("video_id"),
            status=current_state.get("status", "COMPLETED"),
            progress_percent=current_state.get("progress_percent", 100),
            current_stage=current_state.get("current_stage", "Completed"),
            error_message=current_state.get("error_message"),
            metadata=current_state.get("metadata"),
            transcript_sample=current_state.get("transcript_text", "")[:500] if current_state.get("transcript_text") else None,
            transcript_word_count=len((current_state.get("transcript_text") or "").split()),
            safety_report=current_state.get("safety_report"),
            summary=current_state.get("summary"),
            action_items=current_state.get("action_items") or [],
            hitl_state=current_state.get("hitl_state"),
            pdf_filename=current_state.get("pdf_filename"),
            pdf_download_url=current_state.get("pdf_download_url"),
            cached=current_state.get("cached", False),
            model_logs=current_state.get("model_logs") or []
        )
        if final_job.status == "COMPLETED":
            final_job.completed_at = datetime.now(timezone.utc)
            # Save to cache
            cache_key = f"{final_job.video_id}_{target_language}_{custom_focus or 'none'}"
            await redis_service.set_cached_result(cache_key, final_job.model_dump())

        await redis_service.save_job_state(job_id, final_job.model_dump())
        return final_job

    except Exception as e:
        logger.exception(f"LangGraph pipeline error: {e}")
        error_job = PipelineJobResult(
            job_id=job_id,
            url=url,
            status="FAILED",
            progress_percent=100,
            current_stage="LangGraph Execution Failed",
            error_message=str(e),
            model_logs=current_state.get("model_logs") or []
        )
        await redis_service.save_job_state(job_id, error_job.model_dump())
        return error_job
