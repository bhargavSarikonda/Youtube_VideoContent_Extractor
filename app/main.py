import os
import uuid
import json
import asyncio
from pathlib import Path
from typing import Optional
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks, status
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.models.schemas import (
    VideoExtractionRequest,
    HITLReviewActionRequest,
    PipelineJobResult,
    GroundedSummary,
    ActionItem
)
from app.agents.url_agent import YouTubeURLAgent
from app.agents.hitl_manager import HITLManager
from app.services.redis_service import redis_service
from app.services.pdf_generator import PDFGeneratorService
from app.tasks.pipeline_tasks import run_pipeline_direct, process_youtube_pipeline
from app.celery_app import celery_app

# Initialize FastAPI App
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Agentic AI YouTube Intelligence Suite with Strict Grounding, Multilingual Support, Guardrails, HITL, and PDF Generation."
)

# Enable CORS for high-scale distributed clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    await redis_service.connect()


@app.on_event("shutdown")
async def shutdown_event():
    await redis_service.close()


# =============================================================================
# API Endpoints
# =============================================================================

@app.post("/api/extract", status_code=status.HTTP_202_ACCEPTED)
async def extract_youtube_content(
    req: VideoExtractionRequest,
    request: Request,
    background_tasks: BackgroundTasks
):
    """
    Submits a YouTube video URL into the Agentic Pipeline.
    Strictly validates YouTube domain and enqueues task asynchronously.
    """
    # 1. Rate Limiting Check (Distributed Token Bucket)
    client_ip = request.client.host if request.client else "unknown"
    is_allowed = await redis_service.check_rate_limit(
        client_identifier=client_ip,
        limit=settings.RATE_LIMIT_PER_MINUTE,
        window_seconds=60
    )
    if not is_allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please wait a moment before submitting another request."
        )

    # 2. Strict YouTube URL Validation Agent Check
    is_valid, video_id, canonical_url, err = YouTubeURLAgent.validate_and_extract(req.url)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=err or "Invalid YouTube URL. Please provide a valid youtube.com or youtu.be link."
        )

    job_id = str(uuid.uuid4())

    # Initialize Job in Redis
    initial_job = PipelineJobResult(
        job_id=job_id,
        url=canonical_url,
        video_id=video_id,
        status="QUEUED",
        progress_percent=5,
        current_stage="Job submitted to queue",
        created_at=datetime.now(timezone.utc)
    )
    await redis_service.save_job_state(job_id, initial_job.model_dump())

    # 3. Task Dispatcher (Celery or Direct Async Pool)
    # Use direct async background tasks if Celery broker is local or fallback
    try:
        # Try Celery dispatch first
        task = process_youtube_pipeline.delay(
            job_id=job_id,
            url=canonical_url,
            target_language=req.target_language or "en",
            require_human_review=req.require_human_review,
            custom_focus=req.custom_focus
        )
    except Exception:
        # Fallback to FastAPI native async background task
        background_tasks.add_task(
            run_pipeline_direct,
            job_id=job_id,
            url=canonical_url,
            target_language=req.target_language or "en",
            require_human_review=req.require_human_review,
            custom_focus=req.custom_focus
        )

    return {
        "job_id": job_id,
        "video_id": video_id,
        "status": "QUEUED",
        "message": "YouTube Video ingestion accepted.",
        "status_url": f"/api/status/{job_id}",
        "stream_url": f"/api/stream/{job_id}"
    }


@app.get("/api/status/{job_id}")
async def get_job_status(job_id: str):
    """
    Returns the real-time execution state of the agentic pipeline for a job.
    """
    job_data = await redis_service.get_job_state(job_id)
    if not job_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job with ID '{job_id}' not found."
        )
    return job_data


@app.get("/api/stream/{job_id}")
async def stream_job_progress(job_id: str):
    """
    Server-Sent Events (SSE) stream for real-time UI updates without aggressive polling.
    """
    async def event_generator():
        last_status = None
        last_percent = -1
        retry_count = 0

        while retry_count < 120:  # 2 minutes max stream
            job_data = await redis_service.get_job_state(job_id)
            if job_data:
                current_status = job_data.get("status")
                current_percent = job_data.get("progress_percent", 0)

                # Send event if state changed
                if current_status != last_status or current_percent != last_percent:
                    last_status = current_status
                    last_percent = current_percent
                    payload = json.dumps(job_data, default=str)
                    yield f"data: {payload}\n\n"

                if current_status in ("COMPLETED", "FAILED", "REJECTED", "WAITING_HUMAN_REVIEW"):
                    if current_status != "WAITING_HUMAN_REVIEW":
                        break

            await asyncio.sleep(0.5)
            retry_count += 1

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.post("/api/hitl/review/{job_id}")
async def hitl_review_action(job_id: str, req: HITLReviewActionRequest):
    """
    Human-in-the-Loop review endpoint.
    Permits human operators to approve, edit, or reject the video analysis.
    """
    job_data = await redis_service.get_job_state(job_id)
    if not job_data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")

    job = PipelineJobResult(**job_data)

    if not job.summary or not job.safety_report or not job.metadata:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Job has incomplete agent outputs and cannot be reviewed."
        )

    # Process Human Action
    current_hitl = job.hitl_state or HITLManager.evaluate_initial_state(job.safety_report)
    updated_hitl, final_summary, final_action_items = HITLManager.apply_human_action(
        current_state=current_hitl,
        review_request=req,
        current_summary=job.summary,
        current_action_items=job.action_items or []
    )

    job.hitl_state = updated_hitl
    job.summary = final_summary
    job.action_items = final_action_items

    if updated_hitl.status in ("APPROVED", "EDITED"):
        # Compile final PDF with verified stamp
        pdf_path = PDFGeneratorService.generate_pdf(
            job_id=job_id,
            metadata=job.metadata,
            safety_report=job.safety_report,
            summary=job.summary,
            action_items=job.action_items,
            hitl_state=job.hitl_state
        )
        job.status = "COMPLETED"
        job.progress_percent = 100
        job.current_stage = f"Human Review Completed ({updated_hitl.status}). PDF Compiled."
        job.pdf_filename = f"report_{job.metadata.video_id}_{job_id[:8]}.pdf"
        job.pdf_download_url = f"/api/pdf/{job_id}"
        job.completed_at = datetime.now(timezone.utc)

    elif updated_hitl.status == "REJECTED":
        job.status = "REJECTED"
        job.progress_percent = 100
        job.current_stage = "Job rejected by Human Supervisor."
        job.completed_at = datetime.now(timezone.utc)

    await redis_service.save_job_state(job_id, job.model_dump())
    return job.model_dump()


@app.post("/api/cancel/{job_id}")
async def cancel_job(job_id: str):
    """
    Stops and cancels an in-flight video intelligence pipeline execution upon user request.
    """
    job_data = await redis_service.get_job_state(job_id)
    if not job_data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")

    job_data["status"] = "CANCELLED"
    job_data["progress_percent"] = 100
    job_data["current_stage"] = "Analysis stopped by user."
    job_data["error_message"] = "Pipeline execution was stopped by user request."
    job_data["completed_at"] = datetime.now(timezone.utc).isoformat()

    await redis_service.save_job_state(job_id, job_data)
    return {"status": "CANCELLED", "job_id": job_id, "message": "Pipeline analysis stopped."}


@app.get("/api/pdf/{job_id}")
async def download_pdf_report(job_id: str, inline: bool = False):
    """
    Downloads or renders the generated vector PDF report inline in the browser.
    """
    job_data = await redis_service.get_job_state(job_id)
    if not job_data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")

    video_id = job_data.get("video_id", "video")
    filename = job_data.get("pdf_filename") or f"report_{video_id}_{job_id[:8]}.pdf"
    file_path = Path(settings.PDF_OUTPUT_DIR) / filename

    if not file_path.exists():
        # If not generated yet, try generating on-the-fly if summary exists
        job = PipelineJobResult(**job_data)
        if job.metadata and job.safety_report and job.summary:
            PDFGeneratorService.generate_pdf(
                job_id=job_id,
                metadata=job.metadata,
                safety_report=job.safety_report,
                summary=job.summary,
                action_items=job.action_items or [],
                hitl_state=job.hitl_state
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="PDF report file is not available yet."
            )

    disposition_type = "inline" if inline else "attachment"
    return FileResponse(
        path=str(file_path),
        media_type="application/pdf",
        filename=filename,
        headers={"Content-Disposition": f'{disposition_type}; filename="{filename}"'}
    )


@app.get("/api/logs/{job_id}")
async def get_job_model_logs(job_id: str):
    """
    Returns the granular execution logs and performance metrics for each model invoked during the job.
    """
    job_data = await redis_service.get_job_state(job_id)
    if not job_data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")

    return {
        "job_id": job_id,
        "model_logs": job_data.get("model_logs", []),
        "total_models_executed": len(job_data.get("model_logs", []))
    }


@app.get("/api/audit-logs")
async def get_system_audit_logs(limit: int = 50):
    """
    Returns the latest global system model execution audit logs from file.
    """
    from app.services.audit_logger import AUDIT_LOG_FILE
    logs = []
    if AUDIT_LOG_FILE.exists():
        try:
            with open(AUDIT_LOG_FILE, "r", encoding="utf-8") as f:
                lines = f.readlines()
                for line in reversed(lines[-limit:]):
                    try:
                        logs.append(json.loads(line.strip()))
                    except Exception:
                        pass
        except Exception as e:
            logger.warning(f"Failed reading audit log file: {e}")

    return {
        "count": len(logs),
        "logs": logs
    }


@app.get("/api/health")
async def health_check():
    """
    Health check diagnostic endpoint.
    """
    return {
        "status": "HEALTHY",
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "orchestrator": "LangGraph StateGraph",
        "openai_configured": bool(settings.OPENAI_API_KEY and not settings.OPENAI_API_KEY.startswith("your_")),
        "openai_model": settings.OPENAI_MODEL,
        "redis_connected": redis_service._connected,
        "max_concurrency": settings.MAX_CONCURRENT_TASKS
    }


# Mount Static Frontend
frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
