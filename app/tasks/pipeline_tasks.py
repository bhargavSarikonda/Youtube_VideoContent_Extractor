import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from app.celery_app import celery_app
from app.models.schemas import PipelineJobResult
from app.agents.graph_pipeline import execute_langgraph_pipeline
from app.services.redis_service import redis_service

logger = logging.getLogger("agentic_pipeline.tasks")


async def run_pipeline_direct(
    job_id: str,
    url: str,
    target_language: str = "en",
    require_human_review: bool = False,
    custom_focus: Optional[str] = None
) -> PipelineJobResult:
    """
    Direct asynchronous execution of the LangGraph StateGraph Agentic Pipeline.
    """
    logger.info(f"Starting LangGraph Agent Pipeline for job {job_id}...")
    return await execute_langgraph_pipeline(
        job_id=job_id,
        url=url,
        target_language=target_language,
        require_human_review=require_human_review,
        custom_focus=custom_focus
    )


@celery_app.task(bind=True, name="process_youtube_pipeline")
def process_youtube_pipeline(
    self,
    job_id: str,
    url: str,
    target_language: str = "en",
    require_human_review: bool = False,
    custom_focus: Optional[str] = None
) -> Dict[str, Any]:
    """
    Celery task wrapper executing the LangGraph StateGraph in distributed worker pool.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(
            execute_langgraph_pipeline(
                job_id=job_id,
                url=url,
                target_language=target_language,
                require_human_review=require_human_review,
                custom_focus=custom_focus
            )
        )
        return result.model_dump()
    finally:
        loop.close()
