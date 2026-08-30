import pytest
from app.agents.graph_pipeline import (
    build_langgraph_pipeline,
    execute_langgraph_pipeline,
    PipelineGraphState
)


@pytest.mark.asyncio
async def test_langgraph_graph_build_and_nodes():
    app = build_langgraph_pipeline()
    assert app is not None
    # Verify compiled graph has required nodes
    nodes = app.get_graph().nodes
    assert "node_validate_url" in nodes
    assert "node_ingest_transcript" in nodes
    assert "node_guardrail_safety" in nodes
    assert "node_grounded_summarizer" in nodes
    assert "node_action_items" in nodes
    assert "node_hitl_gate" in nodes
    assert "node_pdf_compiler" in nodes


@pytest.mark.asyncio
async def test_langgraph_execution_with_model_logging():
    # Test valid YouTube URL
    job = await execute_langgraph_pipeline(
        job_id="test_langgraph_job_1",
        url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        target_language="en",
        require_human_review=False
    )

    assert job.status in ("COMPLETED", "WAITING_HUMAN_REVIEW")
    assert job.video_id == "dQw4w9WgXcQ"
    assert job.summary is not None
    assert 6 <= job.summary.line_count <= 10
    assert job.safety_report is not None
    
    # Verify model logs are populated for every executed model
    assert len(job.model_logs) >= 3
    agent_names = [log.agent_name for log in job.model_logs]
    assert "Safety & Guardrail Auditor" in agent_names
    assert "Grounded 6-10 Line Summarizer" in agent_names
    assert "Action Items & Implementation Extractor" in agent_names

    # Check metrics
    for log in job.model_logs:
        assert log.latency_ms >= 0.0
        assert log.model_name is not None
        assert log.grounding_guaranteed is True


@pytest.mark.asyncio
async def test_langgraph_invalid_url_fails_cleanly():
    job = await execute_langgraph_pipeline(
        job_id="test_langgraph_invalid",
        url="https://not-youtube.com/video/123",
        target_language="en"
    )
    assert job.status == "FAILED"
    assert "Invalid domain" in (job.error_message or "")
