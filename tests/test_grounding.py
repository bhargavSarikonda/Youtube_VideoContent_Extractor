import pytest
from app.agents.summarizer_agent import GroundedSummarizerAgent
from app.services.llm_service import LLMService


@pytest.mark.asyncio
async def test_summarizer_line_count_constraint():
    llm_service = LLMService()
    agent = GroundedSummarizerAgent(llm_service)
    
    transcript = (
        "Welcome to this lecture on distributed computing. "
        "First, we discuss message brokers and queue partitioning. "
        "Second, we analyze worker thread pools and asynchronous event loops. "
        "Third, we introduce Redis caching strategies to reduce backend database load. "
        "Fourth, we cover rate limiting mechanisms including token bucket algorithms. "
        "Fifth, we explore human-in-the-loop oversight for critical automated actions. "
        "Sixth, we measure throughput under high concurrency of ten thousand users. "
        "Finally, we summarize the deployment architecture using Docker and Celery."
    )
    
    summary = await agent.summarize(transcript, video_id="test_vid_123", target_language="en")
    
    # Strictly between 6 and 10 lines
    assert 6 <= summary.line_count <= 10
    assert len(summary.lines) == summary.line_count
    assert summary.grounding_verified is True
    assert summary.source_video_id == "test_vid_123"


def test_line_bounds_enforcer():
    llm_service = LLMService()
    
    # Case 1: Too few lines (e.g. 3 lines) -> Should be padded to at least 6
    short_lines = ["Point A", "Point B", "Point C"]
    sample_text = "Sentence 1. Sentence 2. Sentence 3. Sentence 4. Sentence 5. Sentence 6. Sentence 7."
    bounded_short = llm_service._enforce_line_bounds(short_lines, sample_text)
    assert 6 <= len(bounded_short) <= 10

    # Case 2: Too many lines (e.g. 15 lines) -> Should be capped at 10
    long_lines = [f"Bullet {i}" for i in range(1, 16)]
    bounded_long = llm_service._enforce_line_bounds(long_lines, sample_text)
    assert len(bounded_long) == 10
