import pytest
from app.agents.guardrail_agent import GuardrailAgent
from app.services.llm_service import LLMService


@pytest.mark.asyncio
async def test_guardrail_safe_content():
    llm_service = LLMService()
    agent = GuardrailAgent(llm_service)
    
    safe_transcript = (
        "In this Python tutorial, we will learn how to build an asynchronous web API with FastAPI. "
        "We will explore dependency injection, Pydantic schemas, routing, and database connections. "
        "By following these best practices, your application will achieve high throughput."
    )
    report = await agent.audit_content(safe_transcript)
    assert report.is_safe is True
    assert report.badge_status == "SAFE"
    assert len(report.harm_flags_detected) == 0
    assert report.requires_human_intervention is False


@pytest.mark.asyncio
async def test_guardrail_detects_harm_and_violence():
    llm_service = LLMService()
    agent = GuardrailAgent(llm_service)
    
    harmful_transcript = (
        "In this video we demonstrate mass violence, terrorist attack techniques, and how to carry out "
        "a brutal massacre against unsuspecting targets with weapons."
    )
    report = await agent.audit_content(harmful_transcript)
    assert report.is_safe is False
    assert report.badge_status == "FLAGGED"
    assert "violence" in report.harm_flags_detected
    assert report.requires_human_intervention is True
