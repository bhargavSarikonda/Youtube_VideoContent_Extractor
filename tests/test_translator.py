import pytest
from app.agents.translator_agent import TranslatorAgent
from app.models.schemas import GroundedSummary, ActionItem


@pytest.mark.asyncio
async def test_translator_agent_instantiation_and_execution():
    agent = TranslatorAgent()
    summary = GroundedSummary(
        source_video_id="test_video_123",
        lines=[
            "1. Artificial Intelligence improves automation.",
            "2. Data engineering is essential for scaling.",
            "3. Machine learning models require clean data.",
            "4. Cloud computing enables distributed processing.",
            "5. Monitoring ensures system reliability.",
            "6. Security is paramount across all tiers."
        ],
        line_count=6,
        grounded=True,
        hallucination_score=0.0
    )
    action_items = [
        ActionItem(id=1, task="Set up distributed queue", category="Implementation", priority="High")
    ]

    translated_summary, translated_actions, log = await agent.translate(
        summary=summary,
        action_items=action_items,
        target_language="te"
    )

    assert translated_summary is not None
    assert translated_summary.line_count == 6
    assert translated_summary.source_video_id == "test_video_123"
    assert translated_actions is not None
    assert log is not None
    assert "Translation" in log.agent_name
