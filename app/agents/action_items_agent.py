import logging
from typing import List
from app.models.schemas import ActionItem
from app.services.llm_service import LLMService

logger = logging.getLogger("agentic_pipeline.action_items_agent")


class ActionItemsAgent:
    """
    Agent 5: Action Items & Key Takeaways Extractor Agent.
    Identifies concrete tasks, implementation steps, referenced tools,
    and strategic recommendations directly cited in the transcript.
    """

    def __init__(self, llm_service: LLMService = None):
        self.llm_service = llm_service or LLMService()

    async def extract_actions(self, transcript_text: str, video_id: str) -> List[ActionItem]:
        """
        Executes action items extraction from video transcript.
        """
        logger.info(f"Action Items Agent starting for video {video_id}...")
        actions, log = await self.llm_service.extract_action_items(transcript_text, video_id)
        logger.info(f"Action Items Agent extracted {len(actions)} actionable items.")
        return actions
