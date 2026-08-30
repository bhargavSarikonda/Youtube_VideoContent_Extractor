import logging
from typing import List, Optional, Tuple
from app.models.schemas import GroundedSummary, ActionItem, ModelExecutionLog
from app.services.llm_service import LLMService

logger = logging.getLogger("agentic_pipeline.translator_agent")


class TranslatorAgent:
    """
    Agent 6: Multilingual Translation & Localization Specialist Agent.
    Specialized in converting grounded executive summaries, action items,
    and metadata into culturally accurate, native-script target languages
    (e.g., Telugu, Hindi, Spanish, French, German, Japanese, Tamil, etc.).
    """

    def __init__(self, llm_service: Optional[LLMService] = None):
        self.llm_service = llm_service or LLMService()

    async def translate(
        self,
        summary: Optional[GroundedSummary],
        action_items: Optional[List[ActionItem]],
        target_language: str
    ) -> Tuple[Optional[GroundedSummary], Optional[List[ActionItem]], ModelExecutionLog]:
        """
        Executes dedicated localization and translation into the target language.
        """
        logger.info(f"Translator Agent starting translation to '{target_language}'...")
        return await self.llm_service.translate_and_localize(
            summary=summary,
            action_items=action_items,
            target_language=target_language
        )
