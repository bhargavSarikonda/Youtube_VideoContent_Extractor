import logging
from typing import Optional
from app.models.schemas import GroundedSummary
from app.services.llm_service import LLMService

logger = logging.getLogger("agentic_pipeline.summarizer_agent")


class GroundedSummarizerAgent:
    """
    Agent 4: Strictly Grounded 6-10 Line Video Summarizer.
    Guarantees:
    - 100% Grounding in the source transcript (zero external hallucinations).
    - Strict line count bounds (between 6 and 10 lines).
    - Multilingual synthesis in requested target language.
    """

    def __init__(self, llm_service: LLMService = None):
        self.llm_service = llm_service or LLMService()

    async def summarize(
        self,
        transcript_text: str,
        video_id: str,
        target_language: str = "en",
        custom_focus: Optional[str] = None
    ) -> GroundedSummary:
        """
        Executes grounded summarization on the video transcript.
        """
        logger.info(f"Summarizer Agent starting for video {video_id} (Target Language: {target_language})...")
        summary, log = await self.llm_service.generate_grounded_summary(
            transcript_text=transcript_text,
            video_id=video_id,
            target_language=target_language,
            custom_focus=custom_focus
        )
        logger.info(f"Summarizer Agent generated {summary.line_count} lines of grounded summary.")
        return summary
