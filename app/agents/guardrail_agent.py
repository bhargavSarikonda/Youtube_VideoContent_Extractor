import logging
from typing import Tuple
from app.models.schemas import SafetyAuditReport
from app.services.llm_service import LLMService

logger = logging.getLogger("agentic_pipeline.guardrail_agent")


class GuardrailAgent:
    """
    Agent 3: Safety & Content Moderation Guardrail Auditor.
    Specialized agent that examines the extracted video transcript for:
    1. Violence, weapons, physical threats, self-harm.
    2. Sexual abuse, non-consensual sexual content, child safety violations.
    3. Cyberbullying, hate speech, targeted harassment.
    4. Dangerous activities & illegal exploitation.
    
    Generates a structured Safety Audit Report with verified badge status.
    """

    def __init__(self, llm_service: LLMService = None):
        self.llm_service = llm_service or LLMService()

    async def audit_content(self, transcript_text: str) -> SafetyAuditReport:
        """
        Executes deep safety inspection across all moderation dimensions.
        """
        logger.info("Executing Content Moderation & Safety Audit...")
        report, log = await self.llm_service.audit_safety(transcript_text)
        logger.info(f"Safety Audit Complete. Result: is_safe={report.is_safe}, flags={report.harm_flags_detected}")
        return report
