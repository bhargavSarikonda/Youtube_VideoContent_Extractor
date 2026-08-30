from datetime import datetime, timezone
from typing import Optional, List, Tuple
from app.models.schemas import HITLState, HITLReviewActionRequest, SafetyAuditReport, GroundedSummary, ActionItem


class HITLManager:
    """
    Agent 6: Human-in-the-Loop (HITL) Gatekeeper & Review Manager.
    Enables interactive human verification, content editing, approval,
    or rejection whenever content safety flags are raised or manual oversight is requested.
    """

    @classmethod
    def evaluate_initial_state(
        cls,
        safety_report: SafetyAuditReport,
        require_human_review: bool = False
    ) -> HITLState:
        """
        Determines whether the pipeline should pause for human review.
        """
        if require_human_review:
            return HITLState(
                status="PENDING",
                triggered_reason="Manual human review was requested by the user."
            )

        if not safety_report.is_safe or safety_report.requires_human_intervention:
            reasons = ", ".join(safety_report.harm_flags_detected) or "Potential safety/harm violation"
            return HITLState(
                status="PENDING",
                triggered_reason=f"Safety Guardrail Alert triggered: {reasons}."
            )

        return HITLState(
            status="NONE",
            triggered_reason=None
        )

    @classmethod
    def apply_human_action(
        cls,
        current_state: HITLState,
        review_request: HITLReviewActionRequest,
        current_summary: GroundedSummary,
        current_action_items: List[ActionItem]
    ) -> Tuple[HITLState, GroundedSummary, List[ActionItem]]:
        """
        Processes human review action ('approve', 'edit', 'reject').
        Returns updated (HITLState, GroundedSummary, List[ActionItem]).
        """
        action = review_request.action.lower()
        now = datetime.now(timezone.utc)

        if action == "approve":
            updated_state = HITLState(
                status="APPROVED",
                triggered_reason=current_state.triggered_reason,
                reviewed_by=review_request.reviewer_name or "Human Supervisor",
                reviewed_at=now,
                review_comments=review_request.comments or "Content manually inspected and approved for PDF compilation."
            )
            return updated_state, current_summary, current_action_items

        elif action == "edit":
            # Apply edited summary if provided
            final_summary = current_summary
            if review_request.edited_summary:
                # Ensure 6 to 10 lines
                clean_lines = [l.strip() for l in review_request.edited_summary if l.strip()]
                final_summary = GroundedSummary(
                    lines=clean_lines,
                    line_count=len(clean_lines),
                    grounding_verified=True,
                    source_video_id=current_summary.source_video_id,
                    language=current_summary.language
                )

            # Apply edited action items if provided
            final_action_items = review_request.edited_action_items if review_request.edited_action_items is not None else current_action_items

            updated_state = HITLState(
                status="EDITED",
                triggered_reason=current_state.triggered_reason,
                reviewed_by=review_request.reviewer_name or "Human Supervisor",
                reviewed_at=now,
                review_comments=review_request.comments or "Content edited and approved by human supervisor.",
                edited_summary=final_summary.lines,
                edited_action_items=final_action_items
            )
            return updated_state, final_summary, final_action_items

        elif action == "reject":
            updated_state = HITLState(
                status="REJECTED",
                triggered_reason=current_state.triggered_reason,
                reviewed_by=review_request.reviewer_name or "Human Supervisor",
                reviewed_at=now,
                review_comments=review_request.comments or "Content rejected due to policy violations."
            )
            return updated_state, current_summary, current_action_items

        else:
            raise ValueError(f"Unknown HITL action '{action}'. Permitted actions: 'approve', 'edit', 'reject'.")
