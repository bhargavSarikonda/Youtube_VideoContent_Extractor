import pytest
from app.agents.hitl_manager import HITLManager
from app.models.schemas import (
    SafetyAuditReport, GroundedSummary, ActionItem, HITLReviewActionRequest
)


def test_hitl_trigger_on_unsafe():
    unsafe_report = SafetyAuditReport(
        is_safe=False,
        harm_flags_detected=["violence", "harassment"],
        requires_human_intervention=True
    )
    hitl_state = HITLManager.evaluate_initial_state(unsafe_report, require_human_review=False)
    assert hitl_state.status == "PENDING"
    assert "violence" in hitl_state.triggered_reason


def test_hitl_action_approval():
    unsafe_report = SafetyAuditReport(is_safe=False, harm_flags_detected=["violence"])
    initial_hitl = HITLManager.evaluate_initial_state(unsafe_report)

    summary = GroundedSummary(
        lines=["Line 1", "Line 2", "Line 3", "Line 4", "Line 5", "Line 6"],
        line_count=6,
        source_video_id="abc12345678"
    )
    action_items = [ActionItem(id=1, task="Review logs", category="Testing", priority="High")]

    req = HITLReviewActionRequest(
        action="approve",
        reviewer_name="Safety Lead",
        comments="Approved after contextual inspection"
    )
    updated_state, final_summary, final_actions = HITLManager.apply_human_action(
        initial_hitl, req, summary, action_items
    )

    assert updated_state.status == "APPROVED"
    assert updated_state.reviewed_by == "Safety Lead"
    assert len(final_summary.lines) == 6


def test_hitl_action_edit():
    unsafe_report = SafetyAuditReport(is_safe=False, harm_flags_detected=["violence"])
    initial_hitl = HITLManager.evaluate_initial_state(unsafe_report)

    summary = GroundedSummary(
        lines=["Old Line 1", "Old Line 2", "Old Line 3", "Old Line 4", "Old Line 5", "Old Line 6"],
        line_count=6,
        source_video_id="abc12345678"
    )
    action_items = []

    new_lines = [
        "New Grounded Point 1",
        "New Grounded Point 2",
        "New Grounded Point 3",
        "New Grounded Point 4",
        "New Grounded Point 5",
        "New Grounded Point 6",
        "New Grounded Point 7"
    ]

    req = HITLReviewActionRequest(
        action="edit",
        reviewer_name="Supervisor Jane",
        comments="Refined summary lines",
        edited_summary=new_lines
    )
    updated_state, final_summary, final_actions = HITLManager.apply_human_action(
        initial_hitl, req, summary, action_items
    )

    assert updated_state.status == "EDITED"
    assert final_summary.line_count == 7
    assert final_summary.lines[0] == "New Grounded Point 1"
