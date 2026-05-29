import json
import logging

from state.case_state import CaseState

logger = logging.getLogger("agentic_it_support")


def log_turn(case: CaseState) -> None:
    logger.info(json.dumps({
        "event": "turn",
        "case_id": case.case_id,
        "phase": case.phase.value,
        "confidence": case.confidence,
        "tool_calls_total": case.tool_calls_total,
        "tool_calls_current": case.tool_calls_current_investigation,
        "budget_mode": case.budget_mode.value,
        "missing_info": case.missing_info,
    }))


def log_case_closed(case: CaseState) -> None:
    logger.info(json.dumps({
        "event": "case_closed",
        "case_id": case.case_id,
        "phase": case.phase.value,
        "escalated": bool(case.escalation_context),
        "tool_calls_total": case.tool_calls_total,
        "resolution_attempts": case.resolution_attempts,
        "final_confidence": case.confidence,
        "facts": case.facts,
        "escalation_context": case.escalation_context or None,
    }))
