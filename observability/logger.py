import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from state.case_state import CaseState

logger = logging.getLogger("agentic_it_support")


@dataclass
class Event:
    type: str
    case_id: str
    phase: str
    confidence: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    details: dict[str, Any] = field(default_factory=dict)


class InMemoryEventLog:
    def __init__(self) -> None:
        self._events: list[Event] = []

    def record(self, event: Event) -> None:
        self._events.append(event)

    def events(self) -> list[Event]:
        return list(self._events)

    def of_type(self, event_type: str) -> list[Event]:
        return [e for e in self._events if e.type == event_type]


# ── helpers called by the controller ─────────────────────────────────────────

def record_turn_start(log: InMemoryEventLog, case: CaseState) -> None:
    log.record(Event(
        type="turn_start",
        case_id=case.case_id,
        phase=case.phase.value,
        confidence=case.confidence,
    ))


def record_tool_call(
    log: InMemoryEventLog,
    case: CaseState,
    tool_name: str,
    success: bool,
    inputs: dict[str, Any],
) -> None:
    log.record(Event(
        type="tool_call",
        case_id=case.case_id,
        phase=case.phase.value,
        confidence=case.confidence,
        details={"tool_name": tool_name, "success": success, "inputs": inputs},
    ))


def record_phase_transition(
    log: InMemoryEventLog,
    case: CaseState,
    from_phase: str,
    to_phase: str,
) -> None:
    log.record(Event(
        type="phase_transition",
        case_id=case.case_id,
        phase=to_phase,
        confidence=case.confidence,
        details={"from_phase": from_phase, "to_phase": to_phase},
    ))


def record_escalation(
    log: InMemoryEventLog,
    case: CaseState,
    reason: str,
) -> None:
    log.record(Event(
        type="escalation",
        case_id=case.case_id,
        phase=case.phase.value,
        confidence=case.confidence,
        details={"reason": reason},
    ))


# ── fire-and-forget logging (kept for backward compatibility) ─────────────────

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
