from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


"""
Runtime observability records a bounded event timeline, not a full case dump.

CaseState owns durable conversation, tool traces, and handoff context. Events
capture process metadata that explains why the agent moved: LLM calls, guard
retries, tool calls, phase changes, escalations, and handoff writes. Do not log
full prompts, secrets, full user-directory records, or full tool outputs by
default.
"""


@dataclass
class Event:
    type: str
    case_id: str
    phase: str
    confidence: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    details: dict[str, Any] = field(default_factory=dict)


class InMemoryEventLog:
    def __init__(self, max_events: int | None = None) -> None:
        # Optional bounded in-memory event buffer.
        self._events: deque[Event] = deque(maxlen=max_events)

    def record(self, event: Event) -> None:
        self._events.append(event)

    def events(self) -> list[Event]:
        return list(self._events)

    def of_type(self, event_type: str) -> list[Event]:
        return [e for e in self._events if e.type == event_type]


# ── helpers called by the controller ─────────────────────────────────────────

def record_turn_start(
    log: InMemoryEventLog,
    case_id: str,
    phase: str,
    confidence: float,
) -> None:
    log.record(Event(type="turn_start", case_id=case_id, phase=phase, confidence=confidence))


def record_turn_end(
    log: InMemoryEventLog,
    case_id: str,
    phase: str,
    confidence: float,
    outcome: str,
) -> None:
    log.record(Event(
        type="turn_end",
        case_id=case_id,
        phase=phase,
        confidence=confidence,
        details={"outcome": outcome},
    ))


def record_tool_call(
    log: InMemoryEventLog,
    case_id: str,
    phase: str,
    confidence: float,
    tool_name: str,
    success: bool,
    inputs: dict[str, Any],
) -> None:
    log.record(Event(
        type="tool_call",
        case_id=case_id,
        phase=phase,
        confidence=confidence,
        details={"tool_name": tool_name, "success": success, "inputs": inputs},
    ))


def record_guard_retry(
    log: InMemoryEventLog,
    case_id: str,
    phase: str,
    confidence: float,
    action: str,
    reason: str,
) -> None:
    log.record(Event(
        type="guard_retry",
        case_id=case_id,
        phase=phase,
        confidence=confidence,
        details={"action": action, "reason": reason},
    ))


def record_llm_parse_error(
    log: InMemoryEventLog,
    case_id: str,
    phase: str,
    confidence: float,
    error: str,
) -> None:
    log.record(Event(
        type="llm_parse_error",
        case_id=case_id,
        phase=phase,
        confidence=confidence,
        details={"error": error},
    ))


def record_llm_call(
    log: InMemoryEventLog,
    case_id: str,
    phase: str,
    confidence: float,
    prompt_tokens: int,
    completion_tokens: int,
    latency_ms: float,
) -> None:
    log.record(Event(
        type="llm_call",
        case_id=case_id,
        phase=phase,
        confidence=confidence,
        details={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "latency_ms": latency_ms,
        },
    ))


def record_runtime_limit_hit(
    log: InMemoryEventLog,
    case_id: str,
    phase: str,
    confidence: float,
    limit: str,
) -> None:
    log.record(Event(
        type="runtime_limit_hit",
        case_id=case_id,
        phase=phase,
        confidence=confidence,
        details={"limit": limit},
    ))


def record_phase_transition(
    log: InMemoryEventLog,
    case_id: str,
    confidence: float,
    from_phase: str,
    to_phase: str,
) -> None:
    log.record(Event(
        type="phase_transition",
        case_id=case_id,
        phase=to_phase,
        confidence=confidence,
        details={"from_phase": from_phase, "to_phase": to_phase},
    ))


def record_escalation(
    log: InMemoryEventLog,
    case_id: str,
    phase: str,
    confidence: float,
    reason: str,
) -> None:
    log.record(Event(
        type="escalation",
        case_id=case_id,
        phase=phase,
        confidence=confidence,
        details={"reason": reason},
    ))


def record_handoff_written(
    log: InMemoryEventLog,
    case_id: str,
    phase: str,
    confidence: float,
    path: str,
) -> None:
    log.record(Event(
        type="handoff_written",
        case_id=case_id,
        phase=phase,
        confidence=confidence,
        details={"path": path},
    ))
