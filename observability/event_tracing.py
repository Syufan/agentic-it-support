from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


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
        # max_events=None keeps the log unbounded; a positive cap turns it into a
        # ring buffer that drops the oldest events so it can't grow without limit.
        self._events: deque[Event] = deque(maxlen=max_events)

    def record(self, event: Event) -> None:
        self._events.append(event)

    def events(self) -> list[Event]:
        return list(self._events)

    def of_type(self, event_type: str) -> list[Event]:
        return [e for e in self._events if e.type == event_type]


# ── helpers called by the controller ─────────────────────────────────────────
# These take raw scalars (case_id / phase / confidence), not a CaseState, so the
# observability layer has no dependency on the domain. The runtime extracts them.

def record_turn_start(
    log: InMemoryEventLog,
    case_id: str,
    phase: str,
    confidence: float,
) -> None:
    log.record(Event(type="turn_start", case_id=case_id, phase=phase, confidence=confidence))


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
