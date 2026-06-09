import json
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agentic_it_support.state.case_state import CaseState


@dataclass
class Event:
    """A trace event. case_id is the primary lens for reading the flow back."""
    case_id: str
    phase: str
    event_type: str
    confidence: float
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class InMemoryEventLog:
    """In-memory event sink. Assembled in the (main/cli)"""

    def __init__(self, max_events: int | None = None) -> None:
        self._events: deque[Event] = deque(maxlen=max_events)

    def record(self, event: Event) -> None:
        self._events.append(event)

    def get_events_for_case(self, case_id: str, limit: int | None = None) -> list[Event]:
        events = [event for event in self._events if event.case_id == case_id]
        if limit is not None:
            return events[-limit:]
        return events


def write_case_trace(log: InMemoryEventLog, case_id: str, output_dir: Path) -> Path:
    """Persist a case's full event trace to {output_dir}/{case_id}.json on close."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{case_id}.json"
    events = [
        {
            "event_type": event.event_type,
            "phase": event.phase,
            "confidence": event.confidence,
            "details": event.details,
            "timestamp": event.timestamp.isoformat(),
        }
        for event in log.get_events_for_case(case_id)
    ]
    path.write_text(json.dumps(events, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _emit(log: InMemoryEventLog, case: CaseState, event_type: str, **details: Any) -> None:
    log.record(Event(
        event_type=event_type,
        case_id=case.case_id,
        phase=case.phase.value,
        confidence=case.confidence,
        details=details,
    ))


def record_turn_start(log: InMemoryEventLog, case: CaseState, user_message: str) -> None:
    _emit(log, case, "turn_start", user_message=user_message)


def record_turn_end(log: InMemoryEventLog, case: CaseState, agent_reply: str) -> None:
    _emit(log, case, "turn_end", agent_reply=agent_reply)


def record_llm_call(log: InMemoryEventLog, case: CaseState, proposed_action: str, latency_ms: float,
                    prompt_tokens: int, completion_tokens: int, total_tokens: int) -> None:
    _emit(log, case, "llm_call", proposed_action=proposed_action, latency_ms=latency_ms,
          prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=total_tokens)


def record_llm_parse_error(log: InMemoryEventLog, case: CaseState, error: str) -> None:
    _emit(log, case, "llm_parse_error", error=error)


def record_tool_start(log: InMemoryEventLog, case: CaseState, tool_name: str, inputs: dict[str, Any]) -> None:
    _emit(log, case, "tool_start", tool_name=tool_name, inputs=inputs)


def record_tool_end(log: InMemoryEventLog, case: CaseState, tool_name: str, success: bool, output: Any, conf_before: float) -> None:
    _emit(log, case, "tool_end", tool_name=tool_name, success=success, output=output, conf_before=conf_before)


def record_guard(log: InMemoryEventLog, case: CaseState, action: str, verdict: str, reason: str | None = None) -> None:
    _emit(log, case, "guard", agent_proposal=action, verdict=verdict, reason=reason)


def record_phase_transition(log: InMemoryEventLog, case: CaseState, from_phase: str, to_phase: str, action: str) -> None:
    _emit(log, case, "phase_transition", from_phase=from_phase, to_phase=to_phase, action=action)


def record_limit_hit(log: InMemoryEventLog, case: CaseState, limit: str) -> None:
    _emit(log, case, "limit_hit", limit=limit)


def record_escalation(log: InMemoryEventLog, case: CaseState, reason: str) -> None:
    _emit(log, case, "escalation", reason=reason)


def record_handoff_written(log: InMemoryEventLog, case: CaseState, path: str) -> None:
    _emit(log, case, "handoff_written", path=path)
