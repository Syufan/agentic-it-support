from datetime import datetime

from agent.llm import MockLLMClient
from agent.proposals import AgentAction, AgentProposal
from observability.logger import Event, InMemoryEventLog
from runtime.controller import run_turn
from state.case_state import CaseState, MissingInfoSource, Phase
from tools.base import BaseTool, ToolResult
from typing import Any


def _proposal(**kwargs) -> AgentProposal:
    defaults = {
        "action": AgentAction.ASK_USER,
        "confidence": 0.6,
        "reasoning_summary": "test",
        "message": "What OS?",
    }
    return AgentProposal(**(defaults | kwargs))


class MockTool(BaseTool):
    name = "kb_search"
    description = "mock"

    def run(self, inputs: dict[str, Any]) -> ToolResult:
        return ToolResult(success=True, data={"results": []})


# ── Event dataclass ───────────────────────────────────────────────────────────

def test_event_has_type_and_phase():
    e = Event(type="turn_start", case_id="abc", phase="intake", confidence=0.5)
    assert e.type == "turn_start"
    assert e.phase == "intake"


def test_event_timestamp_is_datetime():
    e = Event(type="turn_start", case_id="abc", phase="intake", confidence=0.5)
    assert isinstance(e.timestamp, datetime)


def test_event_details_defaults_to_empty():
    e = Event(type="turn_start", case_id="abc", phase="intake", confidence=0.5)
    assert e.details == {}


# ── InMemoryEventLog basic ────────────────────────────────────────────────────

def test_empty_log_has_no_events():
    log = InMemoryEventLog()
    assert log.events() == []


def test_record_appends_event():
    log = InMemoryEventLog()
    e = Event(type="turn_start", case_id="x", phase="intake", confidence=0.5)
    log.record(e)
    assert len(log.events()) == 1


def test_events_returns_copy():
    log = InMemoryEventLog()
    log.record(Event(type="turn_start", case_id="x", phase="intake", confidence=0.5))
    copy = log.events()
    copy.clear()
    assert len(log.events()) == 1


def test_of_type_filters_by_event_type():
    log = InMemoryEventLog()
    log.record(Event(type="turn_start", case_id="x", phase="intake", confidence=0.5))
    log.record(Event(type="tool_call", case_id="x", phase="investigating", confidence=0.6))
    assert len(log.of_type("tool_call")) == 1
    assert log.of_type("tool_call")[0].type == "tool_call"


def test_of_type_returns_empty_for_unknown_type():
    log = InMemoryEventLog()
    assert log.of_type("nonexistent") == []


# ── controller integration ────────────────────────────────────────────────────

def test_turn_start_event_recorded():
    case = CaseState()
    log = InMemoryEventLog()
    run_turn(case, "VPN broken", MockLLMClient([_proposal()]), {}, event_log=log)
    assert any(e.type == "turn_start" for e in log.events())


def test_tool_call_event_recorded():
    case = CaseState(phase=Phase.INVESTIGATING)
    case.confidence = 0.6
    log = InMemoryEventLog()

    proposals = [
        _proposal(action=AgentAction.CALL_TOOL, confidence=0.6,
                  tool_name="kb_search", tool_input={"query": "vpn"},
                  message=None, missing_info_source=MissingInfoSource.TOOL),
        _proposal(action=AgentAction.RESOLVE, confidence=0.9, message="Fix"),
    ]
    run_turn(case, "VPN broken", MockLLMClient(proposals), {"kb_search": MockTool()}, event_log=log)

    tool_events = log.of_type("tool_call")
    assert len(tool_events) == 1
    assert tool_events[0].details["tool_name"] == "kb_search"


def test_tool_call_event_records_success():
    case = CaseState(phase=Phase.INVESTIGATING)
    case.confidence = 0.6
    log = InMemoryEventLog()

    proposals = [
        _proposal(action=AgentAction.CALL_TOOL, confidence=0.6,
                  tool_name="kb_search", tool_input={"query": "vpn"},
                  message=None, missing_info_source=MissingInfoSource.TOOL),
        _proposal(action=AgentAction.RESOLVE, confidence=0.9, message="Fix"),
    ]
    run_turn(case, "VPN broken", MockLLMClient(proposals), {"kb_search": MockTool()}, event_log=log)

    assert log.of_type("tool_call")[0].details["success"] is True


def test_phase_transition_event_recorded():
    case = CaseState()
    log = InMemoryEventLog()
    run_turn(case, "VPN broken", MockLLMClient([_proposal(
        missing_info_source=MissingInfoSource.USER, missing_info=["OS"],
    )]), {}, event_log=log)

    transitions = log.of_type("phase_transition")
    assert len(transitions) >= 1
    assert "to_phase" in transitions[0].details


def test_escalation_event_recorded():
    case = CaseState(phase=Phase.INVESTIGATING)
    log = InMemoryEventLog()

    run_turn(case, "VPN broken", MockLLMClient([
        _proposal(action=AgentAction.ESCALATE, confidence=0.3,
                  escalation_reason="Needs admin", message=None),
    ]), {}, event_log=log)

    esc_events = log.of_type("escalation")
    assert len(esc_events) == 1
    assert esc_events[0].details["reason"] == "Needs admin"


def test_no_event_log_does_not_raise():
    case = CaseState()
    run_turn(case, "VPN broken", MockLLMClient([_proposal()]), {})  # no event_log


def test_event_case_id_matches_case():
    case = CaseState()
    log = InMemoryEventLog()
    run_turn(case, "VPN broken", MockLLMClient([_proposal()]), {}, event_log=log)
    assert all(e.case_id == case.case_id for e in log.events())
