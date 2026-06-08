from datetime import datetime

from agentic_it_support.observability.event_tracing import (
    Event,
    InMemoryEventLog,
    record_escalation,
    record_guard,
    record_handoff_written,
    record_limit_hit,
    record_llm_call,
    record_llm_parse_error,
    record_phase_transition,
    record_tool_end,
    record_tool_start,
    record_turn_end,
    record_turn_start,
)
from agentic_it_support.state.case_state import CaseState, Phase


def _case(case_id="case-1", phase=Phase.INTAKE, confidence=0.0) -> CaseState:
    return CaseState(case_id=case_id, phase=phase, confidence=confidence)


# ── Event dataclass ───────────────────────────────────────────────────────────

def test_event_carries_core_fields():
    e = Event(event_type="turn_start", case_id="abc", phase="intake", confidence=0.0)
    assert e.event_type == "turn_start"
    assert e.phase == "intake"
    assert e.confidence == 0.0


def test_event_timestamp_is_datetime():
    e = Event(event_type="turn_start", case_id="abc", phase="intake", confidence=0.0)
    assert isinstance(e.timestamp, datetime)


def test_event_details_defaults_to_empty():
    e = Event(event_type="turn_start", case_id="abc", phase="intake", confidence=0.0)
    assert e.details == {}


# ── InMemoryEventLog ──────────────────────────────────────────────────────────

def _evt(case_id="x", event_type="turn_start"):
    return Event(event_type=event_type, case_id=case_id, phase="intake", confidence=0.0)


def test_empty_log_has_no_events():
    assert InMemoryEventLog().get_events_for_case("case-1") == []


def test_record_appends_event():
    log = InMemoryEventLog()
    log.record(_evt())
    assert len(log.get_events_for_case("x")) == 1


def test_get_events_for_case_returns_copy():
    log = InMemoryEventLog()
    log.record(_evt())
    log.get_events_for_case("x").clear()
    assert len(log.get_events_for_case("x")) == 1


def test_get_events_for_case_filters_by_case_id():
    log = InMemoryEventLog()
    log.record(_evt(case_id="case-1"))
    log.record(_evt(case_id="case-2", event_type="tool_end"))
    assert [e.case_id for e in log.get_events_for_case("case-1")] == ["case-1"]


def test_get_events_for_case_can_limit_results():
    log = InMemoryEventLog()
    for i in range(3):
        log.record(_evt(case_id="case-1", event_type=str(i)))
    assert [e.event_type for e in log.get_events_for_case("case-1", limit=2)] == ["1", "2"]


def test_unbounded_by_default():
    log = InMemoryEventLog()
    for _ in range(100):
        log.record(_evt())
    assert len(log.get_events_for_case("x")) == 100


def test_max_events_drops_oldest():
    log = InMemoryEventLog(max_events=2)
    for i in range(3):
        log.record(_evt(event_type=str(i)))
    assert [e.event_type for e in log.get_events_for_case("x")] == ["1", "2"]


# ── probes read identity/phase/confidence from the case ───────────────────────

def test_emit_pulls_core_fields_from_case():
    log = InMemoryEventLog()
    record_turn_start(log, _case(phase=Phase.CLARIFYING, confidence=0.35), "hi")
    e = log.get_events_for_case("case-1")[0]
    assert (e.case_id, e.phase, e.confidence) == ("case-1", "clarifying", 0.35)


def test_record_turn_start_captures_user_message():
    log = InMemoryEventLog()
    record_turn_start(log, _case(), "my vpn is down")
    e = log.get_events_for_case("case-1")[0]
    assert e.event_type == "turn_start"
    assert e.details["user_message"] == "my vpn is down"


def test_record_turn_end_captures_reply_and_phase_as_outcome():
    log = InMemoryEventLog()
    record_turn_end(log, _case(phase=Phase.RESOLVING), "try these steps")
    e = log.get_events_for_case("case-1")[0]
    assert e.details["agent_reply"] == "try these steps"
    # the terminal phase IS the outcome; no separate outcome field
    assert e.phase == "resolving"
    assert "outcome" not in e.details


def test_record_llm_call_captures_action_and_latency():
    log = InMemoryEventLog()
    record_llm_call(log, _case(), "call_tool", 123.4)
    e = log.get_events_for_case("case-1")[0]
    assert e.details["proposed_action"] == "call_tool"
    assert e.details["latency_ms"] == 123.4


def test_record_llm_parse_error_captures_error():
    log = InMemoryEventLog()
    record_llm_parse_error(log, _case(), "invalid json")
    assert log.get_events_for_case("case-1")[0].details["error"] == "invalid json"


def test_record_tool_start_captures_name_and_inputs():
    log = InMemoryEventLog()
    record_tool_start(log, _case(phase=Phase.INVESTIGATING), "kb_search", {"query": "vpn"})
    e = log.get_events_for_case("case-1")[0]
    assert e.event_type == "tool_start"
    assert e.details["tool_name"] == "kb_search"
    assert e.details["inputs"] == {"query": "vpn"}


def test_record_tool_end_captures_success_output_and_conf_before():
    log = InMemoryEventLog()
    record_tool_end(log, _case(confidence=0.35), "kb_search", True, {"results": ["vpn guide"]}, conf_before=0.0)
    e = log.get_events_for_case("case-1")[0]
    assert e.details["success"] is True
    assert e.details["output"] == {"results": ["vpn guide"]}
    assert e.details["conf_before"] == 0.0
    # the post-tool confidence rides the top-level field
    assert e.confidence == 0.35


def test_record_guard_captures_action_verdict_reason():
    log = InMemoryEventLog()
    record_guard(log, _case(), "resolve", "retry", "confidence below threshold")
    e = log.get_events_for_case("case-1")[0]
    assert e.details["agent_proposal"] == "resolve"
    assert e.details["verdict"] == "retry"
    assert e.details["reason"] == "confidence below threshold"


def test_record_guard_allow_has_no_reason():
    log = InMemoryEventLog()
    record_guard(log, _case(), "resolve", "allow")
    assert log.get_events_for_case("case-1")[0].details["reason"] is None


def test_record_phase_transition_tracks_from_to_and_trigger():
    log = InMemoryEventLog()
    record_phase_transition(log, _case(phase=Phase.CLARIFYING), "intake", "clarifying", "ask_user")
    e = log.get_events_for_case("case-1")[0]
    assert e.details["from_phase"] == "intake"
    assert e.details["to_phase"] == "clarifying"
    assert e.details["action"] == "ask_user"
    assert e.phase == "clarifying"


def test_record_limit_hit_captures_limit_name():
    log = InMemoryEventLog()
    record_limit_hit(log, _case(), "max_corrections")
    assert log.get_events_for_case("case-1")[0].details["limit"] == "max_corrections"


def test_record_escalation_captures_reason():
    log = InMemoryEventLog()
    record_escalation(log, _case(), "LLM repeatedly failed guard checks")
    assert log.get_events_for_case("case-1")[0].details["reason"] == "LLM repeatedly failed guard checks"


def test_record_handoff_written_captures_path():
    log = InMemoryEventLog()
    record_handoff_written(log, _case(phase=Phase.ESCALATING), "handoffs/case-1.json")
    assert log.get_events_for_case("case-1")[0].details["path"] == "handoffs/case-1.json"
