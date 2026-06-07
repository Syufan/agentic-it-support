from datetime import datetime

from agentic_it_support.observability.event_tracing import (
    Event,
    InMemoryEventLog,
    record_escalation,
    record_phase_transition,
    record_tool_call,
    record_turn_start,
)


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
    log.record(Event(type="turn_start", case_id="x", phase="intake", confidence=0.5))
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


def test_unbounded_by_default():
    log = InMemoryEventLog()
    for _ in range(100):
        log.record(Event(type="t", case_id="x", phase="intake", confidence=0.0))
    assert len(log.events()) == 100


def test_max_events_drops_oldest():
    log = InMemoryEventLog(max_events=2)
    for i in range(3):
        log.record(Event(type=str(i), case_id="x", phase="intake", confidence=0.0))
    assert [e.type for e in log.events()] == ["1", "2"]


# ── record_* helpers ──────────────────────────────────────────────────────────

def test_record_turn_start_writes_turn_start_event():
    log = InMemoryEventLog()
    record_turn_start(log, "case-1", "intake", 0.0)
    events = log.of_type("turn_start")
    assert len(events) == 1
    assert events[0].case_id == "case-1"
    assert events[0].phase == "intake"


def test_record_tool_call_captures_details():
    log = InMemoryEventLog()
    record_tool_call(log, "case-1", "investigating", 0.6, "kb_search", True, {"query": "vpn"})
    event = log.of_type("tool_call")[0]
    assert event.details["tool_name"] == "kb_search"
    assert event.details["success"] is True
    assert event.details["inputs"] == {"query": "vpn"}


def test_record_phase_transition_tracks_from_and_to():
    log = InMemoryEventLog()
    record_phase_transition(log, "case-1", 0.5, "intake", "clarifying")
    event = log.of_type("phase_transition")[0]
    assert event.details["from_phase"] == "intake"
    assert event.details["to_phase"] == "clarifying"
    assert event.phase == "clarifying"


def test_record_escalation_captures_reason():
    log = InMemoryEventLog()
    record_escalation(log, "case-1", "investigating", 0.3, "needs human review")
    event = log.of_type("escalation")[0]
    assert event.details["reason"] == "needs human review"
