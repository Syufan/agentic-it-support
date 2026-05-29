import json
import logging

import pytest

from observability.logger import log_case_closed, log_turn
from state.case_state import BudgetMode, CaseState, Phase


def _case(**kwargs) -> CaseState:
    c = CaseState()
    for k, v in kwargs.items():
        setattr(c, k, v)
    return c


# ── log_turn ──────────────────────────────────────────────────────────────────

def test_log_turn_emits_one_record(caplog):
    with caplog.at_level(logging.INFO, logger="agentic_it_support"):
        log_turn(_case())
    assert len(caplog.records) == 1


def test_log_turn_record_is_valid_json(caplog):
    with caplog.at_level(logging.INFO, logger="agentic_it_support"):
        log_turn(_case())
    json.loads(caplog.records[0].message)  # must not raise


def test_log_turn_event_field(caplog):
    with caplog.at_level(logging.INFO, logger="agentic_it_support"):
        log_turn(_case())
    data = json.loads(caplog.records[0].message)
    assert data["event"] == "turn"


def test_log_turn_includes_case_id(caplog):
    case = _case()
    with caplog.at_level(logging.INFO, logger="agentic_it_support"):
        log_turn(case)
    data = json.loads(caplog.records[0].message)
    assert data["case_id"] == case.case_id


def test_log_turn_includes_phase(caplog):
    case = _case(phase=Phase.INVESTIGATING)
    with caplog.at_level(logging.INFO, logger="agentic_it_support"):
        log_turn(case)
    data = json.loads(caplog.records[0].message)
    assert data["phase"] == "investigating"


def test_log_turn_includes_tool_counts(caplog):
    case = _case(tool_calls_total=3, tool_calls_current_investigation=2)
    with caplog.at_level(logging.INFO, logger="agentic_it_support"):
        log_turn(case)
    data = json.loads(caplog.records[0].message)
    assert data["tool_calls_total"] == 3
    assert data["tool_calls_current"] == 2


# ── log_case_closed ───────────────────────────────────────────────────────────

def test_log_case_closed_event_field(caplog):
    with caplog.at_level(logging.INFO, logger="agentic_it_support"):
        log_case_closed(_case(phase=Phase.CLOSED))
    data = json.loads(caplog.records[0].message)
    assert data["event"] == "case_closed"


def test_log_case_closed_escalated_false_when_no_context(caplog):
    with caplog.at_level(logging.INFO, logger="agentic_it_support"):
        log_case_closed(_case())
    data = json.loads(caplog.records[0].message)
    assert data["escalated"] is False


def test_log_case_closed_escalated_true_when_context_present(caplog):
    case = _case(escalation_context={"reason": "needs admin"})
    with caplog.at_level(logging.INFO, logger="agentic_it_support"):
        log_case_closed(case)
    data = json.loads(caplog.records[0].message)
    assert data["escalated"] is True


def test_log_case_closed_includes_facts(caplog):
    case = _case(facts={"os": "macOS"})
    with caplog.at_level(logging.INFO, logger="agentic_it_support"):
        log_case_closed(case)
    data = json.loads(caplog.records[0].message)
    assert data["facts"]["os"] == "macOS"
