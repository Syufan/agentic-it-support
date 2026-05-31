import pytest
from fastapi import HTTPException

from api.routes import (
    require_case,
    resolve_or_create_case,
    run_chat_turn,
    to_case_view,
    to_chat_response,
)
from llm.client import LLMProviderError
from state.case_state import CaseState, Phase
from state.session import SessionStore


# ── resolve_or_create_case ────────────────────────────────────────────────────

def test_resolve_creates_new_case_when_no_id():
    store = SessionStore()
    case = resolve_or_create_case(store, None)
    assert isinstance(case, CaseState)
    assert store.get(case.case_id) is case


def test_resolve_returns_existing_case_by_id():
    store = SessionStore()
    existing = store.create()
    assert resolve_or_create_case(store, existing.case_id) is existing


def test_resolve_raises_404_for_unknown_id():
    store = SessionStore()
    with pytest.raises(HTTPException) as exc:
        resolve_or_create_case(store, "nope")
    assert exc.value.status_code == 404


# ── require_case ──────────────────────────────────────────────────────────────

def test_require_case_returns_existing():
    store = SessionStore()
    existing = store.create()
    assert require_case(store, existing.case_id) is existing


def test_require_case_raises_404_for_unknown_id():
    store = SessionStore()
    with pytest.raises(HTTPException) as exc:
        require_case(store, "missing")
    assert exc.value.status_code == 404


# ── run_chat_turn ─────────────────────────────────────────────────────────────

def test_run_chat_turn_returns_turn_runner_output():
    case = CaseState()
    result = run_chat_turn(case, "VPN broken", llm=None, tools={},
                           turn_runner=lambda c, m, l, t: "handled")
    assert result == "handled"


def test_run_chat_turn_maps_llm_error_to_503():
    def failing(c, m, l, t):
        raise LLMProviderError("provider down")

    case = CaseState()
    with pytest.raises(HTTPException) as exc:
        run_chat_turn(case, "VPN broken", llm=None, tools={}, turn_runner=failing)
    assert exc.value.status_code == 503


# ── response mapping ──────────────────────────────────────────────────────────

def test_to_chat_response_maps_fields():
    case = CaseState(phase=Phase.INVESTIGATING)
    resp = to_chat_response(case, "here is the answer")
    assert resp.case_id == case.case_id
    assert resp.message == "here is the answer"
    assert resp.phase == "investigating"
    assert resp.is_closed is False


def test_to_chat_response_is_closed_when_phase_closed():
    assert to_chat_response(CaseState(phase=Phase.CLOSED), "done").is_closed is True


def test_to_case_view_maps_fields_and_nulls_empty_context():
    case = CaseState(phase=Phase.INVESTIGATING)
    case.confidence = 0.6
    case.tool_calls_total = 2
    case.facts = {"os": "macOS"}
    view = to_case_view(case)
    assert view.confidence == 0.6
    assert view.tool_calls_total == 2
    assert view.facts == {"os": "macOS"}
    assert view.escalation_context is None  # empty dict -> None


def test_to_case_view_keeps_escalation_context_when_present():
    case = CaseState(phase=Phase.ESCALATING)
    case.escalation_context = {"escalation_reason": "needs admin"}
    view = to_case_view(case)
    assert view.escalation_context == {"escalation_reason": "needs admin"}
