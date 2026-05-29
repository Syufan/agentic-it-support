import pytest

from agent.proposals import AgentAction, AgentProposal
from policy import check
from state.case_state import BudgetMode, CaseState, MissingInfoSource, Phase


def _proposal(**kwargs) -> AgentProposal:
    defaults = {
        "action": AgentAction.RESOLVE,
        "confidence": 0.6,
        "reasoning_summary": "test",
        "message": "Try this",
        "escalation_reason": None,
    }
    return AgentProposal(**(defaults | kwargs))


def _case(**kwargs) -> CaseState:
    defaults = {"phase": Phase.INVESTIGATING}
    c = CaseState(**defaults)
    for k, v in kwargs.items():
        setattr(c, k, v)
    return c


# ── premature escalation guard ────────────────────────────────────────────────

def test_escalate_blocked_when_budget_remains_and_confidence_above_low():
    case = _case(tool_calls_current_investigation=0, budget_mode=BudgetMode.MAIN)
    proposal = _proposal(action=AgentAction.ESCALATE, confidence=0.6,
                         escalation_reason="needs help", message=None)
    decision = check(case, proposal)
    assert not decision.allowed
    assert "premature escalation" in decision.reason


def test_escalate_allowed_when_budget_exhausted():
    case = _case(tool_calls_current_investigation=5, budget_mode=BudgetMode.MAIN)
    proposal = _proposal(action=AgentAction.ESCALATE, confidence=0.6,
                         escalation_reason="needs help", message=None)
    decision = check(case, proposal)
    assert decision.allowed


def test_escalate_allowed_when_confidence_below_low():
    case = _case(tool_calls_current_investigation=0, budget_mode=BudgetMode.MAIN)
    proposal = _proposal(action=AgentAction.ESCALATE, confidence=0.3,
                         escalation_reason="needs help", message=None)
    decision = check(case, proposal)
    assert decision.allowed


# ── zero-effort resolve guard ─────────────────────────────────────────────────

def test_resolve_blocked_when_no_tools_called_and_confidence_below_low():
    case = _case(tool_calls_total=0)
    proposal = _proposal(action=AgentAction.RESOLVE, confidence=0.3, message="Try this")
    decision = check(case, proposal)
    assert not decision.allowed
    assert "resolve blocked" in decision.reason


def test_resolve_allowed_when_tools_called_even_with_low_confidence():
    case = _case(tool_calls_total=2)
    proposal = _proposal(action=AgentAction.RESOLVE, confidence=0.3, message="Try this")
    decision = check(case, proposal)
    assert decision.allowed


def test_resolve_allowed_when_no_tools_but_high_confidence():
    case = _case(tool_calls_total=0)
    proposal = _proposal(action=AgentAction.RESOLVE, confidence=0.6, message="Try this")
    decision = check(case, proposal)
    assert decision.allowed


# ── pass-through for unguarded actions ───────────────────────────────────────

def test_ask_user_always_allowed():
    case = _case()
    proposal = _proposal(action=AgentAction.ASK_USER, confidence=0.5, message="What OS?")
    assert check(case, proposal).allowed


def test_call_tool_always_allowed():
    case = _case()
    proposal = _proposal(action=AgentAction.CALL_TOOL, confidence=0.5,
                         tool_name="kb_search", tool_input={"query": "vpn"}, message=None)
    assert check(case, proposal).allowed
