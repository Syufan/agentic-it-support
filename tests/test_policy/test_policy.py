import pytest

from agent.proposals import AgentAction, AgentProposal
from policy.engine import check_business_policy as _check_business_policy
from policy.engine import find_policy_rules, load_policy_rules
from runtime import limits
from runtime.diagnosis_policy import check_diagnosis_policy as check
from state.case_state import CaseState, Phase


def check_business_policy(case, proposal):
    """Adapt the decoupled engine signature (action, text) to these proposal-based
    tests; `case` is unused, mirroring how the runtime extracts the inputs."""
    text = " ".join(
        part for part in (
            proposal.message or "",
            proposal.reasoning_summary,
            proposal.escalation_reason or "",
        )
        if part
    )
    return _check_business_policy(proposal.action.value, text)


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

def test_escalate_blocked_when_tool_limit_not_reached_and_confidence_above_low():
    case = _case(tool_calls_total=0)
    proposal = _proposal(action=AgentAction.ESCALATE, confidence=0.6,
                         escalation_reason="needs help", message=None)
    decision = check(case, proposal)
    assert not decision.allowed
    assert "premature escalation" in decision.reason


def test_escalate_blocked_when_tool_limit_not_reached_even_with_low_confidence():
    case = _case(tool_calls_total=1)
    proposal = _proposal(action=AgentAction.ESCALATE, confidence=0.3,
                         escalation_reason="needs help", message=None)
    decision = check(case, proposal)
    assert not decision.allowed
    assert "premature escalation" in decision.reason


def test_escalate_allowed_when_tool_case_limit_reached():
    case = _case(tool_calls_total=limits.MAX_TOOL_CALLS_PER_CASE)
    proposal = _proposal(action=AgentAction.ESCALATE, confidence=0.6,
                         escalation_reason="needs help", message=None)
    decision = check(case, proposal)
    assert decision.allowed


def test_escalate_allowed_when_already_in_escalating_phase():
    case = _case(phase=Phase.ESCALATING, tool_calls_total=0)
    proposal = _proposal(action=AgentAction.ESCALATE, confidence=0.6,
                         escalation_reason="needs help", message=None)
    assert check(case, proposal).allowed


def test_escalate_blocked_before_any_tool_for_investigable_issue():
    case = _case(tool_calls_total=0)
    proposal = _proposal(
        action=AgentAction.ESCALATE,
        confidence=0.3,
        escalation_reason="VPN connection issue requires further investigation",
        message=None,
    )
    decision = check(case, proposal)
    assert not decision.allowed
    assert "policy boundary" in decision.reason


def test_escalate_allowed_before_tool_for_direct_handoff_reason():
    case = _case(tool_calls_total=0)
    proposal = _proposal(
        action=AgentAction.ESCALATE,
        confidence=0.3,
        escalation_reason="third-party app outside our supported scope",
        message=None,
    )
    assert check(case, proposal).allowed


# ── zero-effort resolve guard ─────────────────────────────────────────────────

def test_resolve_blocked_when_no_tools_called_and_confidence_below_low():
    case = _case(tool_calls_total=0)
    proposal = _proposal(action=AgentAction.RESOLVE, confidence=0.3, message="Try this")
    decision = check(case, proposal)
    assert not decision.allowed
    assert "resolve blocked" in decision.reason


def test_resolve_blocked_when_no_tools_called_even_with_medium_confidence():
    case = _case(tool_calls_total=0)
    proposal = _proposal(action=AgentAction.RESOLVE, confidence=0.6, message="Try this")
    decision = check(case, proposal)
    assert not decision.allowed
    assert "resolve blocked" in decision.reason


def test_resolve_blocked_when_no_tools_called_even_with_high_confidence():
    case = _case(tool_calls_total=0)
    proposal = _proposal(action=AgentAction.RESOLVE, confidence=0.95, message="Try this")
    decision = check(case, proposal)
    assert not decision.allowed
    assert "resolve blocked" in decision.reason


def test_resolve_allowed_when_tools_called_even_with_low_confidence():
    case = _case(tool_calls_total=2)
    proposal = _proposal(action=AgentAction.RESOLVE, confidence=0.3, message="Try this")
    decision = check(case, proposal)
    assert decision.allowed


# ── high-confidence resolve guard ────────────────────────────────────────────

def test_high_confidence_resolve_blocked_with_single_user_turn_and_one_tool():
    # guard reads runtime-computed case.confidence (set directly here)
    case = _case(tool_calls_total=1, confidence=0.9)
    case.conversation = [{"role": "user", "content": "VPN broken"}]
    proposal = _proposal(action=AgentAction.RESOLVE, message="Try this")
    decision = check(case, proposal)
    assert not decision.allowed
    assert "insufficient investigation" in decision.reason


def test_high_confidence_resolve_allowed_after_user_clarification():
    case = _case(tool_calls_total=1)
    case.conversation = [
        {"role": "user", "content": "VPN broken"},
        {"role": "assistant", "content": "What OS?"},
        {"role": "user", "content": "macOS"},
    ]
    proposal = _proposal(action=AgentAction.RESOLVE, confidence=0.9, message="Try this")
    assert check(case, proposal).allowed


def test_high_confidence_resolve_allowed_after_thorough_tool_investigation():
    case = _case(tool_calls_total=2)
    case.conversation = [{"role": "user", "content": "VPN broken"}]
    proposal = _proposal(action=AgentAction.RESOLVE, confidence=0.9, message="Try this")
    assert check(case, proposal).allowed


def test_medium_confidence_resolve_not_blocked_by_this_rule():
    case = _case(tool_calls_total=1)
    case.conversation = [{"role": "user", "content": "VPN broken"}]
    proposal = _proposal(action=AgentAction.RESOLVE, confidence=0.65, message="Try this")
    assert check(case, proposal).allowed


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


# ── business authorization policy ──────────────────────────────────────────────

def test_business_policy_rules_load_from_policy_data():
    rules = load_policy_rules()
    assert any(rule.action == "reset_mfa_device" for rule in rules)
    assert any(rule.authorization == "human" for rule in rules)


def test_business_policy_find_rules_by_query():
    matches = find_policy_rules("mfa")
    assert [rule.action for rule in matches] == ["reset_mfa_device"]


def test_business_policy_blocks_human_only_resolution():
    case = _case(tool_calls_total=1)
    proposal = _proposal(
        action=AgentAction.RESOLVE,
        confidence=0.9,
        message="I can reset your MFA device now.",
    )
    decision = check_business_policy(case, proposal)
    assert not decision.allowed
    assert decision.matched_rule is not None
    assert decision.matched_rule.action == "reset_mfa_device"
    assert "human approval required" in decision.reason


def test_business_policy_blocks_direct_access_grant_without_approval():
    case = _case(tool_calls_total=1)
    proposal = _proposal(
        action=AgentAction.RESOLVE,
        confidence=0.9,
        message="I will grant software access directly.",
    )
    decision = check_business_policy(case, proposal)
    assert not decision.allowed
    assert decision.matched_rule is not None
    assert decision.matched_rule.action == "grant_software_access"
    assert "approval required" in decision.reason


def test_business_policy_allows_agent_authorized_guidance():
    case = _case(tool_calls_total=1)
    proposal = _proposal(
        action=AgentAction.RESOLVE,
        confidence=0.9,
        message="Use the self-service password reset flow.",
    )
    assert check_business_policy(case, proposal).allowed


# ── decoupled engine contract: (action, text), no proposal/case ───────────────

def test_engine_takes_action_and_text_directly():
    decision = _check_business_policy("resolve", "I can reset your MFA device now.")
    assert not decision.allowed
    assert decision.matched_rule.action == "reset_mfa_device"


def test_engine_passes_through_non_resolve_actions():
    assert _check_business_policy("ask_user", "anything at all").allowed
