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


# ── escalation authorization (engine, matched on the employee's own words) ──────
# Whether a case may be handed off is a business-authority decision in policy/engine,
# not a keyword match on the model's reason. diag no longer judges it.

def test_engine_blocks_escalation_without_human_authorization():
    # a forgotten-password lockout is agent-authorized — it must not escalate
    decision = _check_business_policy("escalate", "I forgot my password and got locked out")
    assert not decision.allowed
    assert "premature escalation" in decision.reason


def test_engine_blocks_escalation_for_investigable_issue():
    decision = _check_business_policy("escalate", "my VPN keeps timing out")
    assert not decision.allowed
    assert "premature escalation" in decision.reason


def test_engine_allows_escalation_for_human_authorized_situation():
    decision = _check_business_policy("escalate", "I think my work laptop has malware")
    assert decision.allowed
    assert decision.matched_rule.action == "handle_security_incident"


def test_engine_allows_escalation_for_lost_mfa_device():
    decision = _check_business_policy("escalate", "I lost my phone with the authenticator, can't get past MFA")
    assert decision.allowed
    assert decision.matched_rule.action == "reset_mfa_device"


def test_engine_routes_access_grant_escalation_to_approval_not_human():
    decision = _check_business_policy("escalate", "please grant me software access to Adobe")
    assert not decision.allowed
    assert decision.matched_rule.action == "grant_software_access"


def test_diagnosis_policy_no_longer_judges_escalation_authority():
    # diag is pass-through for escalate; authority is the engine's job now
    case = _case(tool_calls_total=1)
    proposal = _proposal(action=AgentAction.ESCALATE, confidence=0.3,
                         escalation_reason="needs help", message=None)
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


def test_resolve_allowed_once_confidence_clears_the_bar():
    # Model B: the gate is evidence-based confidence, not the tool-call counter.
    case = _case(confidence=0.35)
    proposal = _proposal(action=AgentAction.RESOLVE, confidence=0.3, message="Try this")
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


def test_business_policy_allows_approval_path_guidance_for_access_grant():
    decision = _check_business_policy(
        "resolve",
        (
            "I can't grant Snowflake write access directly. Submit an IT portal "
            "request with business justification; approval is required."
        ),
    )
    assert decision.allowed
    assert decision.matched_rule.action == "grant_data_access"


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
