from pathlib import Path

from agentic_it_support.agent.proposals import AgentAction, AgentProposal
from agentic_it_support.runtime.guards.business import check_business
from agentic_it_support.state.case_state import CaseState, Phase

_POLICY_FILE = Path(__file__).resolve().parents[2] / "data" / "policies" / "policies.json"


def _proposal(**kwargs) -> AgentProposal:
    defaults = {
        "action": AgentAction.RESOLVE,
        "message": "Try this",
    }
    return AgentProposal(**(defaults | kwargs))


def _case(user_text: str | None = None) -> CaseState:
    case = CaseState(phase=Phase.INVESTIGATING)
    if user_text is not None:
        case.conversation = [{"role": "user", "content": user_text}]
    return case


def check(case, proposal):
    return check_business(case, proposal, _POLICY_FILE)


# ── pass-through for unguarded actions ───────────────────────────────────────

def test_ask_user_always_allowed():
    proposal = _proposal(action=AgentAction.ASK_USER, message="What OS?")
    assert check(_case(), proposal).allowed


def test_call_tool_always_allowed():
    proposal = _proposal(action=AgentAction.CALL_TOOL, tool_name="kb_search",
                         tool_input={"query": "vpn"}, message=None)
    assert check(_case(), proposal).allowed


# ── resolve authorization (matched on the proposal's resolution text) ─────────

def test_resolve_blocks_human_only_action():
    proposal = _proposal(action=AgentAction.RESOLVE, message="I can reset your MFA device now.")
    decision = check(_case(), proposal)
    assert not decision.allowed
    assert decision.matched_rule is not None
    assert decision.matched_rule.action == "reset_mfa_device"
    assert "human approval required" in decision.reason


def test_resolve_blocks_direct_access_grant_without_approval():
    proposal = _proposal(action=AgentAction.RESOLVE, message="I will grant software access directly.")
    decision = check(_case(), proposal)
    assert not decision.allowed
    assert decision.matched_rule is not None
    assert decision.matched_rule.action == "grant_software_access"
    assert "approval required" in decision.reason


def test_resolve_allows_approval_path_guidance():
    proposal = _proposal(
        action=AgentAction.RESOLVE,
        message=(
            "I can't grant Snowflake write access directly. Submit an IT portal "
            "request with business justification; approval is required."
        ),
    )
    decision = check(_case(), proposal)
    assert decision.allowed
    assert decision.matched_rule.action == "grant_data_access"


def test_resolve_allows_agent_authorized_guidance():
    proposal = _proposal(action=AgentAction.RESOLVE, message="Use the self-service password reset flow.")
    assert check(_case(), proposal).allowed


def test_resolve_allows_unmatched_guidance():
    proposal = _proposal(action=AgentAction.RESOLVE, message="Try restarting the VPN client.")
    assert check(_case(), proposal).allowed


# ── escalation authorization (matched on the employee's own words) ────────────

def test_escalation_allowed_for_human_authorized_situation():
    proposal = _proposal(action=AgentAction.ESCALATE, message=None,
                         escalation_reason="needs security review")
    decision = check(_case("I think my work laptop has malware"), proposal)
    assert decision.allowed
    assert decision.matched_rule.action == "handle_security_incident"


def test_escalation_allowed_for_lost_mfa_device():
    proposal = _proposal(action=AgentAction.ESCALATE, message=None,
                         escalation_reason="cannot pass MFA")
    decision = check(_case("I lost my phone with the authenticator, can't get past MFA"), proposal)
    assert decision.allowed
    assert decision.matched_rule.action == "reset_mfa_device"


def test_escalation_blocked_without_human_authorization():
    proposal = _proposal(action=AgentAction.ESCALATE, message=None,
                         escalation_reason="user is locked out")
    decision = check(_case("I forgot my password and got locked out"), proposal)
    assert not decision.allowed
    assert "premature escalation" in decision.reason


def test_escalation_blocked_for_investigable_issue():
    proposal = _proposal(action=AgentAction.ESCALATE, message=None,
                         escalation_reason="vpn down")
    decision = check(_case("my VPN keeps timing out"), proposal)
    assert not decision.allowed
    assert "premature escalation" in decision.reason


def test_escalation_for_access_grant_routes_to_approval_not_human():
    proposal = _proposal(action=AgentAction.ESCALATE, message=None,
                         escalation_reason="needs access")
    decision = check(_case("please grant me software access to Adobe"), proposal)
    assert not decision.allowed
    assert decision.matched_rule.action == "grant_software_access"
