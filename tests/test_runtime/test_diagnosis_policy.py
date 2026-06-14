from agentic_it_support.agent.proposals import AgentAction, AgentProposal
from agentic_it_support.config.settings import ConfidenceSettings
from agentic_it_support.runtime.guards.diagnosis import check_diagnosis
from agentic_it_support.state.case_state import CaseState, Phase

_CONFIDENCE = ConfidenceSettings()


def _proposal(**kwargs) -> AgentProposal:
    defaults = {
        "action": AgentAction.ASK_USER,
        "message": "What OS?",
    }
    return AgentProposal(**(defaults | kwargs))


def _case(**kwargs) -> CaseState:
    case = CaseState()
    for key, value in kwargs.items():
        setattr(case, key, value)
    return case


def _check(case, proposal):
    return check_diagnosis(case, proposal, _CONFIDENCE)


# ── diagnosis only gates RESOLVE; everything else passes through ──────────────

def test_ask_user_passes_through():
    assert _check(_case(phase=Phase.INVESTIGATING), _proposal(action=AgentAction.ASK_USER)).allowed


def test_call_tool_passes_through():
    proposal = _proposal(action=AgentAction.CALL_TOOL, tool_name="kb_search",
                         tool_input={"query": "vpn"}, message=None)
    assert _check(_case(phase=Phase.INVESTIGATING), proposal).allowed


def test_diagnosis_is_pass_through_for_escalation():
    # Escalation authority is the business policy's job (matched on the employee's
    # words); diagnosis allows the ESCALATE action regardless of confidence/reason.
    case = _case(phase=Phase.INVESTIGATING, tool_calls_total=1, confidence=0.0)
    proposal = _proposal(
        action=AgentAction.ESCALATE,
        message=None,
        escalation_reason="VPN needs human intervention",
    )
    assert _check(case, proposal).allowed


# ── resolve grounding gate ────────────────────────────────────────────────────

def test_resolve_blocked_without_evidence():
    # target is named, but no successful tool source → confidence 0 → below the bar → blocked
    case = _case(phase=Phase.INVESTIGATING, confidence=0.0)
    case.conversation = [{"role": "user", "content": "my vpn keeps timing out"}]
    decision = _check(case, _proposal(action=AgentAction.RESOLVE, message="Try this"))
    assert decision.allowed is False
    assert "resolve threshold" in decision.reason


def test_resolve_allowed_with_sufficient_confidence():
    # confidence at/above the bar (≈ one successful source) authorizes the fix
    case = _case(phase=Phase.INVESTIGATING, confidence=0.35)
    case.conversation = [{"role": "user", "content": "my vpn keeps timing out"}]
    assert _check(case, _proposal(action=AgentAction.RESOLVE, message="Try this")).allowed


def test_resolve_blocked_without_an_identified_target():
    # a vague "I can't connect" names no app/service/device/network, so a fix is not
    # yet possible — clarify instead of resolving.
    case = _case(phase=Phase.INVESTIGATING, confidence=0.7)
    case.conversation = [{"role": "user", "content": "I can't connect."}]
    decision = _check(case, _proposal(action=AgentAction.RESOLVE, message="Try restarting."))
    assert decision.allowed is False
    assert "no affected app/service/device/network" in decision.reason


def test_resolve_allowed_once_target_is_named():
    # a named target ("VPN") clears the completeness gate; grounded confidence allows resolve
    case = _case(phase=Phase.INVESTIGATING, confidence=0.35)
    case.conversation = [
        {"role": "user", "content": "I can't connect."},
        {"role": "user", "content": "I mean the company VPN, it keeps timing out."},
    ]
    assert _check(case, _proposal(action=AgentAction.RESOLVE, message="Switch the server region to Auto.")).allowed


def test_diagnosis_does_not_dictate_vpn_resolution_method():
    # diagnosis only enforces evidence grounding via the confidence threshold; how to
    # resolve (ask for OS first, etc.) is the LLM's prompt-guided choice.
    case = _case(phase=Phase.INVESTIGATING, confidence=0.35)
    case.conversation = [
        {"role": "user", "content": "The company VPN keeps timing out when I try to connect."},
    ]
    assert _check(case, _proposal(action=AgentAction.RESOLVE, message="Restart the VPN app.")).allowed


def test_diagnosis_does_not_dictate_access_grant_method():
    case = _case(phase=Phase.INVESTIGATING, confidence=0.35)
    case.conversation = [
        {"role": "user", "content": "I need write access to Snowflake. Can you give me access?"},
    ]
    assert _check(case, _proposal(action=AgentAction.ASK_USER, message="What is your user ID?")).allowed
    assert _check(case, _proposal(action=AgentAction.RESOLVE, message="Submit a request in the IT portal.")).allowed
