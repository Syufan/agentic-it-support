from agent.proposals import AgentAction, AgentProposal
from runtime import limits
from runtime.diagnosis_policy import (
    check_diagnosis_policy,
    has_usable_issue_description,
    needs_issue_description,
)
from state.case_state import CaseState, Phase


def _proposal(**kwargs) -> AgentProposal:
    defaults = {
        "action": AgentAction.ASK_USER,
        "confidence": 0.6,
        "reasoning_summary": "test",
        "message": "What OS?",
        "escalation_reason": None,
    }
    return AgentProposal(**(defaults | kwargs))


def _case(**kwargs) -> CaseState:
    case = CaseState()
    for key, value in kwargs.items():
        setattr(case, key, value)
    return case


def test_vague_initial_message_needs_issue_description():
    case = CaseState()
    case.conversation.append({"role": "user", "content": "hey"})
    assert needs_issue_description(case, "hey") is True


def test_short_issue_does_not_skip_clarification():
    case = CaseState()
    case.conversation = [{"role": "user", "content": "VPN broken"}]
    assert has_usable_issue_description(case) is False


def test_actionable_issue_description_is_usable():
    case = CaseState()
    case.conversation = [
        {"role": "user", "content": "hey"},
        {
            "role": "user",
            "content": "shadowrocket is connected but I cannot visit google on macos right now",
        },
    ]
    assert has_usable_issue_description(case) is True


def test_diagnosis_policy_is_pass_through_for_escalation():
    # Escalation authority moved to policy/engine.py (matched on the employee's words).
    # diag no longer keyword-scans the model's reason, so it allows the ESCALATE action
    # regardless of confidence or the reason text.
    case = _case(phase=Phase.INVESTIGATING, tool_calls_total=1)
    proposal = _proposal(
        action=AgentAction.ESCALATE,
        confidence=0.3,
        message=None,
        escalation_reason="VPN needs human intervention",
    )
    assert check_diagnosis_policy(case, proposal).allowed


def test_resolve_blocked_without_evidence():
    # no successful tool source → confidence 0 → below the resolve bar → blocked
    case = _case(phase=Phase.INVESTIGATING, confidence=0.0)
    decision = check_diagnosis_policy(
        case,
        _proposal(action=AgentAction.RESOLVE, confidence=0.7, message="Try this"),
    )
    assert decision.allowed is False
    assert "resolve threshold" in decision.reason


def test_resolve_allowed_with_sufficient_confidence():
    # confidence at/above the bar (≈ one successful source) authorizes the fix
    case = _case(phase=Phase.INVESTIGATING, confidence=0.35)
    assert check_diagnosis_policy(
        case,
        _proposal(action=AgentAction.RESOLVE, message="Try this"),
    ).allowed


def test_vpn_timeout_resolve_blocked_until_environment_context_exists():
    case = _case(phase=Phase.INVESTIGATING, confidence=0.35)
    case.conversation = [
        {"role": "user", "content": "The company VPN keeps timing out when I try to connect."},
    ]
    decision = check_diagnosis_policy(
        case,
        _proposal(action=AgentAction.RESOLVE, message="Restart the VPN app."),
    )
    assert decision.allowed is False
    assert "vpn environment" in decision.reason.lower()


def test_vpn_timeout_resolve_allowed_with_environment_context():
    case = _case(phase=Phase.INVESTIGATING, confidence=0.35)
    case.conversation = [
        {"role": "user", "content": "The company VPN keeps timing out on macOS with Cisco AnyConnect."},
    ]
    assert check_diagnosis_policy(
        case,
        _proposal(action=AgentAction.RESOLVE, message="Restart the VPN app."),
    ).allowed


def test_access_grant_ask_user_blocked_until_policy_boundary_explained():
    case = _case(phase=Phase.INVESTIGATING, confidence=0.35)
    case.conversation = [
        {"role": "user", "content": "I need write access to Snowflake. Can you give me access?"},
    ]
    decision = check_diagnosis_policy(
        case,
        _proposal(action=AgentAction.ASK_USER, message="What is your user ID?"),
    )
    assert decision.allowed is False
    assert "approval path" in decision.correction.lower()


def test_access_grant_resolution_requires_no_direct_grant_boundary():
    case = _case(phase=Phase.INVESTIGATING, confidence=0.35)
    case.conversation = [
        {"role": "user", "content": "I need write access to Snowflake. Can you give me access?"},
    ]
    decision = check_diagnosis_policy(
        case,
        _proposal(action=AgentAction.RESOLVE, message="Submit a request in the IT portal."),
    )
    assert decision.allowed is False
    assert "cannot directly grant" in decision.correction.lower()


def test_tool_case_limit_reached_blocks_ordinary_clarifying_question():
    case = _case(
        phase=Phase.INVESTIGATING,
        tool_calls_total=limits.MAX_TOOL_CALLS_PER_CASE,
    )
    decision = check_diagnosis_policy(case, _proposal(action=AgentAction.ASK_USER))
    assert decision.allowed is False
    assert "tool-call limit" in decision.reason


def test_tool_case_limit_reached_allows_resolution_or_escalation_not_more_tools_or_questions():
    case = _case(
        phase=Phase.INVESTIGATING,
        tool_calls_total=limits.MAX_TOOL_CALLS_PER_CASE,
        confidence=0.6,  # evidence gathered → resolve is authorized
    )
    assert check_diagnosis_policy(
        case,
        _proposal(action=AgentAction.RESOLVE, confidence=0.6, message="Try this"),
    ).allowed
    assert check_diagnosis_policy(
        case,
        _proposal(action=AgentAction.ESCALATE, message=None, escalation_reason="still unresolved"),
    ).allowed


def test_human_handoff_signal_allows_escalation_before_tool_case_limit():
    case = _case(phase=Phase.INVESTIGATING, tool_calls_total=1)
    case.conversation = [
        {"role": "user", "content": "i clicked a suspicious link and now my account sends weird emails"},
    ]
    decision = check_diagnosis_policy(
        case,
        _proposal(
            action=AgentAction.ESCALATE,
            confidence=0.3,
            message=None,
            escalation_reason="account may be compromised",
        ),
    )
    assert decision.allowed is True
