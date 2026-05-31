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


def test_escalation_blocked_when_tool_limit_not_reached_without_direct_handoff_reason():
    case = _case(
        phase=Phase.INVESTIGATING,
        tool_calls_total=1,
    )
    proposal = _proposal(
        action=AgentAction.ESCALATE,
        confidence=0.3,
        message=None,
        escalation_reason="VPN needs human intervention",
    )
    decision = check_diagnosis_policy(case, proposal)
    assert decision.allowed is False
    assert "premature escalation" in decision.reason


def test_escalation_allowed_when_tool_case_limit_reached():
    case = _case(
        phase=Phase.INVESTIGATING,
        tool_calls_total=limits.MAX_TOOL_CALLS_PER_CASE,
    )
    proposal = _proposal(
        action=AgentAction.ESCALATE,
        message=None,
        escalation_reason="still unresolved",
    )
    assert check_diagnosis_policy(case, proposal).allowed


def test_escalation_allowed_for_direct_handoff_reason():
    case = _case(phase=Phase.CLARIFYING, tool_calls_total=0)
    proposal = _proposal(
        action=AgentAction.ESCALATE,
        message=None,
        escalation_reason="hardware replacement required",
    )
    assert check_diagnosis_policy(case, proposal).allowed


def test_resolve_blocked_without_tool_evidence():
    case = _case(phase=Phase.INVESTIGATING, tool_calls_total=0)
    decision = check_diagnosis_policy(
        case,
        _proposal(action=AgentAction.RESOLVE, confidence=0.7, message="Try this"),
    )
    assert decision.allowed is False
    assert "tool lookup" in decision.reason


def test_high_confidence_resolve_needs_user_clarification_or_multiple_tools():
    # the guard reads the runtime-computed case.confidence (set directly here)
    case = _case(phase=Phase.INVESTIGATING, tool_calls_total=1, confidence=0.9)
    case.conversation = [{"role": "user", "content": "VPN broken"}]
    decision = check_diagnosis_policy(
        case,
        _proposal(action=AgentAction.RESOLVE, message="Try this"),
    )
    assert decision.allowed is False
    assert "insufficient investigation" in decision.reason


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
    )
    assert check_diagnosis_policy(
        case,
        _proposal(action=AgentAction.RESOLVE, confidence=0.6, message="Try this"),
    ).allowed
    assert check_diagnosis_policy(
        case,
        _proposal(action=AgentAction.ESCALATE, message=None, escalation_reason="still unresolved"),
    ).allowed


def test_direct_handoff_signal_allows_escalation_before_tool_case_limit():
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


