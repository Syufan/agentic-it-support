from agent.proposals import AgentAction, AgentProposal
from runtime.diagnosis_policy import (
    check_diagnosis_policy,
    has_direct_handoff_reason,
    has_direct_handoff_signal,
    has_service_wide_signal,
    has_usable_issue_description,
    needs_issue_description,
)
from state.case_state import BudgetMode, CaseState, Phase


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


def test_repeated_pre_tool_question_is_blocked_after_actionable_description():
    case = _case(phase=Phase.CLARIFYING, clarification_attempts=1)
    case.conversation = [
        {"role": "user", "content": "hey"},
        {
            "role": "user",
            "content": "shadowrocket is connected but I cannot visit google on macos right now",
        },
    ]
    decision = check_diagnosis_policy(case, _proposal(action=AgentAction.ASK_USER))
    assert decision.allowed is False
    assert "described an actionable issue" in decision.reason


def test_first_clarifying_question_is_allowed_for_actionable_description():
    case = _case(phase=Phase.CLARIFYING, clarification_attempts=0)
    case.conversation = [
        {
            "role": "user",
            "content": "shadowrocket is connected but I cannot visit google on macos right now",
        },
    ]
    assert check_diagnosis_policy(case, _proposal(action=AgentAction.ASK_USER)).allowed


def test_escalation_blocked_when_budget_remains_without_direct_handoff_reason():
    case = _case(
        phase=Phase.INVESTIGATING,
        budget_mode=BudgetMode.MAIN,
        tool_calls_current_investigation=1,
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


def test_escalation_allowed_when_budget_exhausted():
    case = _case(
        phase=Phase.INVESTIGATING,
        budget_mode=BudgetMode.MAIN,
        tool_calls_current_investigation=5,
        tool_calls_total=5,
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


def test_direct_handoff_reason_detection():
    assert has_direct_handoff_reason("requires admin approval") is True
    assert has_direct_handoff_reason("needs human intervention") is False


def test_budget_exhausted_blocks_ordinary_clarifying_question():
    case = _case(
        phase=Phase.INVESTIGATING,
        budget_mode=BudgetMode.MAIN,
        tool_calls_current_investigation=5,
        tool_calls_total=5,
    )
    decision = check_diagnosis_policy(case, _proposal(action=AgentAction.ASK_USER))
    assert decision.allowed is False
    assert "budget exhausted" in decision.reason


def test_budget_exhausted_allows_resolution_or_escalation_not_more_tools_or_questions():
    case = _case(
        phase=Phase.INVESTIGATING,
        budget_mode=BudgetMode.MAIN,
        tool_calls_current_investigation=5,
        tool_calls_total=5,
    )
    assert check_diagnosis_policy(
        case,
        _proposal(action=AgentAction.RESOLVE, confidence=0.6, message="Try this"),
    ).allowed
    assert check_diagnosis_policy(
        case,
        _proposal(action=AgentAction.ESCALATE, message=None, escalation_reason="still unresolved"),
    ).allowed


def test_security_user_message_is_direct_handoff_signal():
    case = CaseState()
    case.conversation = [
        {"role": "user", "content": "i clicked a suspicious link and now my account sends weird emails"},
    ]
    assert has_direct_handoff_signal(case) is True


def test_direct_handoff_signal_allows_escalation_before_budget_exhaustion():
    case = _case(phase=Phase.INVESTIGATING, tool_calls_total=1, tool_calls_current_investigation=1)
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


def test_service_wide_issue_is_detected():
    case = CaseState()
    case.conversation = [
        {
            "role": "user",
            "content": "salesforce is slow since yesterday and my teammates in chicago see the same issue",
        },
    ]
    assert has_service_wide_signal(case) is True


def test_service_wide_issue_blocks_user_question_before_status_check():
    case = _case(phase=Phase.INVESTIGATING, tool_calls_total=0)
    case.conversation = [
        {
            "role": "user",
            "content": "salesforce is slow since yesterday and my teammates in chicago see the same issue",
        },
    ]
    decision = check_diagnosis_policy(
        case,
        _proposal(action=AgentAction.ASK_USER, message="Any error message?"),
    )
    assert decision.allowed is False
    assert "status_api" in decision.correction
