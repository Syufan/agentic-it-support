from runtime import limits
from state.case_state import CaseState


def test_tool_turn_limit_uses_this_turn_counter():
    case = CaseState(tool_calls_this_turn=limits.MAX_TOOL_CALLS_PER_TURN)

    assert limits.tool_turn_limit_reached(case) is True


def test_tool_turn_limit_allows_remaining_turn_calls():
    case = CaseState(tool_calls_this_turn=limits.MAX_TOOL_CALLS_PER_TURN - 1)

    assert limits.tool_turn_limit_reached(case) is False


def test_tool_case_limit_uses_total_counter():
    case = CaseState(tool_calls_total=limits.MAX_TOOL_CALLS_PER_CASE)

    assert limits.tool_case_limit_reached(case) is True


def test_llm_case_limit_uses_total_counter():
    case = CaseState(llm_calls_total=limits.MAX_LLM_CALLS_PER_CASE)

    assert limits.llm_case_limit_reached(case) is True


def test_clarification_limit_uses_attempt_counter():
    case = CaseState(clarification_attempts=limits.MAX_CLARIFICATION_ATTEMPTS)

    assert limits.clarification_limit_reached(case) is True


def test_context_message_limit_uses_conversation_length():
    case = CaseState(
        conversation=[
            {"role": "user", "content": str(i)}
            for i in range(limits.MAX_CONTEXT_MESSAGES)
        ]
    )

    assert limits.context_message_limit_reached(case) is True
