from agentic_it_support.config.settings import RuntimeLimits
from agentic_it_support.runtime import limits
from agentic_it_support.state.case_state import CaseState

_LIMITS = RuntimeLimits()


def test_tool_turn_limit_uses_this_turn_counter():
    case = CaseState(tool_calls_this_turn=_LIMITS.max_tool_calls_per_turn)

    assert limits.tool_turn_limit_reached(case, _LIMITS) is True


def test_tool_turn_limit_allows_remaining_turn_calls():
    case = CaseState(tool_calls_this_turn=_LIMITS.max_tool_calls_per_turn - 1)

    assert limits.tool_turn_limit_reached(case, _LIMITS) is False


def test_tool_case_limit_uses_total_counter():
    case = CaseState(tool_calls_total=_LIMITS.max_tool_calls_per_case)

    assert limits.tool_case_limit_reached(case, _LIMITS) is True


def test_llm_case_limit_uses_total_counter():
    case = CaseState(llm_calls_total=_LIMITS.max_llm_calls_per_case)

    assert limits.llm_case_limit_reached(case, _LIMITS) is True


def test_clarification_limit_uses_attempt_counter():
    case = CaseState(clarification_attempts=_LIMITS.max_clarification_attempts)

    assert limits.clarification_limit_reached(case, _LIMITS) is True


def test_context_message_limit_uses_conversation_length():
    case = CaseState(
        conversation=[
            {"role": "user", "content": str(i)}
            for i in range(_LIMITS.max_context_messages)
        ]
    )

    assert limits.context_message_limit_reached(case, _LIMITS) is True


def test_inner_iteration_limit_uses_iteration_count():
    assert limits.inner_iteration_limit_reached(_LIMITS.max_inner_iterations, _LIMITS) is True
    assert limits.inner_iteration_limit_reached(_LIMITS.max_inner_iterations - 1, _LIMITS) is False


# ── CorrectionBudget ────────────────────────────────────────────────────────

def test_correction_budget_signals_exhaustion_at_cap():
    budget = limits.CorrectionBudget(max_corrections=2)
    assert budget.record_correction() is False  # 1
    assert budget.record_correction() is True   # 2 -> reached cap


def test_correction_budget_counts_each_record():
    budget = limits.CorrectionBudget(max_corrections=3)
    budget.record_correction()
    budget.record_correction()
    assert budget.corrections == 2
