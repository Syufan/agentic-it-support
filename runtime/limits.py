"""Mechanical runtime ceilings.

This module is intentionally business-blind: it does not decide whether to
resolve, escalate, or ask the user. It only answers whether a runtime ceiling has
been reached.
"""

from state.case_state import CaseState

MAX_INNER_ITERATIONS = 6
MAX_TOOL_CALLS_PER_TURN = 3
MAX_TOOL_CALLS_PER_CASE = 6
MAX_LLM_CALLS_PER_CASE = 12
MAX_CLARIFICATION_ATTEMPTS = 3
MAX_CONTEXT_MESSAGES = 30


def inner_iteration_limit_reached(iteration: int) -> bool:
    return iteration >= MAX_INNER_ITERATIONS


def tool_turn_limit_reached(case: CaseState) -> bool:
    return case.tool_calls_this_turn >= MAX_TOOL_CALLS_PER_TURN


def tool_case_limit_reached(case: CaseState) -> bool:
    return case.tool_calls_total >= MAX_TOOL_CALLS_PER_CASE


def llm_case_limit_reached(case: CaseState) -> bool:
    return case.llm_calls_total >= MAX_LLM_CALLS_PER_CASE


def clarification_limit_reached(case: CaseState) -> bool:
    return case.clarification_attempts >= MAX_CLARIFICATION_ATTEMPTS


def context_message_limit_reached(case: CaseState) -> bool:
    return len(case.conversation) >= MAX_CONTEXT_MESSAGES
