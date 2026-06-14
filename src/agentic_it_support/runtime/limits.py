from dataclasses import dataclass
from agentic_it_support.config.settings import RuntimeLimits
from agentic_it_support.state.case_state import CaseState

@dataclass
class CorrectionBudget:
    max_corrections: int
    corrections: int = 0

    def record_correction(self) -> bool:
        self.corrections += 1
        return self.corrections >= self.max_corrections


def inner_iteration_limit_reached(iteration: int, runtime_limits: RuntimeLimits) -> bool:
    # Per-turn agent loop limit
    return iteration >= runtime_limits.max_inner_iterations


def tool_turn_limit_reached(case: CaseState, runtime_limits: RuntimeLimits) -> bool: 
    # Tool budget for the current user turn
    return case.tool_calls_this_turn >= runtime_limits.max_tool_calls_per_turn


def tool_case_limit_reached(case: CaseState, runtime_limits: RuntimeLimits) -> bool:
    # Tool budget for the whole case
    return case.tool_calls_total >= runtime_limits.max_tool_calls_per_case


def llm_case_limit_reached(case: CaseState, runtime_limits: RuntimeLimits) -> bool:
    # LLM budget for the whole case
    return case.llm_calls_total >= runtime_limits.max_llm_calls_per_case


def clarification_limit_reached(case: CaseState, runtime_limits: RuntimeLimits) -> bool:
    # Consecutive clarification budget
    return case.clarification_attempts >= runtime_limits.max_clarification_attempts


def context_message_limit_reached(case: CaseState, runtime_limits: RuntimeLimits) -> bool:
    # Conversation history budget
    return len(case.conversation) >= runtime_limits.max_context_messages
