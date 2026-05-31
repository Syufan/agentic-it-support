from dataclasses import dataclass

from agent.proposals import AgentAction
from runtime import budget as budget_
from runtime.constants import CONFIDENCE_HIGH, MAX_RESOLUTION_ATTEMPTS
from state.case_state import BudgetMode, CaseState, Phase


@dataclass
class TransitionResult:
    next_phase: Phase
    budget_mode: BudgetMode | None = None
    reset_tool_counter: bool = False
    set_exception_used: bool = False


def _result(phase: Phase, **kwargs) -> TransitionResult:
    return TransitionResult(next_phase=phase, **kwargs)


def evaluate_transition(case: CaseState, action: AgentAction) -> TransitionResult:
    """Decide the next phase from (current phase, action, guard fields).

    `action` is the discrete event that just got accepted; the case fields
    (confidence, budget, resolution_attempts, ...) are the guards. No LLM call.
    """
    match case.phase:
        case Phase.INTAKE:
            return _from_intake(action)
        case Phase.CLARIFYING:
            return _from_clarifying(action)
        case Phase.INVESTIGATING:
            return _from_investigating(case, action)
        case Phase.RESOLVING:
            return _from_resolving(case)
        case Phase.ESCALATING:
            return _from_escalating(case)
        case Phase.CLOSED:
            return _result(Phase.CLOSED)


def _from_intake(action: AgentAction) -> TransitionResult:
    if action == AgentAction.ASK_USER:
        return _result(Phase.CLARIFYING)   # T1
    return _result(Phase.INVESTIGATING)    # T2


def _from_clarifying(action: AgentAction) -> TransitionResult:
    if action == AgentAction.ASK_USER:
        return _result(Phase.CLARIFYING)
    return _result(Phase.INVESTIGATING)    # T3


def _from_investigating(case: CaseState, action: AgentAction) -> TransitionResult:
    # A tool call keeps us investigating so the LLM can synthesize the result on
    # the next turn; it short-circuits the confidence/budget guards on purpose.
    if action == AgentAction.CALL_TOOL:
        return _result(Phase.INVESTIGATING)                  # T6

    if case.confidence >= CONFIDENCE_HIGH:
        return _result(Phase.RESOLVING)                      # T4

    budget_done = budget_.exhausted(case.budget_mode, case.tool_calls_current_investigation)

    if budget_done:
        if case.has_safe_low_risk_guidance:
            return _result(Phase.RESOLVING)                  # T9
        if action == AgentAction.ASK_USER:
            return _result(Phase.CLARIFYING)                 # T7 (budget exhausted)
        return _result(Phase.ESCALATING)                     # T8

    if action == AgentAction.ASK_USER:
        return _result(Phase.CLARIFYING)                     # T7

    return _result(Phase.INVESTIGATING)                      # keep investigating


def _from_resolving(case: CaseState) -> TransitionResult:
    if case.user_confirmed_resolution is True:
        return _result(Phase.CLOSED)                             # T10

    if case.user_confirmed_resolution is False:
        if case.resolution_attempts < MAX_RESOLUTION_ATTEMPTS:
            return _result(                                      # T11
                Phase.INVESTIGATING,
                budget_mode=BudgetMode.RETRY,
                reset_tool_counter=True,
            )

        if case.new_critical_fact_added and not case.exception_used:
            return _result(                                      # T12
                Phase.CLARIFYING,
                budget_mode=BudgetMode.EXCEPTION,
                reset_tool_counter=True,
                set_exception_used=True,
            )

        return _result(Phase.ESCALATING)                         # T13

    return _result(Phase.RESOLVING)


def _from_escalating(case: CaseState) -> TransitionResult:
    if case.handoff_completed:
        return _result(Phase.CLOSED)       # T14
    return _result(Phase.ESCALATING)
