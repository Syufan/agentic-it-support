from dataclasses import dataclass

from runtime.constants import CONFIDENCE_HIGH, MAX_RESOLUTION_ATTEMPTS
from runtime import budget as budget_
from state.case_state import BudgetMode, CaseState, MissingInfoSource, Phase


@dataclass
class TransitionResult:
    next_phase: Phase
    budget_mode: BudgetMode | None = None
    reset_tool_counter: bool = False
    set_exception_used: bool = False


def _result(phase: Phase, **kwargs) -> TransitionResult:
    return TransitionResult(next_phase=phase, **kwargs)


def evaluate_transition(case: CaseState) -> TransitionResult:
    match case.phase:
        case Phase.INTAKE:
            return _from_intake(case)
        case Phase.CLARIFYING:
            return _from_clarifying(case)
        case Phase.INVESTIGATING:
            return _from_investigating(case)
        case Phase.RESOLVING:
            return _from_resolving(case)
        case Phase.ESCALATING:
            return _from_escalating(case)
        case Phase.CLOSED:
            return _result(Phase.CLOSED)


def _from_intake(case: CaseState) -> TransitionResult:
    if case.missing_info_source == MissingInfoSource.USER and case.missing_info:
        return _result(Phase.CLARIFYING)   # T1
    return _result(Phase.INVESTIGATING)    # T2


def _from_clarifying(case: CaseState) -> TransitionResult:
    if not case.missing_info:
        return _result(Phase.INVESTIGATING)  # T3
    return _result(Phase.CLARIFYING)


def _from_investigating(case: CaseState) -> TransitionResult:
    if case.confidence >= CONFIDENCE_HIGH:
        return _result(Phase.RESOLVING)    # T4

    budget_done = budget_.exhausted(case.budget_mode, case.tool_calls_current_investigation)

    if budget_done:
        if case.has_safe_low_risk_guidance:
            return _result(Phase.RESOLVING)                      # T9
        if case.missing_info_source == MissingInfoSource.USER:
            return _result(Phase.CLARIFYING)                     # T7 (budget exhausted)
        return _result(Phase.ESCALATING)                         # T8

    if case.missing_info_source == MissingInfoSource.TOOL:
        return _result(Phase.INVESTIGATING)                      # T6
    return _result(Phase.CLARIFYING)                             # T7 (non-tool source)


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
