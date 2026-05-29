from config import CONFIDENCE_HIGH, CONFIDENCE_LOW, MAX_RESOLUTION_ATTEMPTS
from state import budget as budget_
from state.case_state import CaseState, MissingInfoSource, Phase


def evaluate_transition(case: CaseState) -> Phase:
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
            return Phase.CLOSED


def _from_intake(case: CaseState) -> Phase:
    if case.missing_info_source == MissingInfoSource.USER and case.missing_info:
        return Phase.CLARIFYING  # T1
    return Phase.INVESTIGATING   # T2


def _from_clarifying(case: CaseState) -> Phase:
    if not case.missing_info:
        return Phase.INVESTIGATING  # T3
    return Phase.CLARIFYING


def _from_investigating(case: CaseState) -> Phase:
    if case.confidence >= CONFIDENCE_HIGH:
        return Phase.RESOLVING   # T4

    if case.confidence < CONFIDENCE_LOW:
        return Phase.ESCALATING  # T5

    # medium confidence (CONFIDENCE_LOW <= c < CONFIDENCE_HIGH)
    budget_done = budget_.exhausted(case.budget_mode, case.tool_calls_current_investigation)

    if budget_done:
        if case.has_safe_low_risk_guidance:
            return Phase.RESOLVING   # T9
        if case.missing_info_source == MissingInfoSource.USER:
            return Phase.CLARIFYING  # T7 (budget exhausted, ask user)
        return Phase.ESCALATING      # T8

    if case.missing_info_source == MissingInfoSource.TOOL:
        return Phase.INVESTIGATING   # T6
    return Phase.CLARIFYING          # T7 (non-tool source)


def _from_resolving(case: CaseState) -> Phase:
    if case.user_confirmed_resolution is True:
        return Phase.CLOSED          # T10

    if case.user_confirmed_resolution is False:
        if case.resolution_attempts < MAX_RESOLUTION_ATTEMPTS:
            return Phase.INVESTIGATING  # T11

        if case.new_critical_fact_added and not case.exception_used:
            return Phase.CLARIFYING     # T12

        return Phase.ESCALATING         # T13

    return Phase.RESOLVING


def _from_escalating(case: CaseState) -> Phase:
    if case.handoff_completed:
        return Phase.CLOSED      # T14
    return Phase.ESCALATING
