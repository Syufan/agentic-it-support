from dataclasses import dataclass

from agentic_it_support.agent.proposals import AgentAction
from agentic_it_support.state.case_state import CaseState, Phase


@dataclass
class TransitionResult:
    next_phase: Phase

def evaluate_transition(case: CaseState, action: AgentAction) -> TransitionResult:
    """Decide the next phase from (current phase, action, guard fields).

    `action` is the discrete event that just got accepted; the case fields
    (confidence, counters, resolution_attempts, ...) are the guards. No LLM call.
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
'''
    INTAKE
        ASK_USER → CLARIFYING (T1)
        其他：INVESTIGATING

    CLARIFYING
        ASK_USER → CLARIFYING
        其他：其他：INVESTIGATING (T3)

    INVESTIGATING
        RESOLVE → RESOLVING (T4)
        ASK_USER  → CLARIFYING (T7)
        CALL_TOOL → INVESTIGATING (T6)
    
    RESOLVING
        只看 user_confirmed_resolution
        true -> Closed
        False -> INVESTIGATING
        False + 超过上限 -> Escalation
        None -> 等待
    
    ESCALATING
        只看 handoff_completed
        True  → CLOSED (T14)
        False → ESCALATING     # 还在处理移交，等着
'''


def _from_intake(action: AgentAction) -> TransitionResult:
    if action == AgentAction.ASK_USER:
        return _result(Phase.CLARIFYING)   # T1
    return _result(Phase.INVESTIGATING)    # T2


def _from_clarifying(action: AgentAction) -> TransitionResult:
    if action == AgentAction.ASK_USER:
        return _result(Phase.CLARIFYING)
    return _result(Phase.INVESTIGATING)    # T3


def _from_investigating(case: CaseState, action: AgentAction) -> TransitionResult:
    if action == AgentAction.RESOLVE:
        return _result(Phase.RESOLVING)        # T4 — propose a fix, await confirmation
    if action == AgentAction.ASK_USER:
        return _result(Phase.CLARIFYING)       # T7 — need user-only info
    return _result(Phase.INVESTIGATING)        # T6 — a tool call (or default) keeps investigating


def _from_resolving(case: CaseState) -> TransitionResult:
    if case.user_confirmed_resolution is True:
        return _result(Phase.CLOSED)                             # T10

    if case.user_confirmed_resolution is False:               # T11
        return _result(Phase.INVESTIGATING)

    return _result(Phase.RESOLVING)


def _from_escalating(case: CaseState) -> TransitionResult:
    if case.handoff_completed:
        return _result(Phase.CLOSED)       # T14
    return _result(Phase.ESCALATING)


def _result(phase: Phase, **kwargs) -> TransitionResult:
    return TransitionResult(next_phase=phase, **kwargs)