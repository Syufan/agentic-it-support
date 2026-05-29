from agent.decisions import AgentDecision
from state.case_state import CaseState


class ValidationResult:
    pass


def validate_decision(case: CaseState, decision: AgentDecision) -> ValidationResult:
    raise NotImplementedError
