from dataclasses import dataclass

from agent.proposals import AgentAction, AgentProposal
from state.case_state import CaseState, Phase

VALID_TOOLS = {"kb_search", "status_api", "user_directory", "resolution_history"}

_ALLOWED_ACTIONS: dict[Phase, set[AgentAction]] = {
    Phase.INTAKE:        {AgentAction.ASK_USER, AgentAction.CALL_TOOL},
    Phase.CLARIFYING:    {AgentAction.ASK_USER, AgentAction.CALL_TOOL},
    Phase.INVESTIGATING: {AgentAction.ASK_USER, AgentAction.CALL_TOOL, AgentAction.RESOLVE, AgentAction.ESCALATE},
    Phase.RESOLVING:     {AgentAction.RESOLVE, AgentAction.ASK_USER},
    Phase.ESCALATING:    {AgentAction.ESCALATE},
    Phase.CLOSED:        set(),
}


@dataclass
class ValidationResult:
    valid: bool
    reason: str | None = None


def validate_decision(case: CaseState, decision: AgentProposal) -> ValidationResult:
    if decision.action not in _ALLOWED_ACTIONS[case.phase]:
        return ValidationResult(False, f"{decision.action} not allowed in phase {case.phase}")

    match decision.action:
        case AgentAction.ASK_USER:
            if not decision.message:
                return ValidationResult(False, "ask_user requires message")

        case AgentAction.CALL_TOOL:
            if not decision.tool_name:
                return ValidationResult(False, "call_tool requires tool_name")
            if decision.tool_name not in VALID_TOOLS:
                return ValidationResult(False, f"unknown tool: {decision.tool_name}")

        case AgentAction.RESOLVE:
            if not decision.message:
                return ValidationResult(False, "resolve requires message")

        case AgentAction.ESCALATE:
            if not decision.escalation_reason:
                return ValidationResult(False, "escalate requires escalation_reason")

    return ValidationResult(True)
