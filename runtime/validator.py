from dataclasses import dataclass

from agent.proposals import AgentAction, AgentProposal
from state.case_state import CaseState, Phase

VALID_TOOLS = {"kb_search", "status_api", "user_directory", "resolution_history", "policy_lookup"}

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


def validate_proposal(case: CaseState, proposal: AgentProposal) -> ValidationResult:
    if proposal.action not in _ALLOWED_ACTIONS[case.phase]:
        return ValidationResult(False, f"{proposal.action} not allowed in phase {case.phase}")

    match proposal.action:
        case AgentAction.ASK_USER:
            if not proposal.message:
                return ValidationResult(False, "ask_user requires message")

        case AgentAction.CALL_TOOL:
            if not proposal.tool_name:
                return ValidationResult(False, "call_tool requires tool_name")
            if proposal.tool_name not in VALID_TOOLS:
                return ValidationResult(False, f"unknown tool: {proposal.tool_name}")

        case AgentAction.RESOLVE:
            if not proposal.message:
                return ValidationResult(False, "resolve requires message")

        case AgentAction.ESCALATE:
            if not proposal.escalation_reason:
                return ValidationResult(False, "escalate requires escalation_reason")

    return ValidationResult(True)
