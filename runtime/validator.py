from dataclasses import dataclass

from agent.proposals import AgentAction, AgentProposal
from runtime import limits
from state.case_state import CaseState, Phase

_ALLOWED_ACTIONS: dict[Phase, set[AgentAction]] = {
    Phase.INTAKE:        {AgentAction.ASK_USER, AgentAction.CALL_TOOL},
    Phase.CLARIFYING:    {AgentAction.ASK_USER, AgentAction.CALL_TOOL, AgentAction.ESCALATE},
    Phase.INVESTIGATING: {AgentAction.ASK_USER, AgentAction.CALL_TOOL, AgentAction.RESOLVE, AgentAction.ESCALATE},
    Phase.RESOLVING:     {AgentAction.RESOLVE, AgentAction.ASK_USER},
    Phase.ESCALATING:    {AgentAction.ESCALATE},
    Phase.CLOSED:        set(),
}


@dataclass
class ValidationResult:
    valid: bool
    reason: str | None = None


def validate_proposal(
    case: CaseState,
    proposal: AgentProposal,
    valid_tools: set[str],
) -> ValidationResult:
    """Validate a proposal. `valid_tools` is the single source of truth for which
    tools the LLM may call — the caller passes the injected tool registry's keys,
    so the validator keeps no separate hardcoded list that could drift from it."""
    if proposal.action not in _ALLOWED_ACTIONS[case.phase]:
        return ValidationResult(False, f"{proposal.action} not allowed in phase {case.phase}")

    match proposal.action:
        case AgentAction.ASK_USER:
            if not proposal.message:
                return ValidationResult(False, "ask_user requires message")

        case AgentAction.CALL_TOOL:
            if not proposal.tool_name:
                return ValidationResult(False, "call_tool requires tool_name")
            if proposal.tool_name not in valid_tools:
                return ValidationResult(False, f"unknown tool: {proposal.tool_name}")
            if limits.tool_turn_limit_reached(case):
                return ValidationResult(False, "turn tool-call limit reached")
            if limits.tool_case_limit_reached(case):
                return ValidationResult(False, "case tool-call limit reached")

        case AgentAction.RESOLVE:
            if not proposal.message:
                return ValidationResult(False, "resolve requires message")

        case AgentAction.ESCALATE:
            if not proposal.escalation_reason:
                return ValidationResult(False, "escalate requires escalation_reason")

    return ValidationResult(True)
