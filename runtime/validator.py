from dataclasses import dataclass

from agent.proposals import AgentAction, AgentProposal
from runtime import limits
from state.case_state import CaseState, Phase

# Structural action permissions by workflow phase.
_ALLOWED_ACTIONS: dict[Phase, set[AgentAction]] = {
    Phase.INTAKE:        {AgentAction.ASK_USER, AgentAction.CALL_TOOL},
    Phase.CLARIFYING:    {AgentAction.ASK_USER, AgentAction.CALL_TOOL, AgentAction.ESCALATE},
    Phase.INVESTIGATING: {AgentAction.ASK_USER, AgentAction.CALL_TOOL, AgentAction.RESOLVE, AgentAction.ESCALATE},
    Phase.RESOLVING:     {AgentAction.RESOLVE, AgentAction.ASK_USER, AgentAction.ESCALATE},
    Phase.ESCALATING:    set(),
    Phase.CLOSED:        set(),
}


@dataclass
class ValidationResult:
    valid: bool
    reason: str | None = None
    # Optional re-prompt for correctable validation failures.
    correction: str | None = None


def validate_proposal(
    case: CaseState,
    proposal: AgentProposal,
    valid_tools: set[str],
) -> ValidationResult:
    """Validate proposal structure, phase legality, tool names, and tool budgets."""

    # Enforce phase-level action legality.
    if proposal.action not in _ALLOWED_ACTIONS[case.phase]:
        return ValidationResult(False, f"{proposal.action} not allowed in phase {case.phase}")

    match proposal.action:
        # ASK_USER must return a question/message to the user.
        case AgentAction.ASK_USER:
            if not proposal.message:
                return ValidationResult(False, "ask_user requires message")

        case AgentAction.CALL_TOOL:
            # Tool calls must name a tool.
            if not proposal.tool_name:
                return ValidationResult(False, "call_tool requires tool_name")
            # Only allow tools from the injected registry.
            if proposal.tool_name not in valid_tools:
                return ValidationResult(False, f"unknown tool: {proposal.tool_name}")

            # Re-prompt instead of hard-failing when tool budgets are exhausted.
            if limits.tool_turn_limit_reached(case):
                return ValidationResult(
                    False,
                    "turn tool-call limit reached",
                    correction=(
                            "Tool-call limit reached for this turn. Do not propose another CALL_TOOL. "
                            "Propose RESOLVE if current evidence is sufficient, or ASK_USER if more user input is needed. "
                    ),
                )
            if limits.tool_case_limit_reached(case):
                return ValidationResult(
                    False,
                    "case tool-call limit reached",
                    correction=(
                        "The case limit has been reached. Tool use is no longer allowed. "
                        "Choose a valid non-tool action based only on the existing case evidence."
                    ),
                )
        # Resolution needs a response message.
        case AgentAction.RESOLVE:
            if not proposal.message:
                return ValidationResult(False, "resolve requires message")

        # ESCALATE must include the handoff reason.
        case AgentAction.ESCALATE:
            if not proposal.escalation_reason:
                return ValidationResult(False, "escalate requires escalation_reason")

    return ValidationResult(True)
