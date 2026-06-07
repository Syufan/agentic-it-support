from dataclasses import dataclass

from agentic_it_support.agent.proposals import AgentAction, AgentProposal
from agentic_it_support.config.settings import RuntimeLimits
from agentic_it_support.runtime import limits
from agentic_it_support.state.case_state import CaseState, Phase

# Structural action permissions by workflow phase
_ALLOWED_ACTIONS: dict[Phase, set[AgentAction]] = {
    Phase.INTAKE:        {AgentAction.ASK_USER, AgentAction.CALL_TOOL},
    Phase.CLARIFYING:    {AgentAction.ASK_USER, AgentAction.CALL_TOOL, AgentAction.ESCALATE},
    Phase.INVESTIGATING: {AgentAction.ASK_USER, AgentAction.CALL_TOOL, AgentAction.RESOLVE, AgentAction.ESCALATE},
    Phase.RESOLVING:     {AgentAction.ASK_USER},
    Phase.ESCALATING:    set(),
    Phase.CLOSED:        set(),
}


@dataclass(frozen=True)
class ValidationResult:
    """Validation outcome; invalid results must include both reason and correction."""
    allowed: bool
    reason: str | None = None
    correction: str | None = None

    @property
    def valid(self) -> bool:
        return self.allowed


def validate_proposal(case: CaseState, proposal: AgentProposal, valid_tools: set[str], runtime_limits: RuntimeLimits) -> ValidationResult:
    """Validate proposal structure, phase legality, tool names, and tool budgets."""

    # Enforce phase-level action permissions
    if proposal.action not in _ALLOWED_ACTIONS[case.phase]:
        return ValidationResult(
            False, 
            f"{proposal.action} not allowed in phase {case.phase}",
            "Choose an action that is valid for the current workflow phase."
        )
    
    # Only accept resolution confirmation while waiting for the employee's confirmation
    if proposal.user_confirmed_resolution is not None and case.phase != Phase.RESOLVING:
        return ValidationResult(
            False,
            "user_confirmed_resolution is only valid while awaiting resolution confirmation",
            "Do not set user_confirmed_resolution unless the case is waiting for the employee to confirm a proposed resolution.",
        )

    match proposal.action:
        # ASK_USER must return a question/message to the user
        case AgentAction.ASK_USER:
            if not proposal.message:
                return ValidationResult(
                    False, 
                    "ask_user requires message",
                    "ASK_USER must include a user-facing question or message."
                )

        case AgentAction.CALL_TOOL:
            # Tool calls must name a tool
            if not proposal.tool_name:
                return ValidationResult(
                    False, 
                    "call_tool requires tool_name",
                    "CALL_TOOL must include the name of a registered tool."
                )
            
            # Tool calls must reference a registered tool
            if proposal.tool_name not in valid_tools:
                return ValidationResult(
                    False, 
                    f"unknown tool: {proposal.tool_name}",
                    "Choose one of the available registered tools."
                )


            # Re-prompt instead of hard-failing when the per-turn tool budget is exhausted
            if limits.tool_turn_limit_reached(case, runtime_limits):
                return ValidationResult(
                    False,
                    "turn tool-call limit reached",
                    (
                        "Tool-call limit reached for this turn. Do not propose another CALL_TOOL. "
                        "Propose RESOLVE if current evidence is sufficient, or ASK_USER if more user input is needed. "
                    )
                )
            
            # Re-prompt instead of hard-failing when the case-level tool budget is exhausted
            if limits.tool_case_limit_reached(case, runtime_limits):
                return ValidationResult(
                    False,
                    "case tool-call limit reached",
                    (
                        "The case limit has been reached. Tool use is no longer allowed. "
                        "Choose a valid non-tool action based only on the existing case evidence."
                    )
                )

        # Resolution needs a response message
        case AgentAction.RESOLVE:
            if not proposal.message:
                return ValidationResult(
                    False, 
                    "resolve requires message",
                    "RESOLVE must include the resolution message to send to the user."
                )

        # ESCALATE must include the handoff reason
        case AgentAction.ESCALATE:
            if not proposal.escalation_reason:
                return ValidationResult(
                    False, 
                    "escalate requires escalation_reason",
                    "ESCALATE must include a clear handoff reason."
                )

    return ValidationResult(True)
