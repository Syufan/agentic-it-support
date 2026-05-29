from dataclasses import dataclass

from agent.proposals import AgentAction, AgentProposal
from config import CONFIDENCE_HIGH, CONFIDENCE_LOW
from state import budget as budget_
from state.case_state import CaseState, Phase


@dataclass
class PolicyDecision:
    allowed: bool
    reason: str | None = None


def check(case: CaseState, proposal: AgentProposal) -> PolicyDecision:
    if proposal.action == AgentAction.ESCALATE and case.phase != Phase.ESCALATING:
        budget_ok = not budget_.exhausted(case.budget_mode, case.tool_calls_current_investigation)
        if budget_ok and proposal.confidence >= CONFIDENCE_LOW:
            return PolicyDecision(
                False,
                "premature escalation: budget not exhausted and confidence above minimum threshold",
            )

    if proposal.action == AgentAction.RESOLVE:
        # The grounding gate applies when the agent first proposes a fix
        # (INVESTIGATING). In RESOLVING the action only records the employee's
        # confirmation, so it must not be gated on tool use.
        if case.phase != Phase.RESOLVING and case.tool_calls_total == 0:
            return PolicyDecision(
                False,
                "resolve blocked: ground the diagnosis in at least one tool lookup before resolving",
            )
        if proposal.confidence >= CONFIDENCE_HIGH:
            user_turns = sum(1 for m in case.conversation if m["role"] == "user")
            if user_turns <= 1 and case.tool_calls_total < 2:
                return PolicyDecision(
                    False,
                    "insufficient investigation: high-confidence resolve requires either user clarification or multiple tool calls",
                )

    return PolicyDecision(True)
