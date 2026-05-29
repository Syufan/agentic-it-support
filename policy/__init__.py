from dataclasses import dataclass

from agent.proposals import AgentAction, AgentProposal
from config import CONFIDENCE_LOW
from state import budget as budget_
from state.case_state import CaseState


@dataclass
class PolicyDecision:
    allowed: bool
    reason: str | None = None


def check(case: CaseState, proposal: AgentProposal) -> PolicyDecision:
    if proposal.action == AgentAction.ESCALATE:
        budget_ok = not budget_.exhausted(case.budget_mode, case.tool_calls_current_investigation)
        if budget_ok and proposal.confidence >= CONFIDENCE_LOW:
            return PolicyDecision(
                False,
                "premature escalation: budget not exhausted and confidence above minimum threshold",
            )

    if proposal.action == AgentAction.RESOLVE:
        if case.tool_calls_total == 0 and proposal.confidence < CONFIDENCE_LOW:
            return PolicyDecision(
                False,
                "resolve blocked: no investigation performed and confidence below minimum threshold",
            )

    return PolicyDecision(True)
