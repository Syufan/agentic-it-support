from dataclasses import dataclass

from agent.proposals import AgentProposal
from policy.engine import check_business_policy
from runtime.diagnosis_policy import check_diagnosis_policy
from runtime.validator import validate_proposal
from state.case_state import CaseState
from tools.base import BaseTool

MAX_CORRECTIONS = 3


@dataclass
class GuardState:
    corrections: int = 0


@dataclass
class GuardDecision:
    allowed: bool
    correction: str | None = None
    escalation_reason: str | None = None


def check_workflow_guard(
    case: CaseState,
    proposal: AgentProposal,
    tool_registry: dict[str, BaseTool],
    state: GuardState,
) -> GuardDecision:
    validation = validate_proposal(case, proposal, valid_tools=set(tool_registry))
    if not validation.valid:
        return _reject(
            state,
            f"repeated invalid proposals: {validation.reason}",
            (
                f"Your previous response was rejected: {validation.reason}. "
                "Choose an action that is valid in the current phase and try again."
            ),
        )

    diagnosis_policy = check_diagnosis_policy(case, proposal)
    if not diagnosis_policy.allowed:
        return _reject(
            state,
            f"repeated diagnosis policy violations: {diagnosis_policy.reason}",
            diagnosis_policy.correction or (
                f"Your previous response violated diagnosis policy: {diagnosis_policy.reason}. "
                "Choose a valid next diagnostic step and try again."
            ),
        )

    business_policy = check_business_policy(
        proposal.action.value,
        _proposal_text(proposal),
    )
    if not business_policy.allowed:
        return _reject(
            state,
            f"repeated business policy violations: {business_policy.reason}",
            business_policy.correction or (
                f"Your previous response violated business policy: {business_policy.reason}. "
                "Choose an authorized next step."
            ),
        )

    return GuardDecision(True)


def _reject(
    state: GuardState,
    escalation_reason: str,
    correction: str,
) -> GuardDecision:
    state.corrections += 1
    if state.corrections > MAX_CORRECTIONS:
        return GuardDecision(False, escalation_reason=escalation_reason)
    return GuardDecision(False, correction=correction)


def _proposal_text(proposal: AgentProposal) -> str:
    """Free text for business-policy matching.

    policy/ deliberately does not know AgentProposal's shape; the runtime guard
    extracts the text surface it wants policy rules to evaluate.
    """
    return " ".join(
        part for part in (
            proposal.message or "",
            proposal.reasoning_summary,
            proposal.escalation_reason or "",
        )
        if part
    )
