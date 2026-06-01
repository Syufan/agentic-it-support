from dataclasses import dataclass

from agent.proposals import AgentAction, AgentProposal
from policy.engine import BusinessPolicyDecision, check_business_policy
from runtime import limits
from runtime.diagnosis_policy import check_diagnosis_policy
from runtime.validator import validate_proposal
from state.case_state import CaseState
from tools.base import BaseTool

MAX_CORRECTIONS = 3


@dataclass
class GuardState:
    # Tracks correction attempts within one turn.
    corrections: int = 0


@dataclass
class GuardDecision:
    allowed: bool
    # Correction sent back to the LLM when the proposal is recoverable.
    correction: str | None = None
    # Runtime fallback reason when correction attempts are exhausted.
    escalation_reason: str | None = None


def check_workflow_guard(
    case: CaseState,
    proposal: AgentProposal,
    tool_registry: dict[str, BaseTool],
    state: GuardState,
) -> GuardDecision:

    # 1. Validate proposal shape, phase legality, tool names, and tool budgets.
    validation = validate_proposal(case, proposal, valid_tools=set(tool_registry))
    if not validation.valid:
        return _reject(
            state,
            f"repeated invalid proposals: {validation.reason}",
            validation.correction or (
                f"Your previous response was rejected: {validation.reason}. "
                "Choose an action that is valid in the current phase and try again."
            ),
        )

    # 2. Enforce diagnosis-level safety rules.
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

    # 3. Enforce business authorization boundaries.
    business_policy = _check_business_authorization(case, proposal)
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
    # Re-prompt until the correction budget is exhausted.
    state.corrections += 1
    if state.corrections > MAX_CORRECTIONS:
        return GuardDecision(False, escalation_reason=escalation_reason)
    return GuardDecision(False, correction=correction)


def _check_business_authorization(
    case: CaseState,
    proposal: AgentProposal,
) -> BusinessPolicyDecision:
    """Choose the text surface used for business policy checks."""
    if proposal.action == AgentAction.ESCALATE:
        # Escalation is authorized from the user's situation, not the LLM's wording.
        if limits.tool_case_limit_reached(case):
            return BusinessPolicyDecision(True)
        return check_business_policy("escalate", _user_conversation_text(case))
    # Resolve is checked against the proposed user-facing action.
    return check_business_policy(proposal.action.value, _proposal_text(proposal))


def _proposal_text(proposal: AgentProposal) -> str:
    """Text used to evaluate proposed non-escalation actions."""
    return " ".join(
        part for part in (
            proposal.message or "",
            proposal.escalation_reason or "",
        )
        if part
    )


def _user_conversation_text(case: CaseState) -> str:
    """User-provided text used to evaluate escalation authorization."""
    return " ".join(
        m["content"] for m in case.conversation if m["role"] == "user"
    )
