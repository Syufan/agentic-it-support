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
            validation.correction or (
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
    state.corrections += 1
    if state.corrections > MAX_CORRECTIONS:
        return GuardDecision(False, escalation_reason=escalation_reason)
    return GuardDecision(False, correction=correction)


def _check_business_authorization(
    case: CaseState,
    proposal: AgentProposal,
) -> BusinessPolicyDecision:
    """Run the policy engine with the right text surface per action.

    - escalate: judged on the employee's objective situation (their own words), not
      the model's escalation_reason — so a request the agent can handle cannot talk
      its way into a handoff. Exception: once the case tool budget is exhausted, a
      handoff is the legitimate runtime fallback, authorized regardless of policy.
    - everything else (resolve): judged on what the model proposes to do, to catch an
      over-reaching self-service fix.
    """
    if proposal.action == AgentAction.ESCALATE:
        if limits.tool_case_limit_reached(case):
            return BusinessPolicyDecision(True)
        return check_business_policy("escalate", _user_conversation_text(case))
    return check_business_policy(proposal.action.value, _proposal_text(proposal))


def _proposal_text(proposal: AgentProposal) -> str:
    """Free text for resolve-path business-policy matching.

    policy/ deliberately does not know AgentProposal's shape; the runtime guard
    extracts the text surface it wants policy rules to evaluate.
    """
    return " ".join(
        part for part in (
            proposal.message or "",
            proposal.escalation_reason or "",
        )
        if part
    )


def _user_conversation_text(case: CaseState) -> str:
    """The employee's own words — the objective basis for an escalation decision."""
    return " ".join(
        m["content"] for m in case.conversation if m["role"] == "user"
    )
