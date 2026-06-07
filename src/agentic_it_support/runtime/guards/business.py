import json
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

from agentic_it_support.agent.proposals import AgentAction, AgentProposal
from agentic_it_support.state.case_state import CaseState

@dataclass(frozen=True)
class PolicyRule:
    """Policy rule loaded from the JSON policy data source."""
    action: str
    description: str
    authorization: str
    notes: str
    match_patterns: list[list[str]]

@dataclass(frozen=True)
class BusinessPolicyResult:
    """Business policy outcome; matched_rule records the policy rule used to explain the decision when available."""
    allowed: bool
    reason: str | None = None
    correction: str | None = None
    matched_rule: PolicyRule | None = None



def check_business(case: CaseState, proposal: AgentProposal, policy_file: Path) -> BusinessPolicyResult:
    """Check business authorization for the proposed action."""
    
    # Business policy only guards resolve and escalate actions
    if proposal.action not in (AgentAction.RESOLVE, AgentAction.ESCALATE):
        return BusinessPolicyResult(True)
    
    rules = _load_policy_rules(policy_file)

    if proposal.action == AgentAction.RESOLVE:
        assert proposal.message
        proposal_text = proposal.message
        matched = _find_matching_policy(proposal_text, rules)
        return _authorize_resolution(matched, proposal_text)

    if proposal.action == AgentAction.ESCALATE:
        user_text = _user_conversation_text(case)
        matched = _find_matching_policy(user_text, rules)
        return _authorize_escalation(matched)
    
    raise ValueError(f"unsupported business policy action: {proposal.action}")

def _load_policy_rules(path: Path) -> list[PolicyRule]:
    """Load policy rules from the JSON policy source."""
    # TODO: validate the policy JSON shape with Pydantic
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        PolicyRule(
            action=item["action"],
            description=item["description"],
            authorization=item["authorization"],
            notes=item["notes"],
            match_patterns=item.get("match_patterns", [])
        )
        for item in data["actions"]
    ]

def _find_matching_policy(text: str, rules: list[PolicyRule]) -> PolicyRule | None:
    """Return the first policy rule that matches the text."""
    normalized_text = text.lower()
    for rule in rules:
        if _rule_matches_text(normalized_text, rule):
            return rule
    return None

def _rule_matches_text(text: str, rule: PolicyRule) -> bool:
    """Lightweight keyword matcher for the mock policy engine."""
    return any(
        all(keyword.lower() in text for keyword in pattern) for pattern in rule.match_patterns
    )

def _authorize_resolution(matched: PolicyRule | None, proposal_text: str) -> BusinessPolicyResult:
    if matched is None:
        return BusinessPolicyResult(True)
    
    if matched.authorization == "agent":
        return BusinessPolicyResult(True, matched_rule=matched)
    
    if matched.authorization == "approval":
        if _is_approval_path_guidance(proposal_text):
            return BusinessPolicyResult(True, matched_rule=matched)
        return BusinessPolicyResult(
            False,
            f"approval required for policy action '{matched.action}'",
            (
                "Do not claim the agent can directly grant or complete this action. "
                "Explain the request and approval path instead."
            ),
            matched
        )
    if matched.authorization == "human":
        return BusinessPolicyResult(
            False,
            f"human approval required for policy action '{matched.action}'",
            (
                "Do not provide this as a self-service resolution. Escalate to human IT "
                "and include the policy boundary in the handoff reason."
            ),
            matched
        )
    return _raise_unknown_authorization(matched)

def _is_approval_path_guidance(text: str) -> bool:
    # Approval guidance must include both the path and a direct-grant denial.
    _APPROVAL_PATH_MARKERS = (
        "approval",
        "it portal",
        "request",
    )
    _DIRECT_GRANT_DENIAL_MARKERS = (
        "can't grant",
        "cannot grant",
        "not able to grant",
        "may not grant",
        "do not grant",
    )
    normalized_text = text.lower()
    has_approval_path = any(marker in normalized_text for marker in _APPROVAL_PATH_MARKERS)
    has_direct_grant_denial = any(marker in normalized_text for marker in _DIRECT_GRANT_DENIAL_MARKERS)
    return has_approval_path and has_direct_grant_denial

def _user_conversation_text(case: CaseState) -> str:
    return " ".join(
        message["content"]
        for message in case.conversation
        if message["role"] == "user"
    )

def _authorize_escalation(matched: PolicyRule | None) -> BusinessPolicyResult:
    """Authorize escalation by matched policy authority."""
    if matched is None:
        return BusinessPolicyResult(
            False,
            "premature escalation: no human-authorization policy boundary",
            (
                "Escalation is not authorized. Low confidence is a diagnosis signal, not a "
                "handoff trigger. Continue investigating with a tool, resolve with safe "
                "knowledge-base guidance, or ask the employee for missing information."
            ),
            matched
        )
    
    if matched.authorization == "agent":
        return BusinessPolicyResult(
            False,
            f"agent-authorized action '{matched.action}' should be handled by the agent",
            (
                "Do not escalate this request. Continue with the authorized self-service "
                "guidance or ask for the missing information needed to proceed."
            ),
            matched,
        )

    if matched.authorization == "approval":
        return BusinessPolicyResult(
            False,
            f"approval action '{matched.action}' routes through approval, not human escalation",
            (
                "Do not escalate to a human for this. Explain the approval path and that "
                "the agent cannot directly grant the access."
            ),
            matched
        )

    if matched.authorization == "human":
        return BusinessPolicyResult(True, matched_rule=matched)
    
    return _raise_unknown_authorization(matched)

def _raise_unknown_authorization(rule:PolicyRule) -> NoReturn:
    raise ValueError(
        f"unknown policy authorization '{rule.authorization}' for action '{rule.action}'"
    )
