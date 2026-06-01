import json
from dataclasses import dataclass
from pathlib import Path

_POLICY_FILE = Path(__file__).parent.parent / "data" / "policies" / "policies.json"


@dataclass(frozen=True)
class PolicyRule:
    action: str
    description: str
    authorization: str
    notes: str


@dataclass(frozen=True)
class BusinessPolicyDecision:
    allowed: bool
    reason: str | None = None
    correction: str | None = None
    matched_rule: PolicyRule | None = None


def load_policy_rules(path: Path = _POLICY_FILE) -> list[PolicyRule]:
    # Load policy rules from the JSON policy source.
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        PolicyRule(
            action=item["action"],
            description=item["description"],
            authorization=item["authorization"],
            notes=item["notes"],
        )
        for item in data["actions"]
    ]


def find_policy_rules(query: str, path: Path = _POLICY_FILE) -> list[PolicyRule]:
    # Search policy rules by action or description.
    normalized = query.strip().lower()
    if not normalized:
        return load_policy_rules(path)
    return [
        rule for rule in load_policy_rules(path)
        if normalized in rule.action.lower() or normalized in rule.description.lower()
    ]


def check_business_policy(
    action: str,
    text: str,
    rules: list[PolicyRule] | None = None,
) -> BusinessPolicyDecision:
    """Check whether a proposed resolve/escalate action is authorized."""
    if action not in ("resolve", "escalate"):
        return BusinessPolicyDecision(True)

    matched = _match_policy_rule(text, rules or load_policy_rules())

    # Escalation is allowed only for human-authorized policy actions.
    if action == "escalate":
        return _check_escalation_authorization(matched)
    
    # Agent-authorized or unmatched resolve actions are allowed.
    if matched is None or matched.authorization == "agent":
        return BusinessPolicyDecision(True, matched_rule=matched)

    # Approval-gated actions must explain the approval path, not claim direct grant.
    if matched.authorization == "approval":
        if _is_approval_path_guidance(text):
            return BusinessPolicyDecision(True, matched_rule=matched)
        return BusinessPolicyDecision(
            False,
            f"approval required for policy action '{matched.action}'",
            (
                "Do not claim the agent can directly grant or complete this action. "
                "Explain the request/approval path or escalate if the employee needs "
                "urgent access."
            ),
            matched,
        )
    
    # Human-only actions must not be resolved as self-service.
    return BusinessPolicyDecision(
        False,
        f"human approval required for policy action '{matched.action}'",
        (
            "Do not provide this as a self-service resolution. Escalate to human IT "
            "and include the policy boundary in the handoff reason."
        ),
        matched,
    )


def _check_escalation_authorization(matched: PolicyRule | None) -> BusinessPolicyDecision:
    """Authorize escalation by matched policy authority."""
    if matched is not None and matched.authorization == "human":
        return BusinessPolicyDecision(True, matched_rule=matched)

    if matched is not None and matched.authorization == "approval":
        return BusinessPolicyDecision(
            False,
            f"approval action '{matched.action}' routes through approval, not human escalation",
            (
                "Do not escalate to a human for this. Explain the approval path and that "
                "the agent cannot directly grant the access."
            ),
            matched,
        )

    return BusinessPolicyDecision(
        False,
        "premature escalation: no human-authorization policy boundary",
        (
            "Escalation is not authorized. Low confidence is a diagnosis signal, not a "
            "handoff trigger. Continue investigating with a tool, resolve with safe "
            "knowledge-base guidance, or ask the employee for missing information."
        ),
        matched,
    )


def _match_policy_rule(
    text: str,
    rules: list[PolicyRule],
) -> PolicyRule | None:
    # Return the first policy rule that matches the text.
    lowered = text.lower()
    for rule in rules:
        if _rule_matches_text(rule, lowered):
            return rule
    return None


def _rule_matches_text(rule: PolicyRule, text: str) -> bool:
    # Lightweight keyword matcher for the mock policy engine.
    action = rule.action.lower()
    if action in text:
        return True

    match action:
        case "unlock_account":
            return "unlock" in text and "account" in text
        case "reset_mfa_device":
            # Match how employees actually describe MFA trouble — they say "lost my
            # authenticator", rarely the word "reset". Resolve-path messages that are
            # safe agent guidance ("re-scan the QR code") won't carry these words.
            return ("mfa" in text or "authenticator" in text) and any(
                word in text
                for word in (
                    "reset", "re-enroll", "reenroll", "lost", "lose", "stolen",
                    "new phone", "can't get past", "cannot get past", "no backup",
                )
            )
        case "handle_security_incident":
            return any(
                word in text
                for word in (
                    "malware", "virus", "ransomware", "phishing", "phishing email",
                    "suspicious link", "weird emails", "compromised", "hacked",
                    "breach", "stolen laptop", "stolen device",
                )
            )
        case "grant_software_access":
            return "grant" in text and "access" in text and "software" in text
        case "grant_data_access":
            return "grant" in text and "access" in text and any(
                system in text for system in ("snowflake", "grafana", "database", "dashboard")
            )
        case "modify_network_hardware":
            return any(word in text for word in ("vpn gateway", "network hardware", "router config"))
        case _:
            return False


def _is_approval_path_guidance(text: str) -> bool:
    # Approval guidance must include both the path and a direct-grant denial.
    lowered = text.lower()
    has_approval_path = "approval" in lowered or "it portal" in lowered or "request" in lowered
    has_direct_grant_denial = any(
        marker in lowered
        for marker in (
            "can't grant",
            "cannot grant",
            "can’t grant",
            "not able to grant",
            "may not grant",
            "do not grant",
        )
    )
    return has_approval_path and has_direct_grant_denial
