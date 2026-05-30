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
    """Validate business authorization rules for a proposed action.

    This layer is decoupled from the agent/state layers: the runtime extracts the
    action and the relevant text from its proposal and passes them in. policy/ only
    knows PolicyRule/BusinessPolicyDecision and its own data file.
    """
    if action != "resolve":
        return BusinessPolicyDecision(True)

    matched = _match_policy_rule(text, rules or load_policy_rules())
    if matched is None or matched.authorization == "agent":
        return BusinessPolicyDecision(True, matched_rule=matched)

    if matched.authorization == "approval":
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

    return BusinessPolicyDecision(
        False,
        f"human approval required for policy action '{matched.action}'",
        (
            "Do not provide this as a self-service resolution. Escalate to human IT "
            "and include the policy boundary in the handoff reason."
        ),
        matched,
    )


def _match_policy_rule(
    text: str,
    rules: list[PolicyRule],
) -> PolicyRule | None:
    lowered = text.lower()
    for rule in rules:
        if _rule_matches_text(rule, lowered):
            return rule
    return None


def _rule_matches_text(rule: PolicyRule, text: str) -> bool:
    action = rule.action.lower()
    if action in text:
        return True

    match action:
        case "unlock_account":
            return "unlock" in text and "account" in text
        case "reset_mfa_device":
            return "mfa" in text and any(word in text for word in ("reset", "re-enroll", "reenroll"))
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
