from dataclasses import dataclass
import re

from agent.proposals import AgentAction, AgentProposal
from config import CONFIDENCE_HIGH
from state import budget as budget_
from state.case_state import CaseState, Phase

_VAGUE_INTAKE_MESSAGES = {
    "can you help me",
    "good afternoon",
    "good morning",
    "hello",
    "hello there",
    "hey",
    "hey there",
    "hi",
    "hi there",
    "help",
    "help me",
    "i need help",
    "need help",
    "problem",
    "issue",
    "yo",
}

_ISSUE_SYMPTOMS = {
    "broken",
    "can't",
    "cant",
    "cannot",
    "crash",
    "disconnect",
    "doesn't work",
    "doesnt work",
    "error",
    "fail",
    "failing",
    "freeze",
    "frozen",
    "hang",
    "hanging",
    "locked",
    "not working",
    "stuck",
    "timeout",
    "timed out",
    "unable",
}

_CONTEXT_MARKERS = {
    "right now",
    "today",
    "yesterday",
    "started",
    "happening",
    "connected",
    "no error",
    "error message",
    "mac",
    "macos",
    "windows",
    "website",
    "google",
}

_DIRECT_HANDOFF_REASONS = {
    "admin",
    "administrator",
    "approval",
    "compromised",
    "hardware",
    "replacement",
    "security",
    "breach",
    "unsupported",
    "out of scope",
    "outside our supported scope",
    "human approval",
}

_DIRECT_HANDOFF_SIGNALS = {
    "account sends weird emails",
    "account sending weird emails",
    "compromised",
    "cracked",
    "hardware replacement",
    "malware",
    "phishing",
    "screen is cracked",
    "suspicious link",
    "weird emails",
}

_SERVICE_WIDE_MARKERS = {
    "everyone",
    "multiple users",
    "my team",
    "my teammates",
    "same issue",
    "team",
    "teammates",
}

_STATUS_CHECK_SERVICES = {
    "grafana",
    "okta",
    "salesforce",
    "snowflake",
    "tableau",
    "vpn",
}


@dataclass
class DiagnosisPolicyDecision:
    allowed: bool
    reason: str | None = None
    correction: str | None = None


def check_diagnosis_policy(
    case: CaseState,
    proposal: AgentProposal,
) -> DiagnosisPolicyDecision:
    """Validate diagnosis workflow boundaries.

    This layer owns diagnosis/escalation evidence rules. It does not validate
    schema shape, execute tools, or generate user-facing responses.
    """
    if _repeated_pre_tool_question_after_actionable_issue(case, proposal):
        return DiagnosisPolicyDecision(
            False,
            "employee already described an actionable issue",
            (
                "The employee has already described an actionable issue. "
                "Do not ask for more pre-tool clarification. Call a tool now, "
                "prefer `kb_search` or `resolution_history` using the app/service "
                "and symptom from the conversation."
            ),
        )

    if _service_wide_question_before_status_check(case, proposal):
        return DiagnosisPolicyDecision(
            False,
            "service-wide issue should check service status before asking user",
            (
                "This looks like a service-wide issue. Call `status_api` for the "
                "affected service before asking the employee for more local details."
            ),
        )

    if _budget_exhausted_question(case, proposal):
        return DiagnosisPolicyDecision(
            False,
            "budget exhausted: choose resolution or escalation instead of more clarifying",
            (
                "The investigation tool budget is exhausted. Do not ask another "
                "ordinary clarifying question. Provide safe low-risk guidance if "
                "available, or escalate with the evidence already gathered."
            ),
        )

    if proposal.action == AgentAction.ESCALATE and case.phase != Phase.ESCALATING:
        budget_done = budget_.exhausted(
            case.budget_mode,
            case.tool_calls_current_investigation,
        )
        if (
            not budget_done
            and not has_direct_handoff_reason(proposal.escalation_reason or "")
            and not has_direct_handoff_signal(case)
        ):
            return DiagnosisPolicyDecision(
                False,
                "premature escalation: continue investigation unless a policy boundary requires handoff",
                (
                    "Escalation is not permitted yet. Low confidence is a diagnosis "
                    "signal, not an escalation trigger. Continue investigation with "
                    "a tool or ask for user-only missing information."
                ),
            )

    if proposal.action == AgentAction.RESOLVE:
        if case.phase != Phase.RESOLVING and case.tool_calls_total == 0:
            return DiagnosisPolicyDecision(
                False,
                "resolve blocked: ground the diagnosis in at least one tool lookup before resolving",
                "Ground the diagnosis in at least one tool lookup before resolving.",
            )
        if proposal.confidence >= CONFIDENCE_HIGH:
            user_turns = sum(1 for m in case.conversation if m["role"] == "user")
            if user_turns <= 1 and case.tool_calls_total < 2:
                return DiagnosisPolicyDecision(
                    False,
                    "insufficient investigation: high-confidence resolve requires either user clarification or multiple tool calls",
                    (
                        "High-confidence resolution needs stronger evidence. "
                        "Ask one useful clarifying question or gather another tool result."
                    ),
                )

    return DiagnosisPolicyDecision(True)


def needs_issue_description(case: CaseState, user_message: str) -> bool:
    if case.phase != Phase.INTAKE:
        return False
    if case.tool_calls_total > 0:
        return False
    if len(case.conversation) != 1:
        return False

    return _normalize(user_message) in _VAGUE_INTAKE_MESSAGES


def has_usable_issue_description(case: CaseState) -> bool:
    user_text = " ".join(
        m["content"].lower()
        for m in case.conversation
        if m["role"] == "user"
    )
    normalized = " ".join(user_text.split())
    if not normalized:
        return False

    if _strip_punctuation(normalized) in _VAGUE_INTAKE_MESSAGES:
        return False

    has_symptom = any(symptom in normalized for symptom in _ISSUE_SYMPTOMS)
    has_app_or_service = bool(re.search(
        r"\b(app|application|vpn|website|site|browser|google|okta|snowflake|grafana|salesforce|shadow\w*)\b",
        normalized,
    ))
    has_context = (
        len(normalized.split()) >= 12
        or any(marker in normalized for marker in _CONTEXT_MARKERS)
    )

    return has_symptom and has_app_or_service and has_context


def has_direct_handoff_reason(reason: str) -> bool:
    normalized = reason.lower()
    return any(marker in normalized for marker in _DIRECT_HANDOFF_REASONS)


def has_direct_handoff_signal(case: CaseState) -> bool:
    text = _conversation_text(case)
    return any(marker in text for marker in _DIRECT_HANDOFF_SIGNALS)


def has_service_wide_signal(case: CaseState) -> bool:
    text = _conversation_text(case)
    has_service = any(service in text for service in _STATUS_CHECK_SERVICES)
    has_group_signal = any(marker in text for marker in _SERVICE_WIDE_MARKERS)
    return has_service and has_group_signal


def _repeated_pre_tool_question_after_actionable_issue(
    case: CaseState,
    proposal: AgentProposal,
) -> bool:
    return (
        proposal.action == AgentAction.ASK_USER
        and _pre_tool_clarifying(case)
        and case.clarification_attempts >= 1
        and has_usable_issue_description(case)
    )


def _pre_tool_clarifying(case: CaseState) -> bool:
    return case.phase in (Phase.INTAKE, Phase.CLARIFYING) and case.tool_calls_total == 0


def _service_wide_question_before_status_check(
    case: CaseState,
    proposal: AgentProposal,
) -> bool:
    return (
        proposal.action == AgentAction.ASK_USER
        and has_service_wide_signal(case)
        and not _status_checked(case)
    )


def _budget_exhausted_question(case: CaseState, proposal: AgentProposal) -> bool:
    return (
        proposal.action == AgentAction.ASK_USER
        and case.phase == Phase.INVESTIGATING
        and budget_.exhausted(case.budget_mode, case.tool_calls_current_investigation)
    )


def _status_checked(case: CaseState) -> bool:
    return any(trace.tool_name == "status_api" and trace.success for trace in case.tool_traces)


def _conversation_text(case: CaseState) -> str:
    return " ".join(
        m["content"].lower()
        for m in case.conversation
        if m["role"] == "user"
    )


def _normalize(text: str) -> str:
    return _strip_punctuation(" ".join(text.lower().strip().split()))


def _strip_punctuation(text: str) -> str:
    return text.strip(".,!?;:()[]{}\"'")
