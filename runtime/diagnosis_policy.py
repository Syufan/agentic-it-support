"""Diagnosis-level guards for evidence grounding and runtime budget behavior."""

from dataclasses import dataclass
import re

from agent.proposals import AgentAction, AgentProposal
from runtime import limits
from runtime.constants import CONFIDENCE_RESOLVE_MIN
from state.case_state import CaseState, Phase


# ── Public decision type & entry point ─────────────────────────────────────────

@dataclass
class DiagnosisPolicyDecision:
    allowed: bool
    reason: str | None = None
    correction: str | None = None


def check_diagnosis_policy(
    case: CaseState,
    proposal: AgentProposal,
) -> DiagnosisPolicyDecision:
    """Return the first diagnosis-policy rejection, or allow the proposal."""
    # Once tool budget is exhausted, stop ordinary clarification loops.
    if _tool_case_limit_question(case, proposal):
        return DiagnosisPolicyDecision(
            False,
            "case tool-call limit reached: choose resolution or escalation instead of more clarifying",
            (
                "The case tool-call limit is reached. Do not ask another "
                "ordinary clarifying question. Provide safe low-risk guidance if "
                "available, or escalate with the evidence already gathered."
            ),
        )

    # RESOLVE requires a minimally complete case before entering RESOLVING.
    if proposal.action == AgentAction.RESOLVE and case.phase != Phase.RESOLVING:
       # A resolution needs an identified affected target.
        if not _names_affected_target(case):
            return DiagnosisPolicyDecision(
                False,
                "resolve blocked: no affected app/service/device/network identified yet",
                (
                    "Don't resolve yet — the case hasn't identified which app, service, "
                    "device, or network is affected. A generic answer is not a resolution. "
                    "Ask the employee which specific system they mean."
                ),
            )
        if case.confidence < CONFIDENCE_RESOLVE_MIN:
            return DiagnosisPolicyDecision(
                False,
                "resolve blocked: evidence-based confidence below the resolve threshold",
                (
                    "Don't propose a fix yet — it isn't grounded in evidence. Call a tool "
                    "and get a successful result first, then resolve."
                ),
            )

    return DiagnosisPolicyDecision(True)


# Target nouns required before resolution; verbs like "connect" are not enough.
_AFFECTED_TARGET = re.compile(
    r"\b("
    r"app|apps|application|applications|software|program|browser|website|site|web|portal|"
    r"dashboard|email|inbox|mailbox|account|password|"
    r"network|wifi|wi-fi|internet|ethernet|vpn|server|gateway|database|db|"
    r"computer|laptop|desktop|machine|pc|mac|macbook|phone|iphone|android|ipad|tablet|"
    r"printer|monitor|screen|display|keyboard|mouse|headset|webcam|camera|device|hardware|drive|disk|"
    r"okta|salesforce|snowflake|grafana|github|aws|jenkins|slack|zoom|jira|confluence|adobe|gmail|outlook|teams|shadowrocket|google"
    r")\b"
)


def _names_affected_target(case: CaseState) -> bool:
    """True once the employee has named some affected app/service/device/network.
    Minimum case completeness for a resolution — not a diagnostic-method rule."""
    text = " ".join(
        m["content"].lower() for m in case.conversation if m["role"] == "user"
    )
    return bool(_AFFECTED_TARGET.search(text))


# Vague first messages can be handled without an LLM call.
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

# Symptom words used to decide whether the user described a real issue.
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

# Context words that make an issue description more actionable.
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

def needs_issue_description(case: CaseState, user_message: str) -> bool:
    # Only fast-path vague first messages during intake.
    if case.phase != Phase.INTAKE:
        return False
    if case.tool_calls_total > 0:
        return False
    if len(case.conversation) != 1:
        return False

    return _normalize(user_message) in _VAGUE_INTAKE_MESSAGES


def has_usable_issue_description(case: CaseState) -> bool:
    # Check whether the conversation contains enough issue detail.
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


def _tool_case_limit_question(case: CaseState, proposal: AgentProposal) -> bool:
    # ASK_USER is blocked after the case-level tool budget is exhausted.
    return (
        proposal.action == AgentAction.ASK_USER
        and case.phase == Phase.INVESTIGATING
        and limits.tool_case_limit_reached(case)
    )


def _normalize(text: str) -> str:
    # Normalize user text for exact vague-message matching.
    return _strip_punctuation(" ".join(text.lower().strip().split()))


def _strip_punctuation(text: str) -> str:
    return text.strip(".,!?;:()[]{}\"'")
