"""Correctable "is this a sensible next step?" guards for an agent proposal.

`check_diagnosis_policy` runs the guards and returns the first rejection (a reason
plus a correction the agent is re-prompted with), or an allow. It governs only
*outcomes and authority* — it never decides diagnostic *method* (which tool to call,
whether to re-ask, what order). That is the LLM's job, guided by the phase prompts.

Three concerns live here, tagged §1–§3. Only §1 truly belongs; the other two are
future-extraction candidates:
  §1 Diagnosis workflow    — the genuine owner
  §2 Escalation gating      — when handoff is authorized        → escalation_policy.py
  §3 Runtime-limit reaction — what to do once tool budget is spent → workflow_guard
"""

from dataclasses import dataclass
import re

from agent.proposals import AgentAction, AgentProposal
from runtime import limits
from runtime.constants import CONFIDENCE_HIGH
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
    """Return the first guard rejection (reason + correction) or an allow. Pure
    decision: no schema validation, no tool execution, no user-facing responses.
    Checks run §3 → §2 → §1, but they are mutually exclusive by action so order
    doesn't matter; the §N numbering tracks ownership, not execution order.
    """
    # §3 — tool budget spent: stop asking, force a decision now
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

    # §2 — escalation only when authorized: budget already spent, or a handoff
    # reason/signal is present; otherwise low confidence is not a reason to hand off
    if proposal.action == AgentAction.ESCALATE and case.phase != Phase.ESCALATING:
        if (
            not limits.tool_case_limit_reached(case)
            and not _has_direct_handoff_reason(proposal.escalation_reason or "")
            and not _has_direct_handoff_signal(case)
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

    # §1 — a resolution must be grounded in tool evidence
    if proposal.action == AgentAction.RESOLVE:
        if case.phase != Phase.RESOLVING and case.tool_calls_total == 0:
            return DiagnosisPolicyDecision(
                False,
                "resolve blocked: ground the diagnosis in at least one tool lookup before resolving",
                "Ground the diagnosis in at least one tool lookup before resolving.",
            )
        if case.confidence >= CONFIDENCE_HIGH:
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


# ── §1. Diagnosis workflow ─────────────────────────────────────────────────────
# Vague-intake and usable-issue-description checks. (has_usable_issue_description is
# also used by action_executor's soft-close.) The resolve-grounding rule is inline
# in check_diagnosis_policy above.

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


# ── §2. Escalation gating ──────────────────────────────────────────────────────
# Handoff-authorization helpers: an explicit reason, or a signal in the conversation.
# The premature-escalation decision that uses them is inline above (§2).

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


def _has_direct_handoff_reason(reason: str) -> bool:
    normalized = reason.lower()
    return any(marker in normalized for marker in _DIRECT_HANDOFF_REASONS)


def _has_direct_handoff_signal(case: CaseState) -> bool:
    text = _conversation_text(case)
    return any(marker in text for marker in _DIRECT_HANDOFF_SIGNALS)


# ── §3. Runtime-limit reaction ─────────────────────────────────────────────────
# Fires only for ASK_USER once the case tool budget is spent (CALL_TOOL at the same
# ceiling is the validator's job). The decision that uses it is inline above (§3).

def _tool_case_limit_question(case: CaseState, proposal: AgentProposal) -> bool:
    return (
        proposal.action == AgentAction.ASK_USER
        and case.phase == Phase.INVESTIGATING
        and limits.tool_case_limit_reached(case)
    )


# ── Shared text helpers ────────────────────────────────────────────────────────
# Trivial string utilities used across the sections above.

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
