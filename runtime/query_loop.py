from collections.abc import Callable

from agent.parser import ProposalParseError
from config.settings import Settings
from llm.client import BaseLLMClient, LLMClientError
from observability.event_tracing import (
    InMemoryEventLog,
    record_llm_call,
    record_turn_start,
)
from runtime import limits
from runtime.action_executor import (
    ask_for_issue_description,
    force_escalate,
    run_accepted_action,
)
from runtime.diagnosis_policy import needs_issue_description
from runtime.message_builder import build_messages
from runtime.workflow_guard import GuardState, check_workflow_guard
from state.case_state import CaseState
from tools.base import BaseTool


class TurnCancelled(Exception):
    """Raised when the user interrupts a turn before it produces a response."""


def run_turn(
    case: CaseState,
    user_message: str,
    llm: BaseLLMClient,
    tool_registry: dict[str, BaseTool],
    event_log: InMemoryEventLog | None = None,
    should_cancel: Callable[[], bool] | None = None,
    settings: Settings | None = None,
) -> str:
    retry_penalty = (settings or Settings()).confidence_retry_penalty

    case.conversation.append({"role": "user", "content": user_message})
    case.tool_calls_this_turn = 0

    if event_log:
        record_turn_start(event_log, case.case_id, case.phase.value, case.confidence)

    if needs_issue_description(case, user_message):
        return ask_for_issue_description(case, event_log)

    correction: str | None = None
    guard_state = GuardState()

    for _ in range(limits.MAX_INNER_ITERATIONS):
        # Pre-step runtime guard，case-level hard limit
        _raise_if_cancelled(should_cancel)
        if limits.llm_case_limit_reached(case):
            return force_escalate(case, "maximum LLM calls reached without resolution")

        # Agent proposal
        try:
            proposal = _call_agent(case, correction, llm, event_log)
        except (LLMClientError, ProposalParseError):
            return force_escalate(case, "LLM provider error during investigation")

        # Interrupt
        _raise_if_cancelled(should_cancel)

        # Workflow guard
        correction = None
        guard = check_workflow_guard(case, proposal, tool_registry, guard_state)
        if guard.escalation_reason:
            return force_escalate(case, guard.escalation_reason)
        if not guard.allowed:
            correction = guard.correction
            continue
        
        # Execute accepted proposal:
        outcome = run_accepted_action(case,proposal,tool_registry,retry_penalty,event_log)
        if outcome.continue_loop:
            continue
        return outcome.message or ""

    return force_escalate(case, "maximum investigation steps reached without resolution")


def _call_agent(
    case: CaseState,
    correction: str | None,
    llm: BaseLLMClient,
    event_log: InMemoryEventLog | None,
):
    llm_input = build_messages(case, correction=correction)
    proposal = llm.call(llm_input)
    case.llm_calls_total += 1
    _record_llm_stats(case, llm, event_log)
    return proposal


def _raise_if_cancelled(should_cancel: Callable[[], bool] | None) -> None:
    if should_cancel and should_cancel():
        raise TurnCancelled()


def _record_llm_stats(
    case: CaseState,
    llm: BaseLLMClient,
    event_log: InMemoryEventLog | None,
) -> None:
    stats = getattr(llm, "last_stats", None)
    if stats is None or event_log is None:
        return
    record_llm_call(
        event_log,
        case.case_id,
        case.phase.value,
        case.confidence,
        prompt_tokens=stats.prompt_tokens,
        completion_tokens=stats.completion_tokens,
        latency_ms=stats.latency_ms,
    )
