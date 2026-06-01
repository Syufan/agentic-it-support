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
from runtime.workflow_guard import GuardState, MAX_CORRECTIONS, check_workflow_guard
from state.case_state import CaseState
from tools.base import BaseTool

# Runtime architecture:
#
# query_loop
#   = orchestrates one user turn and controls the inner agent loop
#
# message_builder
#   = builds LLM input from CaseState + correction feedback
#
# workflow_guard
#   = validates the LLM proposal before execution
#     validator → diagnosis_policy → business_policy
#
# action_executor
#   = executes accepted actions and writes side effects to CaseState
#
# transitions
#   = decides the next Phase from the accepted action and current CaseState
#
# Core rule:
#   LLM proposes. Runtime guards. Executor executes. Transitions move state.


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

    # Initialize this turn: load retry settings, persist the user message, and reset counters
    retry_penalty = (settings or Settings()).confidence_retry_penalty
    case.conversation.append({"role": "user", "content": user_message})
    case.tool_calls_this_turn = 0

    # Initialize per-turn observability
    event_log = event_log or InMemoryEventLog()
    record_turn_start(event_log, case.case_id, case.phase.value, case.confidence)

    # Handle vague first messages without spending an LLM call
    if needs_issue_description(case, user_message):
        return ask_for_issue_description(case, event_log)

    # Initialize proposal-correction state for this turn
    correction: str | None = None
    guard_state = GuardState()

    for _ in range(limits.MAX_INNER_ITERATIONS):

        # Pre-step runtime guard，case-level hard limit
        _raise_if_cancelled(should_cancel)
        if limits.llm_case_limit_reached(case):
            return force_escalate(case, "maximum LLM calls reached without resolution", event_log)

        # Agent proposal
        try:
            proposal = _call_agent(case, correction, llm, event_log)
        except ProposalParseError as exc:
            guard_state.corrections += 1
            if guard_state.corrections > MAX_CORRECTIONS:
                return force_escalate(case, "repeated invalid LLM responses", event_log)
            correction = (
                f"Your previous response could not be parsed as a valid AgentProposal: {exc}. "
                "Respond with a single JSON object that matches the required schema."
            )
            continue
        except LLMClientError:
            return force_escalate(case, "LLM provider error during investigation", event_log)

        # Interrupt
        _raise_if_cancelled(should_cancel)

        # Workflow guard
        correction = None
        guard = check_workflow_guard(case, proposal, tool_registry, guard_state)
        if guard.escalation_reason:
            return force_escalate(case, guard.escalation_reason, event_log)
        if not guard.allowed:
            correction = guard.correction
            continue
        
        # Execute accepted proposal
        outcome = run_accepted_action(case,proposal,tool_registry,retry_penalty,event_log)
        if outcome.continue_loop:
            continue
        return outcome.message or ""

    return force_escalate(case, "maximum investigation steps reached without resolution", event_log)


def _call_agent(
    case: CaseState,
    correction: str | None,
    llm: BaseLLMClient,
    event_log: InMemoryEventLog,
):  
    # Build the current LLM input and request the next proposal.
    llm_input = build_messages(case, correction=correction)
    proposal = llm.call(llm_input)

    # Track LLM usage for limits and observability.
    case.llm_calls_total += 1
    _record_llm_stats(case, llm, event_log)
    return proposal


def _raise_if_cancelled(should_cancel: Callable[[], bool] | None) -> None:
    # Stop the turn if the caller requested cancellation.
    if should_cancel and should_cancel():
        raise TurnCancelled()


def _record_llm_stats(
    case: CaseState,
    llm: BaseLLMClient,
    event_log: InMemoryEventLog,
) -> None:
    # Record token and latency stats when the LLM client exposes them.
    stats = getattr(llm, "last_stats", None)
    if stats is None:
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
