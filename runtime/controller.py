from collections.abc import Callable
from datetime import datetime, timezone

from agent.llm import BaseLLMClient, LLMClientError
from agent.proposals import AgentAction, AgentProposal
from config import CONFIDENCE_HIGH
from observability.logger import (
    InMemoryEventLog,
    record_escalation,
    record_llm_call,
    record_phase_transition,
    record_tool_call,
    record_turn_start,
)
from policy import check as policy_check
from runtime.message_builder import build_messages
from runtime.transitions import TransitionResult, evaluate_transition
from runtime.validator import validate_proposal
from state.case_state import CaseState, Phase, ToolTrace
from tools.base import BaseTool, ToolResult

_MAX_INNER_ITERATIONS = 10
_MAX_CORRECTIONS = 3


class TurnCancelled(Exception):
    """Raised when the user interrupts a turn before it produces a response."""


def run_turn(
    case: CaseState,
    user_message: str,
    llm: BaseLLMClient,
    tool_registry: dict[str, BaseTool],
    event_log: InMemoryEventLog | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> str:
    case.conversation.append({"role": "user", "content": user_message})

    if event_log:
        record_turn_start(event_log, case)

    correction: str | None = None
    corrections = 0
    for _ in range(_MAX_INNER_ITERATIONS):
        if should_cancel and should_cancel():
            raise TurnCancelled()

        llm_input = build_messages(case, correction=correction)
        correction = None
        try:
            proposal = llm.call(llm_input)
        except LLMClientError:
            return _force_escalate(case, "LLM provider error during investigation")

        _record_llm_stats(case, llm, event_log)

        # The provider call is the blocking wait, so re-check here: this is what
        # lets ESC interrupt the common single-call turn before we mutate state.
        if should_cancel and should_cancel():
            raise TurnCancelled()

        # A guardrail violation is correctable: feed the reason back and let the
        # agent revise on the next iteration (bounded by _MAX_INNER_ITERATIONS),
        # rather than terminating the case on the first stumble.
        validation = validate_proposal(case, proposal)
        if not validation.valid:
            corrections += 1
            if corrections > _MAX_CORRECTIONS:
                return _force_escalate(case, f"repeated invalid proposals: {validation.reason}")
            correction = (
                f"Your previous response was rejected: {validation.reason}. "
                "Choose an action that is valid in the current phase and try again."
            )
            continue

        policy = policy_check(case, proposal)
        if not policy.allowed:
            corrections += 1
            if corrections > _MAX_CORRECTIONS:
                return _force_escalate(case, f"repeated policy violations: {policy.reason}")
            correction = (
                f"That action is not permitted yet: {policy.reason}. "
                "Gather more evidence with a tool, ask the user a clarifying "
                "question, or only escalate once you genuinely cannot proceed."
            )
            continue

        _project_to_state(case, proposal)

        if proposal.action == AgentAction.CALL_TOOL:
            _execute_tool(case, proposal, tool_registry)
            if event_log:
                last_trace = case.tool_traces[-1]
                record_tool_call(
                    event_log, case,
                    tool_name=last_trace.tool_name,
                    success=last_trace.success,
                    inputs=last_trace.inputs,
                )
            prev_phase = case.phase
            _apply_transition(case, evaluate_transition(case))
            if event_log and case.phase != prev_phase:
                record_phase_transition(event_log, case, prev_phase.value, case.phase.value)
            continue

        if proposal.action == AgentAction.RESOLVE:
            case.resolution_attempts += 1

        if proposal.action == AgentAction.ESCALATE:
            _build_escalation_context(case, proposal.escalation_reason, proposal.confidence)
            case.handoff_completed = True
            if event_log:
                record_escalation(event_log, case, proposal.escalation_reason or "")

        prev_phase = case.phase
        _apply_transition(case, evaluate_transition(case))
        if event_log and case.phase != prev_phase:
            record_phase_transition(event_log, case, prev_phase.value, case.phase.value)

        response = _format_response(proposal)
        case.conversation.append({"role": "assistant", "content": response})
        return response

    return _force_escalate(case, "maximum investigation steps reached without resolution")


def _force_escalate(case: CaseState, reason: str) -> str:
    _build_escalation_context(case, reason, case.confidence)
    case.phase = Phase.ESCALATING
    case.handoff_completed = True
    _apply_transition(case, evaluate_transition(case))
    msg = (
        "I wasn't able to fully resolve this issue. "
        "I'm connecting you with an IT specialist who will have all the context — "
        "you won't need to repeat yourself."
    )
    case.conversation.append({"role": "assistant", "content": msg})
    return msg


def _record_llm_stats(
    case: CaseState,
    llm: BaseLLMClient,
    event_log: InMemoryEventLog | None,
) -> None:
    stats = getattr(llm, "last_stats", None)
    if stats is None:
        return
    case.llm_calls += 1
    case.prompt_tokens += stats.prompt_tokens
    case.completion_tokens += stats.completion_tokens
    case.llm_latency_ms += stats.latency_ms
    if event_log:
        record_llm_call(
            event_log, case,
            prompt_tokens=stats.prompt_tokens,
            completion_tokens=stats.completion_tokens,
            latency_ms=stats.latency_ms,
        )


def _project_to_state(case: CaseState, proposal: AgentProposal) -> None:
    case.confidence = proposal.confidence
    case.missing_info_source = proposal.missing_info_source
    case.missing_info = list(proposal.missing_info)
    case.has_safe_low_risk_guidance = proposal.has_safe_low_risk_guidance
    case.new_critical_fact_added = proposal.new_critical_fact_added
    if proposal.user_confirmed_resolution is not None:
        case.user_confirmed_resolution = proposal.user_confirmed_resolution


def _execute_tool(
    case: CaseState,
    proposal: AgentProposal,
    tool_registry: dict[str, BaseTool],
) -> None:
    tool_name = proposal.tool_name or ""
    tool = tool_registry.get(tool_name)

    if tool is None:
        result = ToolResult(success=False, data={}, error=f"tool '{tool_name}' not available")
    else:
        result = tool.run(proposal.tool_input)

    case.tool_traces.append(ToolTrace(
        tool_name=tool_name,
        inputs=proposal.tool_input,
        output=result.data if result.success else {"error": result.error},
        success=result.success,
        budget_mode=case.budget_mode,
        timestamp=datetime.now(timezone.utc),
    ))

    if result.success:
        case.facts[f"{tool_name}_result"] = result.data
    else:
        case.facts[f"{tool_name}_error"] = result.error or "unknown error"

    case.tool_calls_current_investigation += 1
    case.tool_calls_total += 1


def _apply_transition(case: CaseState, result: TransitionResult) -> None:
    case.phase = result.next_phase
    if result.budget_mode is not None:
        case.budget_mode = result.budget_mode
    if result.reset_tool_counter:
        case.tool_calls_current_investigation = 0
    if result.set_exception_used:
        case.exception_used = True


def _build_escalation_context(
    case: CaseState,
    reason: str | None,
    confidence: float,
) -> None:
    issue_description = next(
        (m["content"] for m in case.conversation if m["role"] == "user"), ""
    )
    case.escalation_context = {
        "escalation_reason": reason,
        "confidence": confidence,
        "issue_description": issue_description,
        "conversation": list(case.conversation),
        "facts": dict(case.facts),
        "hypotheses": list(case.hypotheses),
        "tool_traces": [
            {
                "tool": t.tool_name,
                "success": t.success,
                "inputs": t.inputs,
                "output": t.output,
            }
            for t in case.tool_traces
        ],
        "failed_resolutions": list(case.failed_resolutions),
        "resolution_attempts": case.resolution_attempts,
    }


def _format_response(proposal: AgentProposal) -> str:
    if proposal.action == AgentAction.ESCALATE:
        return (
            "I wasn't able to fully resolve this issue. "
            "I'm connecting you with an IT specialist who will have all the context — "
            "you won't need to repeat yourself."
        )

    if proposal.action == AgentAction.RESOLVE:
        message = proposal.message or ""
        if proposal.confidence >= CONFIDENCE_HIGH:
            return f"I found a likely fix for your issue: {message}"
        return f"I'm not fully certain, but this is a safe first step to try: {message}"

    return proposal.message or ""
