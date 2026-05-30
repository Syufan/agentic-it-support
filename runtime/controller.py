from collections.abc import Callable
from datetime import datetime, timezone

from config.settings import Settings
from llm.client import BaseLLMClient, LLMClientError
from agent.proposals import AgentAction, AgentProposal
from runtime.constants import CONFIDENCE_HIGH
from observability.logger import (
    InMemoryEventLog,
    record_escalation,
    record_llm_call,
    record_phase_transition,
    record_tool_call,
    record_turn_start,
)
from policy.engine import check_business_policy
from runtime.calibration import calibrate
from runtime.diagnosis_policy import (
    check_diagnosis_policy,
    has_usable_issue_description,
    needs_issue_description,
)
from runtime.message_builder import build_messages
from runtime.transitions import TransitionResult, evaluate_transition
from runtime.validator import validate_proposal
from state.case_state import CaseState, MissingInfoSource, Phase, ToolTrace
from tools.base import BaseTool, ToolResult

_MAX_INNER_ITERATIONS = 10
_MAX_CORRECTIONS = 3
_MAX_CLARIFICATION_ATTEMPTS = 3


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
    # Deployment config is injected by the composition root; default to env-backed
    # values so CLI/tests that don't inject still behave identically.
    retry_penalty = (settings or Settings()).confidence_retry_penalty

    case.conversation.append({"role": "user", "content": user_message})

    if event_log:
        record_turn_start(event_log, case)

    if needs_issue_description(case, user_message):
        return _ask_for_issue_description(case, event_log)

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
        validation = validate_proposal(case, proposal, valid_tools=set(tool_registry))
        if not validation.valid:
            corrections += 1
            if corrections > _MAX_CORRECTIONS:
                return _force_escalate(case, f"repeated invalid proposals: {validation.reason}")
            correction = (
                f"Your previous response was rejected: {validation.reason}. "
                "Choose an action that is valid in the current phase and try again."
            )
            continue

        diagnosis_policy = check_diagnosis_policy(case, proposal)
        if not diagnosis_policy.allowed:
            corrections += 1
            if corrections > _MAX_CORRECTIONS:
                return _force_escalate(
                    case,
                    f"repeated diagnosis policy violations: {diagnosis_policy.reason}",
                )
            correction = diagnosis_policy.correction or (
                f"Your previous response violated diagnosis policy: {diagnosis_policy.reason}. "
                "Choose a valid next diagnostic step and try again."
            )
            continue

        business_policy = check_business_policy(case, proposal)
        if not business_policy.allowed:
            corrections += 1
            if corrections > _MAX_CORRECTIONS:
                return _force_escalate(
                    case,
                    f"repeated business policy violations: {business_policy.reason}",
                )
            correction = business_policy.correction or (
                f"Your previous response violated business policy: {business_policy.reason}. "
                "Choose an authorized next step."
            )
            continue

        _project_to_state(case, proposal)
        case.confidence = calibrate(proposal.confidence, case, retry_penalty)

        if proposal.action == AgentAction.CALL_TOOL:
            case.clarification_attempts = 0  # investigating is progress
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
            case.phase = Phase.INVESTIGATING
            if event_log and case.phase != prev_phase:
                record_phase_transition(event_log, case, prev_phase.value, case.phase.value)
            continue

        prev_phase = case.phase

        if proposal.action == AgentAction.RESOLVE:
            case.resolution_attempts += 1

        if proposal.action == AgentAction.ESCALATE:
            _build_escalation_context(case, proposal.escalation_reason, proposal.confidence)
            case.handoff_completed = True
            # a completed handoff is terminal from any phase: go to ESCALATING so the
            # transition rules close the case (T14), not back to investigating
            case.phase = Phase.ESCALATING
            if event_log:
                record_escalation(event_log, case, proposal.escalation_reason or "")

        _apply_transition(case, evaluate_transition(case))
        if event_log and case.phase != prev_phase:
            record_phase_transition(event_log, case, prev_phase.value, case.phase.value)

        if case.phase == Phase.ESCALATING and not case.handoff_completed:
            return _complete_runtime_handoff(
                case,
                "Investigation budget was exhausted without a safe self-service resolution",
                event_log,
            )

        # Bound the pre-investigation clarifying loop: if we keep asking the user
        # for a usable problem description and get nowhere, stop re-asking forever.
        if (
            proposal.action == AgentAction.ASK_USER
            and _stuck_clarifying(case)
            and not has_usable_issue_description(case)
        ):
            case.clarification_attempts += 1
            if case.clarification_attempts > _MAX_CLARIFICATION_ATTEMPTS:
                # No usable issue was ever described — there is nothing to diagnose
                # or hand off, so soft-close rather than escalate to a specialist.
                return _soft_close(case)
        else:
            case.clarification_attempts = 0

        response = _format_response(proposal, case.confidence)
        case.conversation.append({"role": "assistant", "content": response})
        return response

    return _force_escalate(case, "maximum investigation steps reached without resolution")


def _stuck_clarifying(case: CaseState) -> bool:
    """True while we are still trying to get a usable problem statement: pre-
    investigation (no tool has run) and not yet past the clarifying phases."""
    return case.phase in (Phase.INTAKE, Phase.CLARIFYING) and case.tool_calls_total == 0


def _ask_for_issue_description(
    case: CaseState,
    event_log: InMemoryEventLog | None,
) -> str:
    previous_phase = case.phase
    case.phase = Phase.CLARIFYING
    case.missing_info_source = MissingInfoSource.USER
    case.missing_info = ["issue description"]
    case.clarification_attempts += 1
    if event_log:
        record_phase_transition(event_log, case, previous_phase.value, case.phase.value)

    message = (
        "What IT issue are you running into? "
        "Please include the app or service, what you see, and when it started."
    )
    case.conversation.append({"role": "assistant", "content": message})
    return message


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


def _complete_runtime_handoff(
    case: CaseState,
    reason: str,
    event_log: InMemoryEventLog | None,
) -> str:
    _build_escalation_context(case, reason, case.confidence)
    case.handoff_completed = True
    previous_phase = case.phase
    _apply_transition(case, evaluate_transition(case))
    if event_log:
        record_escalation(event_log, case, reason)
        if case.phase != previous_phase:
            record_phase_transition(event_log, case, previous_phase.value, case.phase.value)

    msg = (
        "I wasn't able to fully resolve this issue. "
        "I'm connecting you with an IT specialist who will have all the context — "
        "you won't need to repeat yourself."
    )
    case.conversation.append({"role": "assistant", "content": msg})
    return msg


def _soft_close(case: CaseState) -> str:
    """Close a case that never produced a usable issue description.

    Distinct from escalation: there is no problem to diagnose and nothing to hand
    off, so we do not build an escalation_context or mark a handoff — we just
    close and invite the user to come back with details.
    """
    case.phase = Phase.CLOSED
    msg = (
        "I don't have enough information to diagnose an IT issue yet, so I'll close "
        "this for now. When you're ready, start a new request and include the affected "
        "app or service, what you're seeing (any error message), and when it started."
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


def _derive_missing_info_source(action: AgentAction) -> MissingInfoSource:
    """Infer where the next missing information comes from, from the action alone.

    The action already encodes who is being asked next, so the runtime derives the
    source instead of trusting an LLM-reported field: asking the user implies the
    user holds the missing info; calling a tool implies a tool does.
    """
    if action == AgentAction.ASK_USER:
        return MissingInfoSource.USER
    if action == AgentAction.CALL_TOOL:
        return MissingInfoSource.TOOL
    return MissingInfoSource.NONE


def _project_to_state(case: CaseState, proposal: AgentProposal) -> None:
    case.missing_info_source = _derive_missing_info_source(proposal.action)
    case.missing_info = list(proposal.missing_info)
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


def _format_response(proposal: AgentProposal, confidence: float) -> str:
    if proposal.action == AgentAction.ESCALATE:
        handoff = (
            "I'm connecting you with an IT specialist who will have all the context — "
            "you won't need to repeat yourself."
        )
        reason = (proposal.escalation_reason or "").strip()
        if reason:
            # tell the employee why, so the handoff isn't a black box
            return f"{reason} {handoff}"
        return f"I wasn't able to fully resolve this issue. {handoff}"

    if proposal.action == AgentAction.RESOLVE:
        message = proposal.message or ""
        # use the runtime's calibrated confidence so the wording the employee
        # sees matches the confidence the runtime actually acted on
        if confidence >= CONFIDENCE_HIGH:
            return f"I found a likely fix for your issue: {message}"
        return f"I'm not fully certain, but this is a safe first step to try: {message}"

    return proposal.message or ""
