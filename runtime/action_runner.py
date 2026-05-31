from dataclasses import dataclass
from datetime import datetime, timezone

from agent.proposals import AgentAction, AgentProposal
from observability.event_tracing import (
    InMemoryEventLog,
    record_escalation,
    record_phase_transition,
    record_tool_call,
)
from runtime import limits
from runtime.constants import CONFIDENCE_HIGH
from runtime.confidence import compute_confidence
from runtime.diagnosis_policy import has_usable_issue_description
from runtime.transitions import TransitionResult, evaluate_transition
from state.case_state import CaseState, Phase, ToolTrace
from tools.base import BaseTool, ToolResult


@dataclass
class ActionOutcome:
    continue_loop: bool
    message: str | None = None


def run_accepted_action(
    case: CaseState,
    proposal: AgentProposal,
    tool_registry: dict[str, BaseTool],
    retry_penalty: float,
    event_log: InMemoryEventLog | None = None,
) -> ActionOutcome:
    _project_to_state(case, proposal)
    case.confidence = compute_confidence(case, retry_penalty)

    if proposal.action == AgentAction.CALL_TOOL:
        return _run_tool_action(case, proposal, tool_registry, event_log)

    return _run_terminal_action(case, proposal, event_log)


def ask_for_issue_description(
    case: CaseState,
    event_log: InMemoryEventLog | None,
) -> str:
    previous_phase = case.phase
    case.phase = Phase.CLARIFYING
    case.missing_info = ["issue description"]
    case.clarification_attempts += 1
    if event_log:
        record_phase_transition(
            event_log,
            case.case_id,
            case.confidence,
            previous_phase.value,
            case.phase.value,
        )

    message = (
        "What IT issue are you running into? "
        "Please include the app or service, what you see, and when it started."
    )
    case.conversation.append({"role": "assistant", "content": message})
    return message


def force_escalate(case: CaseState, reason: str) -> str:
    _build_escalation_context(case, reason, case.confidence)
    case.phase = Phase.ESCALATING
    case.handoff_completed = True
    _apply_transition(case, evaluate_transition(case, AgentAction.ESCALATE))
    msg = (
        "I wasn't able to fully resolve this issue. "
        "I'm connecting you with an IT specialist who will have all the context — "
        "you won't need to repeat yourself."
    )
    case.conversation.append({"role": "assistant", "content": msg})
    return msg


def _run_tool_action(
    case: CaseState,
    proposal: AgentProposal,
    tool_registry: dict[str, BaseTool],
    event_log: InMemoryEventLog | None,
) -> ActionOutcome:
    case.clarification_attempts = 0
    _execute_tool(case, proposal, tool_registry)
    if event_log:
        last_trace = case.tool_traces[-1]
        record_tool_call(
            event_log,
            case.case_id,
            case.phase.value,
            case.confidence,
            tool_name=last_trace.tool_name,
            success=last_trace.success,
            inputs=last_trace.inputs,
        )

    prev_phase = case.phase
    _apply_transition(case, evaluate_transition(case, AgentAction.CALL_TOOL))
    _record_phase_if_changed(case, prev_phase, event_log)
    return ActionOutcome(continue_loop=True)


def _run_terminal_action(
    case: CaseState,
    proposal: AgentProposal,
    event_log: InMemoryEventLog | None,
) -> ActionOutcome:
    prev_phase = case.phase

    if proposal.action == AgentAction.RESOLVE:
        case.resolution_attempts += 1

    if proposal.action == AgentAction.ESCALATE:
        _build_escalation_context(case, proposal.escalation_reason, case.confidence)
        case.handoff_completed = True
        case.phase = Phase.ESCALATING
        if event_log:
            record_escalation(
                event_log,
                case.case_id,
                case.phase.value,
                case.confidence,
                proposal.escalation_reason or "",
            )

    _apply_transition(case, evaluate_transition(case, proposal.action))
    _record_phase_if_changed(case, prev_phase, event_log)

    if case.phase == Phase.ESCALATING and not case.handoff_completed:
        return ActionOutcome(
            continue_loop=False,
            message=_complete_runtime_handoff(
                case,
                "Investigation tool-call limit was reached without a safe self-service resolution",
                event_log,
            ),
        )

    if _should_soft_close(case, proposal):
        return ActionOutcome(continue_loop=False, message=_soft_close(case))

    response = _format_response(proposal, case.confidence)
    case.conversation.append({"role": "assistant", "content": response})
    return ActionOutcome(continue_loop=False, message=response)


def _should_soft_close(case: CaseState, proposal: AgentProposal) -> bool:
    if not (
        proposal.action == AgentAction.ASK_USER
        and _stuck_clarifying(case)
        and not has_usable_issue_description(case)
    ):
        case.clarification_attempts = 0
        return False

    case.clarification_attempts += 1
    return limits.clarification_limit_reached(case)


def _stuck_clarifying(case: CaseState) -> bool:
    return case.phase in (Phase.INTAKE, Phase.CLARIFYING) and case.tool_calls_total == 0


def _complete_runtime_handoff(
    case: CaseState,
    reason: str,
    event_log: InMemoryEventLog | None,
) -> str:
    _build_escalation_context(case, reason, case.confidence)
    case.handoff_completed = True
    previous_phase = case.phase
    _apply_transition(case, evaluate_transition(case, AgentAction.ESCALATE))
    if event_log:
        record_escalation(event_log, case.case_id, case.phase.value, case.confidence, reason)
        _record_phase_if_changed(case, previous_phase, event_log)

    msg = (
        "I wasn't able to fully resolve this issue. "
        "I'm connecting you with an IT specialist who will have all the context — "
        "you won't need to repeat yourself."
    )
    case.conversation.append({"role": "assistant", "content": msg})
    return msg


def _soft_close(case: CaseState) -> str:
    case.phase = Phase.CLOSED
    msg = (
        "I don't have enough information to diagnose an IT issue yet, so I'll close "
        "this for now. When you're ready, start a new request and include the affected "
        "app or service, what you're seeing (any error message), and when it started."
    )
    case.conversation.append({"role": "assistant", "content": msg})
    return msg


def _project_to_state(case: CaseState, proposal: AgentProposal) -> None:
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
        timestamp=datetime.now(timezone.utc),
    ))

    if result.success:
        case.facts[f"{tool_name}_result"] = result.data
    else:
        case.facts[f"{tool_name}_error"] = result.error or "unknown error"

    case.tool_calls_this_turn += 1
    case.tool_calls_total += 1


def _apply_transition(case: CaseState, result: TransitionResult) -> None:
    case.phase = result.next_phase


def _record_phase_if_changed(
    case: CaseState,
    previous_phase: Phase,
    event_log: InMemoryEventLog | None,
) -> None:
    if event_log and case.phase != previous_phase:
        record_phase_transition(
            event_log,
            case.case_id,
            case.confidence,
            previous_phase.value,
            case.phase.value,
        )


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
            return f"{reason} {handoff}"
        return f"I wasn't able to fully resolve this issue. {handoff}"

    if proposal.action == AgentAction.RESOLVE:
        message = proposal.message or ""
        if confidence >= CONFIDENCE_HIGH:
            return f"I found a likely fix for your issue: {message}"
        return f"I'm not fully certain, but this is a safe first step to try: {message}"

    return proposal.message or ""
