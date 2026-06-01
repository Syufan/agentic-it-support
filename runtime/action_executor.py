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

# Shared handoff wording for all escalation paths.
_HANDOFF_TAIL = (
    "I'm connecting you with an IT specialist who will have all the context — "
    "you won't need to repeat yourself."
)
_GENERIC_HANDOFF = f"I wasn't able to fully resolve this issue. {_HANDOFF_TAIL}"


# ── 1. Public API / outcome contract ──────────────────────────────────────────

@dataclass
class ActionOutcome:
    continue_loop: bool
    message: str | None = None


def run_accepted_action(
    case: CaseState,
    proposal: AgentProposal,
    tool_registry: dict[str, BaseTool],
    retry_penalty: float,
    event_log: InMemoryEventLog,
) -> ActionOutcome:
    # Project proposal fields into case state.
    _project_to_state(case, proposal)
    
    # Tool actions add new evidence before confidence is recomputed.
    if proposal.action == AgentAction.CALL_TOOL:
        return _run_tool_action(case, proposal, tool_registry, retry_penalty, event_log)
    
    # Terminal actions use the evidence already recorded on the case.
    case.confidence = compute_confidence(case, retry_penalty)
    return _run_terminal_action(case, proposal, event_log)


def ask_for_issue_description(
    case: CaseState,
    event_log: InMemoryEventLog,
) -> str:
    previous_phase = case.phase
    case.phase = Phase.CLARIFYING
    case.missing_info = ["issue description"]
    case.clarification_attempts += 1
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


def force_escalate(
    case: CaseState,
    reason: str,
    event_log: InMemoryEventLog,
) -> str:
    """Runtime-initiated handoff with a generic user-facing message."""
    previous_phase = case.phase
    _build_escalation_context(
        case,
        _human_safe_escalation_reason(reason),
        case.confidence,
        internal_runtime_reason=reason,
    )
    case.phase = Phase.ESCALATING
    case.handoff_completed = True
    record_escalation(event_log, case.case_id, case.phase.value, case.confidence, reason)
    _apply_transition(case, evaluate_transition(case, AgentAction.ESCALATE))
    _record_phase_if_changed(case, previous_phase, event_log)
    case.conversation.append({"role": "assistant", "content": _GENERIC_HANDOFF})
    return _GENERIC_HANDOFF


# ── 2. Action handlers ─────────────────────────────────────────────────────────

def _run_tool_action(
    case: CaseState,
    proposal: AgentProposal,
    tool_registry: dict[str, BaseTool],
    retry_penalty: float,
    event_log: InMemoryEventLog,
) -> ActionOutcome:
    # Tool use means the case made progress.
    case.clarification_attempts = 0

    # Execute tool and recompute confidence from the new evidence.
    _execute_tool(case, proposal, tool_registry)
    case.confidence = compute_confidence(case, retry_penalty)
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
    event_log: InMemoryEventLog,
) -> ActionOutcome:
    prev_phase = case.phase

    # Track attempted fixes.
    if proposal.action == AgentAction.RESOLVE:
        case.resolution_attempts += 1

    # Agent-requested escalation builds handoff context before transitioning.
    if proposal.action == AgentAction.ESCALATE:
        _build_escalation_context(case, proposal.escalation_reason, case.confidence)
        case.handoff_completed = True
        case.phase = Phase.ESCALATING
        record_escalation(
            event_log,
            case.case_id,
            case.phase.value,
            case.confidence,
            proposal.escalation_reason or "",
        )

    _apply_transition(case, evaluate_transition(case, proposal.action))
    _record_phase_if_changed(case, prev_phase, event_log)

    # Safety fallback if a transition reaches ESCALATING without a handoff.
    if case.phase == Phase.ESCALATING and not case.handoff_completed:
        return ActionOutcome(
            continue_loop=False,
            message=force_escalate(
                case,
                "Investigation tool-call limit was reached without a safe self-service resolution",
                event_log,
            ),
        )

    # Soft-close repeated vague clarification loops.
    _track_clarification_attempt(case, proposal)
    if _is_unproductive_clarification(case, proposal) and limits.clarification_limit_reached(case):
        return ActionOutcome(continue_loop=False, message=_soft_close(case))

    response = _format_response(proposal, case.confidence)
    case.conversation.append({"role": "assistant", "content": response})
    return ActionOutcome(continue_loop=False, message=response)


# ── 3. Terminal helpers ────────────────────────────────────────────────────────

def _track_clarification_attempt(case: CaseState, proposal: AgentProposal) -> None:
    """Track consecutive unproductive clarification attempts."""
    if _is_unproductive_clarification(case, proposal):
        case.clarification_attempts += 1
    else:
        case.clarification_attempts = 0


def _is_unproductive_clarification(case: CaseState, proposal: AgentProposal) -> bool:
    """Return true when clarification has not produced a usable issue."""
    return (
        proposal.action == AgentAction.ASK_USER
        and _stuck_clarifying(case)
        and not has_usable_issue_description(case)
    )


def _stuck_clarifying(case: CaseState) -> bool:
    # Still pre-tool and waiting for a usable issue.
    return case.phase in (Phase.INTAKE, Phase.CLARIFYING) and case.tool_calls_total == 0


def _soft_close(case: CaseState) -> str:
    # Close vague requests that did not produce enough issue detail.
    case.phase = Phase.CLOSED
    msg = (
        "I don't have enough information to diagnose an IT issue yet, so I'll close "
        "this for now. When you're ready, start a new request and include the affected "
        "app or service, what you're seeing (any error message), and when it started."
    )
    case.conversation.append({"role": "assistant", "content": msg})
    return msg


def _format_response(proposal: AgentProposal, confidence: float) -> str:
    # Format the final user-facing response.
    if proposal.action == AgentAction.ESCALATE:
        reason = (proposal.escalation_reason or "").strip()
        if reason:
            return f"{reason} {_HANDOFF_TAIL}"
        return _GENERIC_HANDOFF

    if proposal.action == AgentAction.RESOLVE:
        message = proposal.message or ""
        if confidence >= CONFIDENCE_HIGH:
            return f"I found a likely fix for your issue: {message}"
        return f"I'm not fully certain, but this is a safe first step to try: {message}"

    return proposal.message or ""


# ── 4. Tool execution helpers ──────────────────────────────────────────────────

def _execute_tool(
    case: CaseState,
    proposal: AgentProposal,
    tool_registry: dict[str, BaseTool],
) -> None:
    # Run the requested tool, or record a failed trace if unavailable.
    tool_name = proposal.tool_name or ""
    tool = tool_registry.get(tool_name)

    if tool is None:
        result = ToolResult(success=False, data={}, error=f"tool '{tool_name}' not available")
    else:
        result = tool.run(proposal.tool_input)

    # Store the tool trace and expose the result as case facts.
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

    # Update turn-level and case-level tool budgets.
    case.tool_calls_this_turn += 1
    case.tool_calls_total += 1


# ── 5. State / transition helpers ──────────────────────────────────────────────

def _project_to_state(case: CaseState, proposal: AgentProposal) -> None:
    # Copy proposal state fields onto the case.
    case.missing_info = list(proposal.missing_info)
    if proposal.user_confirmed_resolution is not None:
        case.user_confirmed_resolution = proposal.user_confirmed_resolution


def _apply_transition(case: CaseState, result: TransitionResult) -> None:
    # Apply the state-machine decision.
    case.phase = result.next_phase


def _record_phase_if_changed(
    case: CaseState,
    previous_phase: Phase,
    event_log: InMemoryEventLog,
) -> None:
    # Record phase transitions only when the phase actually changed.
    if case.phase != previous_phase:
        record_phase_transition(
            event_log,
            case.case_id,
            case.confidence,
            previous_phase.value,
            case.phase.value,
        )


# ── 6. Escalation helpers ──────────────────────────────────────────────────────

def _build_escalation_context(
    case: CaseState,
    reason: str | None,
    confidence: float,
    internal_runtime_reason: str | None = None,
) -> None:
    # Package case context for human handoff.
    issue_description = next(
        (m["content"] for m in case.conversation if m["role"] == "user"), ""
    )
    case.escalation_context = {
        "escalation_reason": reason,
        "internal_runtime_reason": internal_runtime_reason,
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


def _human_safe_escalation_reason(reason: str) -> str:
    # Hide internal runtime failure reasons from the user-facing handoff.
    internal_markers = (
        "llm provider error",
        "maximum llm calls",
        "maximum investigation steps",
        "repeated invalid",
        "repeated diagnosis policy violations",
        "repeated business policy violations",
    )
    if any(marker in reason.lower() for marker in internal_markers):
        return "The agent could not safely complete the investigation and needs human review."
    return reason
