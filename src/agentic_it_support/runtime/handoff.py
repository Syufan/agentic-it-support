from agentic_it_support.state.case_state import CaseState, Phase

_HANDOFF_TAIL = (
    "I'm connecting you with an IT specialist who will have all the context, "
    "you won't need to repeat yourself."
)

_REASON = "This needs a closer review before we can continue."

def finalize_handoff(case: CaseState, reason: str) -> str:
    """Finaliza human handoff for any runtime escalation decision."""
    case.escalation_context = _build_handoff_context(
        case,
        user_facing_reason=_REASON,
        internal_reason=reason
    )
    case.phase = Phase.ESCALATING
    case.handoff_completed = True

    return f"{_REASON} {_HANDOFF_TAIL}"

def _build_handoff_context(case: CaseState, *, user_facing_reason: str, internal_reason: str) -> dict:
    return{
        "user_facing_reason": user_facing_reason,
        "internal_reason": internal_reason,
        "confidence": case.confidence,
        "conversation": list(case.conversation),
        "tool_traces": [
            {
                "tool": trace.tool_name,
                "success": trace.success,
                "input": trace.inputs,
                "output": trace.output
            }
            for trace in case.tool_traces
        ],
        "resolution_attempts": case.resolution_attempts
    }
