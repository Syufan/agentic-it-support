import json
from datetime import datetime, timezone
from pathlib import Path

from agentic_it_support.observability.event_tracing import (
    InMemoryEventLog,
    record_handoff_written,
)
from agentic_it_support.state.case_state import CaseState, Phase

_HANDOFF_TAIL = (
    "I'm connecting you with an IT specialist who will have all the context, "
    "you won't need to repeat yourself."
)

_REASON = "This needs a closer review before we can continue."

def finalize_handoff(
    case: CaseState,
    reason: str,
    *,
    output_dir: Path,
    event_log: InMemoryEventLog | None = None,
) -> str:
    """Finaliza human handoff for any runtime escalation decision."""
    case.escalation_context = _build_handoff_context(
        case,
        user_facing_reason=_REASON,
        internal_reason=reason
    )
    path = _write_handoff_json(case.escalation_context, output_dir)
    case.phase = Phase.ESCALATING
    case.handoff_completed = True
    record_handoff_written(event_log, case, str(path))

    return f"{_REASON} {_HANDOFF_TAIL}"

def _build_handoff_context(case: CaseState, *, user_facing_reason: str, internal_reason: str) -> dict:
    return{
        "case_id": case.case_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
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


def _write_handoff_json(context: dict, output_dir: Path) -> Path:
    """Persist the handoff payload locally for demo inspection."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{context['case_id']}.json"
    path.write_text(
        json.dumps(context, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    return path
