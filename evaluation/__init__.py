from dataclasses import dataclass

from agentic_it_support.config.settings import RuntimeLimits
from agentic_it_support.state.case_state import CaseState, Phase

_MAX_TOOL_CALLS_PER_CASE = RuntimeLimits().max_tool_calls_per_case


@dataclass
class EvaluationResult:
    resolved: bool
    escalated: bool
    tool_calls_total: int
    resolution_attempts: int
    final_confidence: float
    final_phase: str
    tool_efficiency: float  # 0.0 (spent all tool calls) -> 1.0 (no tools used)


def evaluate(case: CaseState) -> EvaluationResult:
    escalated = bool(case.escalation_context)
    resolved = case.phase == Phase.CLOSED and not escalated
    efficiency = max(0.0, 1.0 - case.tool_calls_total / _MAX_TOOL_CALLS_PER_CASE)

    return EvaluationResult(
        resolved=resolved,
        escalated=escalated,
        tool_calls_total=case.tool_calls_total,
        resolution_attempts=case.resolution_attempts,
        final_confidence=case.confidence,
        final_phase=case.phase.value,
        tool_efficiency=round(efficiency, 3),
    )
