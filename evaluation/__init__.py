from dataclasses import dataclass

from runtime.constants import MAIN_TOOL_BUDGET
from state.case_state import CaseState, Phase


@dataclass
class EvaluationResult:
    resolved: bool
    escalated: bool
    tool_calls_total: int
    resolution_attempts: int
    final_confidence: float
    final_phase: str
    tool_efficiency: float  # 0.0 (spent all budget) → 1.0 (no tools used)


def evaluate(case: CaseState) -> EvaluationResult:
    escalated = bool(case.escalation_context)
    resolved = case.phase == Phase.CLOSED and not escalated
    efficiency = max(0.0, 1.0 - case.tool_calls_total / MAIN_TOOL_BUDGET)

    return EvaluationResult(
        resolved=resolved,
        escalated=escalated,
        tool_calls_total=case.tool_calls_total,
        resolution_attempts=case.resolution_attempts,
        final_confidence=case.confidence,
        final_phase=case.phase.value,
        tool_efficiency=round(efficiency, 3),
    )
