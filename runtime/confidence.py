"""Runtime-owned confidence derived from successful tool evidence."""

from runtime.constants import CONFIDENCE_HIGH, CONFIDENCE_RESOLVE_MIN
from state.case_state import CaseState


# Reuse runtime thresholds so confidence calculation and gates stay aligned.
_PER_SOURCE = CONFIDENCE_RESOLVE_MIN
_EVIDENCE_CAP = CONFIDENCE_HIGH


def compute_confidence(case: CaseState, retry_penalty: float) -> float:
    # Count distinct tools that returned successful evidence.
    distinct_successful_sources = len(
        {trace.tool_name for trace in case.tool_traces if trace.success}
    )

    # Evidence raises confidence; failed resolution attempts reduce it.
    confidence = min(_EVIDENCE_CAP, _PER_SOURCE * distinct_successful_sources)
    confidence -= retry_penalty * case.resolution_attempts

    return round(max(0.0, confidence), 3)