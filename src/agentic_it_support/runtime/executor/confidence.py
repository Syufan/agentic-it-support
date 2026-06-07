from agentic_it_support.config.settings import ConfidenceSettings
from agentic_it_support.state.case_state import CaseState

_CONFIDENCE_DECIMAL_PLACES = 3

def compute_confidence(case: CaseState, confidence_settings: ConfidenceSettings) -> float:
    # Count distinct tools that returned successful evidence
    distinct_successful_sources = len(
        {trace.tool_name for trace in case.tool_traces if trace.success}
    )

    # Evidence raises confidence; failed resolution attempts reduce it
    confidence = min(confidence_settings.high_threshold, confidence_settings.resolve_threshold * distinct_successful_sources)
    confidence -= confidence_settings.retry_penalty * case.resolution_attempts

    return round(max(0.0, confidence), _CONFIDENCE_DECIMAL_PLACES)