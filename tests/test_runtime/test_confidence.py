from datetime import datetime, timezone

from runtime.confidence import compute_confidence
from state.case_state import CaseState, ToolTrace


def _trace(tool_name: str, success: bool = True) -> ToolTrace:
    return ToolTrace(
        tool_name=tool_name,
        inputs={},
        output={},
        success=success,
        timestamp=datetime.now(timezone.utc),
    )


def test_no_tools_is_zero():
    assert compute_confidence(CaseState(), retry_penalty=0.15) == 0.0


def test_one_successful_source():
    case = CaseState(tool_traces=[_trace("kb_search")])
    assert compute_confidence(case, retry_penalty=0.15) == 0.35


def test_two_distinct_sources_hit_the_cap():
    case = CaseState(tool_traces=[_trace("kb_search"), _trace("status_api")])
    assert compute_confidence(case, retry_penalty=0.15) == 0.7


def test_capped_at_0_7_with_more_sources():
    case = CaseState(tool_traces=[_trace("kb_search"), _trace("status_api"), _trace("user_directory")])
    assert compute_confidence(case, retry_penalty=0.15) == 0.7


def test_same_tool_counts_once():
    case = CaseState(tool_traces=[_trace("kb_search"), _trace("kb_search")])
    assert compute_confidence(case, retry_penalty=0.15) == 0.35


def test_failed_traces_do_not_count():
    case = CaseState(tool_traces=[_trace("kb_search", success=False), _trace("status_api")])
    assert compute_confidence(case, retry_penalty=0.15) == 0.35


def test_retry_penalty_subtracts_per_attempt():
    case = CaseState(tool_traces=[_trace("kb_search"), _trace("status_api")], resolution_attempts=1)
    assert compute_confidence(case, retry_penalty=0.15) == round(0.7 - 0.15, 3)


def test_never_below_zero():
    case = CaseState(tool_traces=[_trace("kb_search")], resolution_attempts=10)
    assert compute_confidence(case, retry_penalty=0.15) == 0.0
