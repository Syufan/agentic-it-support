from runtime.calibration import calibrate
from state.case_state import CaseState


def test_passes_through_when_grounded_and_no_prior_attempts():
    case = CaseState(tool_calls_total=2)
    assert calibrate(0.9, case) == 0.9


def test_caps_confidence_when_no_tools_used():
    # no evidence yet: cannot reach the high-confidence resolve threshold (0.8)
    case = CaseState(tool_calls_total=0)
    assert calibrate(0.95, case) <= 0.5


def test_no_evidence_cap_does_not_inflate_low_confidence():
    case = CaseState(tool_calls_total=0)
    assert calibrate(0.2, case) == 0.2


def test_penalizes_each_prior_resolution_attempt():
    grounded = CaseState(tool_calls_total=2)
    fresh = calibrate(0.9, grounded)

    retried = CaseState(tool_calls_total=2, resolution_attempts=2)
    assert calibrate(0.9, retried) < fresh


def test_never_returns_below_zero():
    case = CaseState(tool_calls_total=2, resolution_attempts=10)
    assert calibrate(0.5, case) >= 0.0


def test_never_exceeds_one():
    case = CaseState(tool_calls_total=2)
    assert calibrate(1.0, case) <= 1.0
