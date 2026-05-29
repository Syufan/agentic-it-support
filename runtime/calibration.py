"""Calibrate the LLM's self-reported confidence against actual evidence.

Raw LLM confidence is unreliable - the model will happily claim 0.9 with no
investigation behind it. The runtime's transition thresholds should react to a
grounded estimate instead, so we discount confidence that isn't backed by tool
use, and discount it further each time a previous resolution did not stick.

These coefficients are deliberately simple and hand-set; with a labelled
evaluation set they could be fit (e.g. isotonic/Platt) from observed
resolution-correctness data.
"""

from config import CONFIDENCE_LOW, CONFIDENCE_RETRY_PENALTY
from state.case_state import CaseState

#: with no tool evidence yet, hold confidence at the borderline threshold - high
#: enough not to force an escalation, but never high enough to auto-resolve.
_NO_EVIDENCE_CAP = CONFIDENCE_LOW


def calibrate(raw: float, case: CaseState) -> float:
    calibrated = raw

    if case.tool_calls_total == 0:
        calibrated = min(calibrated, _NO_EVIDENCE_CAP)

    calibrated -= CONFIDENCE_RETRY_PENALTY * case.resolution_attempts

    return round(max(0.0, min(1.0, calibrated)), 3)
