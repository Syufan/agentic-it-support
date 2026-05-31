"""Runtime-owned, evidence-based confidence.

Replaces the old LLM-self-reported confidence: the runtime derives a grounded
estimate from what it can actually verify — the distinct tools that returned a
successful result — and discounts it for each resolution that did not stick.

Evidence alone never auto-resolves: RESOLVING is entered by the RESOLVE *action*,
not by crossing a confidence threshold. The runtime gates that action on this value
(CONFIDENCE_RESOLVE_MIN), so confidence authorizes a fix rather than triggering one.
"""

from state.case_state import CaseState

#: evidence ceiling — gathered evidence alone tops out here
_EVIDENCE_CAP = 0.7
#: confidence contributed per distinct successful tool source
_PER_SOURCE = 0.35


def compute_confidence(case: CaseState, retry_penalty: float) -> float:
    distinct_successful_sources = len(
        {trace.tool_name for trace in case.tool_traces if trace.success}
    )
    confidence = min(_EVIDENCE_CAP, _PER_SOURCE * distinct_successful_sources)
    confidence -= retry_penalty * case.resolution_attempts
    return round(max(0.0, confidence), 3)
