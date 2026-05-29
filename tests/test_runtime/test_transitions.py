import pytest
from state.case_state import BudgetMode, CaseState, MissingInfoSource, Phase
from runtime.transitions import evaluate_transition


# ── helpers ──────────────────────────────────────────────────────────────────

def intake(**kwargs) -> CaseState:
    return CaseState(phase=Phase.INTAKE, **kwargs)

def clarifying(**kwargs) -> CaseState:
    return CaseState(phase=Phase.CLARIFYING, **kwargs)

def investigating(**kwargs) -> CaseState:
    return CaseState(phase=Phase.INVESTIGATING, **kwargs)

def resolving(**kwargs) -> CaseState:
    return CaseState(phase=Phase.RESOLVING, **kwargs)

def escalating(**kwargs) -> CaseState:
    return CaseState(phase=Phase.ESCALATING, **kwargs)


# ── T1: intake → clarifying ───────────────────────────────────────────────────

def test_t1_intake_to_clarifying():
    case = intake(
        missing_info_source=MissingInfoSource.USER,
        missing_info=["What OS are you using?"],
    )
    assert evaluate_transition(case) == Phase.CLARIFYING


# ── T2: intake → investigating ────────────────────────────────────────────────

def test_t2_intake_to_investigating_no_missing_info():
    case = intake(missing_info_source=MissingInfoSource.NONE)
    assert evaluate_transition(case) == Phase.INVESTIGATING

def test_t2_intake_to_investigating_tool_source():
    case = intake(missing_info_source=MissingInfoSource.TOOL)
    assert evaluate_transition(case) == Phase.INVESTIGATING


# ── T3: clarifying → investigating ───────────────────────────────────────────

def test_t3_clarifying_to_investigating_when_info_provided():
    case = clarifying(
        missing_info=[],
        missing_info_source=MissingInfoSource.NONE,
        budget_mode=BudgetMode.RETRY,  # budget_mode must be preserved
    )
    result = evaluate_transition(case)
    assert result == Phase.INVESTIGATING
    assert case.budget_mode == BudgetMode.RETRY  # not reset by transition eval


# ── T4: investigating → resolving (confidence ≥ 0.8) ─────────────────────────

@pytest.mark.parametrize("confidence", [0.8, 0.9, 1.0])
def test_t4_investigating_to_resolving(confidence):
    case = investigating(confidence=confidence)
    assert evaluate_transition(case) == Phase.RESOLVING


# ── T5: investigating → escalating (confidence < 0.5) ────────────────────────

@pytest.mark.parametrize("confidence", [0.0, 0.3, 0.49])
def test_t5_investigating_to_escalating(confidence):
    case = investigating(confidence=confidence)
    assert evaluate_transition(case) == Phase.ESCALATING


# ── T6: investigating → investigating (medium, tool source, budget remains) ───

def test_t6_investigating_stays_when_tool_source_and_budget_remains():
    case = investigating(
        confidence=0.6,
        missing_info_source=MissingInfoSource.TOOL,
        tool_calls_current_investigation=2,  # main budget = 5, still has room
    )
    assert evaluate_transition(case) == Phase.INVESTIGATING


# ── T7: investigating → clarifying ───────────────────────────────────────────

def test_t7_to_clarifying_when_user_source():
    case = investigating(
        confidence=0.6,
        missing_info_source=MissingInfoSource.USER,
        tool_calls_current_investigation=2,
    )
    assert evaluate_transition(case) == Phase.CLARIFYING

def test_t7_to_clarifying_when_budget_exhausted_and_user_source():
    case = investigating(
        confidence=0.6,
        missing_info_source=MissingInfoSource.USER,
        tool_calls_current_investigation=5,  # main budget exhausted
        has_safe_low_risk_guidance=False,
    )
    assert evaluate_transition(case) == Phase.CLARIFYING


# ── T8: investigating → escalating (budget exhausted, no safe guidance) ───────

def test_t8_to_escalating_when_budget_exhausted_no_guidance():
    case = investigating(
        confidence=0.6,
        missing_info_source=MissingInfoSource.NONE,
        tool_calls_current_investigation=5,
        has_safe_low_risk_guidance=False,
    )
    assert evaluate_transition(case) == Phase.ESCALATING


# ── T9: investigating → resolving (budget exhausted, safe guidance exists) ────

def test_t9_to_resolving_when_budget_exhausted_with_guidance():
    case = investigating(
        confidence=0.6,
        tool_calls_current_investigation=5,
        has_safe_low_risk_guidance=True,
    )
    assert evaluate_transition(case) == Phase.RESOLVING


# ── T10: resolving → closed (user confirms) ───────────────────────────────────

def test_t10_resolving_to_closed_when_confirmed():
    case = resolving(user_confirmed_resolution=True)
    assert evaluate_transition(case) == Phase.CLOSED


# ── T11: resolving → investigating retry (user not resolved, attempts < 2) ────

def test_t11_resolving_to_investigating_retry():
    case = resolving(
        user_confirmed_resolution=False,
        resolution_attempts=1,
    )
    assert evaluate_transition(case) == Phase.INVESTIGATING


# ── T12: resolving → clarifying exception ────────────────────────────────────

def test_t12_resolving_to_clarifying_exception():
    case = resolving(
        user_confirmed_resolution=False,
        resolution_attempts=2,
        new_critical_fact_added=True,
        exception_used=False,
    )
    assert evaluate_transition(case) == Phase.CLARIFYING


# ── T13: resolving → escalating ──────────────────────────────────────────────

def test_t13_to_escalating_when_max_attempts_no_new_fact():
    case = resolving(
        user_confirmed_resolution=False,
        resolution_attempts=2,
        new_critical_fact_added=False,
    )
    assert evaluate_transition(case) == Phase.ESCALATING

def test_t13_to_escalating_when_exception_already_used():
    case = resolving(
        user_confirmed_resolution=False,
        resolution_attempts=2,
        new_critical_fact_added=True,
        exception_used=True,
    )
    assert evaluate_transition(case) == Phase.ESCALATING


# ── T14: escalating → closed (handoff completed) ─────────────────────────────

def test_t14_escalating_to_closed_when_handoff_done():
    case = escalating(handoff_completed=True)
    assert evaluate_transition(case) == Phase.CLOSED

def test_t14_escalating_stays_when_handoff_not_done():
    case = escalating(handoff_completed=False)
    assert evaluate_transition(case) == Phase.ESCALATING


# ── closed is terminal ────────────────────────────────────────────────────────

def test_closed_stays_closed():
    case = CaseState(phase=Phase.CLOSED)
    assert evaluate_transition(case) == Phase.CLOSED
