import pytest

from agent.proposals import AgentAction
from runtime.transitions import TransitionResult, evaluate_transition
from state.case_state import BudgetMode, CaseState, Phase

ASK = AgentAction.ASK_USER
TOOL = AgentAction.CALL_TOOL
RESOLVE = AgentAction.RESOLVE
ESCALATE = AgentAction.ESCALATE


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


# ── T1 / T2: from intake (action-driven) ─────────────────────────────────────

def test_t1_intake_to_clarifying_on_ask_user():
    assert evaluate_transition(intake(), ASK).next_phase == Phase.CLARIFYING

def test_t2_intake_to_investigating_on_tool():
    assert evaluate_transition(intake(), TOOL).next_phase == Phase.INVESTIGATING


# ── T3: from clarifying ──────────────────────────────────────────────────────

def test_t3_clarifying_to_investigating_on_tool():
    case = clarifying(budget_mode=BudgetMode.RETRY)
    result = evaluate_transition(case, TOOL)
    assert result.next_phase == Phase.INVESTIGATING
    assert case.budget_mode == BudgetMode.RETRY  # not reset by transition eval

def test_clarifying_stays_on_ask_user():
    assert evaluate_transition(clarifying(), ASK).next_phase == Phase.CLARIFYING


# ── T4: investigating → resolving (confidence ≥ 0.8) ─────────────────────────

@pytest.mark.parametrize("confidence", [0.8, 0.9, 1.0])
def test_t4_investigating_to_resolving_high_confidence(confidence):
    assert evaluate_transition(investigating(confidence=confidence), RESOLVE).next_phase == Phase.RESOLVING


# ── T6: a tool call short-circuits to investigating, skipping the guards ──────

def test_t6_tool_call_stays_investigating():
    case = investigating(confidence=0.6, tool_calls_current_investigation=2)
    assert evaluate_transition(case, TOOL).next_phase == Phase.INVESTIGATING

def test_tool_call_short_circuits_before_confidence_guard():
    # even with high confidence, a tool call keeps investigating (no auto-resolve)
    assert evaluate_transition(investigating(confidence=0.9), TOOL).next_phase == Phase.INVESTIGATING


# ── Problem 1 fix: low-confidence resolve with budget left stays investigating ─

@pytest.mark.parametrize("confidence", [0.0, 0.3, 0.49])
def test_low_confidence_resolve_stays_investigating(confidence):
    case = investigating(confidence=confidence, tool_calls_current_investigation=1)
    assert evaluate_transition(case, RESOLVE).next_phase == Phase.INVESTIGATING


# ── T7: investigating → clarifying (only on ask_user) ────────────────────────

def test_t7_ask_user_to_clarifying_budget_remains():
    case = investigating(confidence=0.6, tool_calls_current_investigation=2)
    assert evaluate_transition(case, ASK).next_phase == Phase.CLARIFYING

def test_t7_ask_user_to_clarifying_budget_exhausted():
    case = investigating(confidence=0.6, tool_calls_current_investigation=5,
                         has_safe_low_risk_guidance=False)
    assert evaluate_transition(case, ASK).next_phase == Phase.CLARIFYING


# ── T8: investigating → escalating (budget exhausted, non-ask, no guidance) ───

def test_t8_budget_exhausted_resolve_escalates():
    case = investigating(confidence=0.6, tool_calls_current_investigation=5,
                         has_safe_low_risk_guidance=False)
    assert evaluate_transition(case, RESOLVE).next_phase == Phase.ESCALATING


# ── T9: investigating → resolving (budget exhausted, safe guidance) ──────────

def test_t9_budget_exhausted_with_guidance_resolves():
    case = investigating(confidence=0.6, tool_calls_current_investigation=5,
                         has_safe_low_risk_guidance=True)
    assert evaluate_transition(case, RESOLVE).next_phase == Phase.RESOLVING


# ── T10: resolving → closed (user confirms) ──────────────────────────────────

def test_t10_resolving_to_closed_when_confirmed():
    assert evaluate_transition(resolving(user_confirmed_resolution=True), RESOLVE).next_phase == Phase.CLOSED


# ── T11: resolving → investigating retry ─────────────────────────────────────

def test_t11_resolving_to_investigating_retry():
    result = evaluate_transition(resolving(user_confirmed_resolution=False, resolution_attempts=1), RESOLVE)
    assert result.next_phase == Phase.INVESTIGATING

def test_t11_switches_budget_to_retry():
    result = evaluate_transition(resolving(user_confirmed_resolution=False, resolution_attempts=1), RESOLVE)
    assert result.budget_mode == BudgetMode.RETRY

def test_t11_resets_tool_counter():
    result = evaluate_transition(resolving(user_confirmed_resolution=False, resolution_attempts=1), RESOLVE)
    assert result.reset_tool_counter is True

def test_t11_does_not_set_exception_used():
    result = evaluate_transition(resolving(user_confirmed_resolution=False, resolution_attempts=1), RESOLVE)
    assert result.set_exception_used is False


# ── T12: resolving → clarifying exception ────────────────────────────────────

def _t12_case() -> CaseState:
    return resolving(user_confirmed_resolution=False, resolution_attempts=2,
                     new_critical_fact_added=True, exception_used=False)

def test_t12_resolving_to_clarifying_exception():
    assert evaluate_transition(_t12_case(), RESOLVE).next_phase == Phase.CLARIFYING

def test_t12_switches_budget_to_exception():
    assert evaluate_transition(_t12_case(), RESOLVE).budget_mode == BudgetMode.EXCEPTION

def test_t12_resets_tool_counter():
    assert evaluate_transition(_t12_case(), RESOLVE).reset_tool_counter is True

def test_t12_sets_exception_used():
    assert evaluate_transition(_t12_case(), RESOLVE).set_exception_used is True


# ── T13: resolving → escalating ──────────────────────────────────────────────

def test_t13_to_escalating_when_max_attempts_no_new_fact():
    case = resolving(user_confirmed_resolution=False, resolution_attempts=2,
                     new_critical_fact_added=False)
    assert evaluate_transition(case, RESOLVE).next_phase == Phase.ESCALATING

def test_t13_to_escalating_when_exception_already_used():
    case = resolving(user_confirmed_resolution=False, resolution_attempts=2,
                     new_critical_fact_added=True, exception_used=True)
    assert evaluate_transition(case, RESOLVE).next_phase == Phase.ESCALATING


# ── T14: escalating → closed (handoff completed) ─────────────────────────────

def test_t14_escalating_to_closed_when_handoff_done():
    assert evaluate_transition(escalating(handoff_completed=True), ESCALATE).next_phase == Phase.CLOSED

def test_t14_escalating_stays_when_handoff_not_done():
    assert evaluate_transition(escalating(handoff_completed=False), ESCALATE).next_phase == Phase.ESCALATING


# ── closed is terminal ────────────────────────────────────────────────────────

def test_closed_stays_closed():
    assert evaluate_transition(CaseState(phase=Phase.CLOSED), RESOLVE).next_phase == Phase.CLOSED


# ── non-T11/T12 transitions have no budget effects ───────────────────────────

@pytest.mark.parametrize("result_fn", [
    lambda: evaluate_transition(intake(), TOOL),
    lambda: evaluate_transition(investigating(confidence=0.9), RESOLVE),
    lambda: evaluate_transition(investigating(confidence=0.3, tool_calls_current_investigation=1), TOOL),
    lambda: evaluate_transition(escalating(handoff_completed=True), ESCALATE),
])
def test_other_transitions_have_no_budget_effects(result_fn):
    result = result_fn()
    assert result.budget_mode is None
    assert result.reset_tool_counter is False
    assert result.set_exception_used is False
