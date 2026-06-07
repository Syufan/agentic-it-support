import pytest

from agentic_it_support.agent.proposals import AgentAction
from agentic_it_support.runtime.transitions import TransitionResult, evaluate_transition
from agentic_it_support.state.case_state import CaseState, Phase

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
    case = clarifying()
    result = evaluate_transition(case, TOOL)
    assert result.next_phase == Phase.INVESTIGATING

def test_clarifying_stays_on_ask_user():
    assert evaluate_transition(clarifying(), ASK).next_phase == Phase.CLARIFYING


# ── T4: investigating → resolving (the RESOLVE action drives it) ─────────────
# Action-driven: confidence does NOT gate the transition (that's the diagnosis_policy
# resolve gate's job, upstream). The machine just maps RESOLVE → RESOLVING regardless
# of the confidence value.

@pytest.mark.parametrize("confidence", [0.0, 0.5, 0.9])
def test_t4_resolve_action_drives_resolving(confidence):
    assert evaluate_transition(investigating(confidence=confidence), RESOLVE).next_phase == Phase.RESOLVING


# ── T6: a tool call keeps investigating (synthesize the result next turn) ─────

def test_t6_tool_call_stays_investigating():
    case = investigating(confidence=0.6, tool_calls_total=2)
    assert evaluate_transition(case, TOOL).next_phase == Phase.INVESTIGATING


# ── T7: investigating → clarifying (only on ask_user) ────────────────────────

def test_t7_ask_user_to_clarifying():
    case = investigating(confidence=0.6, tool_calls_total=2)
    assert evaluate_transition(case, ASK).next_phase == Phase.CLARIFYING


# ── T10: resolving → closed (user confirms) ──────────────────────────────────

def test_t10_resolving_to_closed_when_confirmed():
    assert evaluate_transition(resolving(user_confirmed_resolution=True), RESOLVE).next_phase == Phase.CLOSED


# ── T11: resolving → escalating when the user reports the fix failed ──────────
# A disconfirmed resolution now routes straight to ESCALATING; the resolution-attempt
# budget is enforced in the executor, not in the transition, so the transition no
# longer sends a failed fix back to INVESTIGATING.

def test_t11_resolving_to_escalating_when_not_confirmed():
    result = evaluate_transition(resolving(user_confirmed_resolution=False, resolution_attempts=1), RESOLVE)
    assert result.next_phase == Phase.ESCALATING


# ── T13: resolving → escalating ──────────────────────────────────────────────

def test_t13_to_escalating_when_max_attempts_reached():
    case = resolving(user_confirmed_resolution=False, resolution_attempts=2)
    assert evaluate_transition(case, RESOLVE).next_phase == Phase.ESCALATING


# ── T14: escalating → closed (handoff completed) ─────────────────────────────

def test_t14_escalating_to_closed_when_handoff_done():
    assert evaluate_transition(escalating(handoff_completed=True), ESCALATE).next_phase == Phase.CLOSED

def test_t14_escalating_stays_when_handoff_not_done():
    assert evaluate_transition(escalating(handoff_completed=False), ESCALATE).next_phase == Phase.ESCALATING


# ── closed is terminal ────────────────────────────────────────────────────────

def test_closed_stays_closed():
    assert evaluate_transition(CaseState(phase=Phase.CLOSED), RESOLVE).next_phase == Phase.CLOSED

