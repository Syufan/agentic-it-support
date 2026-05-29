from evaluation import evaluate
from state.case_state import CaseState, Phase


def _closed_case(escalated: bool = False) -> CaseState:
    case = CaseState()
    case.phase = Phase.CLOSED
    if escalated:
        case.escalation_context = {"escalation_reason": "needs admin"}
    return case


# ── resolved / escalated flags ────────────────────────────────────────────────

def test_resolved_true_when_closed_without_escalation():
    result = evaluate(_closed_case(escalated=False))
    assert result.resolved is True
    assert result.escalated is False


def test_escalated_true_when_closed_with_escalation_context():
    result = evaluate(_closed_case(escalated=True))
    assert result.escalated is True
    assert result.resolved is False


def test_open_case_not_resolved():
    case = CaseState(phase=Phase.INVESTIGATING)
    result = evaluate(case)
    assert result.resolved is False


# ── metrics ───────────────────────────────────────────────────────────────────

def test_tool_calls_total_reflected():
    case = _closed_case()
    case.tool_calls_total = 3
    result = evaluate(case)
    assert result.tool_calls_total == 3


def test_resolution_attempts_reflected():
    case = _closed_case()
    case.resolution_attempts = 2
    result = evaluate(case)
    assert result.resolution_attempts == 2


def test_final_confidence_reflected():
    case = _closed_case()
    case.confidence = 0.9
    result = evaluate(case)
    assert result.final_confidence == 0.9


def test_final_phase_is_string():
    case = _closed_case()
    result = evaluate(case)
    assert result.final_phase == "closed"


# ── efficiency score ──────────────────────────────────────────────────────────

def test_efficiency_is_1_when_no_tools_used():
    case = _closed_case()
    case.tool_calls_total = 0
    result = evaluate(case)
    assert result.tool_efficiency == 1.0


def test_efficiency_decreases_with_more_tool_calls():
    case_few = _closed_case()
    case_few.tool_calls_total = 1
    case_many = _closed_case()
    case_many.tool_calls_total = 4
    assert evaluate(case_few).tool_efficiency > evaluate(case_many).tool_efficiency


def test_efficiency_is_0_when_budget_exhausted():
    case = _closed_case()
    case.tool_calls_total = 5  # MAIN_TOOL_BUDGET
    result = evaluate(case)
    assert result.tool_efficiency == 0.0
