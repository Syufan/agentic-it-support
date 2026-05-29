from state.case_state import BudgetMode, CaseState, MissingInfoSource, Phase


def test_default_phase_is_intake():
    case = CaseState()
    assert case.phase == Phase.INTAKE


def test_default_budget_mode_is_main():
    case = CaseState()
    assert case.budget_mode == BudgetMode.MAIN


def test_default_flags_are_false():
    case = CaseState()
    assert case.exception_used is False
    assert case.has_safe_low_risk_guidance is False
    assert case.new_critical_fact_added is False
    assert case.handoff_completed is False


def test_default_missing_info_source_is_none():
    case = CaseState()
    assert case.missing_info_source == MissingInfoSource.NONE


def test_default_counters_are_zero():
    case = CaseState()
    assert case.tool_calls_current_investigation == 0
    assert case.tool_calls_total == 0
    assert case.resolution_attempts == 0
    assert case.confidence == 0.0


def test_each_case_gets_unique_id():
    case_a = CaseState()
    case_b = CaseState()
    assert case_a.case_id != case_b.case_id


def test_conversation_and_facts_are_independent():
    case_a = CaseState()
    case_b = CaseState()
    case_a.conversation.append({"role": "user", "content": "hello"})
    case_a.facts["os"] = "macOS"
    assert case_b.conversation == []
    assert case_b.facts == {}
