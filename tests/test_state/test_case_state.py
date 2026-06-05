from agentic_it_support.state.case_state import CaseState, Phase


def test_default_phase_is_intake():
    case = CaseState()
    assert case.phase == Phase.INTAKE


def test_default_flags_are_false():
    case = CaseState()
    assert case.handoff_completed is False


def test_default_counters_are_zero():
    case = CaseState()
    assert case.tool_calls_this_turn == 0
    assert case.tool_calls_total == 0
    assert case.llm_calls_total == 0
    assert case.resolution_attempts == 0
    assert case.confidence == 0.0


def test_each_case_gets_unique_id():
    case_a = CaseState()
    case_b = CaseState()
    assert case_a.case_id != case_b.case_id


def test_conversation_is_independent():
    case_a = CaseState()
    case_b = CaseState()
    case_a.conversation.append({"role": "user", "content": "hello"})
    assert case_b.conversation == []
