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


def test_add_user_message_appends_user_turn():
    case = CaseState()
    case.add_user_message("my laptop won't boot")
    assert case.conversation == [{"role": "user", "content": "my laptop won't boot"}]


def test_add_assistant_message_appends_assistant_turn():
    case = CaseState()
    case.add_assistant_message("have you tried restarting?")
    assert case.conversation == [{"role": "assistant", "content": "have you tried restarting?"}]


def test_messages_append_in_order():
    case = CaseState()
    case.add_user_message("hi")
    case.add_assistant_message("hello, how can I help?")
    case.add_user_message("vpn is down")
    assert case.conversation == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello, how can I help?"},
        {"role": "user", "content": "vpn is down"},
    ]
