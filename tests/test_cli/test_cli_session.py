from agent.llm import BaseLLMClient, MockLLMClient
from agent.proposals import AgentAction, AgentProposal
from cli import run_cli_session, typewrite
from state.case_state import CaseState, Phase


def _proposal(**kwargs) -> AgentProposal:
    defaults = {
        "action": AgentAction.ASK_USER,
        "confidence": 0.6,
        "reasoning_summary": "test",
        "message": "What OS?",
    }
    return AgentProposal(**(defaults | kwargs))


def _closing_proposal() -> AgentProposal:
    return _proposal(
        action=AgentAction.RESOLVE,
        confidence=0.9,
        message="Restart your VPN client.",
    )


def _no_clear():
    pass


# ── loop termination ──────────────────────────────────────────────────────────

def test_session_exits_when_case_is_closed():
    case = CaseState(phase=Phase.RESOLVING)
    case.user_confirmed_resolution = True
    case.conversation = [{"role": "user", "content": "VPN broken"}]

    messages = iter(["it worked"])
    run_cli_session(
        case,
        MockLLMClient([_closing_proposal()]),
        {},
        reader=lambda _: next(messages),
        writer=lambda _: None,
        clear=_no_clear,
    )
    assert case.phase == Phase.CLOSED


def test_session_exits_on_keyboard_interrupt():
    case = CaseState()
    output = []

    run_cli_session(case, MockLLMClient([]), {},
                    reader=lambda _: (_ for _ in ()).throw(KeyboardInterrupt()),
                    writer=output.append, clear=_no_clear)

    assert any("goodbye" in str(o).lower() or "bye" in str(o).lower() for o in output)


def test_session_exits_on_eof():
    case = CaseState()
    run_cli_session(case, MockLLMClient([]), {},
                    reader=lambda _: (_ for _ in ()).throw(EOFError()),
                    writer=lambda _: None, clear=_no_clear)


# ── input handling ────────────────────────────────────────────────────────────

def test_empty_input_is_skipped():
    from agent.llm import BaseLLMClient
    from runtime.message_builder import LLMInput

    case = CaseState()
    calls = []
    messages = ["", "VPN broken"]
    idx = [0]

    def _reader(_):
        if idx[0] < len(messages):
            msg = messages[idx[0]]
            idx[0] += 1
            return msg
        raise EOFError

    class _TrackingLLM(BaseLLMClient):
        def call(self, llm_input: LLMInput) -> AgentProposal:
            calls.append(llm_input)
            return _proposal()

    run_cli_session(case, _TrackingLLM(), {}, reader=_reader,
                    writer=lambda _: None, clear=_no_clear)
    assert len(calls) == 1


def test_esc_key_exits_gracefully():
    case = CaseState()
    output = []

    run_cli_session(case, MockLLMClient([]), {},
                    reader=lambda _: "\x1b",
                    writer=output.append, clear=_no_clear)

    assert any("goodbye" in str(o).lower() or "bye" in str(o).lower() for o in output)


# ── output ────────────────────────────────────────────────────────────────────

def test_agent_response_is_written(capsys):
    case = CaseState()
    messages = ["VPN broken"]
    idx = [0]

    def _reader(_):
        if idx[0] < len(messages):
            msg = messages[idx[0]]
            idx[0] += 1
            return msg
        raise EOFError

    run_cli_session(
        case,
        MockLLMClient([_proposal(message="What OS are you using?")]),
        {},
        reader=_reader,
        writer=lambda _: None,
        clear=_no_clear,
    )
    assert "What OS are you using?" in capsys.readouterr().out


def test_phase_displayed_after_response():
    case = CaseState()
    output = []
    messages = ["VPN broken"]
    idx = [0]

    def _reader(_):
        if idx[0] < len(messages):
            msg = messages[idx[0]]
            idx[0] += 1
            return msg
        raise EOFError

    run_cli_session(
        case,
        MockLLMClient([_proposal(message="What OS?")]),
        {},
        reader=_reader,
        writer=output.append,
        clear=_no_clear,
    )
    joined = " ".join(str(o) for o in output)
    assert "[" in joined and "]" in joined


def test_initial_agent_greeting_shown():
    case = CaseState(phase=Phase.CLOSED)
    output = []
    run_cli_session(case, MockLLMClient([]), {},
                    reader=lambda _: "", writer=output.append, clear=_no_clear)
    joined = " ".join(str(o) for o in output).lower()
    assert "agent" in joined or "support" in joined or "hi" in joined


def test_clear_called_on_each_render():
    case = CaseState()
    clears = []
    messages = ["VPN broken"]
    idx = [0]

    def _reader(_):
        if idx[0] < len(messages):
            msg = messages[idx[0]]
            idx[0] += 1
            return msg
        raise EOFError

    run_cli_session(
        case,
        MockLLMClient([_proposal(message="What OS?")]),
        {},
        reader=_reader,
        writer=lambda _: None,
        clear=lambda: clears.append(1),
        get_term_height=lambda: 24,
    )
    assert len(clears) >= 2  # initial render + after user message


def test_padding_pushes_divider_to_bottom():
    case = CaseState(phase=Phase.CLOSED)
    output = []

    run_cli_session(
        case, MockLLMClient([]), {},
        reader=lambda _: "",
        writer=output.append,
        clear=_no_clear,
        get_term_height=lambda: 20,
    )
    # divider must be one of the last few items written
    divider_indices = [i for i, o in enumerate(output) if "─" in str(o)]
    assert divider_indices and divider_indices[-1] >= len(output) - 4


# ── typewrite ─────────────────────────────────────────────────────────────────

def test_typewrite_outputs_all_characters():
    output = []
    typewrite("hello", writer=output.append, delay=0)
    assert "".join(output) == "hello"


def test_typewrite_writes_one_char_at_a_time():
    output = []
    typewrite("hi", writer=output.append, delay=0)
    assert output == ["h", "i"]


def test_typewrite_empty_string_writes_nothing():
    output = []
    typewrite("", writer=output.append, delay=0)
    assert output == []
