from agent.llm import BaseLLMClient, MockLLMClient
from agent.proposals import AgentAction, AgentProposal
from cli import _format_status, run_cli_session, typewrite
from runtime.controller import TurnCancelled
from runtime.message_builder import LLMInput
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


def _reader_from(messages: list[str]):
    idx = [0]

    def _reader(_prompt: str) -> str:
        if idx[0] < len(messages):
            msg = messages[idx[0]]
            idx[0] += 1
            return msg
        raise EOFError

    return _reader


def _no_clear():
    pass


class _NoopSpinner:
    def __init__(self, get_phase):
        self.get_phase = get_phase
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True


def _noop_spinner_factory(_get_phase):
    return _NoopSpinner(_get_phase)


# ── loop termination ──────────────────────────────────────────────────────────

def test_session_exits_when_case_is_closed():
    case = CaseState(phase=Phase.RESOLVING)
    case.user_confirmed_resolution = True
    case.conversation = [{"role": "user", "content": "VPN broken"}]

    run_cli_session(
        case,
        MockLLMClient([_closing_proposal()]),
        {},
        reader=_reader_from(["it worked"]),
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

    assert any("goodbye" in str(o).lower() for o in output)


def test_session_exits_on_eof():
    case = CaseState()
    output = []
    run_cli_session(case, MockLLMClient([]), {},
                    reader=lambda _: (_ for _ in ()).throw(EOFError()),
                    writer=output.append, clear=_no_clear)
    assert any("goodbye" in str(o).lower() for o in output)


def test_cancelled_turn_does_not_exit_and_prompts_again():
    case = CaseState()
    output = []
    calls = [0]

    def _cancelling_runner(case, user_input, llm, tools, event_log):
        calls[0] += 1
        raise TurnCancelled()

    run_cli_session(case, MockLLMClient([]), {},
                    reader=_reader_from(["first", "second"]),
                    writer=output.append, clear=_no_clear,
                    spinner_factory=_noop_spinner_factory,
                    turn_runner=_cancelling_runner)

    # both messages were attempted — cancelling one turn did not end the session
    assert calls[0] == 2
    assert case.phase != Phase.CLOSED


def test_cancelled_turn_prints_a_notice():
    case = CaseState()
    output = []

    def _cancelling_runner(case, user_input, llm, tools, event_log):
        raise TurnCancelled()

    run_cli_session(case, MockLLMClient([]), {},
                    reader=_reader_from(["first"]),
                    writer=output.append, clear=_no_clear,
                    spinner_factory=_noop_spinner_factory,
                    turn_runner=_cancelling_runner)

    assert any("cancel" in str(o).lower() for o in output)


def test_normal_turn_runner_response_is_written():
    case = CaseState()
    output = []

    def _runner(case, user_input, llm, tools, event_log):
        return "here is your answer"

    run_cli_session(case, MockLLMClient([]), {},
                    reader=_reader_from(["help"]),
                    writer=output.append, clear=_no_clear,
                    spinner_factory=_noop_spinner_factory,
                    turn_runner=_runner)

    assert any("here is your answer" in str(o) for o in output)


# ── input handling ────────────────────────────────────────────────────────────

def test_empty_input_is_skipped():
    case = CaseState()
    calls = []

    class _TrackingLLM(BaseLLMClient):
        def call(self, llm_input: LLMInput) -> AgentProposal:
            calls.append(llm_input)
            return _proposal()

    run_cli_session(case, _TrackingLLM(), {}, reader=_reader_from(["", "VPN broken"]),
                    writer=lambda _: None, clear=_no_clear)
    assert len(calls) == 1


def test_greeting_does_not_close_case():
    case = CaseState()
    output = []
    run_cli_session(case, MockLLMClient([]), {}, reader=_reader_from(["hey"]),
                    writer=output.append, clear=_no_clear)
    joined = " ".join(str(o) for o in output).lower()
    assert "what it issue" in joined
    assert "case closed" not in joined
    assert case.phase == Phase.CLARIFYING


def test_unknown_command_does_not_call_llm():
    case = CaseState()
    calls = []

    class _TrackingLLM(BaseLLMClient):
        def call(self, llm_input: LLMInput) -> AgentProposal:
            calls.append(llm_input)
            return _proposal()

    output = []
    run_cli_session(case, _TrackingLLM(), {}, reader=_reader_from(["/unknown"]),
                    writer=output.append, clear=_no_clear)

    assert calls == []
    assert any("unknown command" in str(o).lower() for o in output)


# ── commands ──────────────────────────────────────────────────────────────────

def test_help_command_prints_commands():
    output = []
    run_cli_session(CaseState(), MockLLMClient([]), {},
                    reader=_reader_from(["/help"]), writer=output.append, clear=_no_clear)
    joined = " ".join(str(o) for o in output).lower()
    assert "/status" in joined and "/trace" in joined


def test_status_command_prints_phase():
    output = []
    case = CaseState(phase=Phase.INVESTIGATING)
    run_cli_session(case, MockLLMClient([]), {},
                    reader=_reader_from(["/status"]), writer=output.append, clear=_no_clear)
    joined = " ".join(str(o) for o in output).lower()
    assert "phase=investigating" in joined


def test_trace_command_before_turn_prints_empty_message():
    output = []
    run_cli_session(CaseState(), MockLLMClient([]), {},
                    reader=_reader_from(["/trace"]), writer=output.append, clear=_no_clear)
    joined = " ".join(str(o) for o in output).lower()
    assert "no runtime events" in joined


def test_trace_command_after_turn_shows_events():
    output = []
    run_cli_session(
        CaseState(),
        MockLLMClient([_proposal(message="What OS?")]),
        {},
        reader=_reader_from(["VPN broken", "/trace"]),
        writer=output.append,
        clear=_no_clear,
    )
    joined = " ".join(str(o) for o in output).lower()
    assert "recent runtime events" in joined
    assert "turn_start" in joined


def test_clear_command_calls_clear_and_reprints_header():
    output = []
    clears = []
    run_cli_session(CaseState(), MockLLMClient([]), {},
                    reader=_reader_from(["/clear"]),
                    writer=output.append,
                    clear=lambda: clears.append(1))
    assert clears == [1]
    assert sum(1 for o in output if "Agentic IT Support" in str(o)) >= 2


def test_quit_command_exits():
    output = []
    run_cli_session(CaseState(), MockLLMClient([]), {},
                    reader=_reader_from(["/quit"]), writer=output.append, clear=_no_clear)
    assert any("goodbye" in str(o).lower() for o in output)


# ── output ────────────────────────────────────────────────────────────────────

def test_agent_response_is_written():
    output = []
    run_cli_session(
        CaseState(),
        MockLLMClient([_proposal(message="What OS are you using?")]),
        {},
        reader=_reader_from(["VPN broken"]),
        writer=output.append,
        clear=_no_clear,
    )
    joined = " ".join(str(o) for o in output)
    assert "What OS are you using?" in joined


def test_response_is_not_prefixed_with_fixed_agent_label():
    output = []
    run_cli_session(
        CaseState(),
        MockLLMClient([_proposal(message="What OS?")]),
        {},
        reader=_reader_from(["VPN broken"]),
        writer=output.append,
        clear=_no_clear,
    )
    joined = " ".join(str(o) for o in output)
    assert "Agent:" not in joined


def test_thinking_line_shows_current_phase():
    phases = []

    class _CaptureSpinner:
        def __init__(self, get_phase):
            self.get_phase = get_phase

        def start(self):
            phases.append(self.get_phase())

        def stop(self):
            pass

    run_cli_session(
        CaseState(phase=Phase.CLARIFYING),
        MockLLMClient([_proposal(message="What OS?")]),
        {},
        reader=_reader_from(["VPN broken"]),
        writer=lambda _: None,
        clear=_no_clear,
        spinner_factory=_CaptureSpinner,
    )
    assert phases == ["clarifying"]


def test_spinner_starts_and_stops_for_agent_turn():
    spinners = []

    def _factory(get_phase):
        spinner = _NoopSpinner(get_phase)
        spinners.append(spinner)
        return spinner

    run_cli_session(
        CaseState(),
        MockLLMClient([_proposal(message="What OS?")]),
        {},
        reader=_reader_from(["VPN broken"]),
        writer=lambda _: None,
        clear=_no_clear,
        spinner_factory=_factory,
    )
    assert len(spinners) == 1
    assert spinners[0].started is True
    assert spinners[0].stopped is True


def test_status_not_displayed_after_response_by_default():
    output = []
    run_cli_session(
        CaseState(),
        MockLLMClient([_proposal(message="What OS?")]),
        {},
        reader=_reader_from(["VPN broken"]),
        writer=output.append,
        clear=_no_clear,
    )
    joined = " ".join(str(o) for o in output).lower()
    assert "phase=" not in joined
    assert "confidence=" not in joined


def test_initial_agent_greeting_shown():
    output = []
    run_cli_session(CaseState(phase=Phase.CLOSED), MockLLMClient([]), {},
                    reader=lambda _: "", writer=output.append, clear=_no_clear)
    joined = " ".join(str(o) for o in output).lower()
    assert "agentic it support" in joined


def test_format_status_includes_budget_and_remaining():
    case = CaseState(phase=Phase.INVESTIGATING)
    case.tool_calls_current_investigation = 2
    status = _format_status(case)
    assert "phase=investigating" in status
    assert "remaining=" in status
    assert "budget=main" in status


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
