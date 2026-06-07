from pathlib import Path

from agentic_it_support.agent.proposals import AgentAction, AgentProposal
from agentic_it_support.cli.app import _format_status, run_cli_session, typewrite
from agentic_it_support.config.settings import Settings
from agentic_it_support.llm.client import BaseLLMClient, MockLLMClient
from agentic_it_support.llm.client import LLMInput
from agentic_it_support.state.case_state import CaseState, Phase

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def _settings() -> Settings:
    return Settings(_env_file=None, data_dir=_DATA_DIR)


def _proposal(**kwargs) -> AgentProposal:
    defaults = {"action": AgentAction.ASK_USER, "message": "What OS?"}
    return AgentProposal(**(defaults | kwargs))


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


def _session(case, llm, *, tools=None, reader, writer=lambda _: None, clear=_no_clear, **kwargs):
    run_cli_session(case, llm, tools or {}, _settings(),
                    reader=reader, writer=writer, clear=clear, **kwargs)


# ── loop termination ──────────────────────────────────────────────────────────

def test_session_exits_when_case_is_closed():
    case = CaseState(phase=Phase.INVESTIGATING)

    def _closing_runner(case, user_input, llm, tools, settings):
        case.phase = Phase.CLOSED
        return "done"

    _session(case, MockLLMClient([]), reader=_reader_from(["it worked"]),
             spinner_factory=_noop_spinner_factory, turn_runner=_closing_runner)
    assert case.phase == Phase.CLOSED


def test_session_exits_on_keyboard_interrupt():
    output = []
    _session(CaseState(), MockLLMClient([]),
             reader=lambda _: (_ for _ in ()).throw(KeyboardInterrupt()),
             writer=output.append)
    assert any("goodbye" in str(o).lower() for o in output)


def test_session_exits_on_eof():
    output = []
    _session(CaseState(), MockLLMClient([]),
             reader=lambda _: (_ for _ in ()).throw(EOFError()),
             writer=output.append)
    assert any("goodbye" in str(o).lower() for o in output)


def test_normal_turn_runner_response_is_written():
    output = []

    def _runner(case, user_input, llm, tools, settings):
        return "here is your answer"

    _session(CaseState(), MockLLMClient([]), reader=_reader_from(["help"]),
             writer=output.append, spinner_factory=_noop_spinner_factory, turn_runner=_runner)
    assert any("here is your answer" in str(o) for o in output)


# ── input handling ────────────────────────────────────────────────────────────

def test_empty_input_is_skipped():
    calls = []

    class _TrackingLLM(BaseLLMClient):
        def call(self, llm_input: LLMInput) -> AgentProposal:
            calls.append(llm_input)
            return _proposal()

    _session(CaseState(), _TrackingLLM(), reader=_reader_from(["", "VPN broken"]))
    assert len(calls) == 1


def test_unknown_command_does_not_call_llm():
    calls = []

    class _TrackingLLM(BaseLLMClient):
        def call(self, llm_input: LLMInput) -> AgentProposal:
            calls.append(llm_input)
            return _proposal()

    output = []
    _session(CaseState(), _TrackingLLM(), reader=_reader_from(["/unknown"]), writer=output.append)
    assert calls == []
    assert any("unknown command" in str(o).lower() for o in output)


# ── commands ──────────────────────────────────────────────────────────────────

def test_help_command_prints_commands():
    output = []
    _session(CaseState(), MockLLMClient([]), reader=_reader_from(["/help"]), writer=output.append)
    joined = " ".join(str(o) for o in output).lower()
    assert "/status" in joined and "/trace" in joined


def test_status_command_prints_phase():
    output = []
    _session(CaseState(phase=Phase.INVESTIGATING), MockLLMClient([]),
             reader=_reader_from(["/status"]), writer=output.append)
    joined = " ".join(str(o) for o in output).lower()
    assert "phase=investigating" in joined


def test_trace_command_prints_empty_message_when_no_events():
    output = []
    _session(CaseState(), MockLLMClient([]), reader=_reader_from(["/trace"]), writer=output.append)
    joined = " ".join(str(o) for o in output).lower()
    assert "no runtime events" in joined


def test_clear_command_calls_clear_and_reprints_header():
    output = []
    clears = []
    _session(CaseState(), MockLLMClient([]), reader=_reader_from(["/clear"]),
             writer=output.append, clear=lambda: clears.append(1))
    assert clears == [1]
    assert sum(1 for o in output if "Agentic IT Support" in str(o)) >= 2


def test_quit_command_exits():
    output = []
    _session(CaseState(), MockLLMClient([]), reader=_reader_from(["/quit"]), writer=output.append)
    assert any("goodbye" in str(o).lower() for o in output)


# ── output ────────────────────────────────────────────────────────────────────

def test_agent_response_is_written():
    output = []
    _session(CaseState(), MockLLMClient([_proposal(message="What OS are you using?")]),
             reader=_reader_from(["VPN broken"]), writer=output.append)
    joined = " ".join(str(o) for o in output)
    assert "What OS are you using?" in joined


def test_response_is_prefixed_with_agent_label():
    output = []
    _session(CaseState(), MockLLMClient([_proposal(message="What OS?")]),
             reader=_reader_from(["VPN broken"]), writer=output.append)
    joined = " ".join(str(o) for o in output)
    assert "Agent:" in joined


def test_thinking_line_shows_current_phase():
    phases = []

    class _CaptureSpinner:
        def __init__(self, get_phase):
            self.get_phase = get_phase

        def start(self):
            phases.append(self.get_phase())

        def stop(self):
            pass

    _session(CaseState(phase=Phase.CLARIFYING), MockLLMClient([_proposal(message="What OS?")]),
             reader=_reader_from(["VPN broken"]), spinner_factory=_CaptureSpinner)
    assert phases == ["clarifying"]


def test_spinner_starts_and_stops_for_agent_turn():
    spinners = []

    def _factory(get_phase):
        spinner = _NoopSpinner(get_phase)
        spinners.append(spinner)
        return spinner

    _session(CaseState(), MockLLMClient([_proposal(message="What OS?")]),
             reader=_reader_from(["VPN broken"]), spinner_factory=_factory)
    assert len(spinners) == 1
    assert spinners[0].started is True
    assert spinners[0].stopped is True


def test_status_not_displayed_after_response_by_default():
    output = []
    _session(CaseState(), MockLLMClient([_proposal(message="What OS?")]),
             reader=_reader_from(["VPN broken"]), writer=output.append)
    joined = " ".join(str(o) for o in output).lower()
    assert "phase=" not in joined
    assert "confidence=" not in joined


def test_initial_header_shown():
    output = []
    _session(CaseState(phase=Phase.CLOSED), MockLLMClient([]),
             reader=lambda _: "", writer=output.append)
    joined = " ".join(str(o) for o in output).lower()
    assert "agentic it support" in joined


def test_format_status_includes_tool_call_limits():
    case = CaseState(phase=Phase.INVESTIGATING)
    case.tool_calls_this_turn = 2
    case.tool_calls_total = 4
    status = _format_status(case, _settings())
    assert "phase=investigating" in status
    assert "remaining_turn=" in status
    assert "remaining_case=" in status


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
