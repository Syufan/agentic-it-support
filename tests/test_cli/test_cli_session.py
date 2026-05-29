from agent.llm import MockLLMClient
from agent.proposals import AgentAction, AgentProposal
from cli import run_cli_session
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


# ── loop termination ──────────────────────────────────────────────────────────

def test_session_exits_when_case_is_closed():
    case = CaseState(phase=Phase.RESOLVING)
    case.user_confirmed_resolution = True

    messages = iter(["it worked"])
    output = []

    run_cli_session(
        case,
        MockLLMClient([_closing_proposal()]),
        {},
        reader=lambda _: next(messages),
        writer=output.append,
    )

    assert case.phase == Phase.CLOSED


def test_session_exits_on_keyboard_interrupt():
    case = CaseState()
    output = []

    def _raise(_):
        raise KeyboardInterrupt

    run_cli_session(case, MockLLMClient([]), {}, reader=_raise, writer=output.append)

    joined = " ".join(output).lower()
    assert "goodbye" in joined or "bye" in joined


def test_session_exits_on_eof():
    case = CaseState()
    output = []

    def _raise(_):
        raise EOFError

    run_cli_session(case, MockLLMClient([]), {}, reader=_raise, writer=output.append)


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

    run_cli_session(case, _TrackingLLM(), {}, reader=_reader, writer=lambda _: None)

    assert len(calls) == 1  # only called once, for "VPN broken"


# ── output ────────────────────────────────────────────────────────────────────

def test_agent_response_is_written():
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
        MockLLMClient([_proposal(message="What OS are you using?")]),
        {},
        reader=_reader,
        writer=output.append,
    )

    joined = " ".join(str(o) for o in output)
    assert "What OS are you using?" in joined


def test_esc_key_exits_gracefully():
    case = CaseState()
    output = []

    run_cli_session(
        case, MockLLMClient([]), {},
        reader=lambda _: "\x1b",
        writer=output.append,
    )

    joined = " ".join(str(o) for o in output).lower()
    assert "goodbye" in joined or "bye" in joined


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
    )

    joined = " ".join(str(o) for o in output)
    assert "[" in joined and "phase" in joined.lower()


def test_welcome_message_is_printed():
    case = CaseState(phase=Phase.CLOSED)
    output = []

    run_cli_session(case, MockLLMClient([]), {}, reader=lambda _: "", writer=output.append)

    joined = " ".join(str(o) for o in output).lower()
    assert "support" in joined or "ready" in joined or "agent" in joined
