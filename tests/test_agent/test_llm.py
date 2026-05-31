import pytest
from llm.client import (
    BaseLLMClient,
    LLMConfigurationError,
    LLMProviderError,
    MockLLMClient,
    RealLLMClient,
)
from agent.parser import ProposalParseError
from agent.proposals import AgentAction, AgentProposal
from runtime.message_builder import LLMInput


def _proposal(**kwargs) -> AgentProposal:
    defaults = {
        "action": AgentAction.ASK_USER,
        "confidence": 0.6,
        "reasoning_summary": "test",
        "message": "What OS?",
    }
    return AgentProposal(**(defaults | kwargs))

def _llm_input() -> LLMInput:
    return LLMInput(system="test system", messages=[{"role": "user", "content": "hello"}])


def _echo(raw: str) -> str:
    """Trivial transport-level parser: keeps these tests domain-agnostic."""
    return raw


# ── BaseLLMClient is abstract ─────────────────────────────────────────────────

def test_base_llm_client_cannot_be_instantiated():
    with pytest.raises(TypeError):
        BaseLLMClient()


# ── MockLLMClient ─────────────────────────────────────────────────────────────

def test_mock_returns_proposals_in_order():
    p1 = _proposal(message="First question")
    p2 = _proposal(message="Second question")
    client = MockLLMClient([p1, p2])
    assert client.call(_llm_input()) is p1
    assert client.call(_llm_input()) is p2


def test_mock_raises_when_queue_empty():
    client = MockLLMClient([])
    with pytest.raises(RuntimeError):
        client.call(_llm_input())


def test_mock_returns_different_action_types():
    proposals = [
        _proposal(action=AgentAction.CALL_TOOL, tool_name="kb_search",
                  tool_input={"query": "vpn"}, message=None),
        _proposal(action=AgentAction.RESOLVE, confidence=0.9),
    ]
    client = MockLLMClient(proposals)
    first = client.call(_llm_input())
    second = client.call(_llm_input())
    assert first.action == AgentAction.CALL_TOOL
    assert second.action == AgentAction.RESOLVE


def test_mock_is_llm_client_subclass():
    client = MockLLMClient([])
    assert isinstance(client, BaseLLMClient)


# ── RealLLMClient failures ───────────────────────────────────────────────────

class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, content: str) -> None:
        self.content = content

    def create(self, **kwargs):
        return _FakeResponse(self.content)


class _FakeChat:
    def __init__(self, content: str) -> None:
        self.completions = _FakeCompletions(content)


class _FakeOpenAIClient:
    def __init__(self, content: str) -> None:
        self.chat = _FakeChat(content)


def test_real_llm_raises_when_api_key_missing():
    with pytest.raises(LLMConfigurationError, match="api_key"):
        RealLLMClient(response_parser=_echo, api_key="")


def test_real_llm_delegates_raw_content_to_parser():
    seen: dict[str, str] = {}

    def parser(raw: str) -> str:
        seen["raw"] = raw
        return "parsed"

    client = RealLLMClient(response_parser=parser, api_key="", client=_FakeOpenAIClient('{"x": 1}'))
    result = client.call(_llm_input())

    assert seen["raw"] == '{"x": 1}'
    assert result == "parsed"


def test_real_llm_propagates_parser_error():
    def parser(raw: str) -> str:
        raise ProposalParseError("parser said no")

    client = RealLLMClient(response_parser=parser, api_key="", client=_FakeOpenAIClient("whatever"))

    with pytest.raises(ProposalParseError, match="parser said no"):
        client.call(_llm_input())


class _FakeCompletionsNoChoices:
    def create(self, **kwargs):
        class _Empty:
            choices = []
        return _Empty()


class _FakeOpenAIClientNoChoices:
    class _Chat:
        completions = _FakeCompletionsNoChoices()
    chat = _Chat()


def test_real_llm_raises_for_empty_choices():
    client = RealLLMClient(response_parser=_echo, api_key="", client=_FakeOpenAIClientNoChoices())
    with pytest.raises(LLMProviderError, match="no choices"):
        client.call(_llm_input())


# ── cost / latency stats ──────────────────────────────────────────────────────

class _Usage:
    prompt_tokens = 120
    completion_tokens = 30
    total_tokens = 150


class _FakeResponseWithUsage(_FakeResponse):
    def __init__(self, content: str) -> None:
        super().__init__(content)
        self.usage = _Usage()


class _FakeCompletionsWithUsage:
    def create(self, **kwargs):
        return _FakeResponseWithUsage('{"action": "ask_user", "confidence": 0.6, '
                                      '"reasoning_summary": "x", "message": "hi"}')


class _FakeOpenAIClientWithUsage:
    class _Chat:
        completions = _FakeCompletionsWithUsage()
    chat = _Chat()


def test_real_llm_records_token_usage():
    client = RealLLMClient(response_parser=_echo, api_key="", client=_FakeOpenAIClientWithUsage())
    client.call(_llm_input())
    assert client.last_stats is not None
    assert client.last_stats.prompt_tokens == 120
    assert client.last_stats.completion_tokens == 30


class _CapturingCompletions:
    def __init__(self) -> None:
        self.kwargs: dict = {}

    def create(self, **kwargs):
        self.kwargs = kwargs
        return _FakeResponse('{"action": "ask_user", "reasoning_summary": "x", "message": "hi"}')


def _capturing_client(completions: "_CapturingCompletions"):
    class _Client:
        class _Chat:
            pass
        chat = _Chat()
    _Client.chat.completions = completions
    return _Client()


def test_real_llm_passes_injected_temperature_to_the_api():
    completions = _CapturingCompletions()
    client = RealLLMClient(response_parser=_echo, api_key="", model="m",
                           temperature=0.9, client=_capturing_client(completions))
    client.call(_llm_input())
    assert completions.kwargs["temperature"] == 0.9


def test_real_llm_omits_temperature_when_unset():
    completions = _CapturingCompletions()
    client = RealLLMClient(response_parser=_echo, api_key="", model="m",
                           temperature=None, client=_capturing_client(completions))
    client.call(_llm_input())
    assert "temperature" not in completions.kwargs


def test_real_llm_records_latency():
    client = RealLLMClient(response_parser=_echo, api_key="", client=_FakeOpenAIClientWithUsage())
    client.call(_llm_input())
    assert client.last_stats.latency_ms >= 0.0


def test_real_llm_stats_default_zero_without_usage():
    client = RealLLMClient(response_parser=_echo, api_key="", client=_FakeOpenAIClient(
        '{"action": "ask_user", "confidence": 0.6, "reasoning_summary": "x", "message": "hi"}'))
    client.call(_llm_input())
    assert client.last_stats.prompt_tokens == 0
