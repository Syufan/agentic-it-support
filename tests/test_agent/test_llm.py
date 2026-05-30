import pytest
from llm.client import (
    BaseLLMClient,
    LLMConfigurationError,
    LLMResponseError,
    MockLLMClient,
    RealLLMClient,
)
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
    with pytest.raises(LLMConfigurationError, match="LLM_API_KEY"):
        RealLLMClient(api_key="")


def test_real_llm_raises_for_non_json_response():
    client = RealLLMClient(api_key="", client=_FakeOpenAIClient("not json"))

    with pytest.raises(LLMResponseError, match="non-JSON"):
        client.call(_llm_input())


def test_real_llm_raises_for_invalid_proposal_json():
    client = RealLLMClient(api_key="", client=_FakeOpenAIClient('{"ok": true}'))

    with pytest.raises(LLMResponseError, match="AgentProposal"):
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
    client = RealLLMClient(api_key="", client=_FakeOpenAIClientNoChoices())
    with pytest.raises(LLMResponseError, match="no choices"):
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
    client = RealLLMClient(api_key="", client=_FakeOpenAIClientWithUsage())
    client.call(_llm_input())
    assert client.last_stats is not None
    assert client.last_stats.prompt_tokens == 120
    assert client.last_stats.completion_tokens == 30


def test_real_llm_records_latency():
    client = RealLLMClient(api_key="", client=_FakeOpenAIClientWithUsage())
    client.call(_llm_input())
    assert client.last_stats.latency_ms >= 0.0


def test_real_llm_stats_default_zero_without_usage():
    client = RealLLMClient(api_key="", client=_FakeOpenAIClient(
        '{"action": "ask_user", "confidence": 0.6, "reasoning_summary": "x", "message": "hi"}'))
    client.call(_llm_input())
    assert client.last_stats.prompt_tokens == 0
