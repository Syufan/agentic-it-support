import pytest
from agent.llm import BaseLLMClient, MockLLMClient
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
