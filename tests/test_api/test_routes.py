import pytest
from fastapi.testclient import TestClient

from agent.llm import BaseLLMClient, LLMProviderError, MockLLMClient
from agent.proposals import AgentAction, AgentProposal
from api.routes import app, get_llm, get_store, get_tool_registry
from state.session import SessionStore
from tools.base import BaseTool, ToolResult
from typing import Any


# ── fixtures ──────────────────────────────────────────────────────────────────

def _proposal(**kwargs) -> AgentProposal:
    return AgentProposal(**{
        "action": AgentAction.ASK_USER,
        "confidence": 0.6,
        "reasoning_summary": "test",
        "message": "What OS are you using?",
        **kwargs,
    })


@pytest.fixture
def client():
    store = SessionStore()
    llm = MockLLMClient([_proposal()])
    tools: dict[str, BaseTool] = {}

    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_llm] = lambda: llm
    app.dependency_overrides[get_tool_registry] = lambda: tools

    yield TestClient(app)

    app.dependency_overrides.clear()


@pytest.fixture
def persistent_store():
    return SessionStore()


# ── health ────────────────────────────────────────────────────────────────────

def test_health_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


# ── chat basics ───────────────────────────────────────────────────────────────

def test_chat_returns_200(client):
    response = client.post("/chat", json={"message": "VPN is broken"})
    assert response.status_code == 200


def test_chat_returns_case_id(client):
    response = client.post("/chat", json={"message": "VPN is broken"})
    assert "case_id" in response.json()
    assert response.json()["case_id"] is not None


def test_chat_returns_message(client):
    response = client.post("/chat", json={"message": "VPN is broken"})
    assert "message" in response.json()
    assert response.json()["message"] == "What OS are you using?"


def test_chat_returns_phase(client):
    response = client.post("/chat", json={"message": "VPN is broken"})
    assert "phase" in response.json()


def test_chat_returns_is_closed(client):
    response = client.post("/chat", json={"message": "VPN is broken"})
    assert "is_closed" in response.json()
    assert response.json()["is_closed"] is False


# ── case continuity ───────────────────────────────────────────────────────────

def test_chat_continues_existing_case(persistent_store):
    llm_seq = MockLLMClient([
        _proposal(message="What OS?"),
        _proposal(message="Got it, checking now."),
    ])
    app.dependency_overrides[get_store] = lambda: persistent_store
    app.dependency_overrides[get_llm] = lambda: llm_seq
    app.dependency_overrides[get_tool_registry] = lambda: {}

    c = TestClient(app)
    r1 = c.post("/chat", json={"message": "VPN broken"})
    case_id = r1.json()["case_id"]

    r2 = c.post("/chat", json={"message": "macOS", "case_id": case_id})
    assert r2.json()["case_id"] == case_id
    assert r2.json()["message"] == "Got it, checking now."

    app.dependency_overrides.clear()


def test_chat_unknown_case_id_returns_404(persistent_store):
    llm = MockLLMClient([_proposal()])
    app.dependency_overrides[get_store] = lambda: persistent_store
    app.dependency_overrides[get_llm] = lambda: llm
    app.dependency_overrides[get_tool_registry] = lambda: {}

    c = TestClient(app)
    response = c.post("/chat", json={"message": "VPN broken", "case_id": "nonexistent-id"})
    assert response.status_code == 404

    app.dependency_overrides.clear()


def test_chat_returns_graceful_message_when_llm_fails_mid_turn(persistent_store):
    class FailingLLM(BaseLLMClient):
        def call(self, llm_input):
            raise LLMProviderError("LLM model unavailable")

    app.dependency_overrides[get_store] = lambda: persistent_store
    app.dependency_overrides[get_llm] = lambda: FailingLLM()
    app.dependency_overrides[get_tool_registry] = lambda: {}

    c = TestClient(app)
    response = c.post("/chat", json={"message": "VPN broken"})

    assert response.status_code == 200
    body = response.json()
    assert "specialist" in body["message"].lower() or "technical issue" in body["message"].lower()

    app.dependency_overrides.clear()
