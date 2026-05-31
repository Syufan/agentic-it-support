import pytest
from fastapi.testclient import TestClient

from llm.client import BaseLLMClient, LLMProviderError, MockLLMClient
from agent.proposals import AgentAction, AgentProposal
from api.server import ITSupportWebServer
from runtime.query_loop import run_turn
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


class _StubTool(BaseTool):
    name = "kb_search"
    description = "stub"

    def run(self, inputs: dict[str, Any]) -> ToolResult:
        return ToolResult(success=True, data={"results": []})


@pytest.fixture
def client():
    store = SessionStore()
    llm = MockLLMClient([_proposal()])
    tools: dict[str, BaseTool] = {}
    app = ITSupportWebServer(
        llm=llm,
        tools=tools,
        store=store,
        turn_runner=run_turn,
    ).get_app()

    return TestClient(app)


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


def test_chat_rejects_empty_message(client):
    assert client.post("/chat", json={"message": ""}).status_code == 422


def test_chat_rejects_whitespace_only_message(client):
    assert client.post("/chat", json={"message": "   "}).status_code == 422


# ── case continuity ───────────────────────────────────────────────────────────

def test_chat_continues_existing_case(persistent_store):
    llm_seq = MockLLMClient([
        _proposal(message="What OS?"),
        _proposal(message="Got it, checking now."),
    ])
    app = ITSupportWebServer(
        llm=llm_seq,
        tools={},
        store=persistent_store,
        turn_runner=run_turn,
    ).get_app()
    c = TestClient(app)
    # non-actionable messages so the case stays in clarifying across both turns
    # (an actionable issue would correctly force a tool call, not a 2nd question)
    r1 = c.post("/chat", json={"message": "I'm having some trouble"})
    case_id = r1.json()["case_id"]

    r2 = c.post("/chat", json={"message": "not sure how to explain it", "case_id": case_id})
    assert r2.json()["case_id"] == case_id
    assert r2.json()["message"] == "Got it, checking now."


def test_chat_unknown_case_id_returns_404(persistent_store):
    llm = MockLLMClient([_proposal()])
    app = ITSupportWebServer(
        llm=llm,
        tools={},
        store=persistent_store,
        turn_runner=run_turn,
    ).get_app()
    c = TestClient(app)
    response = c.post("/chat", json={"message": "VPN broken", "case_id": "nonexistent-id"})
    assert response.status_code == 404


# ── case retrieval / escalation handoff ───────────────────────────────────────

def test_get_case_returns_404_for_unknown_id(client):
    assert client.get("/case/nonexistent-id").status_code == 404


def test_get_case_returns_state_and_handoff(persistent_store):
    # drive a turn that investigates then escalates so an escalation_context exists
    llm = MockLLMClient([
        _proposal(action=AgentAction.CALL_TOOL, confidence=0.6, tool_name="kb_search",
                  tool_input={"query": "vpn"}, message=None),
        _proposal(action=AgentAction.ESCALATE, confidence=0.3,
                  escalation_reason="needs admin access", message=None),
    ])
    app = ITSupportWebServer(
        llm=llm,
        tools={"kb_search": _StubTool()},
        store=persistent_store,
        turn_runner=run_turn,
    ).get_app()
    c = TestClient(app)
    case_id = c.post("/chat", json={"message": "VPN broken"}).json()["case_id"]

    r = c.get(f"/case/{case_id}")
    assert r.status_code == 200
    body = r.json()
    assert body["case_id"] == case_id
    assert "phase" in body and "is_closed" in body
    assert body["escalation_context"]["escalation_reason"] == "needs admin access"
    assert "conversation" in body["escalation_context"]


def test_get_case_escalation_context_null_when_not_escalated(persistent_store):
    llm = MockLLMClient([_proposal()])  # ask_user, no escalation
    app = ITSupportWebServer(
        llm=llm,
        tools={},
        store=persistent_store,
        turn_runner=run_turn,
    ).get_app()
    c = TestClient(app)
    case_id = c.post("/chat", json={"message": "VPN broken"}).json()["case_id"]

    body = c.get(f"/case/{case_id}").json()
    assert body["escalation_context"] is None


def test_chat_returns_graceful_message_when_llm_fails_mid_turn(persistent_store):
    class FailingLLM(BaseLLMClient):
        def call(self, llm_input):
            raise LLMProviderError("LLM model unavailable")

    app = ITSupportWebServer(
        llm=FailingLLM(),
        tools={},
        store=persistent_store,
        turn_runner=run_turn,
    ).get_app()
    c = TestClient(app)
    response = c.post("/chat", json={"message": "VPN broken"})

    assert response.status_code == 200
    body = response.json()
    assert "specialist" in body["message"].lower() or "technical issue" in body["message"].lower()
