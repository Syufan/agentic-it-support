import pytest
from fastapi.testclient import TestClient

from agentic_it_support.api.server import ITSupportWebServer
from agentic_it_support.state.session import SessionStore


def _runner(responses: list[str] | None = None):
    queue = list(responses or ["What OS are you using?"])

    def run(case, user_message, llm, tools):
        return queue.pop(0) if queue else "handled"

    return run


@pytest.fixture
def client():
    app = ITSupportWebServer(
        llm=None,
        tools={},
        store=SessionStore(),
        turn_runner=_runner(),
    ).get_app()
    return TestClient(app)


@pytest.fixture
def persistent_store():
    return SessionStore()


def _client(store: SessionStore, responses: list[str] | None = None) -> TestClient:
    app = ITSupportWebServer(
        llm=None,
        tools={},
        store=store,
        turn_runner=_runner(responses),
    ).get_app()
    return TestClient(app)


def test_health_returns_ok(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_chat_returns_200(client):
    response = client.post("/chat", json={"message": "VPN is broken"})

    assert response.status_code == 200


def test_chat_returns_case_id(client):
    response = client.post("/chat", json={"message": "VPN is broken"})

    assert response.json()["case_id"]


def test_chat_returns_message(client):
    response = client.post("/chat", json={"message": "VPN is broken"})

    assert response.json()["message"] == "What OS are you using?"


def test_chat_returns_phase(client):
    response = client.post("/chat", json={"message": "VPN is broken"})

    assert "phase" in response.json()


def test_chat_returns_is_closed(client):
    response = client.post("/chat", json={"message": "VPN is broken"})

    assert response.json()["is_closed"] is False


def test_chat_rejects_empty_message(client):
    assert client.post("/chat", json={"message": ""}).status_code == 422


def test_chat_rejects_whitespace_only_message(client):
    assert client.post("/chat", json={"message": "   "}).status_code == 422


def test_chat_continues_existing_case(persistent_store):
    client = _client(persistent_store, ["What OS?", "Got it, checking now."])
    first = client.post("/chat", json={"message": "I'm having some trouble"})
    case_id = first.json()["case_id"]

    second = client.post(
        "/chat",
        json={"message": "not sure how to explain it", "case_id": case_id},
    )

    assert second.json()["case_id"] == case_id
    assert second.json()["message"] == "Got it, checking now."


def test_chat_unknown_case_id_returns_404(persistent_store):
    client = _client(persistent_store)
    response = client.post(
        "/chat",
        json={"message": "VPN broken", "case_id": "nonexistent-id"},
    )

    assert response.status_code == 404


def test_get_case_returns_404_for_unknown_id(client):
    assert client.get("/case/nonexistent-id").status_code == 404

