import pytest
from fastapi.testclient import TestClient

from agentic_it_support.api.server import ITSupportWebServer
from agentic_it_support.observability.event_tracing import InMemoryEventLog, record_turn_start
from agentic_it_support.state.case_state import CaseState
from agentic_it_support.state.session import SessionStore


def _runner(responses: list[str] | None = None):
    queue = list(responses or ["What OS are you using?"])

    def run(case, user_message):
        return queue.pop(0) if queue else "handled"

    return run


@pytest.fixture
def client():
    app = ITSupportWebServer(
        llm=None,
        tools={},
        store=SessionStore(),
        turn_runner=_runner(),
        event_log=InMemoryEventLog(),
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
        event_log=InMemoryEventLog(),
    ).get_app()
    return TestClient(app)


def _client_with_log(event_log: InMemoryEventLog) -> TestClient:
    app = ITSupportWebServer(
        llm=None,
        tools={},
        store=SessionStore(),
        turn_runner=_runner(),
        event_log=event_log,
    ).get_app()
    return TestClient(app)


def test_trace_returns_recorded_events_for_case():
    log = InMemoryEventLog()
    record_turn_start(log, CaseState(case_id="trace-1"), "vpn down")
    resp = _client_with_log(log).get("/case/trace-1/trace")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 1
    assert body[0]["event_type"] == "turn_start"
    assert body[0]["details"]["user_message"] == "vpn down"


def test_trace_is_empty_for_unknown_case():
    resp = _client_with_log(InMemoryEventLog()).get("/case/nope/trace")
    assert resp.status_code == 200
    assert resp.json() == []


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



