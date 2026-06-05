from fastapi.testclient import TestClient

from agentic_it_support.api.server import ITSupportWebServer
from agentic_it_support.state.session import SessionStore


def _runner(message: str = "handled"):
    def run(case, user_message):
        return message

    return run


def _client(store: SessionStore | None = None, runner=None) -> TestClient:
    app = ITSupportWebServer(
        llm=None,
        tools={},
        store=store or SessionStore(),
        turn_runner=runner or _runner(),
    ).get_app()
    return TestClient(app)


def test_chat_creates_case_when_no_case_id():
    response = _client().post("/chat", json={"message": "VPN broken"})

    assert response.status_code == 200
    assert response.json()["case_id"]


def test_chat_continues_existing_case_by_id():
    client = _client(runner=_runner("next response"))
    first = client.post("/chat", json={"message": "VPN broken"})
    case_id = first.json()["case_id"]

    second = client.post("/chat", json={"case_id": case_id, "message": "macOS"})

    assert second.status_code == 200
    assert second.json()["case_id"] == case_id
    assert second.json()["message"] == "next response"


def test_chat_returns_404_for_unknown_case_id():
    response = _client().post(
        "/chat",
        json={"case_id": "missing", "message": "VPN broken"},
    )

    assert response.status_code == 404


def test_case_view_returns_404_for_unknown_case_id():
    response = _client().get("/case/missing")

    assert response.status_code == 404
