import importlib
import sys

from fastapi.testclient import TestClient


def _load_main(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    sys.modules.pop("agentic_it_support.main", None)
    return importlib.import_module("agentic_it_support.main")


def test_main_app_health_check(monkeypatch):
    main_module = _load_main(monkeypatch)
    client = TestClient(main_module.app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_main_chat_uses_stub_turn_runner(monkeypatch):
    main_module = _load_main(monkeypatch)
    client = TestClient(main_module.app)

    response = client.post("/chat", json={"message": "VPN is broken"})

    assert response.status_code == 200
    assert response.json()["message"] == "API is wired. Runtime is temporarily disabled."
