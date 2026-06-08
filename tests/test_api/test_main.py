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


def test_main_wires_real_turn_runner_into_chat(monkeypatch):
    captured = {}

    def fake_run_turn(case, user_message, *, llm, tools, settings, event_log):
        captured["user_message"] = user_message
        captured["llm"] = llm
        captured["tools"] = tools
        captured["settings"] = settings
        captured["event_log"] = event_log
        return "handled by runtime"

    import agentic_it_support.runtime.turn_runner as turn_runner_module

    monkeypatch.setattr(turn_runner_module, "run_turn", fake_run_turn)

    main_module = _load_main(monkeypatch)
    client = TestClient(main_module.app)

    response = client.post("/chat", json={"message": "VPN is broken"})

    assert response.status_code == 200
    assert response.json()["message"] == "handled by runtime"
    # main must inject llm, tools, settings and the event log into run_turn via partial
    assert captured["user_message"] == "VPN is broken"
    assert captured["llm"] is not None
    assert captured["settings"] is not None
    assert captured["event_log"] is not None
