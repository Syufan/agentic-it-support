import pytest
from pydantic import ValidationError

from agentic_it_support.config.settings import RuntimeLimits, Settings

_ENV_VARS = (
    "API_HOST",
    "API_PORT",
    "LLM_API_KEY",
    "LLM_MODEL",
    "LLM_TEMPERATURE",
    "CONFIDENCE_RETRY_PENALTY",
    "EVENT_LOG_CAPACITY",
)


def test_defaults(monkeypatch):
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    s = Settings(_env_file=None)
    assert s.api_host == "0.0.0.0"
    assert s.api_port == 8000
    assert s.llm_api_key == ""
    assert s.llm_model == ""
    assert s.llm_temperature is None
    assert s.confidence_retry_penalty == 0.15
    assert s.event_log_capacity == 1000


def test_env_override(monkeypatch):
    monkeypatch.setenv("API_HOST", "127.0.0.1")
    monkeypatch.setenv("API_PORT", "9000")
    monkeypatch.setenv("LLM_MODEL", "custom-model")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.9")
    monkeypatch.setenv("CONFIDENCE_RETRY_PENALTY", "0.3")
    monkeypatch.setenv("EVENT_LOG_CAPACITY", "200")
    s = Settings(_env_file=None)
    assert s.api_host == "127.0.0.1"
    assert s.api_port == 9000
    assert s.llm_model == "custom-model"
    assert s.llm_temperature == 0.9
    assert s.confidence_retry_penalty == 0.3
    assert s.event_log_capacity == 200


def test_runtime_limits_defaults():
    limits = Settings(_env_file=None).limits
    assert limits.max_inner_iterations == 6
    assert limits.max_tool_calls_per_turn == 3
    assert limits.max_tool_calls_per_case == 6
    assert limits.max_llm_calls_per_case == 12
    assert limits.max_clarification_attempts == 3
    assert limits.max_context_messages == 30
    assert limits.max_corrections == 3


def test_runtime_limits_env_override(monkeypatch):
    monkeypatch.setenv("LIMITS__MAX_INNER_ITERATIONS", "8")
    monkeypatch.setenv("LIMITS__MAX_TOOL_CALLS_PER_TURN", "5")
    s = Settings(_env_file=None)
    assert s.limits.max_inner_iterations == 8
    assert s.limits.max_tool_calls_per_turn == 5
    # untouched fields keep their defaults
    assert s.limits.max_llm_calls_per_case == 12


@pytest.mark.parametrize("value", [0, -1, 21])
def test_runtime_limits_reject_out_of_bounds(value):
    with pytest.raises(ValidationError):
        RuntimeLimits(max_inner_iterations=value)
