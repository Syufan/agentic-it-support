from config.settings import Settings

_ENV_VARS = (
    "LLM_API_KEY",
    "LLM_MODEL",
    "LLM_TEMPERATURE",
    "CONFIDENCE_RETRY_PENALTY",
    "LLM_PROMPT_COST_PER_1K",
    "LLM_COMPLETION_COST_PER_1K",
)


def test_defaults(monkeypatch):
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    s = Settings(_env_file=None)
    assert s.llm_api_key == ""
    assert s.llm_model == ""
    assert s.llm_temperature is None
    assert s.confidence_retry_penalty == 0.15
    assert s.llm_prompt_cost_per_1k == 0.00015
    assert s.llm_completion_cost_per_1k == 0.0006


def test_env_override(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "custom-model")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.9")
    monkeypatch.setenv("CONFIDENCE_RETRY_PENALTY", "0.3")
    s = Settings(_env_file=None)
    assert s.llm_model == "custom-model"
    assert s.llm_temperature == 0.9
    assert s.confidence_retry_penalty == 0.3
