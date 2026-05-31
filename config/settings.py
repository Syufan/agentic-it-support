"""Deployment / environment configuration.

Only environment-driven, deployment-varying, or tunable knobs live here — NOT
state-machine/domain constants (those are runtime/constants.py). Built once at
the composition root (main.py) and injected into the objects that need it.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Deployment / behaviour choices — no hardcoded default; .env is the source
    # of truth (LLM_API_KEY / LLM_MODEL / LLM_TEMPERATURE). Empty / None means
    # "not provided" — the client omits it and lets the provider default apply.
    llm_api_key: str = ""
    llm_model: str = ""
    llm_temperature: float | None = None

    # Tunable knob — safe default is fine here.
    confidence_retry_penalty: float = 0.15

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
