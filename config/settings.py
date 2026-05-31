"""Deployment / environment configuration.

Only environment-driven, deployment-varying, or tunable knobs live here — NOT
state-machine/domain constants (those are runtime/constants.py). Built once at
the composition root (main.py) and injected into the objects that need it.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Deployment choices — no hardcoded default; .env (LLM_API_KEY / LLM_MODEL)
    # is the source of truth. Empty means "must be provided to make real calls".
    llm_api_key: str = ""
    llm_model: str = ""

    # Tunable knobs / list prices — safe defaults are fine here.
    confidence_retry_penalty: float = 0.15
    llm_prompt_cost_per_1k: float = 0.00015
    llm_completion_cost_per_1k: float = 0.0006

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
