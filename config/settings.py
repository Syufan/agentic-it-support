"""Deployment / environment configuration.

Only environment-driven, deployment-varying, or tunable knobs live here — NOT
state-machine/domain constants (those are runtime/constants.py). Built once at
the composition root (main.py) and injected into the objects that need it.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    llm_api_key: str = ""
    llm_model: str = "gpt-4o-mini-2024-07-18"

    confidence_retry_penalty: float = 0.15

    llm_prompt_cost_per_1k: float = 0.00015
    llm_completion_cost_per_1k: float = 0.0006

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
