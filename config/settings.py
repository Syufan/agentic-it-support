"""Environment-driven runtime settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # LLM provider settings from environment or .env.
    llm_api_key: str = ""
    llm_model: str = ""
    llm_temperature: float | None = None

    # Confidence penalty after failed resolution attempts.
    confidence_retry_penalty: float = 0.15

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")