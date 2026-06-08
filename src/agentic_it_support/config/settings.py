from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_DATA_DIR = Path("data")
_ENV_FILES = (".env", "../.env")
DEFAULT_HANDOFF_OUTPUT_DIR = Path("output/handoffs")
DEFAULT_TRACE_OUTPUT_DIR = Path("output/traces")

class RuntimeLimits(BaseModel):
    max_inner_iterations: int = Field(6, gt=0, le=20)
    max_tool_calls_per_turn: int = Field(3, gt=0, le=20)
    max_tool_calls_per_case: int = Field(6, gt=0, le=50)
    max_llm_calls_per_case: int = Field(12, gt=0, le=50)
    max_clarification_attempts: int = Field(3, gt=0, le=10)
    max_context_messages: int = Field(30, gt=0, le=200)
    max_corrections: int = Field(3, gt=0, le=10)
    max_resolution_attempts: int = Field(2, gt=0, le=10)

class ConfidenceSettings(BaseModel):
    resolve_threshold: float = Field(0.35, ge=0.0, le=1.0)
    high_threshold: float = Field(0.7, ge=0.0, le=1.0)
    retry_penalty: float = Field(0.15, ge=0.0, le=1.0)

class ContextSettings(BaseModel):
    max_tool_traces: int = Field(3, gt=0, le=20)
    tool_output_preview_chars: int = Field(1000, gt=100, le=10000)

class Settings(BaseSettings):
    # API server settings
    api_host: str = "0.0.0.0"
    api_port: int = Field(default=8000, gt=0)

    # LLM provider settings from environment or .env
    llm_api_key: str = ""
    llm_model: str = ""
    llm_temperature: float | None = None

    # Tunable runtime ceilings
    limits: RuntimeLimits = RuntimeLimits()

    # Confidence scoring thresholds and penalties
    confidence: ConfidenceSettings = ConfidenceSettings()

    # LLM message context projection settings
    message_context: ContextSettings = ContextSettings()

    # Repository-backed mock data
    data_dir: Path = DEFAULT_DATA_DIR

    # Business policy file path relative to data_dir
    policy_file: Path = Path("policies/policies.json")

    # Local handoff payload output
    handoff_output_dir: Path = DEFAULT_HANDOFF_OUTPUT_DIR

    # Local event-trace output, written when a case closes
    trace_output_dir: Path = DEFAULT_TRACE_OUTPUT_DIR

    # Observability settings
    event_log_capacity: int = Field(default=1000, gt=0)

    # env_nested_delimiter
    model_config = SettingsConfigDict(env_file=_ENV_FILES, extra="ignore", env_nested_delimiter="__")
