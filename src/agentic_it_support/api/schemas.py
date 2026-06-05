from typing import Any

from pydantic import BaseModel, field_validator


class ChatRequest(BaseModel):
    message: str
    case_id: str | None = None

    @field_validator("message")
    @classmethod
    def _reject_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message must not be empty")
        return value


class ChatResponse(BaseModel):
    case_id: str
    message: str
    phase: str
    is_closed: bool


class CaseView(BaseModel):
    """Reserved for future dashboard."""
    case_id: str
    phase: str
    is_closed: bool
    confidence: float
    tool_calls_total: int
    escalation_context: dict[str, Any] | None
