from datetime import datetime
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


class TraceEventView(BaseModel):
    """One runtime trace event for a case, read back via /case/{id}/trace."""
    event_type: str
    phase: str
    confidence: float
    details: dict[str, Any]
    timestamp: datetime
