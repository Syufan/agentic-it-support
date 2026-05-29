from typing import Any

from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str
    case_id: str | None = None


class ChatResponse(BaseModel):
    case_id: str
    message: str
    phase: str
    is_closed: bool


class CaseView(BaseModel):
    """Full case snapshot, including the human-handoff package when escalated."""

    case_id: str
    phase: str
    is_closed: bool
    confidence: float
    tool_calls_total: int
    facts: dict[str, Any]
    escalation_context: dict[str, Any] | None
