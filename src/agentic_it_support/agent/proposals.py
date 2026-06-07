from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class AgentAction(str, Enum):
    ASK_USER = "ask_user"
    CALL_TOOL = "call_tool"
    RESOLVE = "resolve"
    ESCALATE = "escalate"


class AgentProposal(BaseModel):
    """Structured action proposal returned by the LLM."""

    action: AgentAction

    # User-facing response or question.
    message: str | None = None

    # Tool request fields.
    tool_name: str | None = None
    tool_input: dict[str, Any] = Field(default_factory=dict)

    # LLM interpretation of whether the user confirmed the fix worked.
    user_confirmed_resolution: bool | None = None

    # Human handoff reason.
    escalation_reason: str | None = None