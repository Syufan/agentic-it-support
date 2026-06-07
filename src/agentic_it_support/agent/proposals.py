from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class AgentAction(str, Enum):
    ASK_USER = "ask_user"
    CALL_TOOL = "call_tool"
    RESOLVE = "resolve"
    ESCALATE = "escalate"

'''
INTAKE
- ASK_USER → 转 CLARIFYING ✓
- CALL_TOOL → 转 INVESTIGATING ✓

CLARIFYING
- ASK_USER → 留在 CLARIFYING ✓
- CALL_TOOL → 转 INVESTIGATING ✓
- ESCALATE → 直接升级 ✓

INVESTIGATING
- ASK_USER → 转 CLARIFYING ✓
- CALL_TOOL → 留在 INVESTIGATING ✓
- RESOLVE → 转 RESOLVING ✓
- ESCALATE → 直接升级 ✓

'''


class AgentProposal(BaseModel):
    """Structured action proposal returned by the LLM."""

    action: AgentAction

    # User-facing response or question.
    message: str | None = None

    # Missing information the proposal wants to collect.
    missing_info: list[str] = Field(default_factory=list)

    # Tool request fields.
    tool_name: str | None = None
    tool_input: dict[str, Any] = Field(default_factory=dict)

    # LLM interpretation of whether the user confirmed the fix worked.
    user_confirmed_resolution: bool | None = None

    # Human handoff reason.
    escalation_reason: str | None = None