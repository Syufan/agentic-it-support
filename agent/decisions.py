from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from state.case_state import MissingInfoSource


class Action(str, Enum):
    ASK_USER = "ask_user"
    CALL_TOOL = "call_tool"
    RESOLVE = "resolve"
    ESCALATE = "escalate"


class AgentDecision(BaseModel):
    action: Action
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning_summary: str

    # User communication
    message: str | None = None

    # Missing info signals → CaseState.missing_info_source / missing_info
    missing_info_source: MissingInfoSource = MissingInfoSource.NONE
    missing_info: list[str] = Field(default_factory=list)

    # Tool call
    tool_name: str | None = None
    tool_input: dict[str, Any] = Field(default_factory=dict)

    # Transition flags → CaseState
    has_safe_low_risk_guidance: bool = False
    new_critical_fact_added: bool = False
    user_confirmed_resolution: bool | None = None

    # Escalation
    escalation_reason: str | None = None
