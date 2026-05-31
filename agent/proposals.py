from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class AgentAction(str, Enum):
    ASK_USER = "ask_user"
    CALL_TOOL = "call_tool"
    RESOLVE = "resolve"
    ESCALATE = "escalate"


class AgentProposal(BaseModel):
    """Structured proposal returned by the LLM each turn.

    The runtime validates this object, projects selected fields into CaseState,
    and then evaluates deterministic transition rules.
    The LLM proposes; the Runtime decides and executes.
    """

    action: AgentAction
    reasoning_summary: str

    # User communication
    message: str | None = None

    # Missing info description (the runtime derives the *source* from `action`).
    missing_info: list[str] = Field(default_factory=list)

    # Tool call
    tool_name: str | None = None
    tool_input: dict[str, Any] = Field(default_factory=dict)

    # The LLM's reading of the user's reply (the only proposal-carried signal the
    # runtime still trusts, because interpreting natural language is the model's job).
    user_confirmed_resolution: bool | None = None

    # Escalation
    escalation_reason: str | None = None
