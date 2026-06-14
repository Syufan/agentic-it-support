from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Phase(str, Enum):
    INTAKE = "intake"
    CLARIFYING = "clarifying"
    INVESTIGATING = "investigating"
    RESOLVING = "resolving"
    ESCALATING = "escalating"
    CLOSED = "closed"


@dataclass
class ToolTrace:
    tool_name: str
    inputs: dict[str, Any]
    output: Any
    success: bool


@dataclass
class CaseState:
    # Identity
    case_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Workflow
    phase: Phase = Phase.INTAKE

    # Runtime limits
    tool_calls_this_turn: int = 0
    tool_calls_total: int = 0
    llm_calls_total: int = 0
    total_tokens: int = 0
    clarification_attempts: int = 0

    # Confidence
    confidence: float = 0.0

    # Resolution control
    resolution_attempts: int = 0
    handoff_completed: bool = False
    user_confirmed_resolution: bool | None = None

    # Conversation
    conversation: list[dict[str, str]] = field(default_factory=list)

    # Tool history
    tool_traces: list[ToolTrace] = field(default_factory=list)

    # Escalation
    escalation_context: dict[str, Any] = field(default_factory=dict)

    def add_user_message(self, content: str) -> None:
        self.conversation.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        self.conversation.append({"role": "assistant", "content": content})
