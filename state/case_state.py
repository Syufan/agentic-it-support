from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class Phase(str, Enum):
    INTAKE = "intake"
    CLARIFYING = "clarifying"
    INVESTIGATING = "investigating"
    RESOLVING = "resolving"
    ESCALATING = "escalating"
    CLOSED = "closed"


class BudgetMode(str, Enum):
    MAIN = "main"
    RETRY = "retry"
    EXCEPTION = "exception"


@dataclass
class ToolTrace:
    tool_name: str
    inputs: dict[str, Any]
    output: Any
    success: bool
    budget_mode: BudgetMode
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class CaseState:
    case_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Phase & budget
    phase: Phase = Phase.INTAKE
    budget_mode: BudgetMode = BudgetMode.MAIN
    tool_calls_current_investigation: int = 0
    tool_calls_total: int = 0

    # Confidence & missing info
    confidence: float = 0.0
    missing_info: list[str] = field(default_factory=list)
    clarification_attempts: int = 0  # consecutive clarifying turns without progress

    # Resolution control
    resolution_attempts: int = 0
    exception_used: bool = False
    has_safe_low_risk_guidance: bool = False   # T8 vs T9
    new_critical_fact_added: bool = False      # T12 vs T13
    handoff_completed: bool = False            # T14
    user_confirmed_resolution: bool | None = None  # T10 vs T11

    # Conversation memory
    conversation: list[dict[str, str]] = field(default_factory=list)

    # Investigation state
    facts: dict[str, Any] = field(default_factory=dict)
    hypotheses: list[str] = field(default_factory=list)
    checked_sources: list[str] = field(default_factory=list)

    # Tool history
    tool_traces: list[ToolTrace] = field(default_factory=list)

    # Resolution tracking
    failed_resolutions: list[str] = field(default_factory=list)

    # Escalation
    escalation_context: dict[str, Any] = field(default_factory=dict)
