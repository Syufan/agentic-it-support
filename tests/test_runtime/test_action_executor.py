from typing import Any

from agentic_it_support.agent.proposals import AgentAction, AgentProposal
from agentic_it_support.config.settings import ConfidenceSettings, RuntimeLimits
from agentic_it_support.runtime.executor.executor import execute
from agentic_it_support.runtime.result import Continue, Escalate, Terminate
from agentic_it_support.state.case_state import CaseState, Phase
from agentic_it_support.tools.base import BaseTool, ToolResult

_LIMITS = RuntimeLimits()
_CONFIDENCE = ConfidenceSettings()


class MockTool(BaseTool):
    name = "kb_search"
    description = "mock"

    def __init__(self, result: ToolResult):
        self._result = result

    def run(self, inputs: dict[str, Any]) -> ToolResult:
        return self._result


def _proposal(**kwargs) -> AgentProposal:
    defaults = {"action": AgentAction.ASK_USER, "message": "What OS?"}
    return AgentProposal(**(defaults | kwargs))


def _execute(case, proposal, tools=None):
    return execute(case, proposal, tools or {}, runtime_limits=_LIMITS, confidence_settings=_CONFIDENCE)


# ── CALL_TOOL ─────────────────────────────────────────────────────────────────

def test_call_tool_returns_continue():
    case = CaseState(phase=Phase.INVESTIGATING)
    tools = {"kb_search": MockTool(ToolResult(success=True, data={"hit": 1}))}
    proposal = _proposal(action=AgentAction.CALL_TOOL, tool_name="kb_search",
                         tool_input={"query": "vpn"}, message=None)
    assert isinstance(_execute(case, proposal, tools), Continue)


def test_call_tool_records_trace_and_counters():
    case = CaseState(phase=Phase.INVESTIGATING)
    tools = {"kb_search": MockTool(ToolResult(success=True, data={"hit": 1}))}
    proposal = _proposal(action=AgentAction.CALL_TOOL, tool_name="kb_search",
                         tool_input={"query": "vpn"}, message=None)
    _execute(case, proposal, tools)
    assert len(case.tool_traces) == 1
    assert case.tool_traces[0].tool_name == "kb_search"
    assert case.tool_traces[0].success is True
    assert case.tool_calls_this_turn == 1
    assert case.tool_calls_total == 1


def test_call_tool_resets_clarification_attempts():
    case = CaseState(phase=Phase.INVESTIGATING, clarification_attempts=2)
    tools = {"kb_search": MockTool(ToolResult(success=True, data={}))}
    proposal = _proposal(action=AgentAction.CALL_TOOL, tool_name="kb_search",
                         tool_input={"query": "vpn"}, message=None)
    _execute(case, proposal, tools)
    assert case.clarification_attempts == 0


def test_call_tool_recomputes_confidence_from_evidence():
    case = CaseState(phase=Phase.INVESTIGATING)
    tools = {"kb_search": MockTool(ToolResult(success=True, data={}))}
    proposal = _proposal(action=AgentAction.CALL_TOOL, tool_name="kb_search",
                         tool_input={"query": "vpn"}, message=None)
    _execute(case, proposal, tools)
    # one successful source -> resolve_threshold (0.35)
    assert case.confidence == 0.35


def test_failed_tool_records_failure_trace():
    case = CaseState(phase=Phase.INVESTIGATING)
    tools = {"kb_search": MockTool(ToolResult(success=False, data={}, error="boom"))}
    proposal = _proposal(action=AgentAction.CALL_TOOL, tool_name="kb_search",
                         tool_input={"query": "vpn"}, message=None)
    _execute(case, proposal, tools)
    assert case.tool_traces[0].success is False
    assert case.confidence == 0.0


# ── ASK_USER ──────────────────────────────────────────────────────────────────

def test_ask_user_terminates_with_message():
    case = CaseState(phase=Phase.INTAKE)
    outcome = _execute(case, _proposal(action=AgentAction.ASK_USER, message="What OS?"))
    assert isinstance(outcome, Terminate)
    assert outcome.message == "What OS?"


def test_ask_user_appends_assistant_message_and_transitions():
    case = CaseState(phase=Phase.INTAKE)
    _execute(case, _proposal(action=AgentAction.ASK_USER, message="What OS?"))
    assert case.conversation[-1] == {"role": "assistant", "content": "What OS?"}
    assert case.phase == Phase.CLARIFYING


def test_ask_user_below_limit_asks_and_increments():
    # One below the cap: ask normally, bump the counter, stay in clarifying.
    case = CaseState(phase=Phase.CLARIFYING,
                     clarification_attempts=_LIMITS.max_clarification_attempts - 1)
    outcome = _execute(case, _proposal(action=AgentAction.ASK_USER, message="What OS?"))
    assert isinstance(outcome, Terminate)
    assert outcome.message == "What OS?"
    assert case.phase == Phase.CLARIFYING
    assert case.clarification_attempts == _LIMITS.max_clarification_attempts
    assert case.escalation_context == {}


def test_ask_user_soft_closes_at_clarification_limit():
    # At the cap, another ask_user soft-closes instead of asking again: the case lands
    # CLOSED with a closing message, and it is NOT an escalation.
    case = CaseState(phase=Phase.CLARIFYING,
                     clarification_attempts=_LIMITS.max_clarification_attempts)
    outcome = _execute(case, _proposal(action=AgentAction.ASK_USER, message="What OS?"))
    assert isinstance(outcome, Terminate)
    assert case.phase == Phase.CLOSED
    # soft-close delivers the closing copy, not the agent's clarifying question
    assert outcome.message != "What OS?"
    assert case.conversation[-1] == {"role": "assistant", "content": outcome.message}
    # soft-close is a non-case abort, not a human handoff
    assert case.escalation_context == {}


# ── RESOLVE ───────────────────────────────────────────────────────────────────

def test_resolve_terminates_and_enters_resolving():
    case = CaseState(phase=Phase.INVESTIGATING)
    outcome = _execute(case, _proposal(action=AgentAction.RESOLVE, message="Restart it"))
    assert isinstance(outcome, Terminate)
    assert outcome.message == "Restart it"
    assert case.phase == Phase.RESOLVING


def test_resolve_escalates_when_attempts_exhausted():
    case = CaseState(phase=Phase.INVESTIGATING, resolution_attempts=_LIMITS.max_resolution_attempts)
    outcome = _execute(case, _proposal(action=AgentAction.RESOLVE, message="Restart it"))
    assert isinstance(outcome, Escalate)


# ── ESCALATE ──────────────────────────────────────────────────────────────────

def test_escalate_returns_escalate_with_reason():
    case = CaseState(phase=Phase.INVESTIGATING)
    outcome = _execute(case, _proposal(action=AgentAction.ESCALATE, message=None,
                                       escalation_reason="needs human"))
    assert isinstance(outcome, Escalate)
    assert outcome.reason == "needs human"


# ── resolution confirmation sync ──────────────────────────────────────────────

def test_disconfirmed_resolution_increments_attempts_and_penalizes_confidence():
    # A disconfirmed resolution is synced before the action runs; ESCALATE is used here
    # because it leaves the synced flag untouched (ASK_USER deliberately clears it).
    case = CaseState(phase=Phase.RESOLVING, confidence=0.7, resolution_attempts=0)
    proposal = _proposal(action=AgentAction.ESCALATE, message=None,
                         escalation_reason="fix did not work",
                         user_confirmed_resolution=False)
    _execute(case, proposal)
    assert case.user_confirmed_resolution is False
    assert case.resolution_attempts == 1
    # no successful tool sources, so the retry penalty drives confidence to 0
    assert case.confidence == 0.0
