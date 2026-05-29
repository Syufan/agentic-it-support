from typing import Any

import pytest
from agent.llm import MockLLMClient
from agent.proposals import AgentAction, AgentProposal
from runtime.controller import run_turn
from state.case_state import CaseState, MissingInfoSource, Phase
from tools.base import BaseTool, ToolResult


# ── helpers ───────────────────────────────────────────────────────────────────

class MockTool(BaseTool):
    name = "mock_tool"
    description = "mock"

    def __init__(self, result: ToolResult):
        self._result = result

    def run(self, inputs: dict[str, Any]) -> ToolResult:
        return self._result


def _proposal(**kwargs) -> AgentProposal:
    defaults = {
        "action": AgentAction.ASK_USER,
        "confidence": 0.6,
        "reasoning_summary": "test",
        "message": "What OS?",
    }
    return AgentProposal(**(defaults | kwargs))


# ── conversation management ───────────────────────────────────────────────────

def test_user_message_appended_to_conversation():
    case = CaseState()
    run_turn(case, "VPN is broken", MockLLMClient([_proposal()]), {})
    assert case.conversation[0] == {"role": "user", "content": "VPN is broken"}


def test_response_appended_to_conversation():
    case = CaseState()
    run_turn(case, "VPN is broken", MockLLMClient([_proposal(message="What OS?")]), {})
    assert case.conversation[-1] == {"role": "assistant", "content": "What OS?"}


def test_ask_user_returns_message():
    case = CaseState()
    response = run_turn(case, "VPN is broken", MockLLMClient([_proposal(message="What OS?")]), {})
    assert response == "What OS?"


# ── phase transitions ─────────────────────────────────────────────────────────

def test_phase_transitions_to_clarifying_when_missing_user_info():
    case = CaseState()
    run_turn(case, "VPN broken", MockLLMClient([_proposal(
        missing_info_source=MissingInfoSource.USER,
        missing_info=["OS type"],
    )]), {})
    assert case.phase == Phase.CLARIFYING


def test_phase_transitions_to_investigating_when_no_missing_info():
    case = CaseState()
    run_turn(case, "VPN broken", MockLLMClient([_proposal(
        missing_info_source=MissingInfoSource.NONE,
        missing_info=[],
    )]), {})
    assert case.phase == Phase.INVESTIGATING


def test_phase_transitions_to_resolving_after_high_confidence_resolve():
    case = CaseState(phase=Phase.INVESTIGATING)
    run_turn(case, "Still broken", MockLLMClient([
        _proposal(action=AgentAction.RESOLVE, confidence=0.9, message="Try this"),
    ]), {})
    assert case.phase == Phase.RESOLVING


# ── tool call flow ────────────────────────────────────────────────────────────

def test_tool_call_then_resolve_in_one_turn():
    case = CaseState(phase=Phase.INVESTIGATING)
    case.confidence = 0.6

    proposals = [
        _proposal(
            action=AgentAction.CALL_TOOL,
            confidence=0.6,
            tool_name="kb_search",
            tool_input={"query": "VPN"},
            message=None,
            missing_info_source=MissingInfoSource.TOOL,
        ),
        _proposal(action=AgentAction.RESOLVE, confidence=0.9, message="Restart VPN client"),
    ]
    tool = MockTool(ToolResult(success=True, data={"results": []}))
    response = run_turn(case, "VPN keeps disconnecting", MockLLMClient(proposals), {"kb_search": tool})
    assert response == "Restart VPN client"


def test_tool_trace_recorded():
    case = CaseState(phase=Phase.INVESTIGATING)
    case.confidence = 0.6

    proposals = [
        _proposal(action=AgentAction.CALL_TOOL, confidence=0.6,
                  tool_name="kb_search", tool_input={"query": "VPN"},
                  message=None, missing_info_source=MissingInfoSource.TOOL),
        _proposal(action=AgentAction.RESOLVE, confidence=0.9, message="Fix"),
    ]
    tool = MockTool(ToolResult(success=True, data={"results": ["article"]}))
    run_turn(case, "VPN broken", MockLLMClient(proposals), {"kb_search": tool})

    assert len(case.tool_traces) == 1
    assert case.tool_traces[0].tool_name == "kb_search"
    assert case.tool_traces[0].success is True


def test_tool_counters_incremented():
    case = CaseState(phase=Phase.INVESTIGATING)
    case.confidence = 0.6

    proposals = [
        _proposal(action=AgentAction.CALL_TOOL, confidence=0.6,
                  tool_name="kb_search", tool_input={"query": "vpn"},
                  message=None, missing_info_source=MissingInfoSource.TOOL),
        _proposal(action=AgentAction.RESOLVE, confidence=0.9, message="Fix"),
    ]
    tool = MockTool(ToolResult(success=True, data={}))
    run_turn(case, "VPN broken", MockLLMClient(proposals), {"kb_search": tool})

    assert case.tool_calls_current_investigation == 1
    assert case.tool_calls_total == 1


def test_tool_data_stored_in_facts():
    case = CaseState(phase=Phase.INVESTIGATING)
    case.confidence = 0.6

    proposals = [
        _proposal(action=AgentAction.CALL_TOOL, confidence=0.6,
                  tool_name="kb_search", tool_input={"query": "vpn"},
                  message=None, missing_info_source=MissingInfoSource.TOOL),
        _proposal(action=AgentAction.RESOLVE, confidence=0.9, message="Fix"),
    ]
    tool = MockTool(ToolResult(success=True, data={"results": ["article"]}))
    run_turn(case, "VPN broken", MockLLMClient(proposals), {"kb_search": tool})

    assert "kb_search_result" in case.facts


# ── tool failure handling (P1.5) ─────────────────────────────────────────────

def _failing_proposals() -> list:
    return [
        _proposal(action=AgentAction.CALL_TOOL, confidence=0.6,
                  tool_name="kb_search", tool_input={"query": "vpn"},
                  message=None, missing_info_source=MissingInfoSource.TOOL),
        _proposal(action=AgentAction.RESOLVE, confidence=0.9, message="Fix"),
    ]

def test_failed_tool_trace_has_success_false():
    case = CaseState(phase=Phase.INVESTIGATING)
    case.confidence = 0.6
    tool = MockTool(ToolResult(success=False, data={}, error="service unavailable"))
    run_turn(case, "VPN broken", MockLLMClient(_failing_proposals()), {"kb_search": tool})
    assert case.tool_traces[0].success is False

def test_failed_tool_error_stored_in_facts():
    case = CaseState(phase=Phase.INVESTIGATING)
    case.confidence = 0.6
    tool = MockTool(ToolResult(success=False, data={}, error="service unavailable"))
    run_turn(case, "VPN broken", MockLLMClient(_failing_proposals()), {"kb_search": tool})
    assert "kb_search_error" in case.facts

def test_failed_tool_result_not_stored_in_facts():
    case = CaseState(phase=Phase.INVESTIGATING)
    case.confidence = 0.6
    tool = MockTool(ToolResult(success=False, data={}, error="service unavailable"))
    run_turn(case, "VPN broken", MockLLMClient(_failing_proposals()), {"kb_search": tool})
    assert "kb_search_result" not in case.facts

def test_missing_tool_records_failure_trace():
    case = CaseState(phase=Phase.INVESTIGATING)
    case.confidence = 0.6
    run_turn(case, "VPN broken", MockLLMClient(_failing_proposals()), {})
    assert case.tool_traces[0].success is False

def test_missing_tool_error_stored_in_facts():
    case = CaseState(phase=Phase.INVESTIGATING)
    case.confidence = 0.6
    run_turn(case, "VPN broken", MockLLMClient(_failing_proposals()), {})
    assert "kb_search_error" in case.facts


# ── resolve ───────────────────────────────────────────────────────────────────

def test_resolve_increments_resolution_attempts():
    case = CaseState(phase=Phase.INVESTIGATING)
    run_turn(case, "Still broken", MockLLMClient([
        _proposal(action=AgentAction.RESOLVE, confidence=0.9, message="Try this"),
    ]), {})
    assert case.resolution_attempts == 1


# ── escalate ──────────────────────────────────────────────────────────────────

def test_escalate_sets_handoff_completed():
    case = CaseState(phase=Phase.INVESTIGATING)
    run_turn(case, "VPN broken", MockLLMClient([
        _proposal(action=AgentAction.ESCALATE, confidence=0.3,
                  escalation_reason="Needs admin", message=None),
    ]), {})
    assert case.handoff_completed is True


def _escalate_proposal() -> AgentProposal:
    return _proposal(
        action=AgentAction.ESCALATE,
        confidence=0.3,
        escalation_reason="Needs admin access",
        message=None,
    )

def test_escalate_builds_escalation_context():
    case = CaseState(phase=Phase.INVESTIGATING)
    run_turn(case, "VPN broken", MockLLMClient([_escalate_proposal()]), {})
    assert case.escalation_context != {}
    assert "escalation_reason" in case.escalation_context

def test_escalation_context_includes_confidence():
    case = CaseState(phase=Phase.INVESTIGATING)
    run_turn(case, "VPN broken", MockLLMClient([_escalate_proposal()]), {})
    assert case.escalation_context["confidence"] == 0.3

def test_escalation_context_includes_issue_description():
    case = CaseState(phase=Phase.INVESTIGATING)
    run_turn(case, "VPN keeps disconnecting every 10 minutes", MockLLMClient([_escalate_proposal()]), {})
    assert "issue_description" in case.escalation_context
    assert "VPN keeps disconnecting" in case.escalation_context["issue_description"]

def test_escalation_context_includes_full_conversation():
    case = CaseState(phase=Phase.INVESTIGATING)
    run_turn(case, "VPN broken", MockLLMClient([_escalate_proposal()]), {})
    assert "conversation" in case.escalation_context
    assert isinstance(case.escalation_context["conversation"], list)

def test_escalation_context_includes_facts():
    case = CaseState(phase=Phase.INVESTIGATING)
    case.facts = {"os": "macOS"}
    run_turn(case, "VPN broken", MockLLMClient([_escalate_proposal()]), {})
    assert case.escalation_context["facts"]["os"] == "macOS"

def test_escalation_context_tool_traces_include_output():
    case = CaseState(phase=Phase.INVESTIGATING)
    case.confidence = 0.6
    proposals = [
        _proposal(action=AgentAction.CALL_TOOL, confidence=0.6,
                  tool_name="kb_search", tool_input={"query": "vpn"},
                  message=None, missing_info_source=MissingInfoSource.TOOL),
        _escalate_proposal(),
    ]
    tool = MockTool(ToolResult(success=True, data={"results": ["article"]}))
    run_turn(case, "VPN broken", MockLLMClient(proposals), {"kb_search": tool})
    trace = case.escalation_context["tool_traces"][0]
    assert "output" in trace

def test_escalation_context_includes_resolution_attempts():
    case = CaseState(phase=Phase.INVESTIGATING)
    run_turn(case, "VPN broken", MockLLMClient([_escalate_proposal()]), {})
    assert "resolution_attempts" in case.escalation_context


# ── state projection ──────────────────────────────────────────────────────────

def test_confidence_updated_from_proposal():
    case = CaseState(phase=Phase.INVESTIGATING)
    run_turn(case, "msg", MockLLMClient([
        _proposal(action=AgentAction.RESOLVE, confidence=0.95, message="Fix"),
    ]), {})
    assert case.confidence == 0.95


def test_missing_info_projected_from_proposal():
    case = CaseState()
    run_turn(case, "VPN broken", MockLLMClient([_proposal(
        missing_info_source=MissingInfoSource.USER,
        missing_info=["OS", "VPN version"],
    )]), {})
    assert case.missing_info == ["OS", "VPN version"]
