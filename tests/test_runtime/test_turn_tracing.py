from pathlib import Path
from typing import Any

from agentic_it_support.agent.proposals import AgentAction, AgentProposal
from agentic_it_support.config.settings import Settings
from agentic_it_support.llm.client import MockLLMClient
from agentic_it_support.observability.event_tracing import InMemoryEventLog
from agentic_it_support.runtime.turn_runner import run_turn
from agentic_it_support.state.case_state import CaseState, Phase
from agentic_it_support.tools.base import BaseTool, ToolResult

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, data_dir=_DATA_DIR, **overrides)


class MockTool(BaseTool):
    name = "kb_search"
    description = "mock"

    def __init__(self, result: ToolResult):
        self._result = result

    def run(self, inputs: dict[str, Any]) -> ToolResult:
        return self._result


def _proposal(**kwargs) -> AgentProposal:
    return AgentProposal(**({"action": AgentAction.ASK_USER, "message": "What OS?"} | kwargs))


def _types(log: InMemoryEventLog, case_id: str) -> list[str]:
    return [e.event_type for e in log.get_events_for_case(case_id)]


def test_turn_brackets_emit_start_with_input_and_end_with_outcome():
    log = InMemoryEventLog()
    case = CaseState()
    run_turn(case, "VPN broken", llm=MockLLMClient([_proposal(message="What OS?")]),
             tools={}, settings=_settings(), event_log=log)

    events = log.get_events_for_case(case.case_id)
    assert events[0].event_type == "turn_start"
    assert events[0].details["user_message"] == "VPN broken"
    assert events[-1].event_type == "turn_end"
    assert events[-1].phase == "clarifying"  # the phase is the outcome (an ask)
    assert events[-1].details["agent_reply"] == "What OS?"


def test_tool_turn_emits_llm_tool_guard_and_phase_events():
    log = InMemoryEventLog()
    case = CaseState(phase=Phase.INVESTIGATING)
    case.conversation = [{"role": "user", "content": "my vpn keeps timing out"}]
    proposals = [
        _proposal(action=AgentAction.CALL_TOOL, tool_name="kb_search", tool_input={"query": "vpn"}, message=None),
        _proposal(action=AgentAction.RESOLVE, message="Restart the VPN client"),
    ]
    tools = {"kb_search": MockTool(ToolResult(success=True, data={"results": []}))}
    run_turn(case, "VPN keeps disconnecting", llm=MockLLMClient(proposals), tools=tools,
             settings=_settings(), event_log=log)

    types = _types(log, case.case_id)
    assert {"turn_start", "llm_call", "guard", "tool_start", "tool_end", "phase_transition", "turn_end"} <= set(types)

    tool_end = next(e for e in log.get_events_for_case(case.case_id) if e.event_type == "tool_end")
    assert tool_end.details["success"] is True
    assert tool_end.details["output"] == {"results": []}  # the tool's actual result is recorded
    assert tool_end.details["conf_before"] == 0.0  # confidence before the first successful source

    llm_call = next(e for e in log.get_events_for_case(case.case_id) if e.event_type == "llm_call")
    assert llm_call.details["proposed_action"] == "call_tool"
    assert llm_call.details["latency_ms"] >= 0.0


def test_guard_retry_is_traced_with_reason():
    log = InMemoryEventLog()
    case = CaseState(phase=Phase.INTAKE)  # INTAKE forbids RESOLVE -> guard retry
    bad = _proposal(action=AgentAction.RESOLVE, message="Fix this")
    good = _proposal(action=AgentAction.ASK_USER, message="What OS?")
    run_turn(case, "VPN broken", llm=MockLLMClient([bad, good]), tools={},
             settings=_settings(), event_log=log)

    guards = [e for e in log.get_events_for_case(case.case_id) if e.event_type == "guard"]
    verdicts = [g.details["verdict"] for g in guards]
    assert "retry" in verdicts
    retry = next(g for g in guards if g.details["verdict"] == "retry")
    assert retry.details["reason"]  # a correction string is attached


def test_escalation_emits_escalation_handoff_and_escalated_outcome(tmp_path):
    log = InMemoryEventLog()
    case = CaseState(phase=Phase.INVESTIGATING)
    run_turn(case, "my work laptop is infected with malware",
             llm=MockLLMClient([_proposal(action=AgentAction.ESCALATE, message=None,
                                          escalation_reason="Suspected malware needs security review")]),
             tools={}, settings=_settings(handoff_output_dir=tmp_path), event_log=log)

    types = _types(log, case.case_id)
    assert "escalation" in types
    assert "handoff_written" in types
    assert types[-1] == "turn_end"
    assert log.get_events_for_case(case.case_id)[-1].phase == "escalating"  # phase conveys the outcome


def test_soft_close_traces_phase_transition_to_closed():
    # Clarification budget exhausted -> soft-close records the transition to closed
    # (action="soft_close"), which explains the jump from clarifying to closed.
    log = InMemoryEventLog()
    settings = _settings()
    case = CaseState(phase=Phase.CLARIFYING)
    case.clarification_attempts = settings.limits.max_clarification_attempts
    run_turn(case, "i don't know", llm=MockLLMClient([_proposal(message="Can you be more specific?")]),
             tools={}, settings=settings, event_log=log)

    transitions = [e for e in log.get_events_for_case(case.case_id) if e.event_type == "phase_transition"]
    assert any(e.details["action"] == "soft_close" and e.details["to_phase"] == "closed" for e in transitions)
    assert case.phase == Phase.CLOSED


def test_no_event_log_is_safe():
    # The default event_log=None must record nothing and never raise.
    case = CaseState()
    run_turn(case, "VPN broken", llm=MockLLMClient([_proposal(message="?")]),
             tools={}, settings=_settings())
