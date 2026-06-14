from pathlib import Path
from typing import Any

from agentic_it_support.agent.parser import ProposalParseError
from agentic_it_support.agent.proposals import AgentAction, AgentProposal
from agentic_it_support.config.settings import Settings
from agentic_it_support.llm.client import BaseLLMClient, LLMCallStats, LLMProviderError, MockLLMClient
from agentic_it_support.llm.client import LLMInput
from agentic_it_support.observability.event_tracing import InMemoryEventLog
from agentic_it_support.runtime.turn_runner import run_turn
from agentic_it_support.state.case_state import CaseState, Phase, ToolTrace
from agentic_it_support.tools.base import BaseTool, ToolResult

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def _settings() -> Settings:
    # Point at the repo's real policy data so the business guard can load rules.
    return Settings(_env_file=None, data_dir=_DATA_DIR)


def _run(case, message, llm, tools=None):
    return run_turn(case, message, llm=llm, tools=tools or {}, settings=_settings(), event_log=InMemoryEventLog())


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


# Policy reserves security incidents for a human, so an LLM-proposed escalate is
# authorized and the handoff mechanics can be exercised.
_ESCALATABLE_ISSUE = "my work laptop is infected with malware"


# ── conversation management ───────────────────────────────────────────────────

def test_user_message_appended_to_conversation():
    case = CaseState()
    _run(case, "VPN is broken", MockLLMClient([_proposal()]))
    assert case.conversation[0] == {"role": "user", "content": "VPN is broken"}


def test_ask_user_returns_message():
    case = CaseState()
    response = _run(case, "VPN is broken", MockLLMClient([_proposal(message="What OS?")]))
    assert response == "What OS?"


def test_response_appended_to_conversation():
    case = CaseState()
    _run(case, "VPN is broken", MockLLMClient([_proposal(message="What OS?")]))
    assert case.conversation[-1] == {"role": "assistant", "content": "What OS?"}


def test_ask_user_transitions_to_clarifying():
    case = CaseState()
    _run(case, "VPN broken", MockLLMClient([_proposal(message="What OS?")]))
    assert case.phase == Phase.CLARIFYING


# ── tool call flow ────────────────────────────────────────────────────────────

def test_tool_call_then_resolve_in_one_turn():
    case = CaseState(phase=Phase.INVESTIGATING)
    case.conversation = [{"role": "user", "content": "my vpn keeps timing out"}]
    proposals = [
        _proposal(action=AgentAction.CALL_TOOL, tool_name="kb_search",
                  tool_input={"query": "vpn"}, message=None),
        _proposal(action=AgentAction.RESOLVE, message="Restart VPN client"),
    ]
    tools = {"kb_search": MockTool(ToolResult(success=True, data={"results": []}))}
    response = _run(case, "VPN keeps disconnecting", MockLLMClient(proposals), tools)
    assert response == "Restart VPN client"
    assert case.phase == Phase.RESOLVING


def test_tool_trace_and_counters_recorded():
    case = CaseState(phase=Phase.INVESTIGATING)
    case.conversation = [{"role": "user", "content": "my vpn keeps timing out"}]
    proposals = [
        _proposal(action=AgentAction.CALL_TOOL, tool_name="kb_search",
                  tool_input={"query": "vpn"}, message=None),
        _proposal(action=AgentAction.RESOLVE, message="Fix it"),
    ]
    tools = {"kb_search": MockTool(ToolResult(success=True, data={"results": ["article"]}))}
    _run(case, "VPN broken", MockLLMClient(proposals), tools)
    assert len(case.tool_traces) == 1
    assert case.tool_traces[0].tool_name == "kb_search"
    assert case.tool_calls_this_turn == 1
    assert case.tool_calls_total == 1


def test_failed_tool_trace_has_success_false():
    case = CaseState(phase=Phase.INVESTIGATING)
    case.conversation = [{"role": "user", "content": "my vpn keeps timing out"}]
    proposals = [
        _proposal(action=AgentAction.CALL_TOOL, tool_name="kb_search",
                  tool_input={"query": "vpn"}, message=None),
        _proposal(action=AgentAction.ASK_USER, message="What error do you see?"),
    ]
    tools = {"kb_search": MockTool(ToolResult(success=False, data={}, error="service unavailable"))}
    _run(case, "VPN broken", MockLLMClient(proposals), tools)
    assert case.tool_traces[0].success is False


class _RaisingTool(BaseTool):
    name = "kb_search"
    description = "mock that blows up"

    def run(self, inputs: dict[str, Any]) -> ToolResult:
        raise RuntimeError("disk on fire")


def test_run_turn_survives_a_tool_that_raises():
    # A tool exception must not crash the turn; the agent recovers on the next step.
    case = CaseState(phase=Phase.INVESTIGATING)
    case.conversation = [{"role": "user", "content": "my vpn keeps timing out"}]
    proposals = [
        _proposal(action=AgentAction.CALL_TOOL, tool_name="kb_search",
                  tool_input={"query": "vpn"}, message=None),
        _proposal(action=AgentAction.ASK_USER, message="What error do you see?"),
    ]
    response = _run(case, "VPN broken", MockLLMClient(proposals), {"kb_search": _RaisingTool()})
    assert response == "What error do you see?"
    assert case.tool_traces[0].success is False
    assert case.handoff_completed is False


# ── resolve ───────────────────────────────────────────────────────────────────

def test_proposing_a_resolution_enters_resolving_without_counting_an_attempt():
    # An attempt is only counted when the user later reports the fix failed
    # (user_confirmed_resolution=False); merely proposing a fix does not.
    case = CaseState(phase=Phase.INVESTIGATING)
    case.conversation = [{"role": "user", "content": "my vpn keeps timing out"}]
    case.tool_traces = [ToolTrace(tool_name="kb_search", inputs={}, output={}, success=True)]
    case.confidence = 0.35
    _run(case, "Still broken", MockLLMClient([
        _proposal(action=AgentAction.RESOLVE, message="Try this"),
    ]))
    assert case.phase == Phase.RESOLVING
    assert case.resolution_attempts == 0


def test_resolve_message_passes_through_unchanged():
    case = CaseState(phase=Phase.INVESTIGATING)
    case.conversation = [{"role": "user", "content": "my vpn keeps timing out"}]
    case.tool_traces = [ToolTrace(tool_name="kb_search", inputs={}, output={}, success=True)]
    case.confidence = 0.35
    response = _run(case, "still broken", MockLLMClient([
        _proposal(action=AgentAction.RESOLVE, message="Switch the VPN protocol to TCP."),
    ]))
    assert response == "Switch the VPN protocol to TCP."


# ── escalation / handoff ──────────────────────────────────────────────────────

def test_escalate_sets_handoff_completed():
    case = CaseState(phase=Phase.INVESTIGATING)
    _run(case, _ESCALATABLE_ISSUE, MockLLMClient([
        _proposal(action=AgentAction.ESCALATE, message=None,
                  escalation_reason="Suspected malware needs security review"),
    ]))
    assert case.handoff_completed is True
    # handoff is synchronous, so the case lands terminal (CLOSED), not stuck escalating
    assert case.phase == Phase.CLOSED


def test_escalate_response_is_generic_handoff_message():
    case = CaseState(phase=Phase.INVESTIGATING)
    response = _run(case, _ESCALATABLE_ISSUE, MockLLMClient([
        _proposal(action=AgentAction.ESCALATE, message=None,
                  escalation_reason="internal: malware analysis pipeline triggered"),
    ]))
    assert "specialist" in response.lower()
    # the internal reason must never leak to the user
    assert "malware analysis pipeline" not in response.lower()


def test_escalate_builds_handoff_context():
    case = CaseState(phase=Phase.INVESTIGATING)
    _run(case, _ESCALATABLE_ISSUE, MockLLMClient([
        _proposal(action=AgentAction.ESCALATE, message=None,
                  escalation_reason="Suspected malware needs security review"),
    ]))
    ctx = case.escalation_context
    assert ctx != {}
    assert ctx["internal_reason"] == "Suspected malware needs security review"
    assert "conversation" in ctx
    assert "confidence" in ctx
    assert "tool_traces" in ctx
    assert "resolution_attempts" in ctx


# ── correctable guardrail violations (retry, not instant escalation) ──────────

def test_validation_failure_retries_and_recovers():
    # INTAKE forbids RESOLVE; the agent is re-prompted and recovers with a valid action.
    case = CaseState(phase=Phase.INTAKE)
    bad = _proposal(action=AgentAction.RESOLVE, message="Fix this")
    good = _proposal(action=AgentAction.ASK_USER, message="What OS are you on?")
    response = _run(case, "VPN broken", MockLLMClient([bad, good]))
    assert response == "What OS are you on?"
    assert case.handoff_completed is False
    assert case.escalation_context == {}


def test_policy_block_retries_and_recovers():
    # premature ESCALATE is policy-blocked; agent is re-prompted and recovers
    case = CaseState(phase=Phase.INVESTIGATING)
    case.conversation = [{"role": "user", "content": "my vpn keeps timing out"}]
    blocked = _proposal(action=AgentAction.ESCALATE, message=None, escalation_reason="needs help")
    good = _proposal(action=AgentAction.ASK_USER, message="Which tool times out?")
    response = _run(case, "VPN broken", MockLLMClient([blocked, good]))
    assert response == "Which tool times out?"
    assert case.handoff_completed is False


def test_persistent_guardrail_violation_eventually_escalates():
    # an agent that never produces a valid action falls back to a graceful handoff
    case = CaseState(phase=Phase.INTAKE)
    bad = _proposal(action=AgentAction.RESOLVE, message="Fix this")
    response = _run(case, "VPN broken", MockLLMClient([bad] * 6))
    assert case.handoff_completed is True
    assert "specialist" in response.lower()


def test_corrections_capped_before_consuming_whole_queue():
    case = CaseState(phase=Phase.INTAKE)
    bad = _proposal(action=AgentAction.RESOLVE, message="Fix this")
    llm = MockLLMClient([bad] * 6)
    _run(case, "VPN broken", llm)
    # escalated at the correction cap, before draining all proposals
    assert len(llm._queue) > 0


# ── LLM failure handling ──────────────────────────────────────────────────────

class _FailingLLM(BaseLLMClient):
    def call(self, llm_input: LLMInput) -> AgentProposal:
        raise LLMProviderError("provider down")


class _ParseThenGoodLLM(BaseLLMClient):
    def __init__(self, good: AgentProposal):
        self._good = good
        self.calls = 0

    def call(self, llm_input: LLMInput):
        self.calls += 1
        if self.calls == 1:
            raise ProposalParseError("bad json")
        return self._good, LLMCallStats()


def test_llm_provider_error_escalates_gracefully():
    case = CaseState(phase=Phase.INVESTIGATING)
    response = _run(case, "VPN broken", _FailingLLM())
    assert "specialist" in response.lower()
    assert case.handoff_completed is True
    assert case.escalation_context != {}


def test_llm_provider_error_does_not_raise():
    case = CaseState(phase=Phase.INVESTIGATING)
    try:
        _run(case, "VPN broken", _FailingLLM())
    except Exception as exc:
        raise AssertionError(f"run_turn raised unexpectedly: {exc}") from exc


def test_parse_error_retries_with_correction_and_recovers():
    case = CaseState(phase=Phase.INVESTIGATING)
    good = _proposal(action=AgentAction.ASK_USER, message="What VPN client are you using?")
    response = _run(case, "VPN keeps timing out", _ParseThenGoodLLM(good))
    assert response == "What VPN client are you using?"
    assert case.handoff_completed is False
    assert case.escalation_context == {}


# ── per-turn counters ─────────────────────────────────────────────────────────

def test_tool_calls_this_turn_reset_each_turn():
    case = CaseState(phase=Phase.INVESTIGATING, tool_calls_this_turn=2)
    _run(case, "VPN broken", MockLLMClient([_proposal(action=AgentAction.ASK_USER, message="?")]))
    assert case.tool_calls_this_turn == 0


def test_successful_llm_calls_are_counted():
    case = CaseState(phase=Phase.INVESTIGATING)
    _run(case, "VPN broken", MockLLMClient([_proposal(action=AgentAction.ASK_USER, message="?")]))
    assert case.llm_calls_total == 1
