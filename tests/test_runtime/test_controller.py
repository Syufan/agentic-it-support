from typing import Any

import pytest
from llm.client import BaseLLMClient, LLMProviderError, MockLLMClient
from agent.proposals import AgentAction, AgentProposal
from runtime.controller import TurnCancelled, run_turn
from runtime.message_builder import LLMInput
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


def _case_after_clarification(phase: Phase = Phase.INVESTIGATING) -> CaseState:
    """Case that already has one prior user turn, satisfying the high-confidence resolve policy."""
    case = CaseState(phase=phase)
    case.conversation = [{"role": "user", "content": "VPN broken"}]
    return case


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


def test_vague_initial_greeting_asks_for_issue_without_llm():
    case = CaseState()
    response = run_turn(case, "hey", MockLLMClient([]), {})
    assert "what it issue" in response.lower()
    assert case.phase == Phase.CLARIFYING
    assert case.missing_info == ["issue description"]
    assert case.tool_calls_total == 0


def test_vague_initial_greeting_with_punctuation_is_caught():
    response = run_turn(CaseState(), "hey!", MockLLMClient([]), {})
    assert "what it issue" in response.lower()


def test_short_known_issue_still_goes_to_llm():
    case = CaseState()
    response = run_turn(case, "VPN broken", MockLLMClient([
        _proposal(message="What OS?"),
    ]), {})
    assert response == "What OS?"


def test_short_symptom_phrase_still_goes_to_llm():
    case = CaseState()
    response = run_turn(case, "locked out", MockLLMClient([
        _proposal(message="Which account are you locked out of?"),
    ]), {})
    assert response == "Which account are you locked out of?"


# ── clarification loop cap ─────────────────────────────────────────────────────

def _clarify_ask() -> AgentProposal:
    return _proposal(action=AgentAction.ASK_USER, message="Please describe the issue",
                     missing_info_source=MissingInfoSource.USER, missing_info=["issue description"])


def test_repeated_unproductive_clarifying_soft_closes_without_handoff():
    # user never provides a usable issue: there is nothing to diagnose or hand off,
    # so the case should soft-close (no escalation), not be routed to a specialist.
    case = CaseState(phase=Phase.CLARIFYING)
    case.conversation = [{"role": "user", "content": "hey"}]
    llm = MockLLMClient([_clarify_ask() for _ in range(8)])
    response = ""
    for _ in range(8):
        if case.phase == Phase.CLOSED:
            break
        response = run_turn(case, "no", llm, {})
    assert case.phase == Phase.CLOSED
    assert case.handoff_completed is False
    assert case.escalation_context == {}
    assert "enough information" in response.lower()
    assert "specialist" not in response.lower()


def test_a_few_clarifying_turns_do_not_escalate():
    # asking two or three times is fine — only a persistent dead end escalates
    case = CaseState(phase=Phase.CLARIFYING)
    case.conversation = [{"role": "user", "content": "hey"}]
    llm = MockLLMClient([_clarify_ask(), _clarify_ask()])
    run_turn(case, "no", llm, {})
    run_turn(case, "no", llm, {})
    assert case.phase != Phase.CLOSED


def test_actionable_unknown_app_issue_forces_tool_investigation():
    case = CaseState(phase=Phase.CLARIFYING)
    case.conversation = [{"role": "user", "content": "hey"}]
    case.clarification_attempts = 1
    repeated_question = _proposal(
        action=AgentAction.ASK_USER,
        message="Any error message?",
        missing_info_source=MissingInfoSource.USER,
        missing_info=["error message"],
    )
    tool_call = _proposal(
        action=AgentAction.CALL_TOOL,
        confidence=0.6,
        tool_name="kb_search",
        tool_input={"query": "shadowect vpn website stuck macos"},
        message=None,
        missing_info_source=MissingInfoSource.TOOL,
    )
    resolve = _proposal(
        action=AgentAction.RESOLVE,
        confidence=0.7,
        message="Check the VPN profile and try a different network.",
        has_safe_low_risk_guidance=True,
    )
    tools = {"kb_search": MockTool(ToolResult(success=True, data={"results": ["vpn guide"]}))}

    response = run_turn(
        case,
        "my shadowect app is stuck, vpn is connected, and websites will not load on macos right now",
        MockLLMClient([repeated_question, tool_call, resolve]),
        tools,
    )

    assert "Check the VPN profile" in response
    assert case.tool_calls_total == 1
    assert case.phase != Phase.CLOSED
    assert case.escalation_context == {}


def test_tool_call_resets_clarification_attempts():
    case = CaseState(phase=Phase.CLARIFYING)
    case.clarification_attempts = 2
    tools = {"kb_search": MockTool(ToolResult(success=True, data={"hit": 1}))}
    run_turn(case, "my vpn times out", MockLLMClient([
        _proposal(action=AgentAction.CALL_TOOL, tool_name="kb_search",
                  tool_input={"query": "vpn"}, message=None,
                  missing_info_source=MissingInfoSource.TOOL),
        _proposal(action=AgentAction.RESOLVE, confidence=0.6, message="Switch to TCP"),
    ]), tools)
    assert case.clarification_attempts == 0


def test_vague_greeting_counts_as_a_clarification_attempt():
    case = CaseState()
    run_turn(case, "hey", MockLLMClient([]), {})
    assert case.clarification_attempts == 1


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
    case = _case_after_clarification()
    case.tool_calls_total = 1  # investigation already happened
    run_turn(case, "Still broken", MockLLMClient([
        _proposal(action=AgentAction.RESOLVE, confidence=0.9, message="Try this"),
    ]), {})
    assert case.phase == Phase.RESOLVING


# ── tool call flow ────────────────────────────────────────────────────────────

def test_tool_call_then_resolve_in_one_turn():
    case = _case_after_clarification()
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
    assert "Restart VPN client" in response


def test_final_tool_call_gets_synthesis_chance_before_budget_escalation():
    case = CaseState(
        phase=Phase.INVESTIGATING,
        confidence=0.6,
        tool_calls_current_investigation=4,
        tool_calls_total=4,
    )
    proposals = [
        _proposal(
            action=AgentAction.CALL_TOOL,
            confidence=0.6,
            tool_name="kb_search",
            tool_input={"query": "shadowrocket connected cannot access google"},
            message=None,
            missing_info_source=MissingInfoSource.TOOL,
        ),
        _proposal(
            action=AgentAction.RESOLVE,
            confidence=0.7,
            message="Try switching VPN servers, then reconnect and test another external website.",
            has_safe_low_risk_guidance=True,
        ),
    ]
    tool = MockTool(ToolResult(success=True, data={"results": ["vpn troubleshooting guide"]}))

    response = run_turn(case, "still cannot access google", MockLLMClient(proposals), {"kb_search": tool})

    assert "Try switching VPN servers" in response
    assert case.handoff_completed is False
    assert case.escalation_context == {}


def test_budget_exhausted_question_retries_to_resolution():
    case = CaseState(
        phase=Phase.INVESTIGATING,
        confidence=0.6,
        tool_calls_current_investigation=5,
        tool_calls_total=5,
    )
    ask_again = _proposal(
        action=AgentAction.ASK_USER,
        confidence=0.6,
        message="Can you try reconnecting again?",
    )
    resolve = _proposal(
        action=AgentAction.RESOLVE,
        confidence=0.6,
        message="Try switching VPN servers, reconnect, and test another external site.",
        has_safe_low_risk_guidance=True,
    )

    response = run_turn(case, "still cannot access google", MockLLMClient([ask_again, resolve]), {})

    assert "Try switching VPN servers" in response
    assert case.handoff_completed is False


def test_service_wide_question_retries_to_status_api():
    case = CaseState(phase=Phase.INVESTIGATING)
    case.conversation = [
        {
            "role": "user",
            "content": "salesforce is slow since yesterday and my teammates in chicago see the same issue",
        },
    ]
    ask_error = _proposal(
        action=AgentAction.ASK_USER,
        confidence=0.5,
        message="Any error message?",
    )
    status_call = _proposal(
        action=AgentAction.CALL_TOOL,
        confidence=0.6,
        tool_name="status_api",
        tool_input={"service": "Salesforce"},
        message=None,
        missing_info_source=MissingInfoSource.TOOL,
    )
    resolve = _proposal(
        action=AgentAction.RESOLVE,
        confidence=0.7,
        message="Salesforce appears degraded. Use the web client later or monitor the status page.",
        has_safe_low_risk_guidance=True,
    )
    tools = {"status_api": MockTool(ToolResult(success=True, data={"services": []}))}

    response = run_turn(case, "still slow", MockLLMClient([ask_error, status_call, resolve]), tools)

    assert "Salesforce appears degraded" in response
    assert case.tool_traces[0].tool_name == "status_api"


def test_tool_trace_recorded():
    case = CaseState(phase=Phase.INVESTIGATING)
    case.confidence = 0.6

    proposals = [
        _proposal(action=AgentAction.CALL_TOOL, confidence=0.6,
                  tool_name="kb_search", tool_input={"query": "VPN"},
                  message=None, missing_info_source=MissingInfoSource.TOOL),
        _proposal(action=AgentAction.RESOLVE, confidence=0.6, message="Fix"),
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
        _proposal(action=AgentAction.RESOLVE, confidence=0.6, message="Fix"),
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
        _proposal(action=AgentAction.RESOLVE, confidence=0.6, message="Fix"),
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
        _proposal(action=AgentAction.RESOLVE, confidence=0.6, message="Fix"),
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
    case = _case_after_clarification()
    case.tool_calls_total = 1  # investigation already happened
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


def test_escalate_closes_case_in_one_turn():
    # a completed handoff is terminal — the case should close, not linger open
    case = CaseState(phase=Phase.INVESTIGATING)
    run_turn(case, "VPN broken", MockLLMClient([
        _proposal(action=AgentAction.ESCALATE, confidence=0.3,
                  escalation_reason="Needs admin", message=None),
    ]), {})
    assert case.phase == Phase.CLOSED


def test_escalate_message_surfaces_reason_to_user():
    case = CaseState(phase=Phase.INVESTIGATING)
    response = run_turn(case, "x", MockLLMClient([
        _proposal(action=AgentAction.ESCALATE, confidence=0.3,
                  escalation_reason="this needs admin rights we cannot grant", message=None),
    ]), {})
    assert "this needs admin rights we cannot grant" in response


def test_escalate_allowed_and_closes_from_clarifying():
    # a legitimate (low-confidence) escalation while still clarifying must be a
    # clean handoff, not mangled into a "repeated invalid proposals" force-escalate
    case = CaseState(phase=Phase.CLARIFYING)
    case.conversation = [{"role": "user", "content": "shadow rocket is stuck"}]
    response = run_turn(case, "macOS, just stuck, no error", MockLLMClient([
        _proposal(action=AgentAction.ESCALATE, confidence=0.3,
                  escalation_reason="third-party app outside our supported scope", message=None),
    ]), {})
    assert case.phase == Phase.CLOSED
    assert case.handoff_completed is True
    assert "third-party app outside our supported scope" in response


def test_forced_escalation_message_stays_generic():
    # internal force-escalate reasons (e.g. repeated invalid proposals) must not leak
    case = CaseState(phase=Phase.INTAKE)
    bad = _proposal(action=AgentAction.RESOLVE, confidence=0.9, message="x")  # invalid in intake
    response = run_turn(case, "vpn is down", MockLLMClient([bad] * 6), {})
    assert "specialist" in response.lower()
    assert "repeated" not in response.lower()


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


def test_runtime_budget_escalation_builds_handoff_context():
    case = CaseState(
        phase=Phase.INVESTIGATING,
        confidence=0.6,
        tool_calls_current_investigation=5,
        tool_calls_total=5,
        has_safe_low_risk_guidance=False,
        missing_info_source=MissingInfoSource.NONE,
    )
    response = run_turn(case, "still broken", MockLLMClient([
        _proposal(action=AgentAction.ASK_USER, message="Can you try again?"),
        _proposal(action=AgentAction.ESCALATE, confidence=0.4,
                  escalation_reason="investigation budget exhausted", message=None),
    ]), {})

    assert case.phase == Phase.CLOSED
    assert case.handoff_completed is True
    assert case.escalation_context != {}
    assert "specialist" in response.lower()


# ── confidence transparency (P1.7) ───────────────────────────────────────────

def test_high_confidence_resolve_has_confident_prefix():
    case = _case_after_clarification()
    case.tool_calls_total = 1  # investigation already happened
    response = run_turn(case, "VPN broken", MockLLMClient([
        _proposal(action=AgentAction.RESOLVE, confidence=0.9, message="Restart VPN client."),
    ]), {})
    assert "likely fix" in response.lower() or "found" in response.lower()
    assert "Restart VPN client" in response

def test_medium_confidence_resolve_has_hedging_prefix():
    case = CaseState(phase=Phase.INVESTIGATING)
    case.tool_calls_total = 1  # investigation already happened
    response = run_turn(case, "VPN broken", MockLLMClient([
        _proposal(action=AgentAction.RESOLVE, confidence=0.65, message="Try restarting."),
    ]), {})
    assert "not fully certain" in response.lower() or "safe" in response.lower()
    assert "Try restarting" in response

def test_resolve_prefix_reflects_calibrated_not_raw_confidence():
    # raw confidence is high (0.9) but a prior failed attempt calibrates it down,
    # so the employee sees the hedged wording; confidence shown matches the
    # confidence the runtime actually acted on.
    case = _case_after_clarification()
    case.tool_calls_total = 2
    case.resolution_attempts = 1  # one prior fix did not stick -> -0.15
    response = run_turn(case, "still broken", MockLLMClient([
        _proposal(action=AgentAction.RESOLVE, confidence=0.9, message="Reinstall the client."),
    ]), {})
    assert "not fully certain" in response.lower() or "safe" in response.lower()


def test_ask_user_message_passes_through_unchanged():
    case = CaseState()
    response = run_turn(case, "VPN broken", MockLLMClient([
        _proposal(action=AgentAction.ASK_USER, confidence=0.5, message="What OS are you using?"),
    ]), {})
    assert response == "What OS are you using?"

def test_escalate_response_is_handoff_message():
    case = CaseState(phase=Phase.INVESTIGATING)
    response = run_turn(case, "VPN broken", MockLLMClient([
        _proposal(action=AgentAction.ESCALATE, confidence=0.3,
                  escalation_reason="Needs admin", message=None),
    ]), {})
    assert "specialist" in response.lower()
    assert "repeat" in response.lower()


# ── state projection ──────────────────────────────────────────────────────────

def test_confidence_updated_from_proposal():
    case = _case_after_clarification()
    case.tool_calls_total = 2  # investigation already happened
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


# ── cost / latency accounting ─────────────────────────────────────────────────

class _StatLLM(BaseLLMClient):
    def __init__(self, proposals):
        from collections import deque
        self._q = deque(proposals)
        self.last_stats = None

    def call(self, llm_input):
        from llm.client import LLMCallStats
        self.last_stats = LLMCallStats(prompt_tokens=100, completion_tokens=20,
                                       total_tokens=120, latency_ms=5.0)
        return self._q.popleft()


def test_run_turn_accumulates_llm_token_usage():
    case = CaseState()
    run_turn(case, "VPN broken", _StatLLM([_proposal(message="What OS?")]), {})
    assert case.llm_calls == 1
    assert case.prompt_tokens == 100
    assert case.completion_tokens == 20


def test_run_turn_accumulates_latency():
    case = CaseState()
    run_turn(case, "VPN broken", _StatLLM([_proposal(message="What OS?")]), {})
    assert case.llm_latency_ms >= 5.0


# ── LLMClientError handling ───────────────────────────────────────────────────

class _FailingLLM(BaseLLMClient):
    def call(self, llm_input: LLMInput) -> AgentProposal:
        raise LLMProviderError("provider down")


def test_llm_error_returns_graceful_message():
    case = CaseState()
    response = run_turn(case, "VPN broken", _FailingLLM(), {})
    assert "technical issue" in response.lower() or "specialist" in response.lower()


def test_llm_error_appends_to_conversation():
    case = CaseState()
    run_turn(case, "VPN broken", _FailingLLM(), {})
    assert case.conversation[-1]["role"] == "assistant"


def test_llm_error_does_not_raise():
    case = CaseState()
    try:
        run_turn(case, "VPN broken", _FailingLLM(), {})
    except Exception as exc:
        pytest.fail(f"run_turn raised unexpectedly: {exc}")


# ── turn cancellation (ESC interrupt) ──────────────────────────────────────────

def test_cancel_before_first_iteration_raises_turn_cancelled():
    case = CaseState()
    llm = MockLLMClient([_proposal()])
    with pytest.raises(TurnCancelled):
        run_turn(case, "VPN broken", llm, {}, should_cancel=lambda: True)


def test_cancel_before_first_iteration_does_not_call_llm():
    case = CaseState()
    llm = MockLLMClient([_proposal()])
    with pytest.raises(TurnCancelled):
        run_turn(case, "VPN broken", llm, {}, should_cancel=lambda: True)
    # proposal was never consumed
    assert run_turn(CaseState(), "VPN broken", llm, {}) == "What OS?"


def test_cancel_before_first_iteration_leaves_phase_unchanged():
    case = CaseState(phase=Phase.INTAKE)
    with pytest.raises(TurnCancelled):
        run_turn(case, "VPN broken", MockLLMClient([_proposal()]), {},
                 should_cancel=lambda: True)
    assert case.phase == Phase.INTAKE


def test_cancel_after_llm_call_raises_before_mutating_state():
    # not cancelled at the loop top, cancelled by the time the call returns
    checks = [False, True]
    case = CaseState(phase=Phase.INTAKE)
    llm = MockLLMClient([_proposal()])
    with pytest.raises(TurnCancelled):
        run_turn(case, "VPN broken", llm, {}, should_cancel=lambda: checks.pop(0))
    assert case.phase == Phase.INTAKE
    # the proposal was consumed but discarded
    assert case.conversation[-1] == {"role": "user", "content": "VPN broken"}


def test_should_cancel_false_completes_normally():
    case = CaseState()
    response = run_turn(case, "VPN broken", MockLLMClient([_proposal(message="What OS?")]),
                        {}, should_cancel=lambda: False)
    assert response == "What OS?"


def test_llm_error_closes_case():
    case = CaseState()
    run_turn(case, "VPN broken", _FailingLLM(), {})
    assert case.phase == Phase.CLOSED


def test_llm_error_sets_handoff_completed():
    case = CaseState()
    run_turn(case, "VPN broken", _FailingLLM(), {})
    assert case.handoff_completed is True


def test_llm_error_builds_escalation_context():
    case = CaseState()
    run_turn(case, "VPN broken", _FailingLLM(), {})
    assert case.escalation_context != {}
    assert "escalation_reason" in case.escalation_context


# ── guardrail violations are correctable (retry, not instant escalation) ───────

def test_validation_failure_retries_and_recovers():
    # INTAKE forbids RESOLVE; the agent should be re-prompted and recover with a
    # valid action rather than being escalated on the first stumble.
    case = CaseState(phase=Phase.INTAKE)
    bad = _proposal(action=AgentAction.RESOLVE, confidence=0.9, message="Fix this")
    good = _proposal(action=AgentAction.ASK_USER, message="What OS are you on?")
    response = run_turn(case, "VPN broken", MockLLMClient([bad, good]), {})
    assert response == "What OS are you on?"
    assert case.phase != Phase.CLOSED


def test_validation_failure_recovery_does_not_escalate():
    case = CaseState(phase=Phase.INTAKE)
    bad = _proposal(action=AgentAction.RESOLVE, confidence=0.9, message="Fix this")
    good = _proposal(action=AgentAction.ASK_USER, message="What OS?")
    run_turn(case, "VPN broken", MockLLMClient([bad, good]), {})
    assert case.handoff_completed is False
    assert case.escalation_context == {}


def test_policy_block_retries_and_recovers():
    # premature ESCALATE is policy-blocked; agent is re-prompted and recovers
    case = CaseState(phase=Phase.INVESTIGATING)
    blocked = _proposal(action=AgentAction.ESCALATE, confidence=0.6,
                        escalation_reason="needs help", message=None)
    good = _proposal(action=AgentAction.ASK_USER, message="Which tool times out?")
    response = run_turn(case, "VPN broken", MockLLMClient([blocked, good]), {})
    assert response == "Which tool times out?"
    assert case.handoff_completed is False
    assert case.escalation_context == {}


def test_business_policy_block_retries_and_escalates_to_human():
    case = CaseState(phase=Phase.INVESTIGATING)
    case.tool_calls_total = 1
    blocked = _proposal(
        action=AgentAction.RESOLVE,
        confidence=0.9,
        message="I can reset your MFA device now.",
    )
    escalate = _proposal(
        action=AgentAction.ESCALATE,
        confidence=0.4,
        escalation_reason="MFA reset requires human approval and identity verification",
        message=None,
    )
    response = run_turn(case, "I lost my MFA device", MockLLMClient([blocked, escalate]), {})
    assert "specialist" in response.lower()
    assert case.handoff_completed is True
    assert case.escalation_context["escalation_reason"] == (
        "MFA reset requires human approval and identity verification"
    )


def test_pre_tool_low_confidence_escalation_retries_with_tool():
    case = CaseState(phase=Phase.CLARIFYING)
    case.conversation = [{"role": "user", "content": "hey"}]
    case.clarification_attempts = 1
    blocked = _proposal(
        action=AgentAction.ESCALATE,
        confidence=0.3,
        escalation_reason="VPN connection issue requires further investigation by a human specialist",
        message=None,
    )
    do_tool = _proposal(
        action=AgentAction.CALL_TOOL,
        confidence=0.6,
        tool_name="kb_search",
        tool_input={"query": "shadowrocket vpn connected cannot visit google websites"},
        message=None,
        missing_info_source=MissingInfoSource.TOOL,
    )
    resolve = _proposal(
        action=AgentAction.RESOLVE,
        confidence=0.7,
        message="Try reconnecting the VPN and switching the protocol or server.",
        has_safe_low_risk_guidance=True,
    )
    tools = {"kb_search": MockTool(ToolResult(success=True, data={"results": ["vpn guide"]}))}

    response = run_turn(
        case,
        "my shadowrocket is connected but I cant visit google or other outside website",
        MockLLMClient([blocked, do_tool, resolve]),
        tools,
    )

    assert "Try reconnecting" in response
    assert case.tool_calls_total == 1
    assert case.handoff_completed is False
    assert case.escalation_context == {}


def test_direct_handoff_signal_allows_security_escalation():
    case = CaseState(phase=Phase.INVESTIGATING)
    case.conversation = [
        {"role": "user", "content": "i clicked a suspicious link and now my account sends weird emails"},
    ]
    response = run_turn(case, "still happening", MockLLMClient([
        _proposal(
            action=AgentAction.ESCALATE,
            confidence=0.3,
            escalation_reason="account may be compromised",
            message=None,
        ),
    ]), {})

    assert case.phase == Phase.CLOSED
    assert case.handoff_completed is True
    assert "compromised" in response


def test_zero_tool_resolve_is_corrected_then_grounded():
    # RESOLVE with no tool calls is blocked; agent recovers by calling a tool,
    # then resolves successfully once grounded.
    case = CaseState(phase=Phase.INVESTIGATING)
    case.conversation = [{"role": "user", "content": "VPN broken"}]
    premature = _proposal(action=AgentAction.RESOLVE, confidence=0.7,
                          message="Just reinstall it")
    do_tool = _proposal(action=AgentAction.CALL_TOOL, confidence=0.6,
                        tool_name="kb_search", tool_input={"query": "vpn"}, message=None,
                        missing_info_source=MissingInfoSource.TOOL)
    resolve = _proposal(action=AgentAction.RESOLVE, confidence=0.7,
                        message="Switch the VPN protocol to TCP")
    tools = {"kb_search": MockTool(ToolResult(success=True, data={"hits": ["use TCP"]}))}
    response = run_turn(case, "VPN broken", MockLLMClient([premature, do_tool, resolve]), tools)
    assert "Switch the VPN protocol to TCP" in response
    assert case.tool_calls_total >= 1
    assert case.handoff_completed is False


def test_persistent_guardrail_violation_eventually_escalates():
    # an agent that never produces a valid action falls back to a graceful
    # escalation rather than looping forever.
    case = CaseState(phase=Phase.INTAKE)
    bad = _proposal(action=AgentAction.RESOLVE, confidence=0.9, message="Fix this")
    run_turn(case, "VPN broken", MockLLMClient([bad] * 12), {})
    assert case.phase == Phase.CLOSED
    assert case.handoff_completed is True
    assert "escalation_reason" in case.escalation_context


def test_guardrail_corrections_are_capped_before_loop_exhaustion():
    # correction retries are bounded well under the inner-iteration limit, so a
    # misbehaving model cannot burn the whole loop (and its token cost).
    case = CaseState(phase=Phase.INTAKE)
    bad = _proposal(action=AgentAction.RESOLVE, confidence=0.9, message="Fix this")
    llm = MockLLMClient([bad] * 12)
    run_turn(case, "VPN broken", llm, {})
    assert case.phase == Phase.CLOSED
    assert len(llm._queue) > 0  # escalated before consuming all proposals
