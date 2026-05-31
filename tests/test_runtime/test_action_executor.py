from agent.proposals import AgentAction, AgentProposal
from runtime.action_executor import _execute_tool, _project_to_state
from state.case_state import CaseState, Phase


# ── helpers ───────────────────────────────────────────────────────────────────

def _proposal(**kwargs) -> AgentProposal:
    defaults = {
        "action": AgentAction.ASK_USER,
        "confidence": 0.6,
        "reasoning_summary": "test",
        "message": "What OS?",
    }
    return AgentProposal(**(defaults | kwargs))


# ── state projection ──────────────────────────────────────────────────────────
# Runtime-owned flags can no longer be set by the LLM proposal:
# has_safe_low_risk_guidance (T9) is a runtime judgment, so projecting a proposal
# must not clobber what the runtime set.

def test_project_does_not_clobber_safe_guidance_flag():
    case = CaseState()
    case.has_safe_low_risk_guidance = True
    _project_to_state(case, _proposal(action=AgentAction.RESOLVE, message="try restarting"))
    assert case.has_safe_low_risk_guidance is True


# ── tool execution: missing tool ──────────────────────────────────────────────
# A validated proposal can no longer reference a tool outside the registry
# (validate_proposal checks against set(tool_registry)), so the missing-tool path
# is unreachable via run_turn. _execute_tool still guards it defensively, so we
# unit-test that branch directly rather than through the (now-blocked) turn loop.

def test_missing_tool_records_failure_trace():
    case = CaseState(phase=Phase.INVESTIGATING)
    _execute_tool(case, _proposal(action=AgentAction.CALL_TOOL, tool_name="kb_search",
                                  tool_input={"query": "vpn"}, message=None), {})
    assert case.tool_traces[0].success is False


def test_missing_tool_error_stored_in_facts():
    case = CaseState(phase=Phase.INVESTIGATING)
    _execute_tool(case, _proposal(action=AgentAction.CALL_TOOL, tool_name="kb_search",
                                  tool_input={"query": "vpn"}, message=None), {})
    assert "kb_search_error" in case.facts
