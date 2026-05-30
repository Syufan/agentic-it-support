import pytest
from agent.proposals import AgentAction, AgentProposal
from runtime.validator import validate_proposal as _validate_proposal
from state.case_state import CaseState, MissingInfoSource, Phase

# Production validate_proposal requires the caller to inject the tool registry;
# tests default to the standard set via this thin wrapper so existing call sites
# stay unchanged. Tests that exercise the registry parameter pass valid_tools
# explicitly and bypass the default.
_TEST_TOOLS = {"kb_search", "status_api", "user_directory", "resolution_history"}


def validate_proposal(case, proposal, valid_tools=_TEST_TOOLS):
    return _validate_proposal(case, proposal, valid_tools)


# ── helpers ───────────────────────────────────────────────────────────────────

def proposal(**kwargs) -> AgentProposal:
    defaults = {
        "action": AgentAction.ASK_USER,
        "confidence": 0.6,
        "reasoning_summary": "test",
    }
    return AgentProposal(**(defaults | kwargs))

def case_in(phase: Phase) -> CaseState:
    return CaseState(phase=phase)


# ── ValidationResult contract ─────────────────────────────────────────────────

def test_valid_result_has_valid_true():
    result = validate_proposal(
        case_in(Phase.INTAKE),
        proposal(action=AgentAction.ASK_USER, message="What OS?"),
    )
    assert result.valid is True
    assert result.reason is None

def test_invalid_result_has_valid_false_and_reason():
    result = validate_proposal(
        case_in(Phase.CLOSED),
        proposal(action=AgentAction.ASK_USER, message="What OS?"),
    )
    assert result.valid is False
    assert result.reason is not None


# ── phase × action rules ──────────────────────────────────────────────────────

@pytest.mark.parametrize("action,extra", [
    (AgentAction.ASK_USER, {"message": "What OS?"}),
    (AgentAction.CALL_TOOL, {"tool_name": "kb_search", "tool_input": {"query": "vpn"}}),
])
def test_intake_allows_ask_user_and_call_tool(action, extra):
    result = validate_proposal(case_in(Phase.INTAKE), proposal(action=action, **extra))
    assert result.valid is True

@pytest.mark.parametrize("action,extra", [
    (AgentAction.RESOLVE, {"message": "Try this."}),
    (AgentAction.ESCALATE, {"escalation_reason": "needs admin"}),
])
def test_intake_rejects_resolve_and_escalate(action, extra):
    result = validate_proposal(case_in(Phase.INTAKE), proposal(action=action, **extra))
    assert result.valid is False


@pytest.mark.parametrize("action,extra", [
    (AgentAction.ASK_USER, {"message": "Can you clarify?"}),
    (AgentAction.CALL_TOOL, {"tool_name": "kb_search", "tool_input": {"query": "vpn"}}),
])
def test_clarifying_allows_ask_user_and_call_tool(action, extra):
    result = validate_proposal(case_in(Phase.CLARIFYING), proposal(action=action, **extra))
    assert result.valid is True

def test_clarifying_rejects_resolve():
    # cannot resolve before investigating, but escalate IS allowed from clarifying
    # so a genuine out-of-scope handoff isn't mangled into a forced escalate
    result = validate_proposal(case_in(Phase.CLARIFYING),
                               proposal(action=AgentAction.RESOLVE, message="Try this."))
    assert result.valid is False


def test_clarifying_allows_escalate():
    result = validate_proposal(case_in(Phase.CLARIFYING),
                               proposal(action=AgentAction.ESCALATE, escalation_reason="needs admin"))
    assert result.valid is True


@pytest.mark.parametrize("action,extra", [
    (AgentAction.CALL_TOOL, {"tool_name": "kb_search", "tool_input": {"query": "vpn"}}),
    (AgentAction.RESOLVE, {"message": "Try this.", "confidence": 0.9}),
    (AgentAction.ESCALATE, {"escalation_reason": "needs admin", "confidence": 0.3}),
    (AgentAction.ASK_USER, {"message": "Did this help?"}),
])
def test_investigating_allows_all_actions(action, extra):
    result = validate_proposal(case_in(Phase.INVESTIGATING), proposal(action=action, **extra))
    assert result.valid is True


@pytest.mark.parametrize("action,extra", [
    (AgentAction.RESOLVE, {"message": "Try this."}),
    (AgentAction.ASK_USER, {"message": "Did it work?"}),
])
def test_resolving_allows_resolve_and_ask_user(action, extra):
    result = validate_proposal(case_in(Phase.RESOLVING), proposal(action=action, **extra))
    assert result.valid is True

@pytest.mark.parametrize("action,extra", [
    (AgentAction.CALL_TOOL, {"tool_name": "kb_search", "tool_input": {"query": "vpn"}}),
    (AgentAction.ESCALATE, {"escalation_reason": "needs admin"}),
])
def test_resolving_rejects_call_tool_and_escalate(action, extra):
    result = validate_proposal(case_in(Phase.RESOLVING), proposal(action=action, **extra))
    assert result.valid is False


def test_escalating_allows_escalate():
    result = validate_proposal(
        case_in(Phase.ESCALATING),
        proposal(action=AgentAction.ESCALATE, escalation_reason="needs admin"),
    )
    assert result.valid is True

@pytest.mark.parametrize("action,extra", [
    (AgentAction.ASK_USER, {"message": "What OS?"}),
    (AgentAction.CALL_TOOL, {"tool_name": "kb_search", "tool_input": {"query": "vpn"}}),
    (AgentAction.RESOLVE, {"message": "Try this."}),
])
def test_escalating_rejects_other_actions(action, extra):
    result = validate_proposal(case_in(Phase.ESCALATING), proposal(action=action, **extra))
    assert result.valid is False


@pytest.mark.parametrize("action,extra", [
    (AgentAction.ASK_USER, {"message": "What OS?"}),
    (AgentAction.CALL_TOOL, {"tool_name": "kb_search", "tool_input": {"query": "vpn"}}),
    (AgentAction.RESOLVE, {"message": "Try this."}),
    (AgentAction.ESCALATE, {"escalation_reason": "needs admin"}),
])
def test_closed_rejects_all_actions(action, extra):
    result = validate_proposal(case_in(Phase.CLOSED), proposal(action=action, **extra))
    assert result.valid is False


# ── required field checks ─────────────────────────────────────────────────────

def test_ask_user_without_message_rejected():
    result = validate_proposal(
        case_in(Phase.INTAKE),
        proposal(action=AgentAction.ASK_USER, message=None),
    )
    assert result.valid is False

def test_call_tool_without_tool_name_rejected():
    result = validate_proposal(
        case_in(Phase.INVESTIGATING),
        proposal(action=AgentAction.CALL_TOOL, tool_name=None),
    )
    assert result.valid is False

def test_call_tool_with_invalid_tool_name_rejected():
    result = validate_proposal(
        case_in(Phase.INVESTIGATING),
        proposal(action=AgentAction.CALL_TOOL, tool_name="nonexistent_tool", tool_input={"q": "x"}),
    )
    assert result.valid is False


def test_call_tool_rejected_when_budget_exhausted():
    case = case_in(Phase.INVESTIGATING)
    case.tool_calls_current_investigation = 5
    result = validate_proposal(
        case,
        proposal(action=AgentAction.CALL_TOOL, tool_name="kb_search", tool_input={"query": "vpn"}),
    )
    assert result.valid is False
    assert "budget" in result.reason


@pytest.mark.parametrize("tool_name", [
    "kb_search", "status_api", "user_directory", "resolution_history",
])
def test_call_tool_with_valid_tool_names(tool_name):
    result = validate_proposal(
        case_in(Phase.INVESTIGATING),
        proposal(action=AgentAction.CALL_TOOL, tool_name=tool_name, tool_input={"query": "x"}),
    )
    assert result.valid is True


def test_policy_lookup_is_not_llm_callable_tool():
    result = validate_proposal(
        case_in(Phase.INVESTIGATING),
        proposal(action=AgentAction.CALL_TOOL, tool_name="policy_lookup", tool_input={}),
    )
    assert result.valid is False
    assert "unknown tool" in result.reason


def test_tool_validity_is_driven_by_passed_registry():
    # The set of callable tools is whatever registry the caller injects — there is
    # no second hardcoded source of truth inside the validator.
    case = case_in(Phase.INVESTIGATING)
    custom = proposal(action=AgentAction.CALL_TOOL, tool_name="custom_tool", tool_input={"q": "x"})
    kb = proposal(action=AgentAction.CALL_TOOL, tool_name="kb_search", tool_input={"q": "x"})

    assert validate_proposal(case, custom, valid_tools={"custom_tool"}).valid is True
    assert validate_proposal(case, kb, valid_tools={"custom_tool"}).valid is False

def test_resolve_without_message_rejected():
    result = validate_proposal(
        case_in(Phase.INVESTIGATING),
        proposal(action=AgentAction.RESOLVE, confidence=0.9, message=None),
    )
    assert result.valid is False

def test_escalate_without_reason_rejected():
    result = validate_proposal(
        case_in(Phase.INVESTIGATING),
        proposal(action=AgentAction.ESCALATE, confidence=0.3, escalation_reason=None),
    )
    assert result.valid is False
