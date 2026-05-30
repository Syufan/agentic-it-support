import pytest
from agent.proposals import AgentAction, AgentProposal
from runtime.validator import validate_proposal
from state.case_state import CaseState, MissingInfoSource, Phase


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

@pytest.mark.parametrize("tool_name", [
    "kb_search", "status_api", "user_directory",
])
def test_call_tool_with_valid_tool_names(tool_name):
    result = validate_proposal(
        case_in(Phase.INVESTIGATING),
        proposal(action=AgentAction.CALL_TOOL, tool_name=tool_name, tool_input={"query": "x"}),
    )
    assert result.valid is True

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
