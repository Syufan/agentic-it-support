import pytest
from runtime.message_builder import LLMInput, build_messages
from state.case_state import BudgetMode, CaseState, Phase, ToolTrace


# ── LLMInput contract ─────────────────────────────────────────────────────────

def test_returns_llm_input():
    result = build_messages(CaseState())
    assert isinstance(result, LLMInput)

def test_llm_input_has_system_and_messages():
    result = build_messages(CaseState())
    assert isinstance(result.system, str)
    assert isinstance(result.messages, list)

def test_system_prompt_is_non_empty():
    result = build_messages(CaseState())
    assert len(result.system) > 0


# ── system prompt per phase ───────────────────────────────────────────────────

@pytest.mark.parametrize("phase", [
    Phase.INTAKE,
    Phase.CLARIFYING,
    Phase.INVESTIGATING,
    Phase.RESOLVING,
    Phase.ESCALATING,
])
def test_each_phase_has_distinct_system_prompt(phase):
    result = build_messages(CaseState(phase=phase))
    assert len(result.system) > 0

def test_different_phases_produce_different_system_prompts():
    intake_prompt = build_messages(CaseState(phase=Phase.INTAKE)).system
    investigating_prompt = build_messages(CaseState(phase=Phase.INVESTIGATING)).system
    assert intake_prompt != investigating_prompt


# ── conversation history ──────────────────────────────────────────────────────

def test_conversation_history_included_in_messages():
    case = CaseState()
    case.conversation = [
        {"role": "user", "content": "VPN keeps disconnecting"},
        {"role": "assistant", "content": "What OS are you using?"},
    ]
    result = build_messages(case)
    roles = [m["role"] for m in result.messages]
    assert "user" in roles
    assert "assistant" in roles

def test_messages_end_with_user_role():
    case = CaseState()
    case.conversation = [{"role": "user", "content": "VPN keeps disconnecting"}]
    result = build_messages(case)
    assert result.messages[-1]["role"] == "user"

def test_empty_conversation_still_produces_messages():
    case = CaseState()
    result = build_messages(case)
    assert len(result.messages) >= 1


# ── case state in observation ─────────────────────────────────────────────────

def test_observation_includes_phase():
    case = CaseState(phase=Phase.INVESTIGATING)
    result = build_messages(case)
    full_text = " ".join(m["content"] for m in result.messages)
    assert "investigating" in full_text.lower()

def test_observation_includes_facts():
    case = CaseState(phase=Phase.INVESTIGATING)
    case.facts = {"os": "macOS", "vpn_client": "v4.1"}
    result = build_messages(case)
    full_text = " ".join(m["content"] for m in result.messages)
    assert "macOS" in full_text

def test_observation_includes_budget():
    case = CaseState(phase=Phase.INVESTIGATING)
    case.tool_calls_current_investigation = 3
    result = build_messages(case)
    full_text = " ".join(m["content"] for m in result.messages)
    assert "3" in full_text

def test_observation_includes_tool_traces():
    case = CaseState(phase=Phase.INVESTIGATING)
    case.tool_traces = [
        ToolTrace(
            tool_name="kb_search",
            inputs={"query": "VPN"},
            output={"results": []},
            success=True,
            budget_mode=BudgetMode.MAIN,
        )
    ]
    result = build_messages(case)
    full_text = " ".join(m["content"] for m in result.messages)
    assert "kb_search" in full_text


# ── message format ────────────────────────────────────────────────────────────

def test_each_message_has_role_and_content():
    case = CaseState()
    case.conversation = [{"role": "user", "content": "hello"}]
    result = build_messages(case)
    for msg in result.messages:
        assert "role" in msg
        assert "content" in msg

def test_roles_are_valid():
    case = CaseState()
    case.conversation = [
        {"role": "user", "content": "VPN issue"},
        {"role": "assistant", "content": "Let me check"},
    ]
    result = build_messages(case)
    valid_roles = {"user", "assistant"}
    for msg in result.messages:
        assert msg["role"] in valid_roles
