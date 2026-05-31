import pytest
from runtime.message_builder import LLMInput, build_messages
from state.case_state import CaseState, Phase, ToolTrace


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


# ── correction feedback ───────────────────────────────────────────────────────

def test_correction_is_included_in_system_prompt():
    case = CaseState()
    case.conversation = [{"role": "user", "content": "VPN broken"}]
    result = build_messages(case, correction="resolve blocked: investigate first")
    assert "resolve blocked: investigate first" in result.system
    assert "[Correction]" in result.system


def test_no_correction_section_by_default():
    case = CaseState()
    case.conversation = [{"role": "user", "content": "VPN broken"}]
    result = build_messages(case)
    full_text = result.system + " ".join(m["content"] for m in result.messages)
    assert "Correction" not in full_text


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

def test_observation_does_not_expose_runtime_guard_counters():
    case = CaseState(phase=Phase.INVESTIGATING)
    case.tool_calls_this_turn = 3
    case.tool_calls_total = 4
    result = build_messages(case)
    full_text = " ".join(m["content"] for m in result.messages)
    assert "Tool calls:" not in full_text
    assert "case total" not in full_text


def test_observation_does_not_expose_runtime_confidence():
    case = CaseState(phase=Phase.INVESTIGATING, confidence=0.73)
    result = build_messages(case)
    full_text = " ".join(m["content"] for m in result.messages)
    assert "Confidence:" not in full_text
    assert "0.73" not in full_text

def test_observation_includes_tool_traces():
    case = CaseState(phase=Phase.INVESTIGATING)
    case.tool_traces = [
        ToolTrace(
            tool_name="kb_search",
            inputs={"query": "VPN"},
            output={"results": []},
            success=True,
        )
    ]
    result = build_messages(case)
    full_text = " ".join(m["content"] for m in result.messages)
    assert "kb_search" in full_text


# ── no consecutive same-role messages (P0.4) ─────────────────────────────────

def test_observation_merged_when_conv_ends_with_user():
    case = CaseState(phase=Phase.INVESTIGATING)
    case.conversation = [{"role": "user", "content": "VPN is broken"}]
    result = build_messages(case)
    assert len(result.messages) == 1
    assert "VPN is broken" in result.messages[0]["content"]
    assert "investigating" in result.messages[0]["content"].lower()

def test_no_consecutive_user_messages_when_conv_ends_with_user():
    case = CaseState()
    case.conversation = [{"role": "user", "content": "VPN is broken"}]
    result = build_messages(case)
    for i in range(len(result.messages) - 1):
        assert not (result.messages[i]["role"] == "user" and result.messages[i + 1]["role"] == "user")

def test_observation_appended_when_conv_ends_with_assistant():
    case = CaseState()
    case.conversation = [
        {"role": "user", "content": "VPN is broken"},
        {"role": "assistant", "content": "What OS?"},
    ]
    result = build_messages(case)
    assert len(result.messages) == 3
    assert result.messages[-1]["role"] == "user"

def test_no_consecutive_user_with_mixed_conversation():
    case = CaseState()
    case.conversation = [
        {"role": "user", "content": "VPN is broken"},
        {"role": "assistant", "content": "What OS?"},
        {"role": "user", "content": "macOS"},
    ]
    result = build_messages(case)
    for i in range(len(result.messages) - 1):
        assert result.messages[i]["role"] != result.messages[i + 1]["role"]


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
