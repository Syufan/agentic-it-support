from agent.proposals import AgentAction, AgentProposal


# ── valid decisions per action ────────────────────────────────────────────────

def test_ask_user_decision():
    d = AgentProposal(
        action=AgentAction.ASK_USER,
        confidence=0.6,
        reasoning_summary="Missing OS info",
        message="What operating system are you using?",
        missing_info=["operating system"],
    )
    assert d.action == AgentAction.ASK_USER
    assert d.message == "What operating system are you using?"


def test_call_tool_decision():
    d = AgentProposal(
        action=AgentAction.CALL_TOOL,
        confidence=0.65,
        reasoning_summary="Need to check KB for VPN issues",
        tool_name="kb_search",
        tool_input={"query": "VPN disconnects every 10 minutes"},
    )
    assert d.action == AgentAction.CALL_TOOL
    assert d.tool_name == "kb_search"


def test_resolve_decision():
    d = AgentProposal(
        action=AgentAction.RESOLVE,
        confidence=0.9,
        reasoning_summary="KB article matches issue exactly",
        message="Please try resetting your network adapter.",
    )
    assert d.action == AgentAction.RESOLVE
    assert d.message == "Please try resetting your network adapter."


def test_escalate_decision():
    d = AgentProposal(
        action=AgentAction.ESCALATE,
        confidence=0.3,
        reasoning_summary="Issue requires admin access",
        escalation_reason="Requires Active Directory admin permissions",
    )
    assert d.action == AgentAction.ESCALATE
    assert d.escalation_reason is not None


# ── confidence validation ─────────────────────────────────────────────────────

def test_confidence_is_not_a_proposal_field():
    # confidence is runtime-owned (evidence-based); the LLM no longer reports it
    assert "confidence" not in AgentProposal.model_fields


# ── user_confirmed_resolution ─────────────────────────────────────────────────

def test_user_confirmed_resolution_true():
    d = AgentProposal(
        action=AgentAction.RESOLVE,
        confidence=0.9,
        reasoning_summary="User confirmed",
        user_confirmed_resolution=True,
    )
    assert d.user_confirmed_resolution is True


def test_user_confirmed_resolution_false():
    d = AgentProposal(
        action=AgentAction.RESOLVE,
        confidence=0.9,
        reasoning_summary="User not resolved",
        user_confirmed_resolution=False,
    )
    assert d.user_confirmed_resolution is False


def test_user_confirmed_resolution_defaults_to_none():
    d = AgentProposal(
        action=AgentAction.ASK_USER,
        confidence=0.6,
        reasoning_summary="test",
    )
    assert d.user_confirmed_resolution is None


# ── defaults ──────────────────────────────────────────────────────────────────

def test_defaults():
    d = AgentProposal(
        action=AgentAction.ASK_USER,
        confidence=0.5,
        reasoning_summary="test",
    )
    assert d.message is None
    assert d.missing_info == []
    assert d.tool_name is None
    assert d.tool_input == {}
    assert d.escalation_reason is None
    # runtime-owned flags must not exist on the proposal anymore
    assert "has_safe_low_risk_guidance" not in AgentProposal.model_fields
    assert "new_critical_fact_added" not in AgentProposal.model_fields


# ── json round-trip (LLM output parsing) ─────────────────────────────────────

def test_parse_from_dict():
    raw = {
        "action": "call_tool",
        "confidence": 0.7,
        "reasoning_summary": "Checking KB",
        "tool_name": "kb_search",
        "tool_input": {"query": "password reset"},
    }
    d = AgentProposal.model_validate(raw)
    assert d.action == AgentAction.CALL_TOOL
    assert d.tool_name == "kb_search"


def test_parse_ignores_unknown_runtime_owned_fields():
    """Fields the runtime now derives (e.g. missing_info_source) are tolerated but
    dropped if a model still emits them, rather than failing the parse."""
    raw = {
        "action": "call_tool",
        "confidence": 0.7,
        "reasoning_summary": "Checking KB",
        "tool_name": "kb_search",
        "tool_input": {"query": "password reset"},
        "missing_info_source": "tool",
    }
    d = AgentProposal.model_validate(raw)
    assert not hasattr(d, "missing_info_source")


def test_serialize_to_json():
    d = AgentProposal(
        action=AgentAction.ESCALATE,
        reasoning_summary="Out of scope",
        escalation_reason="Requires hardware replacement",
    )
    data = d.model_dump()
    assert data["action"] == AgentAction.ESCALATE
    assert data["escalation_reason"] == "Requires hardware replacement"
