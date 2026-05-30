import pytest

from agent.parser import parse_proposal
from agent.proposals import AgentAction
from llm.client import LLMResponseError


def test_parse_valid_proposal():
    raw = '{"action": "ask_user", "confidence": 0.6, "reasoning_summary": "x", "message": "hi"}'
    proposal = parse_proposal(raw)
    assert proposal.action == AgentAction.ASK_USER
    assert proposal.message == "hi"


def test_parse_non_json_raises():
    with pytest.raises(LLMResponseError, match="non-JSON"):
        parse_proposal("not json at all")


def test_parse_schema_mismatch_raises():
    with pytest.raises(LLMResponseError, match="AgentProposal"):
        parse_proposal('{"ok": true}')
