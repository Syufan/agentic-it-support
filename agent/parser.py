"""Parse raw LLM text into the domain proposal.

This lives in the agent layer, not the transport layer: turning provider text
into an `AgentProposal` is domain knowledge. `llm/` stays schema-agnostic and
receives this function via `response_parser` injection.
"""

import json
from json import JSONDecodeError

from pydantic import ValidationError

from agent.proposals import AgentProposal
from llm.client import LLMResponseError


def parse_proposal(raw: str) -> AgentProposal:
    try:
        return AgentProposal.model_validate(json.loads(raw))
    except JSONDecodeError as exc:
        raise LLMResponseError("LLM returned non-JSON content") from exc
    except ValidationError as exc:
        raise LLMResponseError("LLM returned JSON that does not match AgentProposal") from exc
