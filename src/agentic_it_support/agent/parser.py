"""Parse raw LLM output into an AgentProposal."""

import json
from json import JSONDecodeError

from pydantic import ValidationError

from agentic_it_support.agent.proposals import AgentProposal


class ProposalParseError(Exception):
    """Raised when LLM output is not a valid AgentProposal."""


def parse_proposal(raw: str) -> AgentProposal:
    # Parse JSON and validate it against the proposal schema.
    try:
        return AgentProposal.model_validate(json.loads(raw))
    except JSONDecodeError as exc:
        raise ProposalParseError("LLM returned non-JSON content") from exc
    except ValidationError as exc:
        raise ProposalParseError("LLM returned JSON that does not match AgentProposal") from exc