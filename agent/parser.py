"""Parse raw LLM text into the domain proposal.

This lives in the agent layer, not the transport layer: turning provider text
into an `AgentProposal` is domain knowledge. `llm/` stays schema-agnostic and
receives this function via `response_parser` injection.
"""

import json
from json import JSONDecodeError

from pydantic import ValidationError

from agent.proposals import AgentProposal


class ProposalParseError(Exception):
    """The model's text could not be turned into a valid AgentProposal.

    A domain-level error (about proposal validity), distinct from the llm
    transport-level errors, so the agent layer owns it and depends on nothing.
    """


def parse_proposal(raw: str) -> AgentProposal:
    try:
        return AgentProposal.model_validate(json.loads(raw))
    except JSONDecodeError as exc:
        raise ProposalParseError("LLM returned non-JSON content") from exc
    except ValidationError as exc:
        raise ProposalParseError("LLM returned JSON that does not match AgentProposal") from exc
