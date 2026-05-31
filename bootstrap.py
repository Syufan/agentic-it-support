"""Composition helpers shared by the entry points (main / cli / evaluation).

Lives at the root, above the leaf packages it wires together, so neither llm/
nor agent/ has to depend on the other. Keep this thin: just construction.
"""

from agent.parser import parse_proposal
from config.settings import Settings
from llm.client import BaseLLMClient, RealLLMClient


def build_llm(settings: Settings) -> BaseLLMClient:
    return RealLLMClient(
        response_parser=parse_proposal,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
    )
