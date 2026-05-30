from collections.abc import Callable

from agent.llm import BaseLLMClient
from state.case_state import CaseState
from tools.base import BaseTool

TurnRunner = Callable[[CaseState, str, BaseLLMClient, dict[str, BaseTool]], str]
