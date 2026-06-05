
from agentic_it_support.config.settings import Settings
from agentic_it_support.llm.client import BaseLLMClient
from agentic_it_support.state.case_state import CaseState
from agentic_it_support.tools.base import BaseTool

def run_turn(case: CaseState, user_message: str, *, llm: BaseLLMClient, tools: dict[str, BaseTool], settings: Settings) -> str:
    # 1. Setup
    case.add_user_message(user_message)

    # 2. Main loop

    # 3. Exit
