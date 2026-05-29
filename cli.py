import sys
from collections.abc import Callable

from agent.llm import BaseLLMClient, RealLLMClient
from observability.spinner import Spinner
from runtime.controller import run_turn
from state.case_state import CaseState, Phase
from tools.kb_search import KBSearchTool
from tools.status_api import StatusAPITool
from tools.user_directory import UserDirectoryTool

_TOOLS = {
    "kb_search": KBSearchTool(),
    "status_api": StatusAPITool(),
    "user_directory": UserDirectoryTool(),
}


def run_cli_session(
    case: CaseState,
    llm: BaseLLMClient,
    tools: dict,
    reader: Callable[[str], str] = input,
    writer: Callable[[str], None] = print,
) -> None:
    writer("IT Support Agent ready. Describe your issue (Ctrl+C to quit).\n")
    while case.phase != Phase.CLOSED:
        try:
            user_input = reader("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            writer("\nGoodbye.")
            break
        if not user_input or user_input == "\x1b":
            if user_input == "\x1b":
                writer("\nGoodbye.")
                break
            continue
        spinner = Spinner(
            get_phase=lambda: case.phase.value,
            writer=lambda s: (sys.stdout.write(s), sys.stdout.flush()),
        )
        spinner.start()
        response = run_turn(case, user_input, llm, tools)
        spinner.stop()
        writer(f"Agent: {response}\n")
        writer(f"[phase: {case.phase.value}]")


if __name__ == "__main__":
    run_cli_session(CaseState(), RealLLMClient(), _TOOLS)
