import os
import sys
import time
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

_DIVIDER = "─" * 52
_GREETING = "Hi! I'm your IT support assistant. What's your issue?"


def typewrite(
    text: str,
    writer: Callable[[str], None] = lambda c: (sys.stdout.write(c), sys.stdout.flush()),
    delay: float = 0.015,
) -> None:
    for char in text:
        writer(char)
        if delay:
            time.sleep(delay)


def _render(
    display: list[tuple[str, str]],
    phase: str,
    clear: Callable[[], None],
    writer: Callable[[str], None],
) -> None:
    clear()
    writer("")
    for role, msg in display:
        if role == "agent":
            writer(f"Agent: {msg}\n")
        else:
            writer(f"> {msg}\n")
    writer(_DIVIDER)
    writer(f"[{phase}]  Ctrl+C to quit")


def run_cli_session(
    case: CaseState,
    llm: BaseLLMClient,
    tools: dict,
    reader: Callable[[str], str] = input,
    writer: Callable[[str], None] = print,
    clear: Callable[[], None] = lambda: os.system("cls" if os.name == "nt" else "clear"),
) -> None:
    display: list[tuple[str, str]] = [("agent", _GREETING)]
    _render(display, case.phase.value, clear, writer)

    while case.phase != Phase.CLOSED:
        try:
            user_input = reader("> ").strip()
        except (KeyboardInterrupt, EOFError):
            writer("\nGoodbye.")
            break

        if not user_input or user_input == "\x1b":
            if user_input == "\x1b":
                writer("\nGoodbye.")
                break
            continue

        display.append(("user", user_input))
        _render(display, case.phase.value, clear, writer)

        spinner = Spinner(
            get_phase=lambda: case.phase.value,
            writer=lambda s: (sys.stdout.write(s), sys.stdout.flush()),
        )
        spinner.start()
        response = run_turn(case, user_input, llm, tools)
        spinner.stop()

        sys.stdout.write("Agent: ")
        sys.stdout.flush()
        typewrite(response)
        sys.stdout.write("\n\n")
        sys.stdout.flush()

        display.append(("agent", response))
        writer(_DIVIDER)
        writer(f"[{case.phase.value}]  Ctrl+C to quit")


if __name__ == "__main__":
    run_cli_session(CaseState(), RealLLMClient(), _TOOLS)
