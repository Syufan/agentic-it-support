import os
import shutil
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


def _content_lines(display: list[tuple[str, str]]) -> int:
    count = 1  # leading blank line
    for _, msg in display:
        count += len(msg.splitlines())  # message (may be multi-line)
        count += 1                      # blank line after
    return count


def _render(
    display: list[tuple[str, str]],
    phase: str,
    clear: Callable[[], None],
    writer: Callable[[str], None],
    term_height: int,
) -> None:
    clear()
    writer("")
    for role, msg in display:
        if role == "agent":
            writer(f"Agent: {msg}")
        else:
            writer(f"> {msg}")
        writer("")

    # pad to push divider to bottom (-3: divider + phase line + input prompt)
    padding = max(0, term_height - _content_lines(display) - 3)
    for _ in range(padding):
        writer("")

    writer(_DIVIDER)
    writer(f"[{phase}]  Ctrl+C to quit")


def run_cli_session(
    case: CaseState,
    llm: BaseLLMClient,
    tools: dict,
    reader: Callable[[str], str] = input,
    writer: Callable[[str], None] = print,
    clear: Callable[[], None] = lambda: os.system("cls" if os.name == "nt" else "clear"),
    get_term_height: Callable[[], int] = lambda: shutil.get_terminal_size().lines,
) -> None:
    display: list[tuple[str, str]] = [("agent", _GREETING)]

    def render() -> None:
        _render(display, case.phase.value, clear, writer, get_term_height())

    render()

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
        render()

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
        sys.stdout.write("\n")
        sys.stdout.flush()

        display.append(("agent", response))
        render()


if __name__ == "__main__":
    run_cli_session(CaseState(), RealLLMClient(), _TOOLS)
