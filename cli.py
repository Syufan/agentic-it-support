import os
import sys
import time
from collections.abc import Callable

from agent.llm import BaseLLMClient, RealLLMClient
from observability.logger import InMemoryEventLog
from observability.spinner import Spinner
from runtime.controller import run_turn
from state import budget as budget_
from state.case_state import CaseState, Phase
from tools.kb_search import KBSearchTool
from tools.status_api import StatusAPITool
from tools.user_directory import UserDirectoryTool

_TOOLS = {
    "kb_search": KBSearchTool(),
    "status_api": StatusAPITool(),
    "user_directory": UserDirectoryTool(),
}

_DIM = "\033[2m"
_BOLD = "\033[1m"
_RESET = "\033[0m"

_HELP = """Commands:
  /help     show commands
  /status   show current case state
  /trace    show recent runtime events
  /clear    clear the terminal
  /quit     exit
"""


def typewrite(
    text: str,
    writer: Callable[[str], None] = lambda c: (sys.stdout.write(c), sys.stdout.flush()),
    delay: float = 0.015,
) -> None:
    for char in text:
        writer(char)
        if delay:
            time.sleep(delay)


def run_cli_session(
    case: CaseState,
    llm: BaseLLMClient,
    tools: dict,
    reader: Callable[[str], str] = input,
    writer: Callable[[str], None] = print,
    clear: Callable[[], None] = lambda: os.system("cls" if os.name == "nt" else "clear"),
    get_term_size: Callable[[], tuple[int, int]] | None = None,
    cursor_up: Callable[[int], None] | None = None,
    spinner_factory: Callable[[Callable[[], str]], object] | None = None,
) -> None:
    del get_term_size, cursor_up  # compatibility hooks for older tests.

    event_log = InMemoryEventLog()
    _print_header(writer)

    while case.phase != Phase.CLOSED:
        try:
            user_input = reader("❯ ").strip()
        except (KeyboardInterrupt, EOFError):
            writer("\nGoodbye.")
            break

        if not user_input:
            continue

        if user_input == "\x1b":
            writer("\nGoodbye.")
            break

        if user_input.startswith("/"):
            if _handle_command(user_input, case, event_log, writer, clear):
                break
            continue

        if spinner_factory:
            spinner = spinner_factory(lambda: case.phase.value)
        else:
            spinner = Spinner(
                get_phase=lambda: case.phase.value,
                writer=lambda text: (sys.stdout.write(f"{_DIM}{text}{_RESET}"), sys.stdout.flush()),
            )
        spinner.start()
        try:
            response = run_turn(case, user_input, llm, tools, event_log=event_log)
        finally:
            spinner.stop()

        writer(response)

    if case.phase == Phase.CLOSED:
        writer(f"{_DIM}case closed: {case.case_id}{_RESET}")


def _print_header(writer: Callable[[str], None]) -> None:
    writer(f"{_BOLD}Agentic IT Support{_RESET}")
    writer("Describe the issue. Type /help for commands, /quit to exit.")
    writer("")


def _handle_command(
    command: str,
    case: CaseState,
    event_log: InMemoryEventLog,
    writer: Callable[[str], None],
    clear: Callable[[], None],
) -> bool:
    match command.lower():
        case "/help":
            writer(_HELP.rstrip())
        case "/status":
            writer(_format_status(case))
        case "/trace":
            writer(_format_trace(event_log))
        case "/clear":
            clear()
            _print_header(writer)
        case "/quit" | "/exit":
            writer("Goodbye.")
            return True
        case _:
            writer("Unknown command. Type /help for available commands.")
    return False


def _format_status(case: CaseState) -> str:
    remaining = budget_.remaining(case.budget_mode, case.tool_calls_current_investigation)
    return (
        f"[phase={case.phase.value} confidence={case.confidence:.2f} "
        f"tools={case.tool_calls_current_investigation} remaining={remaining} "
        f"budget={case.budget_mode.value}]"
    )


def _format_trace(event_log: InMemoryEventLog, limit: int = 8) -> str:
    events = event_log.events()[-limit:]
    if not events:
        return "No runtime events recorded yet."

    lines = ["Recent runtime events:"]
    for event in events:
        detail = ""
        if event.details:
            pairs = ", ".join(f"{key}={value}" for key, value in event.details.items())
            detail = f" ({pairs})"
        lines.append(f"- {event.type}: phase={event.phase} confidence={event.confidence:.2f}{detail}")
    return "\n".join(lines)


if __name__ == "__main__":
    run_cli_session(CaseState(), RealLLMClient(), _TOOLS)
