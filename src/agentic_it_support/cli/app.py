import os
import sys
import threading
import time
from collections.abc import Callable

from agentic_it_support.agent.parser import parse_proposal
from agentic_it_support.config.settings import Settings
from agentic_it_support.llm.client import BaseLLMClient, RealLLMClient
from agentic_it_support.observability.event_tracing import InMemoryEventLog
from agentic_it_support.runtime.turn_runner import run_turn
from agentic_it_support.state.case_state import CaseState, Phase
from agentic_it_support.tools import build_tools

_DIM = "\033[2m"
_BOLD = "\033[1m"
_RESET = "\033[0m"
_ESC = "\x1b"

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


# ── thinking spinner (terminal UI) ────────────────────────────────────────────

_SPINNER_FRAMES = ("●", "○")
_SPINNER_CLEAR = "\r\033[2K"


def format_waiting_line(frame_idx: int, elapsed: float, phase: str) -> str:
    frame = _SPINNER_FRAMES[frame_idx % len(_SPINNER_FRAMES)]
    text = f"  {frame} thinking... {elapsed:.1f}s  [phase: {phase}]  (ESC to cancel)"
    try:
        width = os.get_terminal_size().columns
    except OSError:
        width = 80
    if len(text) >= width:
        text = text[:width - 1]
    return "\r\033[2K" + text

class Spinner:
    def __init__(
        self,
        get_phase: Callable[[], str],
        writer: Callable[[str], None],
        interval: float = 0.6,
    ) -> None:
        self._get_phase = get_phase
        self._writer = writer
        self._interval = interval
        self._running = False
        self._thread: threading.Thread | None = None
        self._start_time: float = 0.0

    def start(self) -> None:
        self._running = True
        self._start_time = time.monotonic()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        self._writer(_SPINNER_CLEAR)

    def _run(self) -> None:
        idx = 0
        while self._running:
            elapsed = time.monotonic() - self._start_time
            self._writer(format_waiting_line(idx, elapsed, self._get_phase()))
            time.sleep(self._interval)
            idx += 1


def run_cli_session(
    case: CaseState,
    llm: BaseLLMClient,
    tools: dict,
    settings: Settings,
    reader: Callable[[str], str] = input,
    writer: Callable[[str], None] = print,
    clear: Callable[[], None] = lambda: os.system("cls" if os.name == "nt" else "clear"),
    get_term_size: Callable[[], tuple[int, int]] | None = None,
    cursor_up: Callable[[int], None] | None = None,
    spinner_factory: Callable[[Callable[[], str]], object] | None = None,
    turn_runner: Callable[..., str] | None = None,
) -> None:
    del get_term_size, cursor_up  # compatibility hooks for older tests.

    run_one_turn = turn_runner or _interruptible_run_turn
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

        if user_input.startswith("/"):
            if _handle_command(user_input, case, event_log, settings, writer, clear):
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
            response = run_one_turn(case, user_input, llm, tools, settings)
        finally:
            spinner.stop()

        writer("")
        writer(f"{_BOLD}Agent:{_RESET}")
        writer(response)
        writer("")

    if case.phase == Phase.CLOSED:
        writer(f"{_DIM}case closed: {case.case_id}{_RESET}")


def _interruptible_run_turn(
    case: CaseState,
    user_input: str,
    llm: BaseLLMClient,
    tools: dict,
    settings: Settings,
) -> str:
    return run_turn(case, user_input, llm=llm, tools=tools, settings=settings)


def _stdin_is_tty() -> bool:
    try:
        return sys.stdin.isatty()
    except (ValueError, AttributeError):
        return False


def _print_header(writer: Callable[[str], None]) -> None:
    writer(f"{_BOLD}Agentic IT Support{_RESET}")
    writer("Describe the issue. Type /help for commands, /quit to exit.")
    writer("")


def _handle_command(
    command: str,
    case: CaseState,
    event_log: InMemoryEventLog,
    settings: Settings,
    writer: Callable[[str], None],
    clear: Callable[[], None],
) -> bool:
    match command.lower():
        case "/help":
            writer(_HELP.rstrip())
        case "/status":
            writer(_format_status(case, settings))
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


def _format_status(case: CaseState, settings: Settings) -> str:
    remaining_turn_tools = max(
        0,
        settings.limits.max_tool_calls_per_turn - case.tool_calls_this_turn,
    )
    remaining_case_tools = max(
        0,
        settings.limits.max_tool_calls_per_case - case.tool_calls_total,
    )
    return (
        f"[phase={case.phase.value} confidence={case.confidence:.2f} "
        f"tools_turn={case.tool_calls_this_turn} remaining_turn={remaining_turn_tools} "
        f"tools_case={case.tool_calls_total} remaining_case={remaining_case_tools}]"
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


def main() -> None:
    settings = Settings()
    run_cli_session(
        CaseState(),
        RealLLMClient(
            response_parser=parse_proposal,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            temperature=settings.llm_temperature,
        ),
        build_tools(settings.data_dir),
        settings,
    )


if __name__ == "__main__":
    main()
