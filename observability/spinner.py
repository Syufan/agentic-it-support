import threading
import time
from collections.abc import Callable

_FRAMES = ("●", "○")
_CLEAR = "\r" + " " * 80 + "\r"


def format_waiting_line(frame_idx: int, elapsed: float, phase: str) -> str:
    frame = _FRAMES[frame_idx % len(_FRAMES)]
    return f"\r{frame} thinking... {elapsed:.1f}s  [phase: {phase}]  (ESC to cancel)"


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
        self._writer(_CLEAR)

    def _run(self) -> None:
        idx = 0
        while self._running:
            elapsed = time.monotonic() - self._start_time
            self._writer(format_waiting_line(idx, elapsed, self._get_phase()))
            time.sleep(self._interval)
            idx += 1
