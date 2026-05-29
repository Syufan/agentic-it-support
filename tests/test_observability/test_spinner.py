import time

from observability.spinner import Spinner, format_waiting_line


# ── format_waiting_line (pure function) ───────────────────────────────────────

def test_starts_with_carriage_return():
    line = format_waiting_line(0, 0.0, "intake")
    assert line.startswith("\r")


def test_contains_elapsed_time():
    line = format_waiting_line(0, 2.5, "investigating")
    assert "2.5" in line


def test_contains_phase():
    line = format_waiting_line(0, 1.0, "clarifying")
    assert "clarifying" in line


def test_frame_cycles():
    line0 = format_waiting_line(0, 0.0, "intake")
    line1 = format_waiting_line(1, 0.0, "intake")
    assert line0 != line1


def test_frame_wraps_around():
    line_a = format_waiting_line(0, 0.0, "intake")
    line_b = format_waiting_line(10, 0.0, "intake")  # 10 % 10 frames = 0
    assert line_a == line_b


# ── Spinner (thread + stop) ───────────────────────────────────────────────────

def test_spinner_writes_at_least_once():
    output = []
    spinner = Spinner(get_phase=lambda: "investigating", writer=output.append)
    spinner.start()
    time.sleep(0.25)
    spinner.stop()
    assert len(output) >= 1


def test_spinner_stop_clears_line():
    output = []
    spinner = Spinner(get_phase=lambda: "investigating", writer=output.append)
    spinner.start()
    time.sleep(0.15)
    spinner.stop()
    last = output[-1]
    assert last.startswith("\r") and last.strip() == ""


def test_spinner_does_not_raise_on_stop_before_start():
    spinner = Spinner(get_phase=lambda: "intake", writer=lambda _: None)
    spinner.stop()  # must not raise


def test_spinner_output_contains_phase():
    output = []
    spinner = Spinner(get_phase=lambda: "resolving", writer=output.append)
    spinner.start()
    time.sleep(0.15)
    spinner.stop()
    assert any("resolving" in line for line in output)
