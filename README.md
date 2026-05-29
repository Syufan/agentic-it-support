# Agentic IT Support

An agentic IT helpdesk assistant built with FastAPI and OpenAI. The agent follows a deterministic state machine (intake → clarifying → investigating → resolving → escalating → closed) and uses a budget-controlled tool loop to diagnose and resolve employee IT issues.

## Setup

```bash
cp .env.example .env
# fill in LLM_API_KEY and LLM_MODEL in .env
uv sync
```

## Run

```bash
uv run python main.py
# server starts at http://localhost:8000
```

## Test

```bash
uv run pytest
```

## Smoke test

Start the server, then send requests with the same `case_id` to continue a conversation:

```bash
# turn 1 — open a new case
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "I cannot connect to the VPN, it keeps timing out"}' | python3 -m json.tool

# turn 2 — continue the case (replace <case_id> with the value from turn 1)
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"case_id": "<case_id>", "message": "Cisco AnyConnect on macOS, home WiFi, already restarted"}' | python3 -m json.tool

# health check
curl http://localhost:8000/health
```

## Evaluation

Run batch evaluation against predefined scenarios (requires `LLM_API_KEY`):

```bash
uv run python -m evaluation.runner
```

Scenarios live in `evaluation/scenarios/*.json`. Each scenario defines user messages and pass/fail criteria (`escalated`, `resolved`, `min_tool_calls`, `max_tool_calls`).

## Observability

There is no external tracing dependency. All diagnostic state is carried in `CaseState` after each turn:

| Field | What it tells you |
|-------|-------------------|
| `tool_traces` | Every tool call: name, inputs, output, success, timestamp |
| `facts` | Accumulated facts extracted from tool results |
| `conversation` | Full message history (user + assistant) |
| `escalation_context` | Complete handoff package when escalated |
| `confidence` | LLM's stated confidence at the last turn |
| `phase` | Current state machine phase |

Use `observability.logger` to emit structured JSON logs at the end of each turn or on case close:

```python
from observability.logger import log_turn, log_case_closed

log_turn(case)        # per-turn snapshot
log_case_closed(case) # final summary when phase reaches CLOSED
```

Logs are emitted via Python's standard `logging` module under the `agentic_it_support` logger.
