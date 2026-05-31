# Agentic IT Support

An agentic IT helpdesk assistant built with FastAPI and OpenAI. The agent follows a deterministic state machine (intake → clarifying → investigating → resolving → escalating → closed) and uses runtime-guarded tool loops to diagnose and resolve employee IT issues.

## Tools, data sources, and policy

The agent grounds its reasoning in four mock information sources (under `data/`), each exposed as a tool:

| Tool | Source | Use |
|------|--------|-----|
| `kb_search` | Knowledge base (Markdown) | Troubleshooting articles and runbooks |
| `status_api` | System status (JSON) | Service health and known incidents |
| `user_directory` | User records (JSON) | Employee dept/role/location/permissions |
| `resolution_history` | Past tickets (JSON) | How similar issues were resolved before |

The agent is required to ground in at least one tool before proposing a resolution
(enforced in `policy/`), so it cannot answer common issues from the model's memory alone.
Business authorization rules live in `data/policies/policies.json`, but they are not exposed
as an LLM tool. The runtime loads them through `policy/engine.py` and blocks unauthorized
resolutions before they become user-visible.

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

# retrieve full case state, including the human-handoff package when escalated
curl http://localhost:8000/case/<case_id> | python3 -m json.tool

# health check
curl http://localhost:8000/health
```

### Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/chat` | Send a message; returns the agent reply, phase, and `is_closed` |
| `GET`  | `/case/{case_id}` | Full case snapshot — phase, confidence, facts, and the `escalation_context` handoff package (`null` until escalated) |
| `GET`  | `/health` | Liveness check |

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
| `confidence` | Evidence-calibrated confidence at the last turn (see below) |
| `phase` | Current state machine phase |
| `llm_calls` / `prompt_tokens` / `completion_tokens` / `llm_latency_ms` | Per-case cost & latency accounting |

`log_case_closed` also emits an `estimated_cost_usd` derived from token counts and the
per-1K pricing in `config.py` (override via `LLM_PROMPT_COST_PER_1K` /
`LLM_COMPLETION_COST_PER_1K`).

### Confidence calibration

The transition thresholds — and the wording the employee sees — run on a *calibrated*
confidence, not the LLM's raw self-report (`runtime/calibration.py`): confidence is
capped at the borderline threshold until at least one tool has been called, and
discounted for each prior resolution attempt the user did not confirm
(`CONFIDENCE_RETRY_PENALTY`, default 0.15). The coefficients are hand-set for the MVP
and tunable via env; with a labelled evaluation set they could be fit from observed
resolution correctness.

Use `observability.logger` to emit structured JSON logs at the end of each turn or on case close:

```python
from observability.logger import log_turn, log_case_closed

log_turn(case)        # per-turn snapshot
log_case_closed(case) # final summary when phase reaches CLOSED
```

Logs are emitted via Python's standard `logging` module under the `agentic_it_support` logger.
