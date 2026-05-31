# Agentic IT Support

An agentic IT helpdesk assistant built with FastAPI and OpenAI. The agent follows a deterministic state machine (intake → clarifying → investigating → resolving → escalating → closed) and uses runtime-guarded tool loops to diagnose and resolve employee IT issues.

## Clear Setup Instructions

### Prerequisites

- Python 3.13 or newer
- `uv` for dependency management
- An OpenAI API key with access to the model configured in `.env`

### 1. Configure environment

Create a local environment file:

```bash
cp .env.example .env
```

Edit `.env`:

```bash
LLM_API_KEY=your_openai_api_key_here
LLM_MODEL=gpt-4o-mini-2024-07-18
LLM_TEMPERATURE=0.2
```

`LLM_API_KEY` is required for the real API and CLI flows. `LLM_MODEL` can be changed to any OpenAI chat model available to your account.

### 2. Install dependencies

```bash
uv sync
```

### 3. Run the FastAPI server

```bash
uv run python main.py
```

The server starts at:

```text
http://localhost:8000
```

Check that it is alive:

```bash
curl http://localhost:8000/health
```

### 4. Send a chat request

Start a new case:

```bash
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "I cannot connect to the VPN, it keeps timing out"}' | python3 -m json.tool
```

Continue the same case by reusing the returned `case_id`:

```bash
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"case_id": "<case_id>", "message": "Cisco AnyConnect on macOS, home WiFi, already restarted"}' | python3 -m json.tool
```

Inspect the full case state:

```bash
curl http://localhost:8000/case/<case_id> | python3 -m json.tool
```

### 5. Run the local CLI

```bash
uv run python cli.py
```

Useful CLI commands:

```text
/status   show current case state
/trace    show recent runtime events
/clear    clear the terminal
/quit     exit
```

### 6. Run tests and evaluation

Run the test suite:

```bash
uv run pytest
```

Run the scenario evaluator:

```bash
uv run python -m evaluation.runner
```

## IT Support Problem

This project focuses on employee-facing IT helpdesk cases: diagnosing common access, connectivity, and service availability issues such as VPN failures, application login problems, MFA/account issues, and suspected service outages.

I chose this problem because these cases are common, repetitive, and often move through a clear support workflow: intake, clarification, investigation, resolution, or escalation. That makes the problem a good fit for an agent with explicit runtime state, tool grounding, and policy-controlled escalation.

## Why an Agentic Approach

A simple FAQ bot is not enough because IT issues are often vague, incomplete, and require checking internal context.

An agentic approach fits because the system can ask follow-up questions, call tools, update case state, and decide whether to resolve or escalate. The runtime controls these steps with validation, policy checks, and explicit workflow state.

## Architecture

The system has five main layers:

1. Input Interface
   The employee sends an issue through the API or CLI. Each message belongs to a case, so the system can continue the same support conversation across turns.

2. Runtime Controller
   The runtime is the main control layer. It builds context, runs the ReAct loop, validates proposed actions, calls tools, updates case state, and decides when to ask the user, continue investigating, resolve, or escalate.

3. LLM Agent
   The LLM does not directly control the workflow. It receives the current case context and returns a structured proposal, such as asking a question, calling a tool, suggesting a resolution, or escalating.

4. Tools and Data Sources
   Tools provide grounded information from mock internal systems, including knowledge base articles, system status, policies, and resolution history.

5. State and Observability
   `CaseState` stores the current phase, conversation, facts, tool traces, confidence, and escalation context. This makes the workflow inspectable and allows the system to hand off context to human IT.

## Design Decisions

The main design decision is to keep the LLM and runtime separate. The LLM proposes what to do next, but the runtime controls whether that action is valid, safe, and allowed in the current workflow state.

`CaseState` is the source of truth for the support process. Instead of relying on hidden model reasoning, the system records the current phase, known facts, tool results, confidence, failed resolutions, and escalation context.

The runtime uses a ReAct-style loop, but tool execution is controlled by the system. The model can request a tool, but the runtime validates the request, checks limits and policy boundaries, executes the tool, and feeds the result back into the next step.

Escalation is handled as a first-class outcome. If the issue is risky, unresolved, outside tool coverage, or below the confidence threshold, the system creates a handoff summary for human IT instead of pretending to solve the case.

This design keeps the agent flexible enough to handle vague IT issues while keeping the workflow explicit, auditable, and bounded by runtime rules.

## Resolution vs. Escalation Boundary

The agent resolves only when the runtime has enough evidence, confidence, and policy approval to provide safe guidance.

The agent escalates when the issue is risky, outside tool coverage, restricted by policy, repeatedly unresolved, or too uncertain to answer safely.

On escalation, the system creates a handoff summary with the case context, tool results, and escalation reason for human IT.

## Simulated Data Sources

The project simulates four internal IT data sources:

- Knowledge base: troubleshooting articles for common issues.
- System status: current service health and known incidents.
- Policy rules: what the agent is allowed to resolve versus what requires human IT.
- Resolution history: similar past cases and how they were handled.

These sources were chosen because real IT support usually depends on internal documentation, live service status, business rules, and previous ticket patterns rather than model knowledge alone.

## Assumptions and Tradeoffs

- Assumption: Most IT support cases follow a predictable workflow.
  Tradeoff: This improves control, testing, and auditability, but is less flexible than a fully autonomous agent.

- Assumption: The LLM should propose actions, while the runtime owns execution, validation, and state transitions.
  Tradeoff: This improves safety and predictability, but requires additional runtime logic such as validators, guards, and policies.

- Assumption: Support decisions should be grounded in tool results and case evidence rather than model intuition alone.
  Tradeoff: This improves reliability and traceability, but increases latency and makes answer quality dependent on tool coverage.

## Evaluation

I evaluated the system with automated test cases and scenario-based runs.

The unit tests check the core runtime behavior: state transitions, action validation, policy boundaries, tool execution, message building, API routes, and CLI behavior.

The evaluation scenarios simulate realistic helpdesk conversations, including VPN issues, account/MFA problems, software access requests, vague intake messages, service degradation, and unresolved cases that should escalate.

The main evaluation criteria are whether the agent resolves safe cases, escalates risky or unsupported cases, uses tools before giving guidance, and preserves enough context for human IT handoff.

## Observability

The system records runtime events, case state, tool traces, confidence, and escalation context. This makes each case inspectable during development and gives human IT enough context when a case is escalated.

## External Dependencies

The project keeps external dependencies small and focused. FastAPI and Uvicorn serve the API, OpenAI provides the LLM client, Pydantic handles structured request/response models, and python-dotenv / pydantic-settings load local configuration.

Development dependencies are limited to pytest for tests and httpx for API route testing.

## Future Improvements

With more time, I would improve the coordination between workflow state, confidence, and policy decisions.

Right now the runtime uses explicit rules to decide when to continue investigating, resolve, or escalate. A stronger version would calibrate confidence with more labeled evaluation data, make policy boundaries more fine-grained, and test more edge cases around repeated failures, ambiguous user input, and partial resolutions.

I would also improve the evaluation harness so it can measure not only final outcomes, but also whether the agent took the right path through the workflow.

## Optional Design Note

The system is currently exposed as a FastAPI service rather than a dedicated UI. This keeps the agent integration-ready: after more load testing and production hardening, the same `/chat` and `/case/{case_id}` APIs could be connected to Slack, Microsoft Teams, or another internal support interface.
