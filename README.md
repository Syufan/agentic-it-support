# Agentic IT Support

## IT Support Problem Scope and why

This project focuses on employee-facing IT helpdesk triage across connectivity,
account/identity, application access, and service availability.

The agent can provide safe guidance for VPN troubleshooting, password reset self-service, and known incident checks. It can explain approval paths for access requests, but must escalate MFA recovery, account unlocks, and infrastructure changes to human IT.

I chose this problem because IT helpdesk triage is a practical area for agents: many cases are repetitive and tool-grounded, while still testing important boundaries between safe resolution, policy-based routing, and human escalation.

## Why an Agentic Approach

A simple FAQ bot can only answer from static knowledge, and a rule-based flow cannot cover the variety of ways employees describe IT issues. Traditional FAQ search also pushes the cognitive burden onto the employee, who may not know the right terms, system names, or troubleshooting path.

An agentic approach fits because the system can gather context through conversation, check company-specific tools such as service status, knowledge base articles, policies, and past tickets, then decide whether to provide guidance, explain an approval path, or escalate. The runtime keeps these steps controlled with validation, policy checks, and explicit workflow state.

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
   `CaseState` stores the current phase, conversation, tool traces, confidence, and escalation context. This makes the workflow inspectable and allows the system to hand off context to human IT.

## Design Decisions

The main design decision is to keep the LLM and runtime separate. The LLM proposes what to do next, but the runtime controls whether that action is valid, safe, and allowed in the current workflow state.

`CaseState` is the source of truth for the support process. Instead of relying on hidden model reasoning, the system records the current phase, conversation, tool results, confidence, failed resolutions, and escalation context.

The runtime uses a ReAct-style loop, but tool execution is controlled by the system. The model can request a tool, but the runtime validates the request, checks limits and policy boundaries, executes the tool, and feeds the result back into the next step.

Escalation is handled as a first-class outcome. If the issue requires human approval, is outside supported scope, repeatedly fails, or hits runtime limits, the system creates a handoff summary for human IT instead of pretending to solve the case.

This design keeps the agent flexible enough to handle vague IT issues while keeping the workflow explicit, auditable, and bounded by runtime rules.

## Resolution vs. Escalation Boundary

The agent resolves only when the answer is grounded in tool evidence and allowed by policy.

Low confidence blocks `resolve`; it does not automatically trigger escalation.

Escalation happens only when policy marks the action as human-only, resolution attempts are exhausted, or runtime safety limits are reached.

Approval-gated requests should be routed through the approval path rather than directly handed off by default.

On escalation, the system creates a handoff summary with the case context, tool results, policy boundary, and escalation reason.

## Simulated Data Sources

The system uses four mocked internal IT data sources:

- Knowledge base: troubleshooting articles for common employee issues.
- System status: service health and known incidents.
- Policy rules: authorization boundaries for what the agent may resolve, what must follow an approval path, and what requires human IT.
- Resolution history: similar past cases and whether they were resolved by the agent or escalated to human IT.

These sources keep the agent grounded in auditable runtime evidence instead of model knowledge alone.

## Assumptions and Tradeoffs

- Assumption: Most helpdesk cases should make progress within a small number of agent steps, tool calls, and clarification attempts.  
  Tradeoff: Fixed runtime limits prevent runaway loops and cost growth, but difficult cases may be escalated or closed earlier than a human would.

- Assumption: IT support answers should be grounded in tool results and case evidence, not model knowledge alone.  
  Tradeoff: This improves reliability and traceability, but increases latency and makes answer quality depend on tool coverage.

- Assumption: Successful tool results provide useful evidence for estimating confidence.  
  Tradeoff: This makes confidence simple and enforceable at runtime, but can overestimate confidence if a tool succeeds while returning weak, stale, or only loosely relevant information.

- Assumption: The LLM should propose actions, while the runtime owns validation, execution, policy checks, and state transitions.  
  Tradeoff: This improves safety and predictability, but requires additional runtime logic such as validators, guards, policies, and transition rules.

- Assumption: Most IT support cases follow a predictable workflow: intake, clarification, investigation, resolution, and escalation.  
  Tradeoff: This improves control, testing, and auditability, but is less flexible than a fully autonomous agent.

## How to run

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

Activate the environment, then run the server directly:

```bash
source .venv/bin/activate
agentic-it-api
```

The server starts at `http://localhost:8000`. Stop it with `Ctrl+C`.

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

With the environment activated (step 3):

```bash
agentic-it-cli
```

Useful CLI commands:

```text
/status   show current case state
/trace    show recent runtime events
/clear    clear the terminal
/quit     exit
```

### 6. Run tests and evaluation

Run the test suite (environment activated, step 3):

```bash
pytest
```

Run the scenario evaluator:

```bash
python -m evaluation.runner
```

## Evaluation

The project uses scenario-based evaluation in addition to unit tests.

Each evaluation scenario simulates a realistic IT support conversation and checks the expected outcome, such as whether the case should resolve, escalate, or use tools before responding. The scenario runner measures final state and basic runtime behavior, including escalation status, resolution status, and tool-call counts.

The evaluation scenarios cover:
- safe self-service guidance, such as VPN troubleshooting and password reset;
- clarification flows where the agent must ask for missing information;
- policy-routed requests, such as access grants that require approval;
- human-only cases, such as MFA recovery and network configuration changes;
- known service incidents, such as degraded Salesforce status.

Unit tests cover the lower-level runtime control flow, including action validation, policy checks, invalid proposal recovery, tool limits, and state transitions.

The goal is not only to check whether the final answer sounds reasonable, but also whether the runtime follows the intended control path:

```text
LLM proposal → workflow guard → policy check → action execution → state transition
```

## Future Improvements

With more time, I would improve evaluation, confidence calibration, case grounding, and policy routing.

### 1. Stronger Evaluation

The current evaluation checks final outcomes and basic runtime behavior. A stronger version would also score the full control path, including action sequence, tool choice, policy compliance, and whether the agent avoided unnecessary clarification or escalation.

### 2. Confidence Calibration

The current confidence score is a simple evidence-based heuristic from successful tool results. I would calibrate it with labeled support cases and account for evidence quality, freshness, and relevance.

### 3. Planning and Structured Clarification

For complex or ambiguous cases, I would add a planning mode before normal investigation begins. The agent would first produce a diagnostic plan with missing context, affected system, risk level, tools to check, and escalation boundary.

I would also replace free-form clarification with a structured clarification tool. Each question would be tied to a missing case field, so the runtime can tell whether the case is ready for investigation or still needs user input.

I would also cross-check LLM-inferred resolution confirmations against the latest user message before updating case state.

### 4. Explicit Case Grounding

I would make case grounding more explicit in state. A production version could track fields such as affected target, symptom, environment, start time, user goal, and whether each field is unknown, inferred, or confirmed.

I would also replace the current keyword-based affected-target check with structured grounding or named-entity extraction.

I did not implement the full grounding state in this prototype because it would require changes to the proposal schema, prompt format, state update logic, ambiguity handling, and tests. Instead, this version uses the existing CaseState, prompt guidance, and a minimum completeness gate before RESOLVE to prevent vague issues from becoming fake grounded answers.

### 5. Policy and Context Improvements

The current policy layer uses a JSON-backed mock policy source. In production, I would replace this with a policy lookup tool or enterprise policy API that returns structured authorization facts for self-service, approval-routed, and human-only actions.

I would also validate the policy data shape at load time. Today `_load_policy_rules` reads the JSON and indexes keys directly, so a malformed policy file fails late at runtime; I would parse it through a Pydantic model so a bad shape is rejected with a clear error when the rules are loaded.

I would also improve context projection. The runtime keeps full case memory, but the model should receive only bounded, decision-relevant context such as phase, confidence, retry attempts, and recent tool evidence.


## Optional Design Note

The system is currently exposed as a FastAPI service rather than a dedicated UI. This keeps the agent integration-ready: after more load testing and production hardening, the same `/chat` and `/case/{case_id}/trace` APIs could be connected to Slack, Microsoft Teams, or another internal support interface.

## Observability

The system records runtime events, case state, tool traces, confidence, and escalation context. This makes each case inspectable during development and gives human IT enough context when a case is escalated.

## External Dependencies

The project keeps external dependencies small and focused. FastAPI and Uvicorn serve the API, OpenAI provides the LLM client, Pydantic handles structured request/response models, and python-dotenv / pydantic-settings load local configuration.

Development dependencies are limited to pytest for tests and httpx for API route testing.
