import json

from agentic_it_support.agent.prompts import clarifying, escalating, intake, investigating, resolving
from agentic_it_support.config.settings import ContextSettings
from agentic_it_support.llm.client import LLMInput
from agentic_it_support.state.case_state import CaseState, Phase

_PROMPTS: dict[Phase, str] = {
    Phase.INTAKE:        intake.SYSTEM_PROMPT,
    Phase.CLARIFYING:    clarifying.SYSTEM_PROMPT,
    Phase.INVESTIGATING: investigating.SYSTEM_PROMPT,
    Phase.RESOLVING:     resolving.SYSTEM_PROMPT,
    Phase.ESCALATING:    escalating.SYSTEM_PROMPT,
}

def build_messages(case: CaseState, *, correction: str | None = None, context_settings: ContextSettings) -> LLMInput:
    
    # Select the phase-specific system prompt and apply any correction.
    system = _PROMPTS.get(case.phase, investigating.SYSTEM_PROMPT)
    if correction:
        system = system + "\n\n[Correction] " + correction
    
    # Copy conversation history and append the current case snapshot.
    messages = [dict(m) for m in case.conversation]
    observation = _build_observation(case, context_settings)

    # Attach the snapshot to the latest user turn, or add it as a user message.
    if messages and messages[-1]["role"] == "user":
        messages[-1]["content"] = messages[-1]["content"] + "\n\n" + observation
    else:
        messages.append({"role": "user", "content": observation})

    return LLMInput(system=system, messages=messages)


def _build_observation(case: CaseState, context_settings: ContextSettings) -> str:
    # Start with the current workflow state.
    lines = [
        "[Case State]",
        f"Phase: {case.phase.value}",
    ]

    # Add the most recent bounded tool results.
    if case.tool_traces:
        lines.append("Tool results:")
        for trace in case.tool_traces[-context_settings.max_tool_traces:]:
            status = "ok" if trace.success else "failed"
            output_preview = json.dumps(trace.output)[:context_settings.tool_output_preview_chars]
            lines.append(f"  [{status}] {trace.tool_name}: {output_preview}")

    return "\n".join(lines)
