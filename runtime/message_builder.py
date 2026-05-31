import json

from agent.prompts import clarifying, escalating, intake, investigating, resolving
from llm.client import LLMInput
from state.case_state import CaseState, Phase

_PROMPTS: dict[Phase, str] = {
    Phase.INTAKE:        intake.SYSTEM_PROMPT,
    Phase.CLARIFYING:    clarifying.SYSTEM_PROMPT,
    Phase.INVESTIGATING: investigating.SYSTEM_PROMPT,
    Phase.RESOLVING:     resolving.SYSTEM_PROMPT,
    Phase.ESCALATING:    escalating.SYSTEM_PROMPT,
}

#: how many recent tool results to surface to the LLM, and how much of each
_MAX_TOOL_TRACES_IN_CONTEXT = 3
_TOOL_OUTPUT_PREVIEW_CHARS = 200


def build_messages(case: CaseState, correction: str | None = None) -> LLMInput:
    system = _PROMPTS.get(case.phase, investigating.SYSTEM_PROMPT)

    # A correction is a runtime instruction ("your last response was rejected,
    # do X"), so it belongs in the system prompt, not buried in the user turn.
    if correction:
        system = system + "\n\n[Correction] " + correction

    messages = [dict(m) for m in case.conversation]
    observation = _build_observation(case)

    if messages and messages[-1]["role"] == "user":
        messages[-1]["content"] = messages[-1]["content"] + "\n\n" + observation
    else:
        messages.append({"role": "user", "content": observation})

    return LLMInput(system=system, messages=messages)


def _build_observation(case: CaseState) -> str:
    lines = [
        "[Case State]",
        f"Phase: {case.phase.value}",
    ]

    if case.facts:
        lines.append(f"Facts: {json.dumps(case.facts)}")

    if case.hypotheses:
        lines.append(f"Hypotheses: {'; '.join(case.hypotheses)}")

    if case.missing_info:
        lines.append(f"Missing info: {', '.join(case.missing_info)}")

    if case.tool_traces:
        lines.append("Tool results:")
        for trace in case.tool_traces[-_MAX_TOOL_TRACES_IN_CONTEXT:]:
            status = "ok" if trace.success else "failed"
            output_preview = json.dumps(trace.output)[:_TOOL_OUTPUT_PREVIEW_CHARS]
            lines.append(f"  [{status}] {trace.tool_name}: {output_preview}")

    if case.failed_resolutions:
        lines.append(f"Failed resolutions: {len(case.failed_resolutions)}")

    return "\n".join(lines)
