import json
from dataclasses import dataclass

from agent.prompts import clarifying, escalating, intake, investigating, resolving
from state import budget as budget_
from state.case_state import CaseState, Phase

_PROMPTS: dict[Phase, str] = {
    Phase.INTAKE:        intake.SYSTEM_PROMPT,
    Phase.CLARIFYING:    clarifying.SYSTEM_PROMPT,
    Phase.INVESTIGATING: investigating.SYSTEM_PROMPT,
    Phase.RESOLVING:     resolving.SYSTEM_PROMPT,
    Phase.ESCALATING:    escalating.SYSTEM_PROMPT,
}


@dataclass
class LLMInput:
    system: str
    messages: list[dict[str, str]]


def build_messages(case: CaseState, correction: str | None = None) -> LLMInput:
    system = _PROMPTS.get(case.phase, investigating.SYSTEM_PROMPT)
    messages = [dict(m) for m in case.conversation]
    observation = _build_observation(case)

    if correction:
        observation = observation + "\n\n[Correction] " + correction

    if messages and messages[-1]["role"] == "user":
        messages[-1]["content"] = messages[-1]["content"] + "\n\n" + observation
    else:
        messages.append({"role": "user", "content": observation})

    return LLMInput(system=system, messages=messages)


def _build_observation(case: CaseState) -> str:
    lines = [
        "[Case State]",
        f"Phase: {case.phase.value}",
        f"Confidence: {case.confidence}",
        f"Tool budget: used {case.tool_calls_current_investigation} / remaining {budget_.remaining(case.budget_mode, case.tool_calls_current_investigation)} (mode: {case.budget_mode.value})",
    ]

    if case.facts:
        lines.append(f"Facts: {json.dumps(case.facts)}")

    if case.hypotheses:
        lines.append(f"Hypotheses: {'; '.join(case.hypotheses)}")

    if case.missing_info:
        lines.append(f"Missing info: {', '.join(case.missing_info)}")

    if case.tool_traces:
        lines.append("Tool results:")
        for trace in case.tool_traces[-3:]:
            status = "ok" if trace.success else "failed"
            output_preview = json.dumps(trace.output)[:200]
            lines.append(f"  [{status}] {trace.tool_name}: {output_preview}")

    if case.failed_resolutions:
        lines.append(f"Failed resolutions: {len(case.failed_resolutions)}")

    return "\n".join(lines)
