from datetime import datetime, timezone

from agent.llm import BaseLLMClient
from agent.proposals import AgentAction, AgentProposal
from config import CONFIDENCE_HIGH
from runtime.message_builder import build_messages
from runtime.transitions import TransitionResult, evaluate_transition
from runtime.validator import validate_proposal
from state.case_state import CaseState, ToolTrace
from tools.base import BaseTool, ToolResult

_MAX_INNER_ITERATIONS = 10


def run_turn(
    case: CaseState,
    user_message: str,
    llm: BaseLLMClient,
    tool_registry: dict[str, BaseTool],
) -> str:
    case.conversation.append({"role": "user", "content": user_message})

    for _ in range(_MAX_INNER_ITERATIONS):
        llm_input = build_messages(case)
        proposal = llm.call(llm_input)

        validation = validate_proposal(case, proposal)
        if not validation.valid:
            response = (
                "I ran into an issue processing your request. "
                "Transferring you to a specialist."
            )
            case.conversation.append({"role": "assistant", "content": response})
            return response

        _project_to_state(case, proposal)

        if proposal.action == AgentAction.CALL_TOOL:
            _execute_tool(case, proposal, tool_registry)
            _apply_transition(case, evaluate_transition(case))
            continue

        if proposal.action == AgentAction.RESOLVE:
            case.resolution_attempts += 1

        if proposal.action == AgentAction.ESCALATE:
            _build_escalation_context(case, proposal)
            case.handoff_completed = True

        _apply_transition(case, evaluate_transition(case))

        response = _format_response(proposal)
        case.conversation.append({"role": "assistant", "content": response})
        return response

    response = (
        "I was unable to resolve this within the allotted steps. "
        "Transferring to a specialist with full context."
    )
    case.conversation.append({"role": "assistant", "content": response})
    return response


def _project_to_state(case: CaseState, proposal: AgentProposal) -> None:
    case.confidence = proposal.confidence
    case.missing_info_source = proposal.missing_info_source
    case.missing_info = list(proposal.missing_info)
    case.has_safe_low_risk_guidance = proposal.has_safe_low_risk_guidance
    case.new_critical_fact_added = proposal.new_critical_fact_added
    if proposal.user_confirmed_resolution is not None:
        case.user_confirmed_resolution = proposal.user_confirmed_resolution


def _execute_tool(
    case: CaseState,
    proposal: AgentProposal,
    tool_registry: dict[str, BaseTool],
) -> None:
    tool_name = proposal.tool_name or ""
    tool = tool_registry.get(tool_name)

    if tool is None:
        result = ToolResult(success=False, data={}, error=f"tool '{tool_name}' not available")
    else:
        result = tool.run(proposal.tool_input)

    case.tool_traces.append(ToolTrace(
        tool_name=tool_name,
        inputs=proposal.tool_input,
        output=result.data if result.success else {"error": result.error},
        success=result.success,
        budget_mode=case.budget_mode,
        timestamp=datetime.now(timezone.utc),
    ))

    if result.success:
        case.facts[f"{tool_name}_result"] = result.data
    else:
        case.facts[f"{tool_name}_error"] = result.error or "unknown error"

    case.tool_calls_current_investigation += 1
    case.tool_calls_total += 1


def _apply_transition(case: CaseState, result: TransitionResult) -> None:
    case.phase = result.next_phase
    if result.budget_mode is not None:
        case.budget_mode = result.budget_mode
    if result.reset_tool_counter:
        case.tool_calls_current_investigation = 0
    if result.set_exception_used:
        case.exception_used = True


def _build_escalation_context(case: CaseState, proposal: AgentProposal) -> None:
    issue_description = next(
        (m["content"] for m in case.conversation if m["role"] == "user"), ""
    )
    case.escalation_context = {
        "escalation_reason": proposal.escalation_reason,
        "confidence": proposal.confidence,
        "issue_description": issue_description,
        "conversation": list(case.conversation),
        "facts": dict(case.facts),
        "hypotheses": list(case.hypotheses),
        "tool_traces": [
            {
                "tool": t.tool_name,
                "success": t.success,
                "inputs": t.inputs,
                "output": t.output,
            }
            for t in case.tool_traces
        ],
        "failed_resolutions": list(case.failed_resolutions),
        "resolution_attempts": case.resolution_attempts,
    }


def _format_response(proposal: AgentProposal) -> str:
    if proposal.action == AgentAction.ESCALATE:
        return (
            "I wasn't able to fully resolve this issue. "
            "I'm connecting you with an IT specialist who will have all the context — "
            "you won't need to repeat yourself."
        )

    if proposal.action == AgentAction.RESOLVE:
        message = proposal.message or ""
        if proposal.confidence >= CONFIDENCE_HIGH:
            return f"I found a likely fix for your issue: {message}"
        return f"I'm not fully certain, but this is a safe first step to try: {message}"

    return proposal.message or ""
