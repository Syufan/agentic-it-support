
from agentic_it_support.config.settings import ConfidenceSettings, RuntimeLimits
from agentic_it_support.state.case_state import CaseState, ToolTrace
from agentic_it_support.agent.proposals import AgentAction, AgentProposal
from agentic_it_support.tools.base import BaseTool
from agentic_it_support.runtime.executor.confidence import compute_confidence
from agentic_it_support.runtime.result import Continue, Escalate, Terminate
from agentic_it_support.runtime.transitions import evaluate_transition


def execute(case: CaseState, proposal: AgentProposal, tools: dict[str, BaseTool], *, runtime_limits: RuntimeLimits, confidence_settings: ConfidenceSettings) -> Continue | Terminate | Escalate:
    _sync_resolution_confirmation(case, proposal, confidence_settings)

    match proposal.action:
        case AgentAction.CALL_TOOL:
            return _execute_tool_action(case, proposal, tools, confidence_settings)

        case AgentAction.ASK_USER:
            return _execute_ask_user(case, proposal)

        case AgentAction.RESOLVE:
            return _execute_resolution(case, proposal, runtime_limits)

        case AgentAction.ESCALATE:
            return _execute_escalation(proposal)
        
        case _:
            raise ValueError(f"unsupported agent action: {proposal.action}")

def _sync_resolution_confirmation(case: CaseState, proposal: AgentProposal, confidence_settings: ConfidenceSettings) -> None:
    # Sync resolution confirmation and apply failure penalty before transition
    if proposal.user_confirmed_resolution is not None:
        case.user_confirmed_resolution = proposal.user_confirmed_resolution
    if proposal.user_confirmed_resolution is False:
        case.resolution_attempts += 1
        case.confidence = compute_confidence(case, confidence_settings)

def _execute_tool_action(case: CaseState, proposal: AgentProposal, tools: dict[str, BaseTool], confidence_settings: ConfidenceSettings) -> Continue:
    # Tool use means the case has enough detail to investigate
    case.clarification_attempts = 0
    
    # Run the requested tool save result to tool traces
    _execute_tool(case, proposal, tools)

    # Calculate evidence from the updated tool traces
    case.confidence = compute_confidence(case, confidence_settings)

    _apply_transition(case, AgentAction.CALL_TOOL)

    return Continue()

def _execute_tool(case: CaseState, proposal: AgentProposal, tools: dict[str, BaseTool]) -> None:
    tool_name = proposal.tool_name
    if tool_name is None:
        raise ValueError("CALL_TOOL proposal missing tool_name after validation")
    
    tool = tools[tool_name]
    result = tool.run(proposal.tool_input)

    case.tool_traces.append(
        ToolTrace(
            tool_name=tool_name,
            inputs=proposal.tool_input,
            output=result.data if result.success else {"error": result.error},
            success=result.success,
        )
    )
    case.tool_calls_this_turn += 1
    case.tool_calls_total += 1

def _apply_transition(case: CaseState, action: AgentAction) -> None:
    result = evaluate_transition(case, action)
    case.phase = result.next_phase

def _execute_ask_user(case: CaseState, proposal: AgentProposal) -> Terminate:
    if proposal.message is None:
        raise ValueError("ASK_USER proposal missing message after validation")

    _apply_transition(case, AgentAction.ASK_USER)
    case.user_confirmed_resolution = None
    case.add_assistant_message(proposal.message)
    return Terminate(proposal.message)

def _execute_resolution(case: CaseState, proposal: AgentProposal, runtime_limits: RuntimeLimits) -> Terminate | Escalate:
    message = proposal.message
    if message is None:
        raise ValueError("RESOLVE proposal missing message after validation")

    if case.resolution_attempts >= runtime_limits.max_resolution_attempts:
        return Escalate("maximum resolution attempts reached before successful confirmation")
    
    _apply_transition(case, AgentAction.RESOLVE)
    case.add_assistant_message(message)
    return Terminate(message)


def _execute_escalation(proposal: AgentProposal) -> Escalate:
    reason = proposal.escalation_reason
    if reason is None:
        raise ValueError("ESCALATE proposal missing escalation_reason after validation")
    return Escalate(reason)

#TODO LLM 可以提取用户确认状态，但 runtime 要用最新用户文本做 最小交叉验证。