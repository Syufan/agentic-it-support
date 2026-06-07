

from agentic_it_support.agent.proposals import AgentProposal
from agentic_it_support.config.settings import Settings
from agentic_it_support.agent.parser import ProposalParseError
from agentic_it_support.llm.client import BaseLLMClient, LLMClientError
from agentic_it_support.runtime import limits
from agentic_it_support.runtime.guards import check_guard
from agentic_it_support.runtime.limits import CorrectionBudget
from agentic_it_support.runtime.message_builder import build_messages
from agentic_it_support.runtime.result import Allow, Escalate, Retry, Terminate, Continue
from agentic_it_support.state.case_state import CaseState
from agentic_it_support.tools.base import BaseTool
from agentic_it_support.runtime.executor import execute
from agentic_it_support.runtime.handoff import finalize_handoff


def run_turn(case: CaseState, user_message: str, *, llm: BaseLLMClient, tools: dict[str, BaseTool], settings: Settings) -> str:
    # 1. Setup
    case.add_user_message(user_message)
    case.tool_calls_this_turn = 0 # Reset per turn tool budget(cap = x/turn)

    # 2. Main loop -> Terminate, Escalate
    decision = _run_agent_loop(case, llm=llm, tools=tools, settings=settings)

    # 3. Exit
    match decision:
        case Terminate(message=message):
            return message

        case Escalate(reason=reason):
            return finalize_handoff(case, reason)

def _run_agent_loop(case: CaseState, *,  llm: BaseLLMClient, tools: dict[str, BaseTool], settings: Settings) -> Terminate | Escalate:
    correction: str | None = None
    correction_budget = CorrectionBudget(max_corrections=settings.limits.max_corrections)
    
    for _ in range(settings.limits.max_inner_iterations):
        
        # Gate 1: runtime preflight checks
        if limits.llm_case_limit_reached(case, settings.limits):
            return Escalate(reason="case-level LLM call limit reached before resolution")

        # Gate 2: request and parse the next agent proposal
        try:
            proposal = _call_agent(case, correction=correction, llm=llm)
        
        except ProposalParseError as exc:
            # Contract failure: retry with correction while budget remains
            if correction_budget.record_correction():
                return Escalate("LLM repeatedly failed to produce a valid AgentProposal")
            correction = (
                f"Your previous response could not be parsed as a valid AgentProposal: {exc}. "
                "Respond with a single JSON object that matches the required schema."
            )
            continue

        except LLMClientError:
            # Provider failure: stop this turn and escalate
            return Escalate("LLM provider error during agent proposal request")
        case.llm_calls_total += 1 # Count only successfully parsed LLM proposals

        # Gate 3: guard proposal
        guard_result = check_guard(case, proposal, tools, runtime_limits=settings.limits, confidence_settings=settings.confidence, policy_file=settings.data_dir/settings.policy_file)

        match guard_result:
            case Allow():
                correction = None

            case Retry(correction=message):
                if correction_budget.record_correction():
                    return Escalate("LLM repeatedly failed guard checks")
                correction = message
                continue

        # Gate 4: executor: CALL_TOOL, ASK_USER, RESOLVE, ESCALATE actions -> Continue, Terminate(ASK_USER, RESOLVE), Escalate
        outcome = execute(case, proposal, tools, runtime_limits=settings.limits, confidence_settings=settings.confidence)

        match outcome:
            case Continue():
                correction = None
                continue
            case Terminate(message=message):
                return Terminate(message)
            case Escalate(reason=reason):
                return Escalate(reason)
        
    return Escalate("maximum investigation steps reached before resolution")


def _call_agent(case: CaseState, correction: str | None, llm: BaseLLMClient) -> AgentProposal:
    llm_input = build_messages(case, correction=correction)
    return llm.call(llm_input)
