import time

from agentic_it_support.agent.proposals import AgentProposal
from agentic_it_support.config.settings import Settings
from agentic_it_support.agent.parser import ProposalParseError
from agentic_it_support.llm.client import BaseLLMClient, LLMClientError
from agentic_it_support.observability.event_tracing import (
    InMemoryEventLog,
    record_escalation,
    record_guard,
    record_limit_hit,
    record_llm_call,
    record_llm_parse_error,
    record_turn_end,
    record_turn_start,
)
from agentic_it_support.runtime import limits
from agentic_it_support.runtime.guards import check_guard
from agentic_it_support.runtime.limits import CorrectionBudget
from agentic_it_support.runtime.message_builder import build_messages
from agentic_it_support.runtime.result import Allow, Escalate, Retry, Terminate, Continue
from agentic_it_support.state.case_state import CaseState
from agentic_it_support.tools.base import BaseTool
from agentic_it_support.runtime.executor import execute
from agentic_it_support.runtime.handoff import finalize_handoff


def run_turn(
    case: CaseState,
    user_message: str,
    *,
    llm: BaseLLMClient,
    tools: dict[str, BaseTool],
    settings: Settings,
    event_log: InMemoryEventLog,
) -> str:
    # 1. Setup
    case.add_user_message(user_message)
    case.tool_calls_this_turn = 0  # Reset per turn tool budget(cap = x/turn)
    record_turn_start(event_log, case, user_message)

    # 2. Main loop -> Terminate, Escalate
    decision = _run_agent_loop(case, llm=llm, tools=tools, settings=settings, event_log=event_log)

    # 3. Exit -> Message or File
    match decision:
        case Terminate(message=message):
            record_turn_end(event_log, case, message)
            return message

        case Escalate(reason=reason):
            record_escalation(event_log, case, reason)
            reply = finalize_handoff(case, reason, output_dir=settings.handoff_output_dir, event_log=event_log)
            record_turn_end(event_log, case, reply)
            return reply

def _run_agent_loop(
    case: CaseState,
    *,
    llm: BaseLLMClient,
    tools: dict[str, BaseTool],
    settings: Settings,
    event_log: InMemoryEventLog,
) -> Terminate | Escalate:
    correction: str | None = None
    correction_budget = CorrectionBudget(max_corrections=settings.limits.max_corrections)

    for _ in range(settings.limits.max_inner_iterations):

        # Gate 1: runtime preflight checks
        if limits.llm_case_limit_reached(case, settings.limits):
            record_limit_hit(event_log, case, "max_llm_calls_per_case")
            return Escalate(reason="case-level LLM call limit reached before resolution")

        # Gate 2: request and parse the next agent proposal
        start = time.perf_counter()
        try:
            proposal = _call_agent(case, correction=correction, llm=llm, settings=settings)

        except ProposalParseError as exc:
            # Contract failure: retry with correction while budget remains
            record_llm_parse_error(event_log, case, str(exc))
            if correction_budget.record_correction():
                record_limit_hit(event_log, case, "max_corrections")
                return Escalate("LLM repeatedly failed to produce a valid AgentProposal")
            correction = (
                f"Your previous response could not be parsed as a valid AgentProposal: {exc}. "
                "Respond with a single JSON object that matches the required schema."
            )
            continue

        except LLMClientError:
            # Provider failure: stop this turn and escalate
            return Escalate("LLM provider error during agent proposal request")
        case.llm_calls_total += 1  # Count only successfully parsed LLM proposals
        record_llm_call(event_log, case, proposal.action.value, (time.perf_counter() - start) * 1000)

        # Gate 3: guard proposal
        guard_result = check_guard(case, proposal, tools, runtime_limits=settings.limits, confidence_settings=settings.confidence, policy_file=settings.data_dir/settings.policy_file)

        match guard_result:
            case Allow():
                record_guard(event_log, case, proposal.action.value, "allow")
                correction = None

            case Retry(correction=message):
                record_guard(event_log, case, proposal.action.value, "retry", message)
                if correction_budget.record_correction():
                    record_limit_hit(event_log, case, "max_corrections")
                    return Escalate("LLM repeatedly failed guard checks")
                correction = message
                continue

        # Gate 4: executor: CALL_TOOL, ASK_USER, RESOLVE, ESCALATE actions -> Continue, Terminate(ASK_USER, RESOLVE), Escalate
        outcome = execute(case, proposal, tools, runtime_limits=settings.limits, confidence_settings=settings.confidence, event_log=event_log)

        match outcome:
            case Continue():
                correction = None
                continue
            case Terminate(message=message):
                return Terminate(message)
            case Escalate(reason=reason):
                return Escalate(reason)

    record_limit_hit(event_log, case, "max_inner_iterations")
    return Escalate("maximum investigation steps reached before resolution")


def _call_agent(case: CaseState, correction: str | None, llm: BaseLLMClient, settings: Settings) -> AgentProposal:
    llm_input = build_messages(case, correction=correction, context_settings=settings.message_context)
    return llm.call(llm_input)
