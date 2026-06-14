from pathlib import Path
from agentic_it_support.agent.proposals import AgentProposal
from agentic_it_support.runtime.guards.business import check_business
from agentic_it_support.runtime.guards.diagnosis import check_diagnosis
from agentic_it_support.config.settings import ConfidenceSettings, RuntimeLimits
from agentic_it_support.runtime.guards.validation import validate_proposal
from agentic_it_support.runtime.result import Allow, Retry
from agentic_it_support.state.case_state import CaseState
from agentic_it_support.tools.base import BaseTool

def check_guard(case: CaseState, proposal: AgentProposal, tools: dict[str, BaseTool], runtime_limits: RuntimeLimits, confidence_settings: ConfidenceSettings, policy_file: Path) -> Allow | Retry:
    
    # 1. Validate proposal structure, phase legality, required fields, and tool budgets
    validation = validate_proposal(case, proposal, valid_tools=set(tools), runtime_limits=runtime_limits)
    if not validation.allowed:
        assert validation.correction is not None
        return Retry(validation.correction)
    
    # 2. Enforce diagnostic grounding before resolution
    diagnosis = check_diagnosis(case, proposal, confidence_settings)
    if not diagnosis.allowed:
        assert diagnosis.correction is not None
        return Retry(diagnosis.correction)

    # 3. Enforce business authorization and escalation boundaries
    business = check_business(case, proposal, policy_file)
    if not business.allowed:
        assert business.correction is not None
        return Retry(business.correction)
    
    # All guards passed; the proposal is allowed to execute
    return Allow()
