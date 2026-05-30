from agent.proposals import AgentProposal
from runtime.diagnosis_policy import DiagnosisPolicyDecision, check_diagnosis_policy
from state.case_state import CaseState


PolicyDecision = DiagnosisPolicyDecision


def check(case: CaseState, proposal: AgentProposal) -> PolicyDecision:
    """Compatibility entry point for existing imports.

    Diagnosis and escalation workflow boundaries now live in
    runtime.diagnosis_policy so they are not duplicated across controller,
    transitions, and policy modules.
    """
    return check_diagnosis_policy(case, proposal)
