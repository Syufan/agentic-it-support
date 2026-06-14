from dataclasses import dataclass
import re

from agentic_it_support.agent.proposals import AgentAction, AgentProposal
from agentic_it_support.config.settings import ConfidenceSettings
from agentic_it_support.state.case_state import CaseState


@dataclass(frozen=True)
class DiagnosisResult:
    """Diagnosis outcome; allowed results have no details, invalid results include both reason and correction."""
    allowed: bool
    reason: str | None = None
    correction: str | None = None


def check_diagnosis(case: CaseState, proposal: AgentProposal, confidence_settings: ConfidenceSettings) -> DiagnosisResult:

    if proposal.action == AgentAction.RESOLVE:
        # A resolution needs an identified affected target, only user-provided text count
        if not _names_affected_target(case):
            return DiagnosisResult(
                False,
                "resolve blocked: no affected app/service/device/network identified yet",
                (
                    "Don't resolve yet - the case hasn't identified which app, service, "
                    "device, or network is affected. A generic answer is not a resolution. "
                    "Ask the employee which specific system they mean."
                )
            )

        # Resolution requires confidence grounded in successful evidence
        if case.confidence < confidence_settings.resolve_threshold:
            return DiagnosisResult(
                False,
                "resolve blocked: evidence-based confidence below the resolve threshold",
                (
                    "Don't propose a fix yet - it isn't grounded in evidence. Call a tool "
                    "and get a successful result first, then resolve."
                ),
            )

    return DiagnosisResult(True)

# for _names_affected_target function
_AFFECTED_TARGET = re.compile(
    r"\b("
    r"app|apps|application|applications|software|program|browser|website|site|web|portal|"
    r"dashboard|email|inbox|mailbox|account|password|crm|"
    r"network|wifi|wi-fi|internet|ethernet|vpn|server|gateway|database|db|"
    r"computer|laptop|desktop|machine|pc|mac|macbook|phone|iphone|android|ipad|tablet|"
    r"printer|monitor|screen|display|keyboard|mouse|headset|webcam|camera|device|hardware|drive|disk|"
    r"okta|salesforce|snowflake|grafana|github|aws|jenkins|slack|zoom|jira|confluence|adobe|gmail|outlook|teams|shadowrocket|google"
    r")\b"
)

def _names_affected_target(case: CaseState) -> bool:
    """Return true when user text names an affected app, service, device, or network."""
    text = " ".join( m["content"].lower() for m in case.conversation if m["role"] == "user")
    return bool(_AFFECTED_TARGET.search(text))
