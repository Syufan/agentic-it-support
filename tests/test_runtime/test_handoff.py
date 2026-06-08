import json

from agentic_it_support.observability.event_tracing import InMemoryEventLog
from agentic_it_support.runtime.handoff import finalize_handoff
from agentic_it_support.state.case_state import CaseState, ToolTrace


def test_finalize_handoff_writes_local_json_payload(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    output_dir = tmp_path / "handoffs"
    case = CaseState(confidence=0.42, resolution_attempts=1)
    case.add_user_message("I lost my MFA device")
    case.tool_traces.append(
        ToolTrace(
            tool_name="kb_search",
            inputs={"query": "mfa lost device"},
            output={"results": ["MFA recovery requires identity verification"]},
            success=True,
        )
    )

    finalize_handoff(
        case,
        "MFA reset requires identity verification",
        output_dir=output_dir,
        event_log=InMemoryEventLog(),
    )

    path = output_dir / f"{case.case_id}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["case_id"] == case.case_id
    assert payload["internal_reason"] == "MFA reset requires identity verification"
    assert payload["conversation"] == case.conversation
    assert payload["tool_traces"][0]["tool"] == "kb_search"
    assert payload["resolution_attempts"] == 1
