import json
from pathlib import Path

from agentic_it_support.agent.proposals import AgentAction, AgentProposal
from agentic_it_support.config.settings import Settings
from agentic_it_support.llm.client import MockLLMClient
from evaluation.runner import run_scenario

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def test_run_scenario_executes_runtime_and_checks_expectations(tmp_path):
    scenario = {
        "name": "asks for missing context",
        "messages": ["Something is broken"],
        "expect": {
            "resolved": False,
            "escalated": False,
            "max_tool_calls": 0,
        },
    }
    path = tmp_path / "scenario.json"
    path.write_text(json.dumps(scenario), encoding="utf-8")
    llm = MockLLMClient([
        AgentProposal(
            action=AgentAction.ASK_USER,
            message="Which app or device is affected?",
        )
    ])

    result = run_scenario(
        path,
        llm,
        Settings(
            _env_file=None,
            data_dir=_DATA_DIR,
            handoff_output_dir=tmp_path / "handoffs",
        ),
    )

    assert result.passed is True
    assert result.responses == ["Which app or device is affected?"]
    assert result.evaluation.escalated is False
    assert result.evaluation.tool_calls_total == 0
