"""Batch evaluation runner.

Usage:
    uv run python -m evaluation.runner
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from agent.llm import RealLLMClient
from evaluation import EvaluationResult, evaluate
from runtime.controller import run_turn
from state.case_state import CaseState
from tools.kb_search import KBSearchTool
from tools.status_api import StatusAPITool
from tools.user_directory import UserDirectoryTool

_SCENARIOS_DIR = Path(__file__).parent / "scenarios"

_TOOLS = {
    "kb_search": KBSearchTool(),
    "status_api": StatusAPITool(),
    "user_directory": UserDirectoryTool(),
}


@dataclass
class ScenarioResult:
    name: str
    passed: bool
    evaluation: EvaluationResult
    responses: list[str]
    failure_reason: str | None = None


def run_scenario(path: Path) -> ScenarioResult:
    scenario = json.loads(path.read_text())
    case = CaseState()
    llm = RealLLMClient()
    responses: list[str] = []

    for message in scenario["messages"]:
        responses.append(run_turn(case, message, llm, _TOOLS))

    result = evaluate(case)
    passed, reason = _check(result, scenario.get("expect", {}))

    return ScenarioResult(
        name=scenario["name"],
        passed=passed,
        evaluation=result,
        responses=responses,
        failure_reason=reason,
    )


def _check(result: EvaluationResult, expect: dict) -> tuple[bool, str | None]:
    if "escalated" in expect and result.escalated != expect["escalated"]:
        return False, f"escalated={result.escalated}, want {expect['escalated']}"
    if "resolved" in expect and result.resolved != expect["resolved"]:
        return False, f"resolved={result.resolved}, want {expect['resolved']}"
    if "min_tool_calls" in expect and result.tool_calls_total < expect["min_tool_calls"]:
        return False, f"tool_calls_total={result.tool_calls_total}, want >= {expect['min_tool_calls']}"
    if "max_tool_calls" in expect and result.tool_calls_total > expect["max_tool_calls"]:
        return False, f"tool_calls_total={result.tool_calls_total}, want <= {expect['max_tool_calls']}"
    return True, None


def run_all(scenarios_dir: Path = _SCENARIOS_DIR) -> list[ScenarioResult]:
    results = []
    for path in sorted(scenarios_dir.glob("*.json")):
        print(f"Running: {path.stem} ...", end=" ", flush=True)
        r = run_scenario(path)
        status = "PASS" if r.passed else f"FAIL ({r.failure_reason})"
        print(status)
        results.append(r)

    passed = sum(1 for r in results if r.passed)
    print(f"\n{passed}/{len(results)} scenarios passed")
    return results


if __name__ == "__main__":
    run_all()
