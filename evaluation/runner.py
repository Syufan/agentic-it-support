"""Batch evaluation runner.

Usage:
    uv run python -m evaluation.runner
"""

from __future__ import annotations

import json
import traceback
from dataclasses import dataclass
from pathlib import Path

from agent.parser import parse_proposal
from config.settings import Settings
from evaluation import EvaluationResult, evaluate
from llm.client import BaseLLMClient, RealLLMClient
from runtime.query_loop import run_turn
from state.case_state import CaseState
from tools import DEFAULT_TOOLS

_SCENARIOS_DIR = Path(__file__).parent / "scenarios"

@dataclass
class ScenarioResult:
    name: str
    passed: bool
    evaluation: EvaluationResult
    responses: list[str]
    failure_reason: str | None = None


def run_scenario(path: Path, llm: BaseLLMClient, settings: Settings) -> ScenarioResult:
    scenario = json.loads(path.read_text())
    case = CaseState()
    responses: list[str] = []

    try:
        for message in scenario["messages"]:
            responses.append(run_turn(case, message, llm, DEFAULT_TOOLS, settings=settings))
    except Exception:
        # Keep per-scenario isolation, but preserve the full traceback so a real
        # bug surfaces its stack instead of being flattened to a one-line message.
        return ScenarioResult(
            name=scenario.get("name", path.stem),
            passed=False,
            evaluation=evaluate(case),
            responses=responses,
            failure_reason=f"exception:\n{traceback.format_exc()}",
        )

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


def run_all(
    scenarios_dir: Path = _SCENARIOS_DIR,
    llm: BaseLLMClient | None = None,
    settings: Settings | None = None,
) -> list[ScenarioResult]:
    settings = settings or Settings()
    if llm is None:
        llm = RealLLMClient(
            response_parser=parse_proposal,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            temperature=settings.llm_temperature,
        )

    results = []
    for path in sorted(scenarios_dir.glob("*.json")):
        print(f"Running: {path.stem} ...", end=" ", flush=True)
        r = run_scenario(path, llm, settings)
        status = "PASS" if r.passed else f"FAIL ({r.failure_reason})"
        print(status)
        results.append(r)

    passed = sum(1 for r in results if r.passed)
    print(f"\n{passed}/{len(results)} scenarios passed")
    return results


if __name__ == "__main__":
    run_all()
