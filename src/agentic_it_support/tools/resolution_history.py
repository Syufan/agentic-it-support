import json
import re
from pathlib import Path
from typing import Any

from agentic_it_support.tools.base import BaseTool, ToolResult


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"\b\w+\b", text.lower()))


class ResolutionHistoryTool(BaseTool):
    name = "resolution_history"
    description = "Search past resolved IT tickets for similar issues and how they were fixed"

    def __init__(self, history_file: Path) -> None:
        self._history_file = history_file

    def run(self, inputs: dict[str, Any]) -> ToolResult:
        query = str(inputs.get("query", "")).strip()
        if not query:
            return ToolResult(success=False, data={}, error="query is required")

        try:
            incidents = json.loads(self._history_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return ToolResult(success=False, data={}, error=f"resolution history unavailable: {exc}")

        terms = _tokenize(query)
        ranked = []
        for incident in incidents:
            haystack = _tokenize(
                f"{incident.get('category', '')} {incident.get('summary', '')} {incident.get('resolution', '')}"
            )
            overlap = len(terms & haystack)
            if overlap > 0:
                score = round(min(overlap / len(terms), 1.0), 3)
                ranked.append({**incident, "score": score})

        ranked.sort(key=lambda i: i["score"], reverse=True)
        return ToolResult(success=True, data={"incidents": ranked})
