import re
from pathlib import Path
from typing import Any

from agentic_it_support.config.settings import DEFAULT_DATA_DIR
from agentic_it_support.tools.base import BaseTool, ToolResult

_KB_DIR = DEFAULT_DATA_DIR / "knowledge_base"


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"\b\w+\b", text.lower()))


class KBSearchTool(BaseTool):
    name = "kb_search"
    description = "Search internal knowledge base articles and troubleshooting guides"

    def __init__(self, kb_dir: Path | None = None) -> None:
        self._kb_dir = kb_dir or _KB_DIR

    def run(self, inputs: dict[str, Any]) -> ToolResult:
        query = str(inputs.get("query", "")).strip()
        if not query:
            return ToolResult(success=False, data={}, error="query is required")

        terms = _tokenize(query)
        results = []

        for path in self._kb_dir.glob("*.md"):
            content = path.read_text(encoding="utf-8")
            words = _tokenize(content)
            matches = len(terms & words)
            if matches > 0:
                score = round(min(matches / len(terms), 1.0), 3)
                results.append({
                    "title": path.stem.replace("_", " ").title(),
                    "content": content,
                    "score": score,
                })

        results.sort(key=lambda a: a["score"], reverse=True)
        return ToolResult(success=True, data={"results": results})
