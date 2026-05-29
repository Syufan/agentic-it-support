from typing import Any
from tools.base import BaseTool, ToolResult


class KBSearchTool(BaseTool):
    name = "kb_search"
    description = "Search internal knowledge base articles and troubleshooting guides"

    def run(self, inputs: dict[str, Any]) -> ToolResult:
        raise NotImplementedError
