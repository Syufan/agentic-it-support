from typing import Any
from tools.base import BaseTool, ToolResult


class StatusAPITool(BaseTool):
    name = "status_api"
    description = "Check current status of internal services and known incidents"

    def run(self, inputs: dict[str, Any]) -> ToolResult:
        raise NotImplementedError
