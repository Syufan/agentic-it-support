from typing import Any
from tools.base import BaseTool, ToolResult


class UserDirectoryTool(BaseTool):
    name = "user_directory"
    description = "Look up employee info, department, role, equipment, and permissions"

    def run(self, inputs: dict[str, Any]) -> ToolResult:
        raise NotImplementedError
