import json
from pathlib import Path
from typing import Any

from agentic_it_support.tools.base import BaseTool, ToolResult


class UserDirectoryTool(BaseTool):
    name = "user_directory"
    description = "Look up employee info, department, role, equipment, and permissions"

    def __init__(self, users_file: Path) -> None:
        self._users_file = users_file

    def run(self, inputs: dict[str, Any]) -> ToolResult:
        user_id = str(inputs.get("user_id", "")).strip()
        email = str(inputs.get("email", "")).strip()

        if not user_id and not email:
            return ToolResult(success=False, data={}, error="user_id or email is required")

        try:
            users = json.loads(self._users_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return ToolResult(success=False, data={}, error=f"user directory unavailable: {exc}")

        if user_id:
            match = next((u for u in users if u["user_id"] == user_id), None)
        else:
            match = next((u for u in users if u["email"].lower() == email.lower()), None)

        if not match:
            identifier = user_id or email
            return ToolResult(success=False, data={}, error=f"user '{identifier}' not found")

        return ToolResult(success=True, data={"user": match})
