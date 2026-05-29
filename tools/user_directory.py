import json
from pathlib import Path
from typing import Any

from tools.base import BaseTool, ToolResult

_USERS_FILE = Path(__file__).parent.parent / "data" / "user_directory" / "users.json"


class UserDirectoryTool(BaseTool):
    name = "user_directory"
    description = "Look up employee info, department, role, equipment, and permissions"

    def run(self, inputs: dict[str, Any]) -> ToolResult:
        user_id = str(inputs.get("user_id", "")).strip()
        email = str(inputs.get("email", "")).strip()

        if not user_id and not email:
            return ToolResult(success=False, data={}, error="user_id or email is required")

        users = json.loads(_USERS_FILE.read_text(encoding="utf-8"))

        if user_id:
            match = next((u for u in users if u["user_id"] == user_id), None)
        else:
            match = next((u for u in users if u["email"].lower() == email.lower()), None)

        if not match:
            identifier = user_id or email
            return ToolResult(success=False, data={}, error=f"user '{identifier}' not found")

        return ToolResult(success=True, data={"user": match})
