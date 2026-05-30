import json
from pathlib import Path
from typing import Any

from tools.base import BaseTool, ToolResult

_POLICY_FILE = Path(__file__).parent.parent / "data" / "policies" / "policies.json"


class PolicyLookupTool(BaseTool):
    name = "policy_lookup"
    description = "Look up what the agent is authorized to do vs. what requires human approval"

    def run(self, inputs: dict[str, Any]) -> ToolResult:
        try:
            data = json.loads(_POLICY_FILE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return ToolResult(success=False, data={}, error=f"policy data unavailable: {exc}")

        actions = data["actions"]
        query = str(inputs.get("query", "")).strip().lower()
        if query:
            actions = [
                a for a in actions
                if query in a["action"].lower() or query in a["description"].lower()
            ]

        return ToolResult(success=True, data={"actions": actions, "updated_at": data["updated_at"]})
