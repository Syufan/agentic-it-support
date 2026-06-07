import json
from pathlib import Path
from typing import Any

from agentic_it_support.config.settings import DEFAULT_DATA_DIR
from agentic_it_support.tools.base import BaseTool, ToolResult

_STATUS_FILE = DEFAULT_DATA_DIR / "system_status" / "status.json"


class StatusAPITool(BaseTool):
    name = "status_api"
    description = "Check current status of internal services and known incidents"

    def __init__(self, status_file: Path | None = None) -> None:
        self._status_file = status_file or _STATUS_FILE

    def run(self, inputs: dict[str, Any]) -> ToolResult:
        try:
            data = json.loads(self._status_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return ToolResult(success=False, data={}, error=f"status data unavailable: {exc}")
        services = data["services"]

        service_filter = str(inputs.get("service", "")).strip()
        if service_filter:
            matches = [s for s in services if s["name"].lower() == service_filter.lower()]
            if not matches:
                return ToolResult(success=False, data={}, error=f"service '{service_filter}' not found")
            services = matches

        return ToolResult(success=True, data={"services": services, "updated_at": data["updated_at"]})
