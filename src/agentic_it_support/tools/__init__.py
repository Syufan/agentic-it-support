from pathlib import Path

from agentic_it_support.config.settings import DEFAULT_DATA_DIR
from agentic_it_support.tools.base import BaseTool
from agentic_it_support.tools.kb_search import KBSearchTool
from agentic_it_support.tools.resolution_history import ResolutionHistoryTool
from agentic_it_support.tools.status_api import StatusAPITool
from agentic_it_support.tools.user_directory import UserDirectoryTool


def build_tools(data_dir: Path = DEFAULT_DATA_DIR) -> dict[str, BaseTool]:
    """Create a fresh default tool registry for one application instance."""
    return {
        "kb_search": KBSearchTool(data_dir / "knowledge_base"),
        "status_api": StatusAPITool(data_dir / "system_status" / "status.json"),
        "user_directory": UserDirectoryTool(data_dir / "user_directory" / "users.json"),
        "resolution_history": ResolutionHistoryTool(data_dir / "resolution_history" / "history.json"),
    }
