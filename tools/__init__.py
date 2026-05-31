from tools.base import BaseTool
from tools.kb_search import KBSearchTool
from tools.resolution_history import ResolutionHistoryTool
from tools.status_api import StatusAPITool
from tools.user_directory import UserDirectoryTool

DEFAULT_TOOLS: dict[str, BaseTool] = {
    "kb_search": KBSearchTool(),
    "status_api": StatusAPITool(),
    "user_directory": UserDirectoryTool(),
    "resolution_history": ResolutionHistoryTool(),
}
