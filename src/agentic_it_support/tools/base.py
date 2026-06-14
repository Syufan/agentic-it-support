from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolResult:
    success: bool
    data: dict[str, Any]
    error: str | None = None


class BaseTool(ABC):
    name: str
    description: str

    @abstractmethod
    def run(self, inputs: dict[str, Any]) -> ToolResult:
        ...
