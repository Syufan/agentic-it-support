from abc import ABC, abstractmethod
from collections import deque

from agent.proposals import AgentProposal
from runtime.message_builder import LLMInput


class BaseLLMClient(ABC):
    @abstractmethod
    def call(self, llm_input: LLMInput) -> AgentProposal:
        ...


class MockLLMClient(BaseLLMClient):
    def __init__(self, proposals: list[AgentProposal]) -> None:
        self._queue: deque[AgentProposal] = deque(proposals)

    def call(self, llm_input: LLMInput) -> AgentProposal:
        if not self._queue:
            raise RuntimeError("MockLLMClient: proposal queue is empty")
        return self._queue.popleft()


class RealLLMClient(BaseLLMClient):
    """Plug in a real provider here once decided."""

    def call(self, llm_input: LLMInput) -> AgentProposal:
        raise NotImplementedError("configure a provider first")
