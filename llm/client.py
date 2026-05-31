import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

from openai import OpenAI, OpenAIError, PermissionDeniedError

#: The parsed domain object a client yields. The transport layer stays agnostic
#: about its shape; callers bind it (e.g. to AgentProposal) via response_parser.
T = TypeVar("T")


@dataclass
class LLMInput:
    """The input contract for an LLM client: a system prompt plus the message
    list. Lives here (the transport layer) so callers like message_builder
    construct it; the client no longer reaches back into runtime for its type."""
    system: str
    messages: list[dict[str, str]]


@dataclass
class LLMCallStats:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0


class BaseLLMClient(ABC, Generic[T]):
    #: stats from the most recent call, or None if this client never tracks them
    last_stats: LLMCallStats | None = None

    @abstractmethod
    def call(self, llm_input: LLMInput) -> T:
        ...


class MockLLMClient(BaseLLMClient[T]):
    def __init__(self, responses: list[T]) -> None:
        self._queue: deque[T] = deque(responses)

    def call(self, llm_input: LLMInput) -> T:
        if not self._queue:
            raise RuntimeError("MockLLMClient: response queue is empty")
        return self._queue.popleft()


class LLMClientError(RuntimeError):
    """Raised when the real LLM provider cannot produce a valid response."""


class LLMConfigurationError(LLMClientError):
    """Misconfiguration (e.g. no api_key) — a fail-fast bug, not retryable."""


class LLMProviderError(LLMClientError):
    """A transient provider failure (network, 429, timeout, empty response)."""


class RealLLMClient(BaseLLMClient[T]):
    def __init__(
        self,
        response_parser: Callable[[str], T],
        api_key: str = "",
        model: str = "",
        client: OpenAI | None = None,
    ) -> None:
        self._parse = response_parser
        self._model = model

        if client is None and not api_key:
            raise LLMConfigurationError("no api_key was injected (settings.llm_api_key is empty)")

        self._client = client or OpenAI(api_key=api_key)

    def call(self, llm_input: LLMInput) -> T:
        started = time.perf_counter()
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "system", "content": llm_input.system}, *llm_input.messages],
                response_format={"type": "json_object"},
                temperature=0.2,
            )
        except PermissionDeniedError as exc:
            raise LLMProviderError(
                f"LLM model '{self._model}' is not available to this API key"
            ) from exc
        except OpenAIError as exc:
            raise LLMProviderError(f"LLM provider request failed: {exc}") from exc

        self.last_stats = _stats_from(response, (time.perf_counter() - started) * 1000)

        if not response.choices:
            raise LLMProviderError("LLM returned no choices")
        raw = response.choices[0].message.content or "{}"
        return self._parse(raw)


def _stats_from(response, latency_ms: float) -> LLMCallStats:
    usage = getattr(response, "usage", None)
    return LLMCallStats(
        prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        total_tokens=getattr(usage, "total_tokens", 0) or 0,
        latency_ms=round(latency_ms, 2),
    )
