import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

from openai import OpenAI, OpenAIError, PermissionDeniedError

# Parsed response type returned by the LLM client.
T = TypeVar("T")

_MS_PER_SECOND = 1000
_LATENCY_ROUND_DP = 2


@dataclass
class LLMInput:
    """LLM request payload."""
    system: str
    messages: list[dict[str, str]]


@dataclass
class LLMCallStats:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0


class BaseLLMClient(ABC, Generic[T]):
    # Stats from the most recent LLM call, when available.
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
        temperature: float | None = None,
        client: OpenAI | None = None,
    ) -> None:
        self._parse = response_parser
        self._model = model
        self._temperature = temperature

        if client is None and not api_key:
            raise LLMConfigurationError("no api_key was injected (settings.llm_api_key is empty)")

        self._client = client or OpenAI(api_key=api_key)

    def call(self, llm_input: LLMInput) -> T:
        started = time.perf_counter()
        create_kwargs: dict = {
            "model": self._model,
            "messages": [{"role": "system", "content": llm_input.system}, *llm_input.messages],
            "response_format": {"type": "json_object"},
        }

        # Leave temperature unset unless explicitly configured.
        if self._temperature is not None:
            create_kwargs["temperature"] = self._temperature
        try:
            response = self._client.chat.completions.create(**create_kwargs)
        except PermissionDeniedError as exc:
            raise LLMProviderError(
                f"LLM model '{self._model}' is not available to this API key"
            ) from exc
        except OpenAIError as exc:
            raise LLMProviderError(f"LLM provider request failed: {exc}") from exc

        self.last_stats = _stats_from(response, (time.perf_counter() - started) * _MS_PER_SECOND)

        if not response.choices:
            raise LLMProviderError("LLM returned no choices")
        raw = response.choices[0].message.content or "{}"
        return self._parse(raw)


def _stats_from(response, latency_ms: float) -> LLMCallStats:
    # Convert provider usage data into runtime stats.
    usage = getattr(response, "usage", None)
    return LLMCallStats(
        prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        total_tokens=getattr(usage, "total_tokens", 0) or 0,
        latency_ms=round(latency_ms, _LATENCY_ROUND_DP),
    )
