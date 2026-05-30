import json
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from json import JSONDecodeError

from openai import OpenAI, OpenAIError, PermissionDeniedError
from pydantic import ValidationError

from agent.proposals import AgentProposal
from config import LLM_API_KEY, LLM_MODEL
from runtime.message_builder import LLMInput


@dataclass
class LLMCallStats:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0


class BaseLLMClient(ABC):
    #: stats from the most recent call, or None if this client never tracks them
    last_stats: LLMCallStats | None = None

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


class LLMClientError(RuntimeError):
    """Raised when the real LLM provider cannot produce a valid proposal."""


class LLMConfigurationError(LLMClientError):
    pass


class LLMProviderError(LLMClientError):
    pass


class LLMResponseError(LLMClientError):
    pass


class RealLLMClient(BaseLLMClient):
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        client: OpenAI | None = None,
    ) -> None:
        self._model = model or LLM_MODEL
        resolved_api_key = api_key if api_key is not None else LLM_API_KEY

        if client is None and not resolved_api_key:
            raise LLMConfigurationError("LLM_API_KEY is not configured")

        self._client = client or OpenAI(api_key=resolved_api_key)

    def call(self, llm_input: LLMInput) -> AgentProposal:
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
            raise LLMResponseError("LLM returned no choices")
        raw = response.choices[0].message.content or "{}"
        try:
            return AgentProposal.model_validate(json.loads(raw))
        except JSONDecodeError as exc:
            raise LLMResponseError("LLM returned non-JSON content") from exc
        except ValidationError as exc:
            raise LLMResponseError("LLM returned JSON that does not match AgentProposal") from exc


def _stats_from(response, latency_ms: float) -> LLMCallStats:
    usage = getattr(response, "usage", None)
    return LLMCallStats(
        prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        total_tokens=getattr(usage, "total_tokens", 0) or 0,
        latency_ms=round(latency_ms, 2),
    )
