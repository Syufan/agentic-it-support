import json
from abc import ABC, abstractmethod
from collections import deque
from json import JSONDecodeError

from openai import OpenAI, OpenAIError, PermissionDeniedError
from pydantic import ValidationError

from agent.proposals import AgentProposal
from config import LLM_API_KEY, LLM_MODEL
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

        if not response.choices:
            raise LLMResponseError("LLM returned no choices")
        raw = response.choices[0].message.content or "{}"
        try:
            return AgentProposal.model_validate(json.loads(raw))
        except JSONDecodeError as exc:
            raise LLMResponseError("LLM returned non-JSON content") from exc
        except ValidationError as exc:
            raise LLMResponseError("LLM returned JSON that does not match AgentProposal") from exc
