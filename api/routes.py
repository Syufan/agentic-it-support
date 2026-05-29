from fastapi import Depends, FastAPI, HTTPException
from functools import lru_cache

from agent.llm import BaseLLMClient, LLMClientError, RealLLMClient
from api.schemas import ChatRequest, ChatResponse
from runtime.controller import run_turn
from state.case_state import Phase
from state.session import SessionStore, store as _default_store
from tools.base import BaseTool
from tools import DEFAULT_TOOLS

app = FastAPI(title="IT Helpdesk Agent")


@lru_cache(maxsize=1)
def _llm_singleton() -> BaseLLMClient:
    return RealLLMClient()


def get_llm() -> BaseLLMClient:
    try:
        return _llm_singleton()
    except LLMClientError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def get_tool_registry() -> dict[str, BaseTool]:
    return DEFAULT_TOOLS


def get_store() -> SessionStore:
    return _default_store


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    llm: BaseLLMClient = Depends(get_llm),
    tools: dict[str, BaseTool] = Depends(get_tool_registry),
    store: SessionStore = Depends(get_store),
) -> ChatResponse:
    case = None
    if request.case_id:
        case = store.get(request.case_id)
        if case is None:
            raise HTTPException(status_code=404, detail="case not found")

    if case is None:
        case = store.create()

    try:
        message = run_turn(case, request.message, llm, tools)
    except LLMClientError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return ChatResponse(
        case_id=case.case_id,
        message=message,
        phase=case.phase.value,
        is_closed=case.phase == Phase.CLOSED,
    )
