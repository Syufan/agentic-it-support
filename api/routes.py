from collections.abc import Callable

from fastapi import APIRouter, HTTPException

from agent.llm import BaseLLMClient, LLMClientError
from api.schemas import CaseView, ChatRequest, ChatResponse
from state.case_state import CaseState, Phase
from state.session import SessionStore
from tools.base import BaseTool


def build_router(
    *,
    llm: BaseLLMClient,
    tools: dict[str, BaseTool],
    store: SessionStore,
    turn_runner: Callable[[CaseState, str, BaseLLMClient, dict[str, BaseTool]], str],
) -> APIRouter:
    router = APIRouter()

    @router.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @router.post("/chat", response_model=ChatResponse)
    def chat(request: ChatRequest) -> ChatResponse:
        case = None
        if request.case_id:
            case = store.get(request.case_id)
            if case is None:
                raise HTTPException(status_code=404, detail="case not found")

        if case is None:
            case = store.create()

        try:
            message = turn_runner(case, request.message, llm, tools)
        except LLMClientError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        return ChatResponse(
            case_id=case.case_id,
            message=message,
            phase=case.phase.value,
            is_closed=case.phase == Phase.CLOSED,
        )

    @router.get("/case/{case_id}", response_model=CaseView)
    def get_case(case_id: str) -> CaseView:
        case = store.get(case_id)
        if case is None:
            raise HTTPException(status_code=404, detail="case not found")

        return CaseView(
            case_id=case.case_id,
            phase=case.phase.value,
            is_closed=case.phase == Phase.CLOSED,
            confidence=case.confidence,
            tool_calls_total=case.tool_calls_total,
            facts=case.facts,
            escalation_context=case.escalation_context or None,
        )

    return router
