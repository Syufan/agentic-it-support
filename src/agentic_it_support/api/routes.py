from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, HTTPException

from agentic_it_support.api.schemas import CaseView, ChatRequest, ChatResponse
from agentic_it_support.state.case_state import Phase


def build_router(*, llm: Any, tools: dict[str, Any], store: Any, turn_runner: Callable[..., str]) -> APIRouter:
    """Build API routes with injected app."""
    router = APIRouter()

    @router.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @router.post("/chat", response_model=ChatResponse)
    def chat(request: ChatRequest) -> ChatResponse:
        case = _get_or_create_case(store, request.case_id)
        message = turn_runner(case, request.message, llm, tools)
        return _to_chat_response(case, message)

    # Reserved for future dashboard.
    @router.get("/case/{case_id}", response_model=CaseView)
    def get_case(case_id: str) -> CaseView:
        return _to_case_view(_require_case(store, case_id))

    return router


# helpers
def _get_or_create_case(store: Any, case_id: str | None) -> Any:
    """Return an existing case, or create a new one."""
    if case_id:
        return _require_case(store, case_id)
    return store.create()


def _require_case(store: Any, case_id: str) -> Any:
    """404 when the requested case does not exist"""
    case = store.get(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="case not found")
    return case


def _to_chat_response(case: Any, message: str) -> ChatResponse:
    """Map case state to chat response DTO."""
    return ChatResponse(
        case_id=case.case_id,
        message=message,
        phase=case.phase.value,
        is_closed=case.phase == Phase.CLOSED,
    )


def _to_case_view(case: Any) -> CaseView:
    """Reserved for future dashboard, map case state to case view."""
    return CaseView(
        case_id=case.case_id,
        phase=case.phase.value,
        is_closed=case.phase == Phase.CLOSED,
        confidence=case.confidence,
        tool_calls_total=case.tool_calls_total,
        escalation_context=case.escalation_context or None,
    )