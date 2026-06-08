from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, HTTPException

from agentic_it_support.api.schemas import ChatRequest, ChatResponse, TraceEventView
from agentic_it_support.state.case_state import Phase


def build_router(*, llm: Any, tools: dict[str, Any], store: Any, turn_runner: Callable[..., str], event_log: Any) -> APIRouter:
    """Build API routes with injected app."""
    router = APIRouter()

    @router.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @router.post("/chat", response_model=ChatResponse)
    def chat(request: ChatRequest) -> ChatResponse:
        case = _get_or_create_case(store, request.case_id)
        # mutates case in place, returns reply
        message = turn_runner(case, request.message)
        return _to_chat_response(case, message)

    # Read back the recorded runtime trace for a case (empty if none recorded).
    @router.get("/case/{case_id}/trace", response_model=list[TraceEventView])
    def get_trace(case_id: str, limit: int | None = None) -> list[TraceEventView]:
        return [_to_trace_view(event) for event in event_log.get_events_for_case(case_id, limit=limit)]

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


def _to_trace_view(event: Any) -> TraceEventView:
    """Map a runtime Event to its API view."""
    return TraceEventView(
        event_type=event.event_type,
        phase=event.phase,
        confidence=event.confidence,
        details=event.details,
        timestamp=event.timestamp,
    )