from collections.abc import Callable

from fastapi import APIRouter, HTTPException

from llm.client import BaseLLMClient, LLMClientError
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
    """Build API routes and inject runtime dependencies."""
    router = APIRouter()

    @router.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @router.post("/chat", response_model=ChatResponse)
    def chat(request: ChatRequest) -> ChatResponse:
        case = resolve_or_create_case(store, request.case_id)
        message = run_chat_turn(case, request.message, llm, tools, turn_runner)
        return to_chat_response(case, message)

    @router.get("/case/{case_id}", response_model=CaseView)
    def get_case(case_id: str) -> CaseView:
        return to_case_view(require_case(store, case_id))

    return router


# Request handling helpers.

def resolve_or_create_case(store: SessionStore, case_id: str | None) -> CaseState:
    """Return an existing case, or create a new one."""
    if case_id:
        return require_case(store, case_id)
    return store.create()


def require_case(store: SessionStore, case_id: str) -> CaseState:
    # 404 when the requested case does not exist.
    case = store.get(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="case not found")
    return case


def run_chat_turn(
    case: CaseState,
    message: str,
    llm: BaseLLMClient,
    tools: dict[str, BaseTool],
    turn_runner: Callable[[CaseState, str, BaseLLMClient, dict[str, BaseTool]], str],
) -> str:
    # Convert LLM failures into HTTP errors.
    try:
        return turn_runner(case, message, llm, tools)
    except LLMClientError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def to_chat_response(case: CaseState, message: str) -> ChatResponse:
    # Map case state to chat response DTO.
    return ChatResponse(
        case_id=case.case_id,
        message=message,
        phase=case.phase.value,
        is_closed=case.phase == Phase.CLOSED,
    )


def to_case_view(case: CaseState) -> CaseView:
    # Map case state to case view DTO.
    return CaseView(
        case_id=case.case_id,
        phase=case.phase.value,
        is_closed=case.phase == Phase.CLOSED,
        confidence=case.confidence,
        tool_calls_total=case.tool_calls_total,
        facts=case.facts,
        escalation_context=case.escalation_context or None,
    )