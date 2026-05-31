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
    """Wire HTTP routes to the request-handling functions below. The router only
    does dispatch; the actual flow lives in module-level functions so it can be
    tested directly and lifted into its own service module later."""
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


# ── request-handling logic ────────────────────────────────────────────────────
# Plain functions (no router/closure capture): the dependencies are passed in,
# so this flow is independent of the route declarations above.

def resolve_or_create_case(store: SessionStore, case_id: str | None) -> CaseState:
    """The case this request operates on: an existing one by id, or a fresh one."""
    if case_id:
        return require_case(store, case_id)
    return store.create()


def require_case(store: SessionStore, case_id: str) -> CaseState:
    case = store.get(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="case not found")
    return case


def run_chat_turn(
    case: CaseState,
    message: str,
    llm: BaseLLMClient,
    tools: dict[str, BaseTool],
    turn_runner: TurnRunner,
) -> str:
    try:
        return turn_runner(case, message, llm, tools)
    except LLMClientError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def to_chat_response(case: CaseState, message: str) -> ChatResponse:
    return ChatResponse(
        case_id=case.case_id,
        message=message,
        phase=case.phase.value,
        is_closed=case.phase == Phase.CLOSED,
    )


def to_case_view(case: CaseState) -> CaseView:
    return CaseView(
        case_id=case.case_id,
        phase=case.phase.value,
        is_closed=case.phase == Phase.CLOSED,
        confidence=case.confidence,
        tool_calls_total=case.tool_calls_total,
        facts=case.facts,
        escalation_context=case.escalation_context or None,
    )
