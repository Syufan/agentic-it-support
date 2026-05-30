from collections.abc import Callable

from fastapi import FastAPI

from agent.llm import BaseLLMClient
from api.routes import build_router
from state.case_state import CaseState
from state.session import SessionStore
from tools.base import BaseTool


class ITSupportWebServer:
    def __init__(
        self,
        *,
        llm: BaseLLMClient,
        tools: dict[str, BaseTool],
        store: SessionStore,
        turn_runner: Callable[[CaseState, str, BaseLLMClient, dict[str, BaseTool]], str],
    ) -> None:
        self._llm = llm
        self._tools = tools
        self._store = store
        self._turn_runner = turn_runner
        self._app = self._build_fastapi()

    def get_app(self) -> FastAPI:
        return self._app

    def _build_fastapi(self) -> FastAPI:
        app = FastAPI(title="IT Helpdesk Agent")
        app.include_router(build_router(
            llm=self._llm,
            tools=self._tools,
            store=self._store,
            turn_runner=self._turn_runner,
        ))
        return app
