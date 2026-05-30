from fastapi import FastAPI

from agent.llm import BaseLLMClient
from api.routes import build_router
from state.session import SessionStore
from tools.base import BaseTool


class ITSupportWebServer:
    def __init__(
        self,
        *,
        llm: BaseLLMClient,
        tools: dict[str, BaseTool],
        store: SessionStore,
    ) -> None:
        self._llm = llm
        self._tools = tools
        self._store = store
        self._app = self._build_fastapi()

    def get_app(self) -> FastAPI:
        return self._app

    def _build_fastapi(self) -> FastAPI:
        app = FastAPI(title="IT Helpdesk Agent")
        app.include_router(build_router(llm=self._llm, tools=self._tools, store=self._store))
        return app
