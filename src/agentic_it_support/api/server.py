from collections.abc import Callable
from typing import Any

from fastapi import FastAPI

from agentic_it_support.api.routes import build_router


class ITSupportWebServer:
    def __init__(self, *, llm: Any, tools: dict[str, Any], store: Any, turn_runner: Callable[..., str]) -> None:
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
