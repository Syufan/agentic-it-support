from functools import partial

import uvicorn

from agent.parser import parse_proposal
from config.settings import Settings
from llm.client import BaseLLMClient, RealLLMClient
from api.server import ITSupportWebServer
from runtime.controller import run_turn
from state.session import SessionStore
from tools import DEFAULT_TOOLS
from tools.base import BaseTool

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000


def _build_webserver() -> ITSupportWebServer:
    settings = Settings()
    llm = RealLLMClient(
        response_parser=parse_proposal,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        temperature=settings.llm_temperature,
    )
    tools = DEFAULT_TOOLS
    store = SessionStore()
    turn_runner = partial(run_turn, settings=settings)

    _validate_dependencies(llm=llm, tools=tools, store=store, turn_runner=turn_runner)

    return ITSupportWebServer(
        llm=llm,
        tools=tools,
        store=store,
        turn_runner=turn_runner,
    )


def _validate_dependencies(
    *,
    llm: BaseLLMClient,
    tools: dict[str, BaseTool],
    store: SessionStore,
    turn_runner,
) -> None:
    if not isinstance(llm, BaseLLMClient):
        raise TypeError("llm must implement BaseLLMClient")

    if not isinstance(store, SessionStore):
        raise TypeError("store must be a SessionStore")

    if not isinstance(tools, dict) or not tools:
        raise ValueError("tools must be a non-empty dict[str, BaseTool]")

    if not callable(turn_runner):
        raise TypeError("turn_runner must be callable")

    for name, tool in tools.items():
        if not isinstance(name, str) or not name:
            raise ValueError("tool registry keys must be non-empty strings")
        if not isinstance(tool, BaseTool):
            raise TypeError(f"tool '{name}' must implement BaseTool")
        if tool.name != name:
            raise ValueError(f"tool registry key '{name}' does not match tool.name '{tool.name}'")


app = _build_webserver().get_app()


def main() -> None:
    uvicorn.run(app, host=DEFAULT_HOST, port=DEFAULT_PORT)


if __name__ == "__main__":
    main()
