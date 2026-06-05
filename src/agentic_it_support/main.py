from functools import partial

import uvicorn

from agentic_it_support.agent.parser import parse_proposal
from agentic_it_support.api.server import ITSupportWebServer
from agentic_it_support.config.settings import Settings
from agentic_it_support.llm.client import RealLLMClient
from agentic_it_support.state.session import SessionStore
from agentic_it_support.tools import build_tools

def _stub_turn_runner(case, message, llm, tools) -> str:
    case.conversation.append({"role": "user", "content": message})
    return "API is wired. Runtime is temporarily disabled."


def _build_webserver():
    settings = Settings()
    llm = RealLLMClient(
        response_parser=parse_proposal,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        temperature=settings.llm_temperature,
    )
    tools = build_tools(settings.data_dir)
    store = SessionStore()
    # turn_runner = partial(run_turn, settings=settings)
   
    return ITSupportWebServer(llm=llm, tools=tools, store=store, turn_runner=_stub_turn_runner).get_app(), settings


app, settings = _build_webserver()


def main() -> None:
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)


if __name__ == "__main__":
    main()
