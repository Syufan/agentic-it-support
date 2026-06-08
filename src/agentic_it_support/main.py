from functools import partial

import uvicorn

from agentic_it_support.agent.parser import parse_proposal
from agentic_it_support.api.server import ITSupportWebServer
from agentic_it_support.config.settings import Settings
from agentic_it_support.llm.client import RealLLMClient
from agentic_it_support.observability.event_tracing import InMemoryEventLog
from agentic_it_support.runtime.turn_runner import run_turn
from agentic_it_support.state.session import SessionStore
from agentic_it_support.tools import build_tools

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
    event_log = InMemoryEventLog(max_events=settings.event_log_capacity)
    turn_runner = partial(run_turn, llm=llm, tools=tools, settings=settings, event_log=event_log)
    return ITSupportWebServer(llm=llm, tools=tools, store=store, turn_runner=turn_runner).get_app(), settings


app, settings = _build_webserver()


def main() -> None:
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)


if __name__ == "__main__":
    main()
