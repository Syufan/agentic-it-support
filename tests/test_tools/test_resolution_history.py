import agentic_it_support.tools.resolution_history as _hist_mod
from agentic_it_support.tools.resolution_history import ResolutionHistoryTool

tool = ResolutionHistoryTool()


def test_query_required():
    result = tool.run({})
    assert result.success is False
    assert result.error is not None


def test_finds_similar_vpn_incidents():
    result = tool.run({"query": "vpn disconnects timeout"})
    assert result.success is True
    incidents = result.data["incidents"]
    assert len(incidents) >= 1
    assert any("vpn" == i["category"] for i in incidents)


def test_results_ranked_by_relevance():
    result = tool.run({"query": "okta password locked out"})
    incidents = result.data["incidents"]
    assert incidents[0]["category"] == "password"


def test_no_match_returns_empty_list_not_error():
    result = tool.run({"query": "zzzzz nonsense quux"})
    assert result.success is True
    assert result.data["incidents"] == []


def test_missing_history_file_returns_error_result(tmp_path, monkeypatch):
    monkeypatch.setattr(_hist_mod, "_HISTORY_FILE", tmp_path / "nope.json")
    result = ResolutionHistoryTool().run({"query": "vpn"})
    assert result.success is False
    assert result.error is not None
