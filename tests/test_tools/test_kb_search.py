from pathlib import Path

from agentic_it_support.tools.kb_search import KBSearchTool

_KB_DIR = Path(__file__).resolve().parents[2] / "data" / "knowledge_base"
tool = KBSearchTool(_KB_DIR)


def test_relevant_query_returns_results():
    result = tool.run({"query": "VPN disconnects"})
    assert result.success is True
    assert len(result.data["results"]) > 0


def test_result_has_expected_fields():
    result = tool.run({"query": "VPN disconnects"})
    article = result.data["results"][0]
    assert "title" in article
    assert "content" in article
    assert "score" in article


def test_score_is_between_0_and_1():
    result = tool.run({"query": "VPN disconnects"})
    for article in result.data["results"]:
        assert 0.0 <= article["score"] <= 1.0


def test_results_sorted_by_score_descending():
    result = tool.run({"query": "password reset"})
    scores = [a["score"] for a in result.data["results"]]
    assert scores == sorted(scores, reverse=True)


def test_irrelevant_query_returns_empty():
    result = tool.run({"query": "xyzzy_nonexistent_topic_abc"})
    assert result.success is True
    assert result.data["results"] == []


def test_missing_query_returns_error():
    result = tool.run({})
    assert result.success is False
    assert result.error is not None
