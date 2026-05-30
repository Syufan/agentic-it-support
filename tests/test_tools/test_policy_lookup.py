import tools.policy_lookup as _pol_mod
from tools.policy_lookup import PolicyLookupTool

tool = PolicyLookupTool()


def test_returns_all_actions_without_query():
    result = tool.run({})
    assert result.success is True
    assert len(result.data["actions"]) >= 1


def test_filters_by_query():
    result = tool.run({"query": "password"})
    assert result.success is True
    actions = result.data["actions"]
    assert actions
    assert all("password" in a["action"].lower() or "password" in a["description"].lower()
               for a in actions)


def test_action_includes_authorization_level():
    result = tool.run({"query": "unlock"})
    actions = result.data["actions"]
    assert actions[0]["authorization"] in {"agent", "human", "approval"}


def test_unknown_query_returns_empty_not_error():
    result = tool.run({"query": "zzzzz-unknown-action"})
    assert result.success is True
    assert result.data["actions"] == []


def test_missing_policy_file_returns_error_result(tmp_path, monkeypatch):
    monkeypatch.setattr(_pol_mod, "_POLICY_FILE", tmp_path / "nope.json")
    result = PolicyLookupTool().run({})
    assert result.success is False
    assert result.error is not None
