import tools.user_directory as _dir_mod
from tools.user_directory import UserDirectoryTool

tool = UserDirectoryTool()


def test_lookup_by_user_id():
    result = tool.run({"user_id": "u001"})
    assert result.success is True
    assert result.data["user"]["name"] == "Alice Johnson"


def test_lookup_by_email():
    result = tool.run({"email": "bob.chen@company.com"})
    assert result.success is True
    assert result.data["user"]["user_id"] == "u002"


def test_user_has_expected_fields():
    result = tool.run({"user_id": "u001"})
    user = result.data["user"]
    for field in ["user_id", "name", "email", "department", "role", "permissions"]:
        assert field in user


def test_unknown_user_id_returns_error():
    result = tool.run({"user_id": "u999"})
    assert result.success is False
    assert result.error is not None


def test_unknown_email_returns_error():
    result = tool.run({"email": "nobody@company.com"})
    assert result.success is False
    assert result.error is not None


def test_no_input_returns_error():
    result = tool.run({})
    assert result.success is False
    assert result.error is not None


def test_permissions_is_list():
    result = tool.run({"user_id": "u001"})
    assert isinstance(result.data["user"]["permissions"], list)


def test_missing_users_file_returns_error_result(tmp_path, monkeypatch):
    monkeypatch.setattr(_dir_mod, "_USERS_FILE", tmp_path / "nonexistent.json")
    result = UserDirectoryTool().run({"user_id": "u001"})
    assert result.success is False
    assert result.error is not None


def test_malformed_users_file_returns_error_result(tmp_path, monkeypatch):
    bad = tmp_path / "users.json"
    bad.write_text("not valid json")
    monkeypatch.setattr(_dir_mod, "_USERS_FILE", bad)
    result = UserDirectoryTool().run({"user_id": "u001"})
    assert result.success is False
    assert result.error is not None
