from tools.status_api import StatusAPITool

tool = StatusAPITool()

VALID_STATUSES = {"operational", "degraded", "outage"}


def test_no_filter_returns_all_services():
    result = tool.run({})
    assert result.success is True
    assert len(result.data["services"]) > 0


def test_each_service_has_expected_fields():
    result = tool.run({})
    for svc in result.data["services"]:
        assert "name" in svc
        assert "status" in svc
        assert svc["status"] in VALID_STATUSES


def test_filter_by_known_service():
    result = tool.run({"service": "VPN"})
    assert result.success is True
    assert len(result.data["services"]) == 1
    assert result.data["services"][0]["name"] == "VPN"


def test_filter_is_case_insensitive():
    result = tool.run({"service": "vpn"})
    assert result.success is True
    assert result.data["services"][0]["name"] == "VPN"


def test_degraded_service_has_incident():
    result = tool.run({"service": "Salesforce"})
    svc = result.data["services"][0]
    assert svc["status"] == "degraded"
    assert svc["incident"] is not None


def test_unknown_service_returns_error():
    result = tool.run({"service": "NonExistentService"})
    assert result.success is False
    assert result.error is not None
