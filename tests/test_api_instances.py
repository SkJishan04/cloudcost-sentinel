"""API-level integration tests covering instances, forecast and agent endpoints."""

def test_health_check(app_client):
    resp = app_client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_create_and_get_instance(app_client):
    payload = {
        "instance_id": "i-api-test-01",
        "name": "api-test-instance",
        "instance_type": "t3.medium",
        "tags": {"criticality": "non-production"},
    }
    create_resp = app_client.post("/api/v1/instances", json=payload)
    assert create_resp.status_code == 201
    assert create_resp.json()["instance_id"] == "i-api-test-01"

    get_resp = app_client.get("/api/v1/instances/i-api-test-01")
    assert get_resp.status_code == 200
    assert get_resp.json()["instance_type"] == "t3.medium"


def test_get_unknown_instance_returns_404(app_client):
    resp = app_client.get("/api/v1/instances/does-not-exist")
    assert resp.status_code == 404
    assert resp.json()["error"] == "InstanceNotFoundError"


def test_instance_cost_endpoint(app_client):
    app_client.post(
        "/api/v1/instances",
        json={"instance_id": "i-cost-test", "name": "cost-test", "instance_type": "m5.xlarge", "tags": {}},
    )
    resp = app_client.get("/api/v1/instances/i-cost-test/cost")
    assert resp.status_code == 200
    body = resp.json()
    assert body["hourly_cost_usd"] > 0
    assert body["estimated_monthly_cost_usd"] == pytest.approx(body["hourly_cost_usd"] * 730, rel=1e-3)


def test_agent_optimize_endpoint_uses_rule_based_by_default(app_client):
    app_client.post(
        "/api/v1/instances",
        json={
            "instance_id": "i-agent-test",
            "name": "agent-test",
            "instance_type": "t3.large",
            "tags": {"criticality": "non-production"},
        },
    )
    resp = app_client.post("/api/v1/agent/optimize", json={"instance_id": "i-agent-test", "dry_run": True})
    assert resp.status_code == 200
    body = resp.json()
    assert body["agent_source"] in {"rule_based"}
    assert body["evaluated_instances"] == 1


import pytest  # noqa: E402  (kept at bottom to avoid reordering test bodies above)
