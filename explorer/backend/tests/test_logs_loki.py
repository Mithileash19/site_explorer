"""Tests for the new /api/v1/logs/* endpoints (Loki-backed)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client():
    """Import the app, inject a mock LokiService, return (TestClient, mock)."""
    from app.main import app
    from app.routes import logs as logs_route

    mock_loki = MagicMock()
    logs_route._loki = mock_loki
    yield TestClient(app), mock_loki
    logs_route._loki = None


# ── /api/v1/logs/environments ──────────────────────────────────────────────


def test_environments_returns_hardcoded_list(client):
    tc, _mock = client
    resp = tc.get("/api/v1/logs/environments")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert "sootballs-prod-logs-loki" in data
    assert "Loki" in data
    assert len(data) == 5


# ── /api/v1/logs/sites ────────────────────────────────────────────────────


def test_sites_calls_label_values(client):
    tc, mock_loki = client
    mock_loki.label_values.return_value = ["denjef001", "actsgm001"]
    resp = tc.get("/api/v1/logs/sites", params={"env": "sootballs-prod-logs-loki"})
    assert resp.status_code == 200
    assert resp.json() == ["denjef001", "actsgm001"]
    mock_loki.label_values.assert_called_once_with("site", env="sootballs-prod-logs-loki")


def test_sites_rejects_bad_input(client):
    tc, mock_loki = client
    mock_loki.label_values.side_effect = ValueError("Invalid characters")
    resp = tc.get("/api/v1/logs/sites", params={"env": "bad{input}"})
    assert resp.status_code == 400


def test_sites_requires_env(client):
    tc, _mock = client
    resp = tc.get("/api/v1/logs/sites")
    assert resp.status_code == 422  # missing required param


# ── /api/v1/logs/hostnames ────────────────────────────────────────────────


def test_hostnames_calls_label_values(client):
    tc, mock_loki = client
    mock_loki.label_values.return_value = ["amr04", "edge01"]
    resp = tc.get(
        "/api/v1/logs/hostnames",
        params={"env": "sootballs-prod-logs-loki", "site": "denjef001"},
    )
    assert resp.status_code == 200
    assert resp.json() == ["amr04", "edge01"]
    mock_loki.label_values.assert_called_once_with(
        "hostname",
        env="sootballs-prod-logs-loki",
        extra_matchers={"site": "denjef001"},
    )


def test_hostnames_requires_env_and_site(client):
    tc, _mock = client
    resp = tc.get("/api/v1/logs/hostnames", params={"site": "denjef001"})
    assert resp.status_code == 422


# ── /api/v1/logs/deployments ─────────────────────────────────────────────


def test_deployments_calls_label_values(client):
    tc, mock_loki = client
    mock_loki.label_values.return_value = ["gbc", "gwm", "ims"]
    resp = tc.get(
        "/api/v1/logs/deployments",
        params={
            "env": "sootballs-prod-logs-loki",
            "site": "denjef001",
            "hostname": "edge01",
        },
    )
    assert resp.status_code == 200
    assert resp.json() == ["gbc", "gwm", "ims"]
    mock_loki.label_values.assert_called_once_with(
        "deployment_name",
        env="sootballs-prod-logs-loki",
        extra_matchers={"site": "denjef001", "hostname": "edge01"},
    )


# ── /api/v1/logs/volume ──────────────────────────────────────────────────


def test_volume_returns_buckets(client):
    tc, mock_loki = client
    mock_loki.query_volume.return_value = [
        {"ts": 1710374400.0, "count": 10},
        {"ts": 1710374405.0, "count": 25},
    ]
    resp = tc.get(
        "/api/v1/logs/volume",
        params={
            "env": "sootballs-prod-logs-loki",
            "site": "denjef001",
            "hostname": "edge01",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 2
    assert data[0]["count"] == 10


# ── /api/v1/logs/query ───────────────────────────────────────────────────


def test_query_returns_lines(client):
    tc, mock_loki = client
    mock_loki.query_logs.return_value = (
        [
            {"ts": "1710374400000000000", "line": "[INFO] started", "labels": {}},
            {"ts": "1710374401000000000", "line": "[ERROR] failed", "labels": {}},
        ],
        2,
    )
    resp = tc.get(
        "/api/v1/logs/query",
        params={
            "env": "sootballs-prod-logs-loki",
            "site": "denjef001",
            "hostname": "edge01",
            "deployment": "gbc",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_count"] == 2
    assert len(data["lines"]) == 2
    assert data["limit"] <= 4000


def test_query_hard_caps_limit(client):
    tc, mock_loki = client
    mock_loki.query_logs.return_value = ([], 0)
    resp = tc.get(
        "/api/v1/logs/query",
        params={
            "env": "x",
            "site": "y",
            "limit": "9999",  # exceeds cap
        },
    )
    # FastAPI should reject limit > 4000 via le=4000 validation
    assert resp.status_code == 422


def test_query_validation_error(client):
    tc, mock_loki = client
    mock_loki.query_logs.side_effect = ValueError("bad chars")
    resp = tc.get(
        "/api/v1/logs/query",
        params={"env": "x", "site": "y"},
    )
    assert resp.status_code == 400


# ── Error classification ─────────────────────────────────────────────────


def test_hostnames_returns_401_on_token_error(client):
    tc, mock_loki = client
    mock_loki.label_values.side_effect = RuntimeError(
        "Grafana token invalid or expired. Regenerate the service account token."
    )
    resp = tc.get(
        "/api/v1/logs/hostnames",
        params={"env": "sootballs-prod-logs-loki", "site": "site1"},
    )
    assert resp.status_code == 401


def test_deployments_returns_503_on_connection_error(client):
    tc, mock_loki = client
    mock_loki.label_values.side_effect = RuntimeError(
        "Cannot reach Grafana at https://grafana.example.com."
    )
    resp = tc.get(
        "/api/v1/logs/deployments",
        params={"env": "sootballs-prod-logs-loki", "site": "s1", "hostname": "h1"},
    )
    assert resp.status_code == 503


# ── Debug endpoints ──────────────────────────────────────────────────────


def test_debug_datasources(client):
    tc, mock_loki = client
    mock_loki.list_datasources_raw.return_value = [
        {"name": "sootballs-prod-logs-loki", "uid": "uid-1", "type": "loki", "url": "http://loki:3100"},
    ]
    resp = tc.get("/api/v1/logs/debug/datasources")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["name"] == "sootballs-prod-logs-loki"


def test_debug_labels(client):
    tc, mock_loki = client
    mock_loki.label_values.return_value = ["edge01", "amr04"]
    mock_loki.get_datasource_uid.return_value = "uid-1"
    resp = tc.get(
        "/api/v1/logs/debug/labels",
        params={"env": "sootballs-prod-logs-loki", "site": "denjef001", "hostname": "edge01"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "hostnames" in data
    assert "deployments" in data
    assert data["datasource_uid"] == "uid-1"
