"""Tests for the /api/v1/logs/* endpoints (legacy + new Loki-based).

The old endpoints (/api/v1/logs/hostnames, /api/v1/logs/deployments,
/api/v1/logs) now require an 'env' parameter and are backed by LokiService.
These tests exercise the new Loki-backed paths.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client():
    """Import the app, inject a mock LokiService, and return a test client."""
    from app.main import app
    from app.routes import logs as logs_route

    mock_loki = MagicMock()
    logs_route._loki = mock_loki

    # Also inject a mock for legacy _grafana_svc
    mock_grafana = MagicMock()
    logs_route._grafana_svc = mock_grafana

    yield TestClient(app), mock_loki, mock_grafana
    logs_route._loki = None
    logs_route._grafana_svc = None


# ── /api/v1/logs/hostnames ──────────────────────────────────────────────────


def test_hostnames_returns_unique_sorted_list(client):
    tc, mock_loki, _ = client
    mock_loki.label_values.return_value = ["amr04", "amr05", "edge01"]
    resp = tc.get("/api/v1/logs/hostnames?env=sootballs-prod-logs-loki&site=site1")
    assert resp.status_code == 200
    assert resp.json() == ["amr04", "amr05", "edge01"]


def test_hostnames_requires_site(client):
    tc, _, _ = client
    resp = tc.get("/api/v1/logs/hostnames?env=some-env")
    assert resp.status_code == 422  # missing required query param


# ── /api/v1/logs/deployments ────────────────────────────────────────────────


def test_deployments_returns_unique_sorted_list(client):
    tc, mock_loki, _ = client
    mock_loki.label_values.return_value = ["gbc", "gwm"]
    resp = tc.get("/api/v1/logs/deployments?env=sootballs-prod-logs-loki&site=site1&hostname=edge01")
    assert resp.status_code == 200
    assert resp.json() == ["gbc", "gwm"]


def test_deployments_requires_site(client):
    tc, _, _ = client
    resp = tc.get("/api/v1/logs/deployments?env=some-env")
    assert resp.status_code == 422


# ── /api/v1/logs (legacy endpoint via grafana service) ──────────────────────


def test_logs_returns_formatted_lines(client):
    tc, _, mock_grafana = client
    from schemas.grafana import GrafanaLogLine, GrafanaLogsResponse

    mock_grafana.fetch_logs.return_value = GrafanaLogsResponse(
        site="denjef001",
        hostname="edge01",
        deployment="gbc",
        from_ms=1000,
        to_ms=2000,
        line_count=1,
        logs=[
            GrafanaLogLine(
                timestamp_ms=1710381676806,
                labels={
                    "hostname": "edge01",
                    "deployment_name": "gbc",
                    "detected_level": "info",
                },
                line="[1710381676.806] [ INFO] [ros.route_manager]: Agent 28 OK",
            ),
        ],
    )
    resp = tc.get("/api/v1/logs?env=sootballs-prod-logs-loki&site=denjef001&hostname=edge01&deployment=gbc")
    assert resp.status_code == 200
    body = resp.json()
    assert body["line_count"] == 1
    assert len(body["logs"]) == 1
    assert body["logs"][0]["labels"]["hostname"] == "edge01"


def test_logs_requires_env_and_site(client):
    tc, _, _ = client
    resp = tc.get("/api/v1/logs")
    assert resp.status_code == 422


def test_logs_search_filter(client):
    tc, _, mock_grafana = client
    from schemas.grafana import GrafanaLogsResponse

    mock_grafana.fetch_logs.return_value = GrafanaLogsResponse(
        site="s1", hostname="e01", deployment=None,
        from_ms=0, to_ms=1, line_count=0, logs=[],
    )
    resp = tc.get("/api/v1/logs?env=loki&site=s1&search=error")
    assert resp.status_code == 200
    mock_grafana.fetch_logs.assert_called_once()
    call_kwargs = mock_grafana.fetch_logs.call_args
    assert call_kwargs.kwargs.get("log_filter") == "error"


def test_logs_exclude_filter(client):
    tc, _, mock_grafana = client
    from schemas.grafana import GrafanaLogLine, GrafanaLogsResponse

    mock_grafana.fetch_logs.return_value = GrafanaLogsResponse(
        site="s1", hostname=".*", deployment=None,
        from_ms=0, to_ms=1, line_count=2,
        logs=[
            GrafanaLogLine(timestamp_ms=1, labels={}, line="Error happened"),
            GrafanaLogLine(timestamp_ms=2, labels={}, line="All good INFO"),
        ],
    )
    resp = tc.get("/api/v1/logs?env=loki&site=s1&exclude=Error")
    assert resp.status_code == 200
    body = resp.json()
    assert body["line_count"] == 1
    assert "Error" not in body["logs"][0]["line"]
