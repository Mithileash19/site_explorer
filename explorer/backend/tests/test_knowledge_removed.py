"""Regression tests: Knowledge Base endpoints are intentionally removed."""
import os
import sys

from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.main import app

client = TestClient(app)


def test_incidents_endpoint_removed() -> None:
    response = client.get("/api/v1/incidents")
    assert response.status_code == 404


def test_ingest_manual_endpoint_removed() -> None:
    response = client.post(
        "/api/v1/ingest/manual",
        json={"title": "x", "description": "y", "tags": []},
    )
    assert response.status_code == 404


def test_ingest_slack_endpoint_removed() -> None:
    response = client.post(
        "/api/v1/ingest/slack",
        json={"channel": "#incidents", "ts": "123.456"},
    )
    assert response.status_code == 404
