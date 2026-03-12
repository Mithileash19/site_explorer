"""Smoke tests — verify the API starts and health endpoint responds."""
import sys
import os

# Make sure the backend package is importable from tests/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_returns_200():
    response = client.get("/api/v1/health")
    assert response.status_code == 200


def test_health_payload():
    response = client.get("/api/v1/health")
    body = response.json()
    assert "status" in body
