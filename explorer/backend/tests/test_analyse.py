"""Tests for POST /api/v1/investigate/analyse."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import MagicMock, patch

from schemas.analyse import AnalyseRequest, LogEntry


def test_analyse_request_validates_description() -> None:
    with pytest.raises(Exception):
        AnalyseRequest(issue_description="abc")  # < 5 chars


def test_analyse_request_accepts_valid_payload() -> None:
    req = AnalyseRequest(
        logs=[
            LogEntry(timestamp_ms=1000000, level="ERROR", message="segfault"),
        ],
        issue_description="Robot stopped unexpectedly during mission",
        site_id="actsgm001",
    )
    assert len(req.logs) == 1
    assert req.logs[0].level == "ERROR"


def test_analyse_response_schema() -> None:
    from schemas.analyse import AnalyseResponse

    resp = AnalyseResponse(
        model_used="qwen2.5:7b",
        has_images=False,
        slack_messages=0,
        log_count=42,
        summary="## What Happened\nRobot stopped.",
    )
    assert resp.model_used == "qwen2.5:7b"
    assert resp.log_count == 42
    assert "What Happened" in resp.summary
