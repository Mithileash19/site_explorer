"""Tests for LokiService input validation and safety."""
from __future__ import annotations

import pytest

from services.grafana.loki_service import _sanitize, _sanitize_search


class TestSanitize:
    """Input validation rejects LogQL injection characters."""

    def test_allows_normal_values(self):
        assert _sanitize("denjef001", "site") == "denjef001"
        assert _sanitize("edge01", "hostname") == "edge01"
        assert _sanitize("amr-04", "hostname") == "amr-04"
        assert _sanitize("otel_collector.edge01", "deployment") == "otel_collector.edge01"
        assert _sanitize("some value", "x") == "some value"

    def test_rejects_curly_braces(self):
        with pytest.raises(ValueError, match="Invalid characters"):
            _sanitize("{bad}", "site")

    def test_rejects_pipe(self):
        with pytest.raises(ValueError, match="Invalid characters"):
            _sanitize("a|b", "site")

    def test_rejects_backslash(self):
        with pytest.raises(ValueError, match="Invalid characters"):
            _sanitize("a\\b", "hostname")

    def test_rejects_backtick(self):
        with pytest.raises(ValueError, match="Invalid characters"):
            _sanitize("a`b", "deployment")

    def test_empty_passes(self):
        assert _sanitize("", "site") == ""

    def test_strips_whitespace(self):
        assert _sanitize("  denjef001  ", "site") == "denjef001"


class TestSanitizeSearch:
    """Search/exclude values are similarly protected."""

    def test_allows_normal_search(self):
        assert _sanitize_search("ERROR timeout", "search") == "ERROR timeout"

    def test_rejects_injection(self):
        with pytest.raises(ValueError, match="Invalid characters"):
            _sanitize_search('{job="evil"}', "search")
