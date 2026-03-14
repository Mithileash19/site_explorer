"""Unit tests for the Grafana-proxied LokiService.

Tests datasource UID discovery, label value queries, log queries and
volume queries — all through the Grafana datasource proxy URL pattern.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from services.grafana.loki_service import LokiService, _cache, _sanitize


# ── Fixtures ────────────────────────────────────────────────────────────────

FAKE_DATASOURCES = [
    {"name": "sootballs-prod-logs-loki-US-latest", "uid": "uid-us", "type": "loki", "url": "http://loki-us:3100"},
    {"name": "sootballs-prod-logs-loki", "uid": "uid-prod", "type": "loki", "url": "http://loki-prod:3100"},
    {"name": "sootballs-staging-logs-loki", "uid": "uid-stg", "type": "loki", "url": "http://loki-stg:3100"},
    {"name": "rio-loki", "uid": "uid-rio", "type": "loki", "url": "http://loki-rio:3100"},
    {"name": "Loki", "uid": "uid-default", "type": "loki", "url": "http://loki:3100"},
    {"name": "Prometheus", "uid": "uid-prom", "type": "prometheus", "url": "http://prom:9090"},
]


@pytest.fixture(autouse=True)
def _clear_cache():
    """Ensure label cache is fresh for every test."""
    _cache.clear()
    yield
    _cache.clear()


@pytest.fixture()
def svc():
    """Return a LokiService with mocked settings."""
    with patch("services.grafana.loki_service.settings") as mock_settings:
        mock_settings.grafana_url = "https://grafana.example.com"
        mock_settings.grafana_service_account_token = "glsa_test_token"
        mock_settings.grafana_api_key = ""
        yield LokiService()


# ── Datasource discovery ───────────────────────────────────────────────────

class TestDatasourceDiscovery:

    def test_loads_loki_datasources_on_first_call(self, svc):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = FAKE_DATASOURCES
        with patch("services.grafana.loki_service.requests.get", return_value=mock_resp):
            uid = svc.get_datasource_uid("sootballs-prod-logs-loki-US-latest")
        assert uid == "uid-us"

    def test_caches_datasource_map(self, svc):
        mock_resp = MagicMock()
        mock_resp.json.return_value = FAKE_DATASOURCES
        with patch("services.grafana.loki_service.requests.get", return_value=mock_resp) as mock_get:
            svc.get_datasource_uid("sootballs-prod-logs-loki-US-latest")
            svc.get_datasource_uid("sootballs-prod-logs-loki")
        # Only one HTTP call — second lookup uses cache
        mock_get.assert_called_once()

    def test_raises_for_unknown_environment(self, svc):
        mock_resp = MagicMock()
        mock_resp.json.return_value = FAKE_DATASOURCES
        with patch("services.grafana.loki_service.requests.get", return_value=mock_resp):
            with pytest.raises(ValueError, match="No Loki datasource found"):
                svc.get_datasource_uid("nonexistent-env")

    def test_ignores_non_loki_datasources(self, svc):
        mock_resp = MagicMock()
        mock_resp.json.return_value = FAKE_DATASOURCES
        with patch("services.grafana.loki_service.requests.get", return_value=mock_resp):
            with pytest.raises(ValueError):
                svc.get_datasource_uid("Prometheus")

    def test_raises_on_401_token_error(self, svc):
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.text = "Unauthorized"
        http_error = requests.exceptions.HTTPError(response=mock_resp)
        mock_resp.raise_for_status.side_effect = http_error
        with patch("services.grafana.loki_service.requests.get", return_value=mock_resp):
            with pytest.raises(RuntimeError, match="token invalid or expired"):
                svc.get_datasource_uid("sootballs-prod-logs-loki")

    def test_raises_on_connection_error(self, svc):
        with patch(
            "services.grafana.loki_service.requests.get",
            side_effect=requests.exceptions.ConnectionError("refused"),
        ):
            with pytest.raises(RuntimeError, match="Cannot reach Grafana"):
                svc.get_datasource_uid("sootballs-prod-logs-loki")

    def test_raises_when_no_loki_datasources_found(self, svc):
        mock_resp = MagicMock()
        mock_resp.json.return_value = [
            {"name": "Prometheus", "uid": "uid-prom", "type": "prometheus"},
        ]
        with patch("services.grafana.loki_service.requests.get", return_value=mock_resp):
            with pytest.raises(RuntimeError, match="No Loki datasources found"):
                svc.get_datasource_uid("anything")

    def test_reload_datasources(self, svc):
        mock_resp = MagicMock()
        mock_resp.json.return_value = FAKE_DATASOURCES
        with patch("services.grafana.loki_service.requests.get", return_value=mock_resp):
            result = svc.reload_datasources()
        assert "sootballs-prod-logs-loki-US-latest" in result
        assert result["sootballs-prod-logs-loki-US-latest"] == "uid-us"

    def test_list_datasources_raw(self, svc):
        mock_resp = MagicMock()
        mock_resp.json.return_value = FAKE_DATASOURCES
        with patch("services.grafana.loki_service.requests.get", return_value=mock_resp):
            result = svc.list_datasources_raw()
        assert len(result) == 6  # includes Prometheus
        assert result[0]["name"] == "sootballs-prod-logs-loki-US-latest"
        assert result[0]["uid"] == "uid-us"


# ── Label value queries ────────────────────────────────────────────────────

class TestLabelValues:

    def _setup_ds(self, svc):
        """Pre-load the datasource map so tests don't need extra mocking."""
        svc._ds_map = {
            "sootballs-prod-logs-loki-US-latest": "uid-us",
            "sootballs-prod-logs-loki": "uid-prod",
        }
        svc._ds_map_loaded = True

    def test_fetches_sites_via_proxy(self, svc):
        self._setup_ds(svc)
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "status": "success",
            "data": ["denjef001", "actsgm001", "bospat002"],
        }
        with patch("services.grafana.loki_service.requests.get", return_value=mock_resp) as mock_get:
            result = svc.label_values("site", env="sootballs-prod-logs-loki-US-latest")

        assert result == ["actsgm001", "bospat002", "denjef001"]
        call_url = mock_get.call_args[0][0]
        assert "/api/datasources/proxy/uid/uid-us/loki/api/v1/label/site/values" in call_url

    def test_fetches_hostnames_with_site_matcher(self, svc):
        self._setup_ds(svc)
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "status": "success",
            "data": ["edge01", "amr04", "amr05"],
        }
        with patch("services.grafana.loki_service.requests.get", return_value=mock_resp) as mock_get:
            result = svc.label_values(
                "hostname",
                env="sootballs-prod-logs-loki-US-latest",
                extra_matchers={"site": "denjef001"},
            )

        assert result == ["amr04", "amr05", "edge01"]
        call_kwargs = mock_get.call_args
        assert 'site="denjef001"' in call_kwargs.kwargs.get("params", {}).get("query", "")

    def test_fetches_deployments_with_site_and_hostname_matchers(self, svc):
        self._setup_ds(svc)
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "status": "success",
            "data": ["gbc", "gwm", "ims"],
        }
        with patch("services.grafana.loki_service.requests.get", return_value=mock_resp):
            result = svc.label_values(
                "deployment_name",
                env="sootballs-prod-logs-loki-US-latest",
                extra_matchers={"site": "denjef001", "hostname": "edge01"},
            )

        assert result == ["gbc", "gwm", "ims"]

    def test_returns_empty_on_failure_status(self, svc):
        self._setup_ds(svc)
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "status": "error",
            "data": [],
        }
        with patch("services.grafana.loki_service.requests.get", return_value=mock_resp):
            result = svc.label_values("site", env="sootballs-prod-logs-loki-US-latest")
        assert result == []

    def test_caches_label_values(self, svc):
        self._setup_ds(svc)
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"status": "success", "data": ["a", "b"]}
        with patch("services.grafana.loki_service.requests.get", return_value=mock_resp) as mock_get:
            svc.label_values("site", env="sootballs-prod-logs-loki-US-latest")
            svc.label_values("site", env="sootballs-prod-logs-loki-US-latest")
        mock_get.assert_called_once()

    def test_rejects_malicious_matcher_value(self, svc):
        self._setup_ds(svc)
        with pytest.raises(ValueError, match="Invalid characters"):
            svc.label_values(
                "hostname",
                env="sootballs-prod-logs-loki-US-latest",
                extra_matchers={"site": "bad{inject}"},
            )


# ── Log query ──────────────────────────────────────────────────────────────

class TestQueryLogs:

    def _setup_ds(self, svc):
        svc._ds_map = {"sootballs-prod-logs-loki-US-latest": "uid-us"}
        svc._ds_map_loaded = True

    def test_query_logs_through_proxy(self, svc):
        self._setup_ds(svc)
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "status": "success",
            "data": {
                "result": [
                    {
                        "stream": {"site": "denjef001", "hostname": "edge01"},
                        "values": [
                            ["1710374400000000000", "[INFO] started"],
                            ["1710374401000000000", "[ERROR] failed"],
                        ],
                    }
                ],
            },
        }
        with patch("services.grafana.loki_service.requests.get", return_value=mock_resp) as mock_get:
            lines, total = svc.query_logs(
                env="sootballs-prod-logs-loki-US-latest",
                site="denjef001",
                hostname="edge01",
                deployment="gbc",
                from_ns=1710374400000000000,
                to_ns=1710374500000000000,
            )

        assert total == 2
        assert len(lines) == 2
        assert lines[0]["line"] == "[INFO] started"
        call_url = mock_get.call_args[0][0]
        assert "/api/datasources/proxy/uid/uid-us/loki/api/v1/query_range" in call_url
        params = mock_get.call_args.kwargs.get("params", {})
        assert params["direction"] == "forward"

    def test_query_logs_hard_caps_limit(self, svc):
        self._setup_ds(svc)
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"status": "success", "data": {"result": []}}
        with patch("services.grafana.loki_service.requests.get", return_value=mock_resp) as mock_get:
            svc.query_logs(
                env="sootballs-prod-logs-loki-US-latest",
                site="s1", hostname="h1", deployment="d1",
                from_ns=0, to_ns=1, limit=9999,
            )
        params = mock_get.call_args.kwargs.get("params", {})
        assert int(params["limit"]) <= 4000

    def test_query_logs_with_search_and_exclude(self, svc):
        self._setup_ds(svc)
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"status": "success", "data": {"result": []}}
        with patch("services.grafana.loki_service.requests.get", return_value=mock_resp) as mock_get:
            svc.query_logs(
                env="sootballs-prod-logs-loki-US-latest",
                site="s1", hostname="h1", deployment="d1",
                from_ns=0, to_ns=1,
                search="error", exclude="debug",
            )
        params = mock_get.call_args.kwargs.get("params", {})
        assert '|= "error"' in params["query"]
        assert '!= "debug"' in params["query"]


# ── Volume query ───────────────────────────────────────────────────────────

class TestQueryVolume:

    def _setup_ds(self, svc):
        svc._ds_map = {"sootballs-prod-logs-loki-US-latest": "uid-us"}
        svc._ds_map_loaded = True

    def test_volume_through_proxy(self, svc):
        self._setup_ds(svc)
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "status": "success",
            "data": {
                "result": [
                    {
                        "values": [
                            [1710374400.0, "10"],
                            [1710374405.0, "25"],
                        ],
                    }
                ],
            },
        }
        with patch("services.grafana.loki_service.requests.get", return_value=mock_resp) as mock_get:
            buckets = svc.query_volume(
                env="sootballs-prod-logs-loki-US-latest",
                site="denjef001", hostname="edge01", deployment="gbc",
                from_ns=1710374400000000000, to_ns=1710374500000000000,
            )

        assert len(buckets) == 2
        assert buckets[0]["ts"] == 1710374400.0
        assert buckets[0]["count"] == 10
        params = mock_get.call_args.kwargs.get("params", {})
        assert "count_over_time" in params["query"]

    def test_auto_step_short_range(self, svc):
        """A 100-second range should use 5s step (minimum)."""
        from_ns = 1710374400_000_000_000
        to_ns = from_ns + 100_000_000_000  # 100 seconds
        step = svc._auto_step(from_ns, to_ns)
        assert step == "5s"

    def test_auto_step_large_range(self, svc):
        """A 24-hour range should use a larger step to stay under 11k points."""
        from_ns = 1710374400_000_000_000
        to_ns = from_ns + 24 * 3600 * 1_000_000_000  # 24 hours
        step = svc._auto_step(from_ns, to_ns)
        # 86400s / 2000 = 43s → should round to "43s" 
        # Verify it's larger than 5s and not minutes-level
        assert step.endswith("s") or step.endswith("m")
        # Compute points: should be under 11,000
        step_val = int(step.rstrip("smh"))
        if step.endswith("m"):
            step_val *= 60
        elif step.endswith("h"):
            step_val *= 3600
        points = 86400 / step_val
        assert points < 11000


# ── Input validation ───────────────────────────────────────────────────────

class TestSanitize:

    def test_rejects_curly_braces(self):
        with pytest.raises(ValueError):
            _sanitize("{malicious}", "field")

    def test_rejects_pipe(self):
        with pytest.raises(ValueError):
            _sanitize("a|b", "field")

    def test_rejects_backtick(self):
        with pytest.raises(ValueError):
            _sanitize("a`b", "field")

    def test_allows_normal_values(self):
        assert _sanitize("denjef001", "site") == "denjef001"
        assert _sanitize("edge01", "hostname") == "edge01"
        assert _sanitize("my-deployment_v2.1", "dep") == "my-deployment_v2.1"

    def test_empty_passthrough(self):
        assert _sanitize("", "field") == ""


# ── Configuration check ───────────────────────────────────────────────────

class TestConfiguration:

    def test_not_configured_without_url(self):
        with patch("services.grafana.loki_service.settings") as mock_settings:
            mock_settings.grafana_url = ""
            mock_settings.grafana_service_account_token = "token"
            mock_settings.grafana_api_key = ""
            svc = LokiService()
            assert not svc.configured

    def test_not_configured_without_token(self):
        with patch("services.grafana.loki_service.settings") as mock_settings:
            mock_settings.grafana_url = "https://grafana.example.com"
            mock_settings.grafana_service_account_token = ""
            mock_settings.grafana_api_key = ""
            svc = LokiService()
            assert not svc.configured

    def test_configured_with_url_and_token(self, svc):
        assert svc.configured

    def test_falls_back_to_api_key(self):
        with patch("services.grafana.loki_service.settings") as mock_settings:
            mock_settings.grafana_url = "https://grafana.example.com"
            mock_settings.grafana_service_account_token = ""
            mock_settings.grafana_api_key = "api_key_fallback"
            svc = LokiService()
            assert svc.configured
