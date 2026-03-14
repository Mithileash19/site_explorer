"""Grafana-proxied Loki query service.

All queries go through Grafana's datasource proxy endpoint:
  GET {GRAFANA_URL}/api/datasources/proxy/uid/{uid}/loki/api/v1/...

This avoids needing direct access to Loki and uses the Grafana service
account token for authentication.  Datasource UIDs are discovered once
from GET /api/datasources and cached for the lifetime of the process.

The service-account token is never returned in any response or logged.
"""
from __future__ import annotations

import re
import threading
import time
from typing import Dict, List, Optional, Tuple

import requests

from core.config import settings
from core.logging import get_logger

logger = get_logger(__name__)

# ── Safety constants ────────────────────────────────────────────────────────
_MAX_LINES = 4000
_QUERY_TIMEOUT = 30   # seconds
_LABEL_TIMEOUT = 15   # seconds
_CACHE_TTL = 300       # 5 minutes for label value cache

# Characters that MUST be rejected (LogQL injection vectors)
_DANGEROUS_CHARS = re.compile(r"[{}|\\`]")

# ── Simple in-memory cache for label values ─────────────────────────────────
_cache: Dict[str, Tuple[float, object]] = {}


def _cache_get(key: str) -> object | None:
    entry = _cache.get(key)
    if entry is None:
        return None
    ts, val = entry
    if time.time() - ts > _CACHE_TTL:
        _cache.pop(key, None)
        return None
    return val


def _cache_set(key: str, val: object) -> None:
    _cache[key] = (time.time(), val)


# ── Input validation ────────────────────────────────────────────────────────

def _sanitize(value: str, field: str) -> str:
    """Validate a filter value.  Reject dangerous chars; allow safe ones."""
    if not value:
        return value
    if _DANGEROUS_CHARS.search(value):
        raise ValueError(
            f"Invalid characters in '{field}': "
            "characters {{ }} | \\ ` are not allowed."
        )
    return value.strip()


def _sanitize_search(value: str, field: str) -> str:
    """Validate search/exclude text — slightly more permissive but still
    rejects LogQL injection chars."""
    if not value:
        return value
    if _DANGEROUS_CHARS.search(value):
        raise ValueError(
            f"Invalid characters in '{field}': "
            "characters {{ }} | \\ ` are not allowed."
        )
    return value.strip()


class LokiService:
    """Grafana-proxied Loki query service.

    Uses Grafana's datasource proxy to route all Loki API requests:
      {GRAFANA_URL}/api/datasources/proxy/uid/{uid}/loki/api/v1/...

    Datasource UIDs are discovered once from ``GET /api/datasources``
    and cached.  The ``env`` parameter in every public method maps to a
    Grafana Loki datasource **name**, which is resolved into a UID.
    """

    def __init__(self) -> None:
        self._grafana_url = (
            settings.grafana_url.rstrip("/") if settings.grafana_url else ""
        )
        self._token = (
            settings.grafana_service_account_token or settings.grafana_api_key
        )
        # Datasource UID cache: {datasource_name: uid}
        self._ds_map: Dict[str, str] = {}
        self._ds_map_loaded = False
        self._ds_lock = threading.Lock()

    # ── Configuration check ─────────────────────────────────────────────────

    @property
    def configured(self) -> bool:
        return bool(self._grafana_url and self._token)

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
        }

    # ── Datasource UID resolution ───────────────────────────────────────────

    def _load_datasource_map(self) -> None:
        """Fetch all Loki datasources from Grafana and cache their UIDs."""
        if not self.configured:
            raise RuntimeError(
                "Grafana is not configured. "
                "Set GRAFANA_URL and GRAFANA_SERVICE_ACCOUNT_TOKEN in backend/.env."
            )
        url = f"{self._grafana_url}/api/datasources"
        try:
            resp = requests.get(url, headers=self._headers(), timeout=_LABEL_TIMEOUT)
            resp.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code
            if status == 401:
                raise RuntimeError(
                    "Grafana token invalid or expired. "
                    "Regenerate the service account token."
                ) from exc
            raise RuntimeError(
                f"Failed to fetch datasources from Grafana (HTTP {status}): "
                f"{exc.response.text[:300]}"
            ) from exc
        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(
                f"Cannot reach Grafana at {self._grafana_url}. "
                "Check GRAFANA_URL configuration."
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise RuntimeError(
                "Grafana datasource lookup timed out."
            ) from exc

        all_ds = resp.json()
        if not isinstance(all_ds, list):
            raise RuntimeError(
                f"Unexpected datasource response format: {type(all_ds).__name__}"
            )

        ds_map: Dict[str, str] = {}
        for ds in all_ds:
            if ds.get("type") == "loki":
                ds_map[ds["name"]] = ds["uid"]

        if not ds_map:
            raise RuntimeError(
                "No Loki datasources found in Grafana. "
                "Ensure Loki datasources are configured."
            )

        logger.info("Loki datasources found: %s", ds_map)
        self._ds_map = ds_map
        self._ds_map_loaded = True

    def _ensure_datasource_map(self) -> None:
        """Thread-safe lazy load of the datasource UID map."""
        if self._ds_map_loaded:
            return
        with self._ds_lock:
            if not self._ds_map_loaded:
                self._load_datasource_map()

    def get_datasource_uid(self, env: str) -> str:
        """Return the datasource UID for the given environment name."""
        self._ensure_datasource_map()
        uid = self._ds_map.get(env)
        if not uid:
            available = list(self._ds_map.keys())
            raise ValueError(
                f"No Loki datasource found for environment '{env}'. "
                f"Available: {available}"
            )
        return uid

    def reload_datasources(self) -> Dict[str, str]:
        """Force-reload the datasource map.  Returns the new map."""
        with self._ds_lock:
            self._ds_map_loaded = False
            self._load_datasource_map()
        return dict(self._ds_map)

    def list_datasources_raw(self) -> list:
        """Return raw datasource info for the debug endpoint."""
        if not self.configured:
            raise RuntimeError(
                "Grafana is not configured. "
                "Set GRAFANA_URL and GRAFANA_SERVICE_ACCOUNT_TOKEN in backend/.env."
            )
        url = f"{self._grafana_url}/api/datasources"
        resp = requests.get(url, headers=self._headers(), timeout=_LABEL_TIMEOUT)
        resp.raise_for_status()
        all_ds = resp.json()
        return [
            {
                "name": ds.get("name"),
                "uid": ds.get("uid"),
                "type": ds.get("type"),
                "url": ds.get("url"),
            }
            for ds in (all_ds if isinstance(all_ds, list) else [])
        ]

    # ── Proxy HTTP helper ───────────────────────────────────────────────────

    def _proxy_get(
        self,
        datasource_uid: str,
        path: str,
        params: Optional[Dict] = None,
        timeout: int = _QUERY_TIMEOUT,
    ) -> dict | list:
        """GET through Grafana's datasource proxy endpoint."""
        if not self.configured:
            raise RuntimeError(
                "Grafana is not configured. "
                "Set GRAFANA_URL and GRAFANA_SERVICE_ACCOUNT_TOKEN in backend/.env."
            )
        url = (
            f"{self._grafana_url}/api/datasources/proxy/uid/"
            f"{datasource_uid}{path}"
        )
        logger.info(
            "Loki proxy query: %s params=%s",
            path,
            {k: v for k, v in (params or {}).items()},
        )
        try:
            resp = requests.get(
                url, headers=self._headers(), params=params, timeout=timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.Timeout as exc:
            raise RuntimeError(f"Loki query timed out ({timeout}s).") from exc
        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(
                f"Cannot reach Grafana at {self._grafana_url}."
            ) from exc
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code
            if status == 401:
                raise RuntimeError(
                    "Grafana token invalid or expired. "
                    "Regenerate the service account token."
                ) from exc
            raise RuntimeError(
                f"Loki proxy error {status}: {exc.response.text[:300]}"
            ) from exc

    # ── Label-value discovery ───────────────────────────────────────────────

    def label_values(
        self,
        label: str,
        env: str,
        extra_matchers: Optional[Dict[str, str]] = None,
    ) -> List[str]:
        """Fetch label values through the Grafana datasource proxy.

        ``env`` is resolved to a datasource UID.  ``extra_matchers`` are
        used to build a LogQL stream selector that narrows the results
        (e.g. ``{"site": "denjef001"}``).
        """
        uid = self.get_datasource_uid(env)

        # Build stream selector from matchers only (env -> UID, not a label)
        parts: list[str] = []
        for k, v in (extra_matchers or {}).items():
            parts.append(f'{k}="{_sanitize(v, k)}"')
        selector = "{" + ", ".join(parts) + "}" if parts else "{}"

        cache_key = f"lv:{env}:{label}:{selector}"
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached  # type: ignore[return-value]

        # Loki expects epoch seconds — not relative strings like "now-24h"
        now = int(time.time())
        start_ts = str(now - 24 * 60 * 60)  # 24 hours ago
        end_ts = str(now)

        result = self._proxy_get(
            uid,
            f"/loki/api/v1/label/{label}/values",
            params={"query": selector, "start": start_ts, "end": end_ts},
            timeout=_LABEL_TIMEOUT,
        )

        if isinstance(result, dict):
            if result.get("status") != "success":
                logger.error("Loki label query failed: %s", result)
                return []
            values = sorted(result.get("data", []))
        else:
            values = []

        if not values:
            logger.warning(
                "No values found for label '%s' with matchers %s on env '%s'",
                label, extra_matchers or {}, env,
            )

        _cache_set(cache_key, values)
        return values

    # ── Log query ───────────────────────────────────────────────────────────

    def query_logs(
        self,
        env: str,
        site: str,
        hostname: str,
        deployment: str,
        from_ns: int,
        to_ns: int,
        search: str = "",
        exclude: str = "",
        limit: int = _MAX_LINES,
    ) -> Tuple[List[dict], int]:
        """Fetch log lines through the Grafana datasource proxy.

        Returns ``(lines, total_count)``.
        Each line dict: ``{"ts": nanosecond_str, "line": str, "labels": dict}``
        """
        uid = self.get_datasource_uid(env)

        site = _sanitize(site, "site")
        hostname = _sanitize(hostname, "hostname")
        deployment = _sanitize(deployment, "deployment")
        search = _sanitize_search(search, "search")
        exclude = _sanitize_search(exclude, "exclude")
        limit = min(limit, _MAX_LINES)

        # Build LogQL — env is the datasource UID, not a label
        parts = [f'site="{site}"']
        if hostname:
            parts.append(f'hostname="{hostname}"')
        if deployment:
            parts.append(f'deployment_name="{deployment}"')
        expr = "{" + ", ".join(parts) + "}"
        if search:
            expr += f' |= "{search}"'
        if exclude:
            expr += f' != "{exclude}"'

        logger.info(
            "Loki query_range: expr=%s from=%s to=%s limit=%d",
            expr, from_ns, to_ns, limit,
        )

        result = self._proxy_get(
            uid,
            "/loki/api/v1/query_range",
            params={
                "query": expr,
                "start": str(from_ns),
                "end": str(to_ns),
                "limit": str(limit),
                "direction": "forward",
            },
        )

        lines: List[dict] = []
        data = result.get("data", {}) if isinstance(result, dict) else {}
        for stream in data.get("result", []):
            labels = stream.get("stream", {})
            for ts_str, line_text in stream.get("values", []):
                lines.append({
                    "ts": ts_str,
                    "line": line_text,
                    "labels": labels,
                })

        # Forward = oldest first; sort by nanosecond timestamp ascending
        lines.sort(key=lambda x: int(x["ts"]))
        total = len(lines)
        return lines[:limit], total

    # ── Volume query ────────────────────────────────────────────────────────

    @staticmethod
    def _auto_step(from_ns: int, to_ns: int) -> str:
        """Pick a step size that keeps total points under Loki's 11k limit.

        Returns a string like ``"5s"``, ``"15s"``, ``"1m"``, ``"5m"``.
        """
        range_s = max((to_ns - from_ns) / 1_000_000_000, 1)
        # Target ~2000 points max (well under Loki's 11,000 cap)
        step_s = max(int(range_s / 2000), 5)
        if step_s < 60:
            return f"{step_s}s"
        if step_s < 3600:
            return f"{step_s // 60}m"
        return f"{step_s // 3600}h"

    def query_volume(
        self,
        env: str,
        site: str,
        hostname: str,
        deployment: str,
        from_ns: int,
        to_ns: int,
        step: str = "",
    ) -> List[dict]:
        """Return time-bucketed log counts through the Grafana proxy.

        Returns list of ``{"ts": epoch_seconds, "count": int}``.
        """
        uid = self.get_datasource_uid(env)

        site = _sanitize(site, "site")
        hostname = _sanitize(hostname, "hostname")
        deployment = _sanitize(deployment, "deployment")

        # Auto-calculate step to stay under Loki's 11,000 point limit
        if not step:
            step = self._auto_step(from_ns, to_ns)

        parts = [f'site="{site}"']
        if hostname:
            parts.append(f'hostname="{hostname}"')
        if deployment:
            parts.append(f'deployment_name="{deployment}"')
        selector = "{" + ", ".join(parts) + "}"

        expr = f"count_over_time({selector}[{step}])"

        result = self._proxy_get(
            uid,
            "/loki/api/v1/query_range",
            params={
                "query": expr,
                "start": str(from_ns),
                "end": str(to_ns),
                "step": step,
                "limit": "1000",
            },
        )

        buckets: List[dict] = []
        data = result.get("data", {}) if isinstance(result, dict) else {}
        for series in data.get("result", []):
            for ts_val, count_str in series.get("values", []):
                buckets.append({
                    "ts": float(ts_val),
                    "count": int(float(count_str)),
                })

        buckets.sort(key=lambda x: x["ts"])
        return buckets
