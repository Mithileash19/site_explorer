"""Read-only Grafana service.

All methods use HTTP GET (or POST /api/ds/query which is a read-only data
query endpoint — it executes no mutations on Grafana).  No DELETE/PUT/PATCH
calls are ever made.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import requests

from core.config import settings
from core.logging import get_logger
from schemas.grafana import (
    GrafanaAnnotation,
    GrafanaAnnotationsResponse,
    GrafanaDashboard,
    GrafanaDashboardsResponse,
    GrafanaLogLine,
    GrafanaLogsResponse,
    GrafanaStatusResponse,
)

logger = get_logger(__name__)

# Loki datasource UIDs mapped by logical name (resolved from datasource list).
# The service discovers these at runtime so adding new Loki instances requires
# no code changes.
_LOKI_TYPE = "loki"


class GrafanaService:
    """Safe, read-only wrapper around the Grafana HTTP API."""

    def __init__(self) -> None:
        self._base = settings.grafana_url.rstrip("/")
        self._token = settings.grafana_api_key

    # ── Internal helpers ────────────────────────────────────────────────────

    @property
    def _configured(self) -> bool:
        return bool(self._base and self._token)

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
        }

    def _get(self, path: str, params: Optional[Dict] = None, timeout: int = 10) -> dict | list:
        """Perform a safe GET request; raises RuntimeError on failure."""
        if not self._configured:
            raise RuntimeError(
                "Grafana is not configured. Set GRAFANA_URL and GRAFANA_API_KEY in backend/.env."
            )
        url = f"{self._base}{path}"
        try:
            resp = requests.get(url, headers=self._headers(), params=params, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(f"Cannot reach Grafana at {self._base}.") from exc
        except requests.exceptions.HTTPError as exc:
            raise RuntimeError(f"Grafana API error {exc.response.status_code}: {exc.response.text[:200]}") from exc
        except Exception as exc:
            raise RuntimeError(f"Grafana request failed: {exc}") from exc

    def _post_query(self, payload: dict, timeout: int = 30) -> dict:
        """POST to /api/ds/query — read-only datasource query endpoint."""
        if not self._configured:
            raise RuntimeError(
                "Grafana is not configured. Set GRAFANA_URL and GRAFANA_API_KEY in backend/.env."
            )
        url = f"{self._base}/api/ds/query"
        try:
            resp = requests.post(url, headers=self._headers(), json=payload, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(f"Cannot reach Grafana at {self._base}.") from exc
        except requests.exceptions.HTTPError as exc:
            raise RuntimeError(f"Grafana query error {exc.response.status_code}: {exc.response.text[:200]}") from exc
        except Exception as exc:
            raise RuntimeError(f"Grafana query failed: {exc}") from exc

    # ── Datasource helpers ──────────────────────────────────────────────────

    def _loki_datasources(self) -> List[Dict]:
        """Return all Loki datasources available in this Grafana instance."""
        try:
            all_ds = self._get("/api/datasources")
            return [d for d in (all_ds if isinstance(all_ds, list) else []) if d.get("type") == _LOKI_TYPE]
        except Exception:
            return []

    def _resolve_loki_uid(self, datasource_name: Optional[str] = None) -> Tuple[str, str]:
        """Resolve a Loki datasource UID.  Returns (uid, name).

        If datasource_name is provided, looks up that specific datasource.
        Otherwise returns the first Loki datasource found.
        """
        loki_ds = self._loki_datasources()
        if not loki_ds:
            raise RuntimeError("No Loki datasource found in Grafana.")

        if datasource_name:
            match = next((d for d in loki_ds if d["name"] == datasource_name), None)
            if not match:
                available = [d["name"] for d in loki_ds]
                raise ValueError(
                    f"Loki datasource '{datasource_name}' not found. "
                    f"Available: {available}"
                )
            return match["uid"], match["name"]

        # Default to sootballs-prod-logs-loki if present, otherwise first Loki
        preferred = next(
            (d for d in loki_ds if "prod" in d["name"] and "sootballs" in d["name"]),
            loki_ds[0],
        )
        return preferred["uid"], preferred["name"]

    # ── Public API ──────────────────────────────────────────────────────────

    def status(self) -> GrafanaStatusResponse:
        """Verify configuration and return Grafana health information."""
        if not self._configured:
            return GrafanaStatusResponse(
                status="unconfigured",
                fix="Set GRAFANA_URL and GRAFANA_API_KEY in backend/.env and recreate the backend container.",
            )
        try:
            health = self._get("/api/health")
            org    = self._get("/api/org")
            loki_names = [d["name"] for d in self._loki_datasources()]
            return GrafanaStatusResponse(
                status="online",
                grafana_version=health.get("version"),
                org_name=org.get("name"),
                loki_datasources=loki_names,
            )
        except RuntimeError as exc:
            return GrafanaStatusResponse(status="offline", fix=str(exc))

    def list_dashboards(self, query: str = "", limit: int = 500) -> GrafanaDashboardsResponse:
        """Return all dashboards, optionally filtered by a search string."""
        params: Dict = {"type": "dash-db", "limit": limit}
        if query:
            params["query"] = query

        raw = self._get("/api/search", params=params)
        dashboards = [
            GrafanaDashboard(
                uid=d["uid"],
                title=d["title"],
                folder=d.get("folderTitle", "General"),
                url=f"{self._base}{d['url']}",
                tags=d.get("tags", []),
            )
            for d in (raw if isinstance(raw, list) else [])
        ]
        return GrafanaDashboardsResponse(total=len(dashboards), dashboards=dashboards)

    def fetch_logs(
        self,
        site: str,
        hostname: str = ".*",
        deployment: Optional[str] = None,
        log_filter: str = "",
        from_ms: Optional[int] = None,
        to_ms: Optional[int] = None,
        max_lines: int = 200,
        datasource_name: Optional[str] = None,
    ) -> GrafanaLogsResponse:
        """Fetch log lines from Loki for a specific site/host via Grafana proxy.

        Uses POST /api/ds/query which is the standard Grafana datasource query
        endpoint — read-only, identical to what the Logs Viewer dashboard calls.
        """
        import time

        to_ms   = to_ms   or int(time.time() * 1000)
        from_ms = from_ms or (to_ms - 15 * 60 * 1000)  # default: last 15 min

        uid, ds_name = self._resolve_loki_uid(datasource_name)

        # Build LogQL expression
        hostname_selector = f'hostname=~"{hostname}"' if hostname != ".*" else 'hostname=~".*"'
        expr = f'{{site="{site}", {hostname_selector}}}'
        if deployment:
            expr = f'{{site="{site}", {hostname_selector}, deployment_name=~"{deployment}"}}'
        if log_filter:
            expr += f' |~ `{log_filter}`'

        payload = {
            "queries": [{
                "datasource": {"uid": uid, "type": "loki"},
                "expr": expr,
                "queryType": "range",
                "maxLines": max_lines,
                "refId": "A",
            }],
            "from": str(from_ms),
            "to":   str(to_ms),
        }

        result = self._post_query(payload, timeout=30)
        frames = result.get("results", {}).get("A", {}).get("frames", [])

        log_lines: List[GrafanaLogLine] = []
        for frame in frames:
            values = frame.get("data", {}).get("values", [])
            # Grafana Loki frame layout (from /api/ds/query):
            #   values[0] = labels dicts per row (type=other)
            #   values[1] = timestamps in milliseconds (type=time)
            #   values[2] = log line text (type=string)
            if len(values) < 3:
                continue
            labels_col    = values[0]
            timestamp_col = values[1]
            line_col      = values[2]

            for raw_labels, ts_ms, line in zip(labels_col, timestamp_col, line_col):
                if not line:
                    continue
                labels = {k: str(v) for k, v in raw_labels.items()} if isinstance(raw_labels, dict) else {}
                ts_ms  = int(ts_ms) if isinstance(ts_ms, (int, float)) else from_ms
                log_lines.append(GrafanaLogLine(timestamp_ms=ts_ms, labels=labels, line=str(line)))

        log_lines.sort(key=lambda x: x.timestamp_ms, reverse=True)

        return GrafanaLogsResponse(
            site=site,
            hostname=hostname,
            deployment=deployment,
            from_ms=from_ms,
            to_ms=to_ms,
            line_count=len(log_lines),
            logs=log_lines,
        )

    def fetch_annotations(
        self,
        site: Optional[str] = None,
        from_ms: Optional[int] = None,
        to_ms: Optional[int] = None,
        tags: Optional[List[str]] = None,
        limit: int = 100,
    ) -> GrafanaAnnotationsResponse:
        """Fetch Grafana annotations in a time window, optionally scoped to a site tag."""
        import time

        to_ms   = to_ms   or int(time.time() * 1000)
        from_ms = from_ms or (to_ms - 60 * 60 * 1000)  # default: last 1 hour

        params: Dict = {"from": from_ms, "to": to_ms, "limit": limit}
        query_tags = list(tags or [])
        if site:
            query_tags.append(site)

        raw = self._get("/api/annotations", params=params)

        annotations = []
        for a in (raw if isinstance(raw, list) else []):
            a_tags = a.get("tags") or []
            # Filter by tags if specified
            if query_tags and not any(t in a_tags for t in query_tags):
                continue
            annotations.append(GrafanaAnnotation(
                id=a.get("id", 0),
                time_ms=a.get("time", 0),
                text=a.get("text", "") or "",
                tags=a_tags,
                dashboard_uid=a.get("dashboardUID"),
            ))

        return GrafanaAnnotationsResponse(
            site=site or "",
            from_ms=from_ms,
            to_ms=to_ms,
            count=len(annotations),
            annotations=annotations,
        )
