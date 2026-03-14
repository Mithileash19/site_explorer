"""Routes for the Grafana-style log viewer.

Provides:
  GET /api/v1/logs/environments      — hardcoded environment list
  GET /api/v1/logs/sites             — Loki label values for site
  GET /api/v1/logs/hostnames         — Loki label values for hostname
  GET /api/v1/logs/deployments       — Loki label values for deployment_name
  GET /api/v1/logs/volume            — time-bucketed log counts
  GET /api/v1/logs/query             — log lines from Loki (hard-capped at 4000)
  GET /api/v1/logs/debug/datasources — debug: list Grafana datasources
  GET /api/v1/logs/debug/labels      — debug: raw label value check

All Loki requests go through Grafana's datasource proxy endpoint so that
we never need direct Loki network access.
"""
from __future__ import annotations

import os
import re
import time
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query

from core.config import settings
from core.logging import get_logger
from schemas.grafana import GrafanaLogsResponse
from services.grafana.loki_service import LokiService

logger = get_logger(__name__)
router = APIRouter()

_loki: LokiService | None = None
_grafana_svc = None  # legacy GrafanaService kept for /api/v1/logs backward compat


def register_singletons(grafana_svc=None) -> None:
    global _loki, _grafana_svc
    _loki = LokiService()
    _grafana_svc = grafana_svc


def _require_loki() -> LokiService:
    if _loki is None:
        raise HTTPException(503, "Loki service not initialised.")
    return _loki


_MAX_LINES = 4000

ENVIRONMENTS = [
    "sootballs-prod-logs-loki-US-latest",
    "sootballs-prod-logs-loki",
    "sootballs-staging-logs-loki",
    "rio-loki",
    "Loki",
]


# ── Error helpers ──────────────────────────────────────────────────────────────

def _classify_error(exc: RuntimeError) -> HTTPException:
    """Map a RuntimeError from LokiService to the correct HTTP status."""
    msg = str(exc)
    if "token invalid" in msg.lower() or "expired" in msg.lower():
        return HTTPException(
            401,
            "Grafana token invalid or expired. "
            "Regenerate the service account token.",
        )
    return HTTPException(503, msg)


def _fallback_sites_from_disk() -> List[str]:
    """Best-effort fallback when Loki label API is unavailable."""
    try:
        roots = [
            settings.sites_root,
            getattr(settings, "sootballs_sites_root", ""),
        ]
        site_ids: set[str] = set()
        for root in roots:
            if not root or not os.path.exists(root):
                continue
            for d in os.listdir(root):
                if os.path.isdir(os.path.join(root, d)) and not d.startswith("."):
                    site_ids.add(d)
        return sorted(site_ids)
    except Exception:
        return []


# ── Environment list (hardcoded) ───────────────────────────────────────────────

@router.get("/api/v1/logs/environments", tags=["logs"])
def list_environments() -> List[str]:
    """Return the static list of Loki environment datasource names."""
    return ENVIRONMENTS


# ── Site discovery ─────────────────────────────────────────────────────────────

@router.get("/api/v1/logs/sites", tags=["logs"])
def list_sites(
    env: str = Query(..., description="Environment / Loki datasource name"),
) -> List[str]:
    """Return sorted site label values for the given environment."""
    loki = _require_loki()
    try:
        return loki.label_values("site", env=env)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        fallback = _fallback_sites_from_disk()
        if fallback:
            logger.warning("Loki site label lookup failed; using disk fallback: %s", exc)
            return fallback
        raise _classify_error(exc) from exc


# ── Hostname discovery ─────────────────────────────────────────────────────────

@router.get("/api/v1/logs/hostnames", tags=["logs"])
def list_hostnames(
    env: str = Query(..., description="Environment"),
    site: str = Query(..., description="Site ID, e.g. denjef001"),
    datasource: Optional[str] = Query(None, description="(legacy, ignored)"),
) -> List[str]:
    """Return sorted unique hostnames for a site."""
    loki = _require_loki()
    try:
        return loki.label_values("hostname", env=env, extra_matchers={"site": site})
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        raise _classify_error(exc) from exc


# ── Deployment discovery ───────────────────────────────────────────────────────

@router.get("/api/v1/logs/deployments", tags=["logs"])
def list_deployments(
    env: str = Query(..., description="Environment"),
    site: str = Query(..., description="Site ID"),
    hostname: str = Query(..., description="Hostname"),
    datasource: Optional[str] = Query(None, description="(legacy, ignored)"),
) -> List[str]:
    """Return sorted unique deployment_names for a site+hostname."""
    loki = _require_loki()
    try:
        return loki.label_values(
            "deployment_name",
            env=env,
            extra_matchers={"site": site, "hostname": hostname},
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        raise _classify_error(exc) from exc


# ── Volume query ───────────────────────────────────────────────────────────────

@router.get("/api/v1/logs/volume", tags=["logs"])
def log_volume(
    env: str = Query(...),
    site: str = Query(...),
    hostname: str = Query(""),
    deployment: str = Query(""),
    from_ms: Optional[int] = Query(None, alias="from"),
    to_ms: Optional[int] = Query(None, alias="to"),
) -> List[dict]:
    """Return time-bucketed log counts for the volume chart."""
    loki = _require_loki()
    now_ms = int(time.time() * 1000)
    to_ms = to_ms or now_ms
    from_ms = from_ms or (to_ms - 15 * 60 * 1000)

    from_ns = from_ms * 1_000_000
    to_ns = to_ms * 1_000_000

    try:
        return loki.query_volume(
            env=env, site=site, hostname=hostname, deployment=deployment,
            from_ns=from_ns, to_ns=to_ns,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        raise _classify_error(exc) from exc


# ── Main log query ─────────────────────────────────────────────────────────────

@router.get("/api/v1/logs/query", tags=["logs"])
def query_logs(
    env: str = Query(...),
    site: str = Query(...),
    hostname: str = Query(""),
    deployment: str = Query(""),
    search: str = Query(""),
    exclude: str = Query(""),
    from_ms: Optional[int] = Query(None, alias="from"),
    to_ms: Optional[int] = Query(None, alias="to"),
    limit: int = Query(4000, ge=1, le=4000),
) -> dict:
    """Fetch log lines from Loki.  Hard-capped at 4000 lines."""
    loki = _require_loki()
    now_ms = int(time.time() * 1000)
    to_ms = to_ms or now_ms
    from_ms = from_ms or (to_ms - 15 * 60 * 1000)

    limit = min(limit, _MAX_LINES)
    from_ns = from_ms * 1_000_000
    to_ns = to_ms * 1_000_000

    try:
        lines, total = loki.query_logs(
            env=env, site=site, hostname=hostname, deployment=deployment,
            from_ns=from_ns, to_ns=to_ns,
            search=search, exclude=exclude, limit=limit,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        raise _classify_error(exc) from exc

    return {
        "lines": lines,
        "total_count": total,
        "limit": limit,
        "from_ms": from_ms,
        "to_ms": to_ms,
    }


# ── Debug endpoints ────────────────────────────────────────────────────────────

@router.get("/api/v1/logs/debug/datasources", tags=["logs-debug"])
def debug_datasources():
    """List all Grafana datasources (verifies token + connectivity).

    WARNING: Remove or protect this endpoint before production use.
    """
    loki = _require_loki()
    try:
        return loki.list_datasources_raw()
    except requests.exceptions.HTTPError as exc:
        raise _classify_error(RuntimeError(str(exc))) from exc
    except RuntimeError as exc:
        raise _classify_error(exc) from exc


@router.get("/api/v1/logs/debug/labels", tags=["logs-debug"])
def debug_labels(
    env: str = Query(..., description="Environment / datasource name"),
    site: str = Query("", description="Site ID (optional)"),
    hostname: str = Query("", description="Hostname (optional)"),
) -> dict:
    """Fetch raw label values for hostname and deployment_name.

    WARNING: Remove or protect this endpoint before production use.
    """
    loki = _require_loki()
    try:
        site_matchers = {"site": site} if site else {}
        hostname_matchers = dict(site_matchers)
        if hostname:
            hostname_matchers["hostname"] = hostname

        hostnames = loki.label_values(
            "hostname", env=env, extra_matchers=site_matchers,
        )
        deployments = loki.label_values(
            "deployment_name", env=env, extra_matchers=hostname_matchers,
        )
        return {
            "env": env,
            "datasource_uid": loki.get_datasource_uid(env),
            "site_filter": site or "(none)",
            "hostname_filter": hostname or "(none)",
            "hostnames": hostnames,
            "deployments": deployments,
        }
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        raise _classify_error(exc) from exc


# Need requests import for the debug endpoint exception handler
import requests  # noqa: E402


# ── Legacy endpoint (backward compat) ─────────────────────────────────────────


@router.get("/api/v1/logs", tags=["logs"], response_model=GrafanaLogsResponse)
def fetch_logs_legacy(
    env: str = Query(..., description="Loki datasource name"),
    site: str = Query(..., description="Site ID"),
    hostname: str = Query(".*", description="Hostname or regex"),
    deployment: Optional[str] = Query(None, description="Deployment name"),
    search: Optional[str] = Query(None, description="Log line include filter"),
    exclude: Optional[str] = Query(None, description="Log line exclude regex"),
    from_ms: Optional[int] = Query(None, description="Start epoch ms"),
    to_ms: Optional[int] = Query(None, description="End epoch ms"),
    max_lines: int = Query(2000, ge=1, le=5000),
) -> GrafanaLogsResponse:
    """Legacy endpoint — proxies through GrafanaService for backward compat."""
    import re as _re
    if _grafana_svc is None:
        raise HTTPException(503, "Grafana service not initialised.")
    try:
        resp = _grafana_svc.fetch_logs(
            site=site,
            hostname=hostname,
            deployment=deployment,
            log_filter=search or "",
            from_ms=from_ms,
            to_ms=to_ms,
            max_lines=max_lines,
            datasource_name=env,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    except Exception as exc:
        logger.error("fetch_logs failed: %s", exc, exc_info=True)
        raise HTTPException(500, f"Failed to fetch logs: {exc}") from exc

    if exclude and exclude.strip():
        try:
            pattern = _re.compile(exclude)
        except _re.error:
            raise HTTPException(400, f"Invalid exclude regex: {exclude}")
        filtered = [log for log in resp.logs if not pattern.search(log.line)]
        resp = GrafanaLogsResponse(
            site=resp.site,
            hostname=resp.hostname,
            deployment=resp.deployment,
            from_ms=resp.from_ms,
            to_ms=resp.to_ms,
            line_count=len(filtered),
            logs=filtered,
        )

    return resp
