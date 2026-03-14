"""Routes for read-only Grafana integration."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from core.logging import get_logger
from schemas.grafana import (
    GrafanaAnnotationsResponse,
    GrafanaDashboardsResponse,
    GrafanaLogsResponse,
    GrafanaStatusResponse,
)
from services.grafana.grafana_service import GrafanaService

logger = get_logger(__name__)
router = APIRouter()

_svc: GrafanaService | None = None


def register_singletons() -> None:
    global _svc
    _svc = GrafanaService()


def _require_svc() -> GrafanaService:
    if _svc is None:
        raise HTTPException(503, "Grafana service not initialised.")
    return _svc


# ── Status ─────────────────────────────────────────────────────────────────────

@router.get("/api/v1/grafana/status", tags=["grafana"], response_model=GrafanaStatusResponse)
def grafana_status() -> GrafanaStatusResponse:
    return _require_svc().status()


# ── Dashboards ─────────────────────────────────────────────────────────────────

@router.get("/api/v1/grafana/dashboards", tags=["grafana"], response_model=GrafanaDashboardsResponse)
def list_dashboards(
    q: str = Query("", description="Optional title search string"),
    limit: int = Query(500, ge=1, le=2000),
) -> GrafanaDashboardsResponse:
    svc = _require_svc()
    try:
        return svc.list_dashboards(query=q, limit=limit)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    except Exception as exc:
        logger.error("list_dashboards failed: %s", exc, exc_info=True)
        raise HTTPException(500, f"Failed to list dashboards: {exc}") from exc


# ── Logs ───────────────────────────────────────────────────────────────────────

@router.get("/api/v1/grafana/logs", tags=["grafana"], response_model=GrafanaLogsResponse)
def fetch_logs(
    site: str                  = Query(...,  description="Site ID, e.g. actsgm001"),
    hostname: str              = Query(".*", description="Hostname or regex, e.g. edge01"),
    deployment: Optional[str]  = Query(None, description="Deployment name filter"),
    filter: Optional[str]      = Query(None, description="Log line substring filter"),
    from_ms: Optional[int]     = Query(None, description="Start epoch milliseconds"),
    to_ms: Optional[int]       = Query(None, description="End epoch milliseconds"),
    max_lines: int             = Query(200,  ge=1, le=2000),
    datasource: Optional[str]  = Query(None, description="Loki datasource name override"),
) -> GrafanaLogsResponse:
    svc = _require_svc()
    try:
        return svc.fetch_logs(
            site=site,
            hostname=hostname,
            deployment=deployment,
            log_filter=filter or "",
            from_ms=from_ms,
            to_ms=to_ms,
            max_lines=max_lines,
            datasource_name=datasource,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    except Exception as exc:
        logger.error("fetch_logs failed: %s", exc, exc_info=True)
        raise HTTPException(500, f"Failed to fetch logs: {exc}") from exc


# ── Annotations ────────────────────────────────────────────────────────────────

@router.get("/api/v1/grafana/annotations", tags=["grafana"], response_model=GrafanaAnnotationsResponse)
def fetch_annotations(
    site: Optional[str]    = Query(None, description="Site tag to filter annotations"),
    from_ms: Optional[int] = Query(None, description="Start epoch milliseconds"),
    to_ms: Optional[int]   = Query(None, description="End epoch milliseconds"),
    limit: int             = Query(100, ge=1, le=1000),
) -> GrafanaAnnotationsResponse:
    svc = _require_svc()
    try:
        return svc.fetch_annotations(site=site, from_ms=from_ms, to_ms=to_ms, limit=limit)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    except Exception as exc:
        logger.error("fetch_annotations failed: %s", exc, exc_info=True)
        raise HTTPException(500, f"Failed to fetch annotations: {exc}") from exc
