"""Pydantic schemas for Grafana read-only integration."""
from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel


class GrafanaStatusResponse(BaseModel):
    status: str          # "online" | "offline" | "unconfigured"
    grafana_version: Optional[str] = None
    org_name: Optional[str] = None
    loki_datasources: List[str] = []
    fix: Optional[str] = None


class GrafanaDashboard(BaseModel):
    uid: str
    title: str
    folder: str
    url: str
    tags: List[str] = []


class GrafanaDashboardsResponse(BaseModel):
    total: int
    dashboards: List[GrafanaDashboard]


class GrafanaLogLine(BaseModel):
    timestamp_ms: int
    labels: Dict[str, str]
    line: str


class GrafanaLogsResponse(BaseModel):
    site: str
    hostname: str
    deployment: Optional[str]
    from_ms: int
    to_ms: int
    line_count: int
    logs: List[GrafanaLogLine]


class GrafanaAnnotation(BaseModel):
    id: int
    time_ms: int
    text: str
    tags: List[str] = []
    dashboard_uid: Optional[str] = None


class GrafanaAnnotationsResponse(BaseModel):
    site: str
    from_ms: int
    to_ms: int
    count: int
    annotations: List[GrafanaAnnotation]
