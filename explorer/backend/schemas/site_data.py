"""
schemas/site_data.py — Pydantic models for site map and fleet data.
"""
from __future__ import annotations
from typing import List, Optional, Any
from pydantic import BaseModel


class MapConfig(BaseModel):
    resolution: float = 0.05
    origin:     List[float] = [0.0, 0.0, 0.0]


class MapImage(BaseModel):
    width:  int
    height: int
    b64:    str  # "data:image/png;base64,..."


class NodeData(BaseModel):
    id:    Any
    label: str
    x:     float
    y:     float


class EdgeData(BaseModel):
    id:    Optional[Any] = None
    start: Any
    end:   Any


class SiteData(BaseModel):
    nodes:   List[NodeData]   = []
    edges:   List[EdgeData]   = []
    spots:   List[dict]       = []
    storage: List[dict]       = []


class SiteInfo(BaseModel):
    site_id: str
    config:  MapConfig


class FleetStatusResponse(BaseModel):
    total_sites:  int
    site_ids:     List[str]
    health:       str  # "ok" | "degraded"
