"""
app/routes/sites.py — Site map + fleet data endpoints.
"""
from __future__ import annotations

import base64
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()

_site_manager = None


def register_singletons(site_mgr):
    global _site_manager
    _site_manager = site_mgr


def _placeholder_map():
    try:
        import cv2
        import numpy as np
        img = np.zeros((600, 800, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".png", img)
        b64 = base64.b64encode(buf).decode("utf-8")
        return {"width": 800, "height": 600, "b64": f"data:image/png;base64,{b64}"}
    except Exception:
        return {"width": 0, "height": 0, "b64": ""}


@router.get("/api/v1/sites", tags=["sites"])
def list_sites():
    if not _site_manager:
        return []
    sites = _site_manager.list_sites()
    # Return SiteInfo array that matches frontend SiteInfo type
    return [
        {"id": s, "name": s.replace("-", " ").replace("_", " ").title()}
        for s in sites
    ]


@router.get("/api/v1/sites/{site_id}/config", tags=["sites"])
def get_site_config(site_id: str):
    if not _site_manager:
        return {"resolution": 0.05, "origin": [0, 0, 0]}
    return _site_manager.get_config(site_id)


@router.get("/api/v1/sites/{site_id}/map", tags=["sites"])
def get_site_map(site_id: str, dark_mode: bool = Query(True)):
    if not _site_manager:
        return _placeholder_map()
    img = _site_manager.get_map_image(site_id, dark_mode)
    return img if img else _placeholder_map()


@router.get("/api/v1/sites/{site_id}/data", tags=["sites"])
def get_site_data(site_id: str):
    if not _site_manager:
        return {"nodes": [], "edges": [], "spots": [], "storage": []}
    return _site_manager.get_site_data(site_id)


@router.get("/api/v1/fleet/status", tags=["fleet"])
def fleet_status(site_id: str = Query("")):
    if not _site_manager:
        return {
            "site_id": site_id,
            "online_robots": 0,
            "total_robots": 0,
            "active_missions": 0,
            "alerts": 0,
        }
    sites   = _site_manager.list_sites()
    target  = site_id if site_id in sites else (sites[0] if sites else "")
    # Attempt to read robot count from site data
    robots  = 0
    try:
        data   = _site_manager.get_site_data(target)
        robots = len(data.get("robots", [])) or len(data.get("nodes", []))
    except Exception:
        pass
    return {
        "site_id":         target,
        "online_robots":   robots,
        "total_robots":    robots,
        "active_missions": 0,
        "alerts":          [],
    }
