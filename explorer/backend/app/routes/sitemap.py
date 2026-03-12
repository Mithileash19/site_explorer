"""
app/routes/sitemap.py
─────────────────────
Interactive site map endpoints.
Serves sootballs_sites data: map images, spots, racks, regions,
robot lists, and bag file listings.
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from core.config import settings
from core.logging import get_logger
from services.sitemap.git_manager import GitRepoManager
from services.sitemap.service import SiteMapService

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/sitemap", tags=["sitemap"])

_svc: Optional[SiteMapService] = None
_git_mgr: Optional[GitRepoManager] = None


def _get_git() -> GitRepoManager:
    global _git_mgr
    if _git_mgr is None:
        _git_mgr = GitRepoManager(settings.sootballs_repo_root)
    return _git_mgr


def _get_svc() -> SiteMapService:
    global _svc
    if _svc is None:
        _svc = SiteMapService(settings.sootballs_sites_root, _get_git())
    return _svc


class BranchOverrideRequest(BaseModel):
    branch: str


# ── Branch endpoints (MUST come before /{site_id}/... to avoid path conflicts) ─

@router.get("/branches")
def list_branches():
    """List all remote branch names available for branch override."""
    return _get_git().list_all_remote_branches()


@router.post("/sync")
def sync_repo():
    """
    Run ``git fetch origin --prune`` to update remote refs.
    Returns the number of site-specific branches found.
    """
    git = _get_git()
    git.fetch(force=True)
    branches = git.list_site_branches()
    return {"branches_found": len(branches)}


@router.get("/{site_id}/branch")
def get_site_branch(site_id: str):
    """
    Return branch resolution info for a site:
    which branch is used, whether it is site-specific or main,
    whether a manual override is active, the last commit, and
    the filtered list of available branches (main + valid site branches only).
    """
    git  = _get_git()
    svc  = _get_svc()
    ref  = git.resolve_branch(site_id)
    short = ref.replace("origin/", "", 1)
    site_branches = git.list_site_branches()
    site_ids = [s["id"] for s in svc.list_sites()]
    return {
        "branch":             short,
        "ref":                ref,
        "is_site_specific":   short in site_branches,
        "is_override":        git.is_override(site_id),
        "last_commit":        git.get_last_commit(ref),
        "available_branches": git.list_clean_branches(site_ids),
    }


@router.post("/{site_id}/branch")
def set_site_branch(site_id: str, body: BranchOverrideRequest):
    """Override which branch to use when loading data for *site_id*."""
    git  = _get_git()
    svc  = _get_svc()
    git.set_override(site_id, body.branch)
    ref   = git.resolve_branch(site_id)
    short = ref.replace("origin/", "", 1)
    site_branches = git.list_site_branches()
    site_ids = [s["id"] for s in svc.list_sites()]
    return {
        "branch":             short,
        "ref":                ref,
        "is_site_specific":   short in site_branches,
        "is_override":        True,
        "last_commit":        git.get_last_commit(ref),
        "available_branches": git.list_clean_branches(site_ids),
    }


@router.delete("/{site_id}/branch")
def clear_site_branch(site_id: str):
    """Remove the branch override for *site_id* (revert to auto-detect)."""
    git  = _get_git()
    svc  = _get_svc()
    git.clear_override(site_id)
    ref   = git.resolve_branch(site_id)
    short = ref.replace("origin/", "", 1)
    site_branches = git.list_site_branches()
    site_ids = [s["id"] for s in svc.list_sites()]
    return {
        "branch":             short,
        "ref":                ref,
        "is_site_specific":   short in site_branches,
        "is_override":        False,
        "last_commit":        git.get_last_commit(ref),
        "available_branches": git.list_clean_branches(site_ids),
    }


# ── Branch cleanup ─────────────────────────────────────────────────────────────

@router.get("/cleanup/plan")
def get_cleanup_plan():
    """
    Dry-run: return which remote-tracking refs would be removed and which
    would be kept.  Nothing is modified.

    A branch is *valid* iff its name is ``main`` or it exactly matches
    a known site ID (e.g. ``mncyok001``).  All other branches (feature
    branches, hotfixes, typos, CI scratch branches …) are *invalid*.
    """
    git      = _get_git()
    svc      = _get_svc()
    site_ids = [s["id"] for s in svc.list_sites()]
    return git.get_branch_cleanup_plan(site_ids)


@router.post("/cleanup")
def run_cleanup():
    """
    Execute the branch cleanup:
    - Keep ``main`` and any branch named after a known site ID.
    - Delete local remote-tracking refs for every other branch.

    **Safe**: only ``refs/remotes/origin/*`` local refs are deleted.
    The actual branches on the remote server are untouched.
    Site configs and working-tree files are not affected.
    """
    git      = _get_git()
    svc      = _get_svc()
    site_ids = [s["id"] for s in svc.list_sites()]
    result   = git.prune_invalid_remote_refs(site_ids)
    logger.info(
        "Branch cleanup: removed=%d kept=%d errors=%d",
        len(result["removed"]), len(result["kept"]), len(result["errors"]),
    )
    return result


# ── Sites ──────────────────────────────────────────────────────────────────────

@router.get("/sites")
def list_sites():
    """List all available sites in the sootballs_sites repository."""
    return _get_svc().list_sites()


@router.get("/{site_id}/map")
def get_site_map(site_id: str, dark_mode: bool = Query(True)):
    """
    Return the site's navigation map as a base64-encoded PNG together with
    its coordinate metadata (resolution, origin, width, height).
    """
    svc  = _get_svc()
    meta = svc.get_map_meta(site_id)
    img  = svc.get_map_image(site_id, dark_mode)
    if img is None:
        raise HTTPException(404, f"No map image found for site '{site_id}'")

    native_res = float(meta.get("resolution", 0.05))
    served_w = int(img.get("width", 0) or 0)
    served_h = int(img.get("height", 0) or 0)

    # Prefer explicit native image dimensions from the service. If unavailable,
    # fall back to metadata dimensions for backward compatibility.
    native_size = svc.get_native_map_size(site_id)
    if native_size is not None:
        native_w, native_h = native_size
    else:
        native_w = int(meta.get("width", 0) or 0)
        native_h = int(meta.get("height", 0) or 0)

    effective_res = native_res
    if served_w > 0 and served_h > 0 and native_w > 0 and native_h > 0:
        scale_x = served_w / native_w
        scale_y = served_h / native_h
        if scale_x > 0 and scale_y > 0:
            # Map rendering should be uniformly scaled. If not, use X scale
            # as the primary conversion factor and emit a warning.
            if abs(scale_x - scale_y) > 1e-6:
                logger.warning(
                    "get_site_map(%s): non-uniform map scaling native=%dx%d served=%dx%d "
                    "(sx=%.6f sy=%.6f)",
                    site_id,
                    native_w,
                    native_h,
                    served_w,
                    served_h,
                    scale_x,
                    scale_y,
                )
            effective_res = native_res / scale_x

    return {
        "resolution": effective_res,
        "origin":     meta["origin"],
        "width":      img["width"],
        "height":     img["height"],
        "b64":        img["b64"],
    }


@router.get("/{site_id}/data")
def get_site_data(site_id: str):
    """
    Return parsed site fixtures: spots, racks, regions, robots, nodes, edges.
    """
    return _get_svc().get_site_data(site_id)


@router.get("/{site_id}/markers")
def get_markers(site_id: str):
    """
    Return AR marker poses from config/param/markers.yaml.
    Positions are in world coordinates (metres, ROS map frame).
    Yaw is in radians.
    """
    return _get_svc().get_markers(site_id)

