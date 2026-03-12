"""
app/routes/bags.py — ROS bag upload, timeline, log analysis, map diff.
"""
from __future__ import annotations

import datetime
import os
import pathlib
import re
import uuid
from typing import Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

from core.config import settings
from core.logging import get_logger
from schemas.bag_analysis import (
    BagLogAnalysisRequest, BagLogAnalysisResponse, BagTimeline, MapDiffRequest,
    MapDiffResponse, LogEntry, TimelineBucket,
)
from services.ros.log_extractor import ROSLogExtractor
from services.ros.log_analyzer_engine import LogAnalyzerEngine
from services.ros.map_processor import process_bag_for_changes

logger = get_logger(__name__)
router = APIRouter()

_llm_service  = None
_site_manager = None

ALLOWED_EXTENSIONS = {".bag", ".db3"}
MAX_UPLOAD_BYTES    = 400 * 1024 * 1024  # 400 MB


def register_singletons(llm, site_mgr):
    global _llm_service, _site_manager
    _llm_service  = llm
    _site_manager = site_mgr


def _save_upload(file: UploadFile) -> tuple[str, int]:
    """Save an uploaded bag file; returns (absolute path, byte count).

    The file is stored using the sanitised original filename so users can
    identify their bags in the UI.  On collision a short hex suffix is appended
    to keep the name recognisable while avoiding overwrites.
    """
    upload_dir = pathlib.Path(settings.bag_upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    original = pathlib.Path(file.filename or "upload.bag")
    suffix = original.suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type: {suffix!r}. Use .bag or .db3")

    # Sanitise: keep word chars, dots, hyphens; collapse sequences of
    # other chars to a single underscore; limit stem length to 80 chars.
    safe_stem = re.sub(r"[^\w.\-]+", "_", original.stem).strip("_")[:80] or "upload"
    dest = upload_dir / f"{safe_stem}{suffix}"
    if dest.exists():
        dest = upload_dir / f"{safe_stem}_{uuid.uuid4().hex[:8]}{suffix}"

    content = file.file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File exceeds 400 MB limit.")
    with open(dest, "wb") as fh:
        fh.write(content)

    logger.info("Saved bag: %s (%d bytes)", dest, len(content))
    return str(dest), len(content)


def _map_log(raw: dict) -> LogEntry:
    """Map extractor's internal dict (log_level/node_name) to LogEntry schema."""
    return LogEntry(
        timestamp = raw["timestamp"],
        datetime  = raw.get("datetime", ""),
        level     = raw.get("log_level", "INFO"),
        node      = raw.get("node_name", ""),
        message   = raw.get("message",   ""),
    )


def _map_bucket(b: dict) -> TimelineBucket:
    """Map extractor's internal bucket dict to TimelineBucket schema."""
    return TimelineBucket(
        t_start     = b["from_ts"],
        t_end       = b["to_ts"],
        count       = b["total"],
        error_count = b["error"],
        warn_count  = b["warn"],
    )


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post("/api/v1/bags/upload", tags=["bags"])
async def upload_bag(file: UploadFile = File(...)):
    bag_path, nbytes = _save_upload(file)
    return {
        "bag_path": bag_path,
        "size_mb":  round(nbytes / (1024 * 1024), 2),
    }


@router.get("/api/v1/bags/timeline", tags=["bags"])
def bag_timeline(
    bag_path:  str = Query(...),
    n_buckets: int = Query(300, ge=10, le=1000),
):
    if not os.path.exists(bag_path):
        raise HTTPException(404, f"Bag not found: {bag_path}")

    try:
        extractor = ROSLogExtractor(bag_path)
        logs      = extractor.extract()
        raw_bkts  = extractor.get_timeline_buckets(logs, n_buckets)
    except Exception as e:
        logger.error("bag_timeline(%s): %s", bag_path, e)
        raise HTTPException(422, f"Could not read bag timeline: {e}")

    return BagTimeline(
        bag_path = bag_path,
        buckets  = [_map_bucket(b) for b in raw_bkts],
    )


@router.post("/api/v1/bags/analyze", tags=["bags"])
def analyze_bag_logs(req: BagLogAnalysisRequest):
    if not os.path.exists(req.bag_path):
        raise HTTPException(404, f"Bag not found: {req.bag_path}")

    # ── 1. Extract logs and apply optional window filter ──────────────────────
    extractor = ROSLogExtractor(req.bag_path)
    all_logs  = extractor.extract()

    if req.window_start is not None and req.window_end is not None:
        filtered = [l for l in all_logs
                    if req.window_start <= l["timestamp"] <= req.window_end]
    else:
        filtered = all_logs

    # ── 2. Compute summary stats ─────────────────────────────────────────────
    duration_secs = 0.0
    if all_logs:
        duration_secs = all_logs[-1]["timestamp"] - all_logs[0]["timestamp"]

    error_count   = sum(1 for l in filtered if l["log_level"] in ("ERROR", "FATAL"))
    warning_count = sum(1 for l in filtered if l["log_level"] == "WARN")

    # ── 3. Rule-based anomaly detection ──────────────────────────────────────
    engine_hypothesis: str  = ""
    try:
        engine        = LogAnalyzerEngine(req.bag_path)
        engine_result = engine.analyze()
        engine_hypothesis = engine_result.get("summary", {}).get("hypothesis", "")
    except Exception as e:
        logger.warning("LogAnalyzerEngine failed: %s", e)

    # ── 4. LLM analysis ───────────────────────────────────────────────────────
    llm_summary = ""
    if _llm_service and filtered:
        incident_ts = (
            (req.window_start + req.window_end) / 2
            if req.window_start is not None and req.window_end is not None
            else (all_logs[0]["timestamp"] if all_logs else 0.0)
        )
        incident_dt = datetime.datetime.utcfromtimestamp(incident_ts).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        try:
            sections = _llm_service.generate_log_incident_summary(
                robot_name        = os.path.basename(req.bag_path),
                incident_time     = incident_dt,
                filtered_logs     = filtered,
                priority_logs     = extractor.priority_logs(filtered),
                issue_description = req.issue_description,
                engine_hypothesis = engine_hypothesis,
            )
            llm_summary = "\n\n".join(
                f"### {k.replace('_', ' ').title()}\n{v}"
                for k, v in sections.items() if v
            )
        except Exception as e:
            logger.error("LLM analysis failed: %s", e)

    # ── 5. Build response ─────────────────────────────────────────────────────
    return BagLogAnalysisResponse(
        status            = "ok",
        bag_path          = req.bag_path,
        duration_secs     = round(duration_secs, 3),
        total_messages    = len(all_logs),
        error_count       = error_count,
        warning_count     = warning_count,
        log_entries       = [_map_log(l) for l in filtered[:2000]],
        engine_hypothesis = engine_hypothesis,
        llm_summary       = llm_summary,
    )


@router.post("/api/v1/bags/mapdiff", tags=["bags"])
def map_diff(req: MapDiffRequest):
    """Compare a bag's map against the reference site map."""
    if not os.path.exists(req.bag_path):
        raise HTTPException(404, f"Bag not found: {req.bag_path}")

    if not _site_manager:
        raise HTTPException(503, "Site manager not available.")

    map_data = _site_manager.get_map_image(req.site_id or "", dark_mode=False) if req.site_id else None
    if not map_data:
        # Return a neutral response when there is no reference map
        return MapDiffResponse(iou_score=0.0, diff_image_b64="", message="No reference map available.")

    cfg = _site_manager.get_config(req.site_id) if req.site_id else {}

    try:
        b64_raw = map_data["b64"].split(",")[-1]
        diff_b64, score = process_bag_for_changes(
            bag_path         = pathlib.Path(req.bag_path),
            original_map_b64 = b64_raw,
            resolution       = cfg.get("resolution", 0.05),
            origin           = cfg.get("origin", [0.0, 0.0, 0.0]),
        )
    except Exception as e:
        logger.error("map_diff failed: %s", e)
        raise HTTPException(500, str(e))

    if diff_b64 is None:
        raise HTTPException(500, "Map diff computation failed.")

    return MapDiffResponse(
        iou_score      = round(min(1.0, max(0.0, score / 100.0)), 4),
        diff_image_b64 = diff_b64,          # raw base64, no data: prefix
        message        = f"IoU score: {score:.1f}%",
    )

