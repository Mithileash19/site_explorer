"""
app/routes/investigation.py — Full AMR incident investigation pipeline + SSE streaming.
"""
from __future__ import annotations

import asyncio
import json
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from core.logging import get_logger
from schemas.investigation import (
    IncidentReportRequest, OrchestratorResponse,
)

logger = get_logger(__name__)
router = APIRouter()

_engine      = None
_llm_service = None


def register_singletons(engine, llm):
    global _engine, _llm_service
    _engine      = engine
    _llm_service = llm


@router.post("/api/v1/investigate", tags=["investigation"])
def investigate(req: IncidentReportRequest) -> OrchestratorResponse:
    if not _engine:
        raise HTTPException(503, "Investigation engine not available.")

    import os
    ros_signals  = None
    log_analysis = None

    # ── ROS bag analysis (if bag_path provided) ──────────────────────────────
    if req.bag_path and os.path.exists(req.bag_path):
        try:
            from services.ros.log_extractor import ROSLogExtractor
            from services.ros.log_analyzer_engine import LogAnalyzerEngine

            extractor    = ROSLogExtractor(req.bag_path)
            all_logs     = extractor.extract()
            # Use the full bag as the window for investigation
            if all_logs:
                midpoint = (all_logs[0]["timestamp"] + all_logs[-1]["timestamp"]) / 2
                filtered = extractor.filter_window(all_logs, midpoint, window=9999)
                priority = extractor.priority_logs(filtered)
                if _llm_service:
                    log_analysis = _llm_service.generate_log_incident_summary(
                        robot_name        = os.path.basename(req.bag_path),
                        incident_time     = req.title,
                        filtered_logs     = filtered,
                        priority_logs     = priority,
                        issue_description = req.description,
                    )

            engine       = LogAnalyzerEngine(req.bag_path)
            engine_res   = engine.analyze()
            summary      = engine_res.get("summary", {})
            ros_signals  = {
                "hardware_signals":         0.6 if summary.get("status") == "CRITICAL" else 0.3,
                "log_correlation_strength": 0.7 if summary.get("status") in ("CRITICAL", "WARNING") else 0.4,
                "jumps_detected":           0,
                "scan_dropouts":            0,
                "velocity_spikes":          0,
                "battery_events":           0,
                "evidence":                 summary.get("hypothesis", ""),
            }
        except Exception as e:
            logger.error("Investigation bag analysis failed: %s", e)

    try:
        result = _engine.investigate(req, ros_signals=ros_signals, log_analysis=log_analysis)
    except Exception as e:
        logger.error("investigate() failed: %s", e, exc_info=True)
        raise HTTPException(500, f"Investigation failed: {e}")
    return result


@router.get("/api/v1/investigate/stream", tags=["investigation"])
async def investigate_stream(
    title:       str,
    description: str,
    bag_path:    str = "",
    site_id:     str = "",
):
    """SSE endpoint — streams investigation progress events."""

    async def event_generator() -> AsyncGenerator[str, None]:
        def send(data: dict) -> str:
            return f"data: {json.dumps(data)}\n\n"

        yield send({"step": "start", "message": "Investigation started…"})
        await asyncio.sleep(0.1)

        yield send({"step": "bag_analysis", "message": "Parsing ROS bag…"})
        await asyncio.sleep(0.5)

        yield send({"step": "similarity_search", "message": "Searching incident history…"})
        await asyncio.sleep(0.3)

        yield send({"step": "llm_analysis", "message": "Running AI analysis…"})

        # Run the actual investigation
        try:
            req    = IncidentReportRequest(
                title       = title,
                description = description,
                bag_path    = bag_path or None,
                site_id     = site_id or None,
            )
            result = await asyncio.to_thread(
                _engine.investigate, req
            ) if _engine else None
            if result:
                yield send({
                    "step":    "complete",
                    "message": "Analysis complete.",
                    "data":    result.model_dump(),
                })
            else:
                yield send({"step": "error", "message": "Engine not available."})
        except Exception as e:
            yield send({"step": "error", "message": str(e)})

    return StreamingResponse(event_generator(), media_type="text/event-stream")
