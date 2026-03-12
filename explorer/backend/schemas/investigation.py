"""
schemas/investigation.py — Pydantic models for the AMR Master AI investigation pipeline.
"""
from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel


class IncidentReportRequest(BaseModel):
    """Submitted by the operator to start an investigation."""
    title:             str
    description:       str
    bag_path:          Optional[str]   = None
    site_id:           Optional[str]   = None
    grafana_link:      Optional[str]   = None
    slack_url:         Optional[str]   = None
    sw_version:        Optional[str]   = None
    config_changed:    bool            = False


class SimilarCase(BaseModel):
    id:          str    # str index
    title:       str    # short thread summary
    description: str    # full thread summary
    similarity:  float  # 0.0 – 1.0
    resolution:  str    # recommended fix


class RankedItem(BaseModel):
    description: str
    confidence:  float        # 0.0 – 1.0
    evidence:    List[str] = []


class OrchestratorResponse(BaseModel):
    status:                      str
    confidence_score:            float   # 0.0 – 1.0
    human_intervention_required: bool
    issue_summary:               str
    similar_cases:               List[SimilarCase]  = []
    log_anomaly_summary:         str                = ""
    ranked_causes:               List[RankedItem]   = []
    ranked_solutions:            List[RankedItem]   = []
    safety_assessment:           str                = ""
    raw_analysis:                str                = ""


