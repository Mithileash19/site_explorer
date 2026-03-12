"""
services/ai/investigation_engine.py
─────────────────────────────────────
Full AMR incident investigation pipeline.
Combines ROS analysis, FAISS similarity, and structured LLM output.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from core.logging import get_logger
from schemas.investigation import (
    OrchestratorResponse, SimilarCase, RankedItem,
    IncidentReportRequest,
)
from services.ai.llm_service import LLMService
from services.ai.vector_db import HistoricalMatcher

logger = get_logger(__name__)


class InvestigationEngine:
    """
    Orchestrates all intelligence signals and produces a structured
    OrchestratorResponse with confidence scoring.
    """

    # Confidence weight constants
    W_HISTORICAL    = 0.40
    W_LOG_CORR      = 0.30
    W_VERSION       = 0.15
    W_CONFIG        = 0.10
    W_HARDWARE      = 0.05
    HUMAN_THRESHOLD = 0.60

    def __init__(self, llm: LLMService, matcher: HistoricalMatcher) -> None:
        self.llm     = llm
        self.matcher = matcher

    def investigate(
        self,
        request:       IncidentReportRequest,
        ros_signals:   Optional[Dict[str, Any]] = None,
        log_analysis:  Optional[Dict[str, Any]] = None,
    ) -> OrchestratorResponse:
        """
        Run the full investigation pipeline.

        ros_signals   — output from ROSAnomalyDetector.detect_all() (if bag provided)
        log_analysis  — output from LLMService.generate_log_incident_summary() (if bag provided)
        """
        # ── 1. FAISS similarity search ────────────────────────────────────────
        query_text   = f"{request.title}\n{request.description}"
        similar_raw  = self.matcher.search(query_text, k=5)
        similar_cases = [
            SimilarCase(
                id          = str(i),
                title       = s["summary"][:100],
                description = s["summary"][:300],
                similarity  = round(s["similarity_pct"] / 100.0, 4),
                resolution  = s["fix"],
            )
            for i, s in enumerate(similar_raw)
        ]

        # ── 2. Historical similarity score ────────────────────────────────────
        hist_score = max((s.similarity for s in similar_cases), default=0.0)

        # ── 3. Log correlation score ──────────────────────────────────────────
        log_corr = 0.5
        if ros_signals:
            log_corr = min(1.0, ros_signals.get("log_correlation_strength", 0.5))

        # ── 4. Version regression risk ────────────────────────────────────────
        version_risk = 0.5  # default neutral; extend with VersionDiffModule

        # ── 5. Config impact ──────────────────────────────────────────────────
        config_impact = 0.8 if request.config_changed else 0.3

        # ── 6. Hardware signal score ──────────────────────────────────────────
        hw_score = 0.5
        if ros_signals:
            hw_score = min(1.0, ros_signals.get("hardware_signals", 0.5))

        # ── 7. Composite confidence ───────────────────────────────────────────
        confidence = (
            self.W_HISTORICAL * hist_score
            + self.W_LOG_CORR  * log_corr
            + self.W_VERSION   * version_risk
            + self.W_CONFIG    * config_impact
            + self.W_HARDWARE  * hw_score
        )
        confidence = round(min(1.0, max(0.0, confidence)), 4)  # 0.0 – 1.0

        # ── 8. Build LLM investigation prompt ────────────────────────────────
        llm_raw = self._build_investigation_prompt(
            request, ros_signals, log_analysis, similar_cases, confidence
        )
        ai_raw = self.llm.generate_investigation_summary(llm_raw)

        # ── 9. Extract ranked causes / solutions from LLM output ─────────────
        ranked_causes    = self._parse_ranked_items(ai_raw, "Root Cause")
        ranked_solutions = self._parse_ranked_items(ai_raw, "Recommended Next Steps")

        # ── 10. Log anomaly summary ────────────────────────────────────────────
        log_anomaly_summary = ""
        if ros_signals:
            evidence = ros_signals.get("evidence", "")
            log_anomaly_summary = (
                f"Jumps: {ros_signals.get('jumps_detected', 0)}, "
                f"Scan dropouts: {ros_signals.get('scan_dropouts', 0)}, "
                f"Velocity spikes: {ros_signals.get('velocity_spikes', 0)}, "
                f"Battery events: {ros_signals.get('battery_events', 0)}. "
                f"{evidence}"
            )

        return OrchestratorResponse(
            status                      = "completed",
            confidence_score            = confidence,
            human_intervention_required = confidence < self.HUMAN_THRESHOLD,
            issue_summary               = f"{request.title}: {request.description[:300]}",
            similar_cases               = similar_cases,
            log_anomaly_summary         = log_anomaly_summary,
            ranked_causes               = ranked_causes,
            ranked_solutions            = ranked_solutions,
            safety_assessment           = (
                "⚠️ Human review required — confidence below 60%."
                if confidence < self.HUMAN_THRESHOLD else
                "✅ High confidence — automated root cause identified."
            ),
            raw_analysis                = ai_raw,
        )

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _build_investigation_prompt(
        self,
        request:       IncidentReportRequest,
        ros_signals:   Optional[Dict],
        log_analysis:  Optional[Dict],
        similar_cases: List[SimilarCase],
        confidence:    float,
    ) -> str:
        lines = [
            f"INCIDENT REPORT",
            f"  Title:       {request.title}",
            f"  Description: {request.description}",
            f"  Site:        {request.site_id or 'N/A'}",
            f"  SW Version:  {request.sw_version or 'N/A'}",
            f"  Config changed: {request.config_changed}",
            "",
        ]
        if similar_cases:
            lines.append("SIMILAR PAST INCIDENTS:")
            for sc in similar_cases[:3]:
                lines.append(f"  [{sc.similarity * 100:.1f}%] {sc.title}")
                lines.append(f"    Fix: {sc.resolution}")
            lines.append("")

        if ros_signals:
            lines.append("ROS SIGNAL ANALYSIS:")
            lines.append(f"  {ros_signals.get('evidence', '')}")
            lines.append("")

        if log_analysis:
            lines.append("LOG ANALYSIS (LLM §5):")
            lines.append(f"  Conclusion: {log_analysis.get('technical_conclusion', '')[:500]}")
            lines.append("")

        lines.append(f"COMPOSITE CONFIDENCE: {confidence * 100:.1f}%")
        lines.append("")
        lines.append(
            "Provide a structured diagnostic report with:\n"
            "1. Root Cause (ranking by likelihood with confidence %)\n"
            "2. Evidence summary\n"
            "3. Recommended Next Steps\n"
            "4. Risk assessment"
        )
        return "\n".join(lines)

    def _parse_ranked_items(self, text: str, section_keyword: str) -> List[RankedItem]:
        """Naive line-by-line parser for numbered ranked items in LLM output."""
        items: List[RankedItem] = []
        in_section = False
        for line in text.split("\n"):
            if section_keyword.lower() in line.lower():
                in_section = True
                continue
            if in_section:
                stripped = line.strip()
                if not stripped:
                    continue
                # Stop at next numbered section header
                if stripped and stripped[0].isdigit() and "." in stripped[:3] and len(stripped) > 10:
                    import re
                    pct_match = re.search(r"(\d+)%", stripped)
                    conf_pct  = float(pct_match.group(1)) if pct_match else 70.0
                    items.append(RankedItem(
                        description = stripped[:300],
                        confidence  = round(min(1.0, conf_pct / 100.0), 4),
                        evidence    = [],
                    ))
                if len(items) >= 5:
                    break
        return items
