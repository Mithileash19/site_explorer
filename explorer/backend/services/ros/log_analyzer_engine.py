"""
services/ros/log_analyzer_engine.py
─────────────────────────────────────
Rule-based ROS bag anomaly detector + LLM prompt builder.
Originated from site_commander; now wired to LLMService.
"""
from __future__ import annotations

import numpy as np
from typing import Any, Dict, List
from core.logging import get_logger

logger = get_logger(__name__)

# Avoid hard import failure if rosbags not installed in dev
try:
    from rosbags.rosbag1 import Reader
    from rosbags.serde import deserialize_cdr, ros1_to_cdr
    _ROSBAGS_AVAILABLE = True
except ImportError:
    _ROSBAGS_AVAILABLE = False
    logger.warning("rosbags not available — LogAnalyzerEngine in stub mode.")


class LogAnalyzerEngine:
    """
    Production-grade engine for extracting semantic events from ROS bags.
    Pipeline: parse → detect anomalies → generate hypothesis → build LLM prompt.
    """

    def __init__(self, bag_path: str) -> None:
        self.bag_path      = bag_path
        self.metadata:     Dict[str, Any]       = {}
        self.topics_index: Dict[str, Any]       = {}
        self.events:       List[Dict[str, Any]] = []
        self.incident_report: Dict[str, Any]    = {}

    # ── Public ────────────────────────────────────────────────────────────────

    def analyze(self) -> Dict[str, Any]:
        """Run the full analysis pipeline and return a structured report dict."""
        if not _ROSBAGS_AVAILABLE:
            return self._stub_result()

        logger.info("LogAnalyzerEngine: starting analysis on %s", self.bag_path)
        self._parse_and_index()
        self._detect_anomalies()
        self._generate_hypotheses()
        llm_prompt = self._construct_llm_prompt()

        return {
            "metadata":   self.metadata,
            "events":     self.events,
            "topics":     self.topics_index,
            "llm_prompt": llm_prompt,
            "summary":    self.incident_report,
        }

    # ── Private pipeline ──────────────────────────────────────────────────────

    def _parse_and_index(self) -> None:
        stats: Dict[str, Any] = {}
        start_t = float("inf")
        end_t   = float("-inf")

        with Reader(self.bag_path) as reader:
            for conn in reader.connections:
                if conn.topic not in stats:
                    stats[conn.topic] = {
                        "msg_type":   conn.msgtype,
                        "count":      0,
                        "timestamps": [],
                    }
            for conn, timestamp, _ in reader.messages():
                t = timestamp * 1e-9
                stats[conn.topic]["count"] += 1
                stats[conn.topic]["timestamps"].append(t)
                if t < start_t: start_t = t
                if t > end_t:   end_t   = t

        for topic, data in stats.items():
            ts = np.array(sorted(data["timestamps"]))
            avg_rate = 0.0
            if len(ts) > 1:
                diffs = np.diff(ts)
                if np.mean(diffs) > 0:
                    avg_rate = 1.0 / np.mean(diffs)
            data["frequency"] = round(avg_rate, 2)
            data["start"]     = float(ts[0])  if len(ts) > 0 else 0.0
            data["end"]       = float(ts[-1]) if len(ts) > 0 else 0.0
            data.pop("timestamps")
            self.topics_index[topic] = data

        self.metadata = {
            "start_time": start_t if start_t != float("inf")  else 0.0,
            "end_time":   end_t   if end_t   != float("-inf") else 0.0,
            "duration":   round(max(0.0, end_t - start_t), 2),
        }

    def _detect_anomalies(self) -> None:
        bag_end = self.metadata.get("end_time", 0.0)
        for topic, info in self.topics_index.items():
            if (bag_end - info["end"]) > 2.0:
                self.events.append({
                    "timestamp": info["end"],
                    "type":      "TOPIC_DIED",
                    "topic":     topic,
                    "severity":  "HIGH",
                    "details":   f"Stopped {round(bag_end - info['end'], 2)}s before bag end.",
                })
            if "scan" in topic and info["frequency"] < 5.0 and info["count"] > 0:
                self.events.append({
                    "timestamp": info["start"],
                    "type":      "LOW_RATE",
                    "topic":     topic,
                    "severity":  "MEDIUM",
                    "details":   f"Rate: {info['frequency']} Hz (expected ≥5 Hz)",
                })
        self.events.sort(key=lambda x: x["timestamp"])

    def _generate_hypotheses(self) -> None:
        if not self.events:
            self.incident_report = {
                "status":     "HEALTHY",
                "hypothesis": "No major anomalies detected.",
                "evidence":   [],
            }
            return

        high_ts = [e["timestamp"] for e in self.events if e["severity"] == "HIGH"]
        if high_ts:
            critical_time  = max(set(high_ts), key=high_ts.count)
            context_events = [e for e in self.events if abs(e["timestamp"] - critical_time) < 1.0]
            topics_died    = [e["topic"] for e in context_events if e["type"] == "TOPIC_DIED"]

            hypothesis = "System Failure"
            if len(topics_died) > 2:
                hypothesis = "Multi-Sensor Loss (Possible USB/Power Bus Failure)"
            elif any("scan" in t for t in topics_died):
                hypothesis = "LiDAR Driver or Connection Failure"

            self.incident_report = {
                "status":     "CRITICAL",
                "timestamp":  critical_time,
                "hypothesis": hypothesis,
                "evidence":   [f"{e['topic']} → {e['type']}" for e in context_events],
            }
        else:
            self.incident_report = {
                "status":     "WARNING",
                "hypothesis": "Degraded performance detected.",
                "evidence":   [],
            }

    def _construct_llm_prompt(self) -> str:
        timeline_str = "\n".join(
            f"  [{e['severity']}] t={round(e['timestamp'], 2)}: {e['topic']} → {e['type']}: {e['details']}"
            for e in self.events
        ) or "  (No anomalous events detected)"

        return (
            f"ACT AS: Expert ROS Diagnostic Engineer.\n\n"
            f"BAG METADATA:\n"
            f"  Duration: {self.metadata.get('duration', 0)}s\n"
            f"  Topics indexed: {len(self.topics_index)}\n\n"
            f"INCIDENT TIMELINE (rule-based detections):\n{timeline_str}\n\n"
            f"AUTOMATED HYPOTHESIS:\n"
            f"  Status: {self.incident_report.get('status', 'UNKNOWN')}\n"
            f"  Hypothesis: {self.incident_report.get('hypothesis', '')}\n"
            f"  Evidence: {', '.join(self.incident_report.get('evidence', []))}\n\n"
            f"USER QUERY:\n"
            f"  Explain what happened. Provide Root Cause, Confidence Level, and Next Steps."
        )

    def _stub_result(self) -> Dict[str, Any]:
        return {
            "metadata":   {"start_time": 0, "end_time": 0, "duration": 0},
            "events":     [],
            "topics":     {},
            "llm_prompt": "rosbags library not available.",
            "summary":    {"status": "UNKNOWN", "hypothesis": "Analysis unavailable.", "evidence": []},
        }
