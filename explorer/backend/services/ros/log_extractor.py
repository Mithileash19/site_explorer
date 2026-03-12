"""
services/ros/log_extractor.py
──────────────────────────────
Extracts /rosout log messages from ROS1 (.bag) and ROS2 (.db3) bag files.
Uses rosbags.highlevel.AnyReader which transparently handles both formats.
"""
from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any, Dict, List

from rosbags.highlevel import AnyReader

ROSOUT_TOPICS = {"/rosout", "/rosout_agg"}

LEVEL_MAP: Dict[int, str] = {
    1:  "DEBUG",
    2:  "INFO",
    4:  "WARN",
    8:  "ERROR",
    16: "FATAL",
}

PRIORITY_ORDER = {"FATAL": 0, "ERROR": 1, "WARN": 2, "INFO": 3, "DEBUG": 4}


class ROSLogExtractor:
    """
    Extracts /rosout messages from ROS1 (.bag) or ROS2 (.db3) bag files.

    extractor = ROSLogExtractor("amr01.bag")
    all_logs  = extractor.extract()
    filtered  = extractor.filter_window(all_logs, incident_ts=..., window=10)
    priority  = extractor.priority_logs(filtered)
    """

    def __init__(self, bag_path: str) -> None:
        self.bag_path = bag_path

    def extract(self) -> List[Dict[str, Any]]:
        """Return all /rosout log entries sorted chronologically."""
        logs: List[Dict[str, Any]] = []
        try:
            with AnyReader([Path(self.bag_path)]) as reader:
                connections = [c for c in reader.connections if c.topic in ROSOUT_TOPICS]
                if not connections:
                    return []
                for connection, timestamp_ns, rawdata in reader.messages(connections):
                    msg = reader.deserialize(rawdata, connection.msgtype)
                    try:
                        sec  = msg.header.stamp.sec
                        nsec = msg.header.stamp.nanosec
                        ts   = float(sec) + nsec * 1e-9
                    except AttributeError:
                        ts = timestamp_ns / 1_000_000_000.0

                    dt = datetime.datetime.utcfromtimestamp(ts)
                    logs.append({
                        "timestamp": ts,
                        "datetime":  dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                        "log_level": LEVEL_MAP.get(getattr(msg, "level", 2), "INFO"),
                        "node_name": getattr(msg, "name", ""),
                        "message":   getattr(msg, "msg",  ""),
                    })
        except Exception as exc:
            return [{
                "timestamp": 0.0,
                "datetime":  "",
                "log_level": "ERROR",
                "node_name": "ros_log_extractor",
                "message":   f"Bag read failed: {exc}",
            }]

        logs.sort(key=lambda x: x["timestamp"])
        return logs

    def filter_window(
        self,
        logs: List[Dict[str, Any]],
        incident_ts: float,
        window: float = 10.0,
    ) -> List[Dict[str, Any]]:
        """Return only entries within ±window seconds of incident_ts."""
        return [log for log in logs if abs(log["timestamp"] - incident_ts) <= window]

    def priority_logs(self, logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Stable-sort: FATAL → ERROR → WARN → INFO → DEBUG."""
        return sorted(logs, key=lambda x: PRIORITY_ORDER.get(x["log_level"], 99))

    def get_timeline_buckets(
        self, logs: List[Dict[str, Any]], n_buckets: int = 300
    ) -> List[Dict[str, Any]]:
        """Split all logs into n_buckets time buckets with per-level counts."""
        if not logs:
            return []
        t_min = logs[0]["timestamp"]
        t_max = logs[-1]["timestamp"]
        if t_max <= t_min:
            t_max = t_min + 1.0
        span = t_max - t_min

        buckets = []
        for i in range(n_buckets):
            b_from = t_min + (i / n_buckets) * span
            b_to   = t_min + ((i + 1) / n_buckets) * span
            entries = [l for l in logs if b_from <= l["timestamp"] < b_to]
            buckets.append({
                "from_ts": round(b_from, 3),
                "to_ts":   round(b_to,   3),
                "total":   len(entries),
                "error":   sum(1 for l in entries if l["log_level"] in ("ERROR", "FATAL")),
                "warn":    sum(1 for l in entries if l["log_level"] == "WARN"),
                "info":    sum(1 for l in entries if l["log_level"] == "INFO"),
                "debug":   sum(1 for l in entries if l["log_level"] == "DEBUG"),
            })
        return buckets
