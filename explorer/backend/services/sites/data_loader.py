"""
services/sites/data_loader.py
──────────────────────────────
SiteDataManager: reads maps, CSV waypoints, and JSON topology graphs.
Source: site_commander/backend/data_loader.py
"""
from __future__ import annotations

import base64
import json
import math
import os
from typing import Any, Dict, List, Optional

from core.logging import get_logger

logger = get_logger(__name__)

try:
    import cv2
    import numpy as np
    import pandas as pd
    import yaml
    _DEPS_OK = True
except ImportError as e:
    logger.warning("SiteDataManager: missing dependency (%s) — site loading limited.", e)
    _DEPS_OK = False


class SiteDataManager:
    """
    Vendor-agnostic site asset loader.
    Scans filesystem to resolve per-site: map image, YAML config, spots/storage CSVs,
    and JSON topology graph. Handles multiple naming conventions and subdirectory layouts.
    """

    def __init__(self, sites_root_path: str) -> None:
        self.root = sites_root_path
        logger.info("SiteDataManager: root=%s", self.root)

    # ── File hunting ───────────────────────────────────────────────────────────

    def _find_file(self, base_path: str, potential_names) -> Optional[str]:
        if isinstance(potential_names, str):
            potential_names = [potential_names]
        subdirs = ["", "config", "config/maps", "config/fixtures", "app/gwm", "gwm", "maps", "data"]
        for name in potential_names:
            for sub in subdirs:
                target = os.path.join(base_path, sub, name)
                if os.path.exists(target):
                    return target
        for root, dirs, files in os.walk(base_path):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for name in potential_names:
                if name in files:
                    return os.path.join(root, name)
        return None

    def _resolve_paths(self, site_id: str) -> Dict[str, Optional[str]]:
        base = os.path.join(self.root, site_id)
        return {
            "map_img":    self._find_file(base, ["map.png", "map.pgm", "map.jpg", "occupancy.png"]),
            "map_config": self._find_file(base, ["map.yaml", "navigation_map.yaml", "config.yaml"]),
            "spots":      self._find_file(base, ["spots.csv", "waypoints.csv", "fixtures.csv", "goals.csv"]),
            "storage":    self._find_file(base, ["storage_locations.csv", "racks.csv", "bins.csv", "rack_mapping.csv"]),
            "graph_json": self._find_file(base, ["maps.json", "graph.json", "site_graph.json", "topology.json"]),
        }

    # ── Data cleaning ──────────────────────────────────────────────────────────

    def _clean_df(self, df):
        try:
            df.replace([float("inf"), float("-inf")], float("nan"), inplace=True)
            return df.where(pd.notnull(df), None)
        except Exception:
            return df

    def _normalize_columns(self, df):
        df.columns = [c.lower().strip() for c in df.columns]
        remap = {}
        for col in df.columns:
            if col in ["pos_x", "coordinate_x", "x_coord", "location_x", "x_pos"]:
                remap[col] = "x"
            if col in ["pos_y", "coordinate_y", "y_coord", "location_y", "y_pos"]:
                remap[col] = "y"
            if col in ["name", "spot_name", "id", "rack_id", "bin_id"]:
                remap[col] = "label"
        return df.rename(columns=remap)

    def _sanitize_structure(self, data: Any) -> Any:
        if isinstance(data, dict):
            return {k: self._sanitize_structure(v) for k, v in data.items()}
        if isinstance(data, list):
            return [self._sanitize_structure(v) for v in data]
        if isinstance(data, float) and (math.isnan(data) or math.isinf(data)):
            return None
        return data

    # ── Public API ─────────────────────────────────────────────────────────────

    def list_sites(self) -> List[str]:
        if not os.path.exists(self.root):
            return []
        return sorted(
            d for d in os.listdir(self.root)
            if os.path.isdir(os.path.join(self.root, d)) and not d.startswith(".")
        )

    def get_config(self, site_id: str) -> Dict[str, Any]:
        paths = self._resolve_paths(site_id)
        res, origin = 0.05, [0, 0, 0]
        if paths["map_config"] and _DEPS_OK:
            try:
                with open(paths["map_config"]) as f:
                    d      = yaml.safe_load(f)
                    res    = d.get("resolution", 0.05)
                    origin = d.get("origin", [0, 0, 0])
            except Exception as e:
                logger.warning("get_config(%s): %s", site_id, e)
        return {"resolution": res, "origin": origin}

    def get_map_image(self, site_id: str, dark_mode: bool = True) -> Optional[Dict[str, Any]]:
        if not _DEPS_OK:
            return None
        paths = self._resolve_paths(site_id)
        if not paths["map_img"]:
            return None
        try:
            img = cv2.imread(paths["map_img"], cv2.IMREAD_GRAYSCALE)
            if img is None:
                return None
            if dark_mode:
                img = cv2.bitwise_not(img)
            h, w = img.shape
            _, buf = cv2.imencode(".png", img)
            b64 = base64.b64encode(buf).decode("utf-8")
            return {"width": w, "height": h, "b64": f"data:image/png;base64,{b64}"}
        except Exception as e:
            logger.error("get_map_image(%s): %s", site_id, e)
            return None

    def get_site_data(self, site_id: str) -> Dict[str, Any]:
        if not _DEPS_OK:
            return {"nodes": [], "edges": [], "spots": [], "storage": []}
        paths = self._resolve_paths(site_id)
        data: Dict[str, Any] = {"nodes": [], "edges": [], "spots": [], "storage": []}

        for key, csv_path_key in [("spots", "spots"), ("storage", "storage")]:
            if paths[csv_path_key]:
                try:
                    df = pd.read_csv(paths[csv_path_key])
                    df = self._normalize_columns(df)
                    if "x" in df.columns:
                        df = self._clean_df(df)
                        data[key] = df.to_dict(orient="records")
                except Exception as e:
                    logger.warning("get_site_data(%s) %s: %s", site_id, key, e)

        if paths["graph_json"]:
            try:
                with open(paths["graph_json"]) as f:
                    content = json.load(f)
                graph = (
                    content.get("maps", [{}])[0]
                    if "maps" in content else
                    content.get("graph", content)
                )
                for n in graph.get("nodes", []):
                    x = y = None
                    if "pos" in n and "coordinates" in n["pos"]:
                        coords = n["pos"]["coordinates"]
                        if len(coords) >= 2:
                            x, y = coords[0], coords[1]
                    elif "x" in n and "y" in n:
                        x, y = n["x"], n["y"]
                    if x is not None and y is not None:
                        try:
                            fx, fy = float(x), float(y)
                            if math.isfinite(fx) and math.isfinite(fy):
                                data["nodes"].append({
                                    "id":    n.get("id"),
                                    "label": n.get("name") or str(n.get("id")),
                                    "x":     fx,
                                    "y":     fy,
                                })
                        except (TypeError, ValueError):
                            pass

                for e in graph.get("edges", []):
                    start = e.get("start_node_id") or e.get("start") or e.get("source")
                    end   = e.get("end_node_id")   or e.get("end")   or e.get("target")
                    if start and end:
                        data["edges"].append({"id": e.get("id"), "from": start, "to": end})
            except Exception as e:
                logger.error("get_site_data(%s) graph: %s", site_id, e)

        return self._sanitize_structure(data)
