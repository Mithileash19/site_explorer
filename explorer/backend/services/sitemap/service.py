"""
services/sitemap/service.py
────────────────────────────
SiteMapService — reads sootballs_sites/sites/ data for interactive map visualization.
Parses: navigation_map.yaml, map.png, spots.csv, rack_mapping.csv, regions.csv,
    robots.json, and app/gwm/maps.json (nav nodes + edges).

When a GitRepoManager is supplied all file reads go through ``git show``
(no git checkout), with pure-filesystem fallback when git is unavailable.
"""
from __future__ import annotations

import base64
import csv
import io
import json
import math
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from core.logging import get_logger

if TYPE_CHECKING:
    from services.sitemap.git_manager import GitRepoManager

logger = get_logger(__name__)

try:
    import cv2
    _CV2_OK = True
except ImportError:
    _CV2_OK = False
    logger.warning("SiteMapService: cv2 not available — map images disabled.")

try:
    import yaml
    _YAML_OK = True
except ImportError:
    _YAML_OK = False

# ── Colour palettes ────────────────────────────────────────────────────────────
SPOT_COLORS: Dict[str, str] = {
    "action_spot":         "#3b82f6",
    "waiting_spot":        "#a855f7",
    "charging_spot":       "#22c55e",
    "loading_spot":        "#eab308",
    "unloading_spot":      "#f97316",
    "exception_spot":      "#ef4444",
    "transport_spot":      "#06b6d4",
    "idle_spot":           "#64748b",
}

REGION_COLORS: Dict[str, str] = {
    "loading":             "rgba(234,179,8,0.18)",
    "unloading":           "rgba(34,197,94,0.18)",
    "exception_unloading": "rgba(239,68,68,0.18)",
    "idle":                "rgba(100,116,139,0.20)",
    "charging":            "rgba(34,197,94,0.20)",
    "aisle":               "rgba(59,130,246,0.08)",
    "replenishment":       "rgba(251,146,60,0.18)",
}


class SiteMapService:
    """
    Reads per-site configuration from the sootballs_sites repository.
    When *git_manager* is provided, all file reads are routed through
    ``git show`` so the correct branch is always used regardless of
    the current working-tree checkout.
    """

    def __init__(
        self,
        sites_root: str,
        git_manager: Optional["GitRepoManager"] = None,
    ) -> None:
        self.root = Path(sites_root).resolve()
        self._git = git_manager
        logger.info(
            "SiteMapService: root=%s git=%s",
            self.root,
            "enabled" if git_manager else "disabled",
        )

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _site_dir(self, site_id: str) -> Path:
        return self.root / site_id

    def _find(self, site_id: str, *relative_paths: str) -> Optional[Path]:
        """Filesystem-only path search (used in git-disabled mode)."""
        base = self._site_dir(site_id)
        for rp in relative_paths:
            p = base / rp
            if p.exists():
                return p
        return None

    _LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"

    def _read_bytes(self, site_id: str, *relative_paths: str) -> Optional[bytes]:
        """
        Return the raw bytes of the first matching file, preferring git
        branch reads when a GitRepoManager is configured.
        Falls back to the filesystem when git is unavailable, returns
        nothing, or returns a Git LFS pointer (unresolved binary).
        """
        if self._git is not None:
            for rp in relative_paths:
                data = self._git.read_file_for_site(site_id, rp)
                if data is not None and not data.startswith(self._LFS_POINTER_PREFIX):
                    return data
        # Filesystem fallback (used when git is disabled, git returned nothing,
        # or git returned an unresolved LFS pointer)
        p = self._find(site_id, *relative_paths)
        return p.read_bytes() if p else None

    def _parse_geom(self, geom_str: str) -> List[Tuple[float, float]]:
        """Parse 'x1 y1|x2 y2|...' into [[x1,y1],[x2,y2],...]"""
        points: List[Tuple[float, float]] = []
        for part in geom_str.strip().split("|"):
            p = part.strip()
            if not p:
                continue
            xy = p.split()
            if len(xy) >= 2:
                try:
                    points.append((float(xy[0]), float(xy[1])))
                except ValueError:
                    pass
        return points

    # ── Public API ─────────────────────────────────────────────────────────────

    def list_sites(self) -> List[Dict[str, str]]:
        # Prefer git ls-tree when a manager is available
        if self._git is not None:
            names = self._git.list_sites_from_git()
            if names:
                return [{"id": n, "name": n} for n in names]
        # Filesystem fallback
        if not self.root.exists():
            logger.warning("SiteMapService: sites root does not exist: %s", self.root)
            return []
        return [
            {"id": d.name, "name": d.name}
            for d in sorted(self.root.iterdir())
            if d.is_dir() and not d.name.startswith(".")
        ]

    def get_map_meta(self, site_id: str) -> Dict[str, Any]:
        """Returns resolution, origin, width, height for the site map."""
        import numpy as np

        res: float = 0.05
        origin: List[float] = [0.0, 0.0, 0.0]
        w = h = 0

        yaml_data = self._read_bytes(
            site_id,
            "config/maps/navigation_map.yaml",
            "config/maps/localization_map.yaml",
            "config/simulation/navigation_map.yaml",
        )
        if yaml_data and _YAML_OK:
            try:
                d = yaml.safe_load(io.BytesIO(yaml_data))
                res = float(d.get("resolution", 0.05))
                origin = [float(v) for v in d.get("origin", [0.0, 0.0, 0.0])]
            except Exception as e:
                logger.warning("get_map_meta(%s) yaml: %s", site_id, e)

        img_data = self._read_bytes(
            site_id,
            "config/maps/map.png",
            "config/simulation/map.png",
        )
        if img_data and _CV2_OK:
            try:
                arr = np.frombuffer(img_data, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    h, w = img.shape
            except Exception as e:
                logger.warning("get_map_meta(%s) img: %s", site_id, e)

        return {"resolution": res, "origin": origin, "width": w, "height": h}

    def get_map_image(self, site_id: str, dark_mode: bool = True) -> Optional[Dict[str, Any]]:
        import numpy as np

        if not _CV2_OK:
            return None
        img_data = self._read_bytes(
            site_id,
            "config/maps/map.png",
            "config/simulation/map.png",
        )
        if not img_data:
            return None
        try:
            arr = np.frombuffer(img_data, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
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

    def get_native_map_size(self, site_id: str) -> Optional[Tuple[int, int]]:
        """
        Return native map image size as ``(width, height)`` in pixels.

        This is the pre-render source size used to derive map resolution from
        map metadata. It is used by the API layer to compute an effective
        resolution if a served image is ever rescaled.
        """
        import numpy as np

        if not _CV2_OK:
            return None

        img_data = self._read_bytes(
            site_id,
            "config/maps/map.png",
            "config/simulation/map.png",
        )
        if not img_data:
            return None

        try:
            arr = np.frombuffer(img_data, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return None
            h, w = img.shape
            return (w, h)
        except Exception as e:
            logger.warning("get_native_map_size(%s): %s", site_id, e)
            return None

    def get_site_data(self, site_id: str) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "spots":   [],
            "racks":   [],
            "regions": [],
            "robots":  [],
            "nodes":   [],
            "edges":   [],
        }

        # ── Spots ──────────────────────────────────────────────────────────────
        spots_data = self._read_bytes(site_id, "config/fixtures/spots.csv")
        if spots_data:
            try:
                f = io.StringIO(spots_data.decode("utf-8", errors="replace"))
                for row in csv.DictReader(f):
                    try:
                        stype = row.get("type", "action_spot").strip()
                        data["spots"].append({
                            "name":  row.get("name", "").strip(),
                            "type":  stype,
                            "x":     float(row.get("x", 0)),
                            "y":     float(row.get("y", 0)),
                            "yaw":   float(row.get("yaw", 0) or 0),
                            "robot": row.get("robot", "").strip(),
                            "color": SPOT_COLORS.get(stype, "#64748b"),
                        })
                    except (ValueError, KeyError):
                        pass
            except Exception as e:
                logger.warning("get_site_data(%s) spots: %s", site_id, e)

        # ── Racks ───────────────────────────────────────────────────────────────
        racks_data = self._read_bytes(site_id, "config/fixtures/rack_mapping.csv")
        if racks_data:
            try:
                f = io.StringIO(racks_data.decode("utf-8", errors="replace"))
                for row in csv.DictReader(f):
                    try:
                        sec = row.get("section", "").strip().strip('"')
                        rid = row.get("row", "").strip().strip('"')
                        x   = float(row.get("x", 0))
                        y   = float(row.get("y", 0))
                        orientation = float(row.get("orientation", 0) or 0)
                        data["racks"].append({
                            "section":     sec,
                            "row":         rid,
                            "label":       f"{sec}-{rid}",
                            "x":           x,
                            "y":           y,
                            "orientation": orientation,
                            "direction":   row.get("direction", "").strip(),
                        })
                    except (ValueError, KeyError):
                        pass
            except Exception as e:
                logger.warning("get_site_data(%s) racks: %s", site_id, e)

        # ── Regions ─────────────────────────────────────────────────────────────
        regions_data = self._read_bytes(site_id, "config/fixtures/regions.csv")
        if regions_data:
            try:
                f = io.StringIO(regions_data.decode("utf-8", errors="replace"))
                for row in csv.DictReader(f):
                    rtype    = row.get("type", "").strip()
                    geom_str = row.get("geom", "")
                    polygon  = self._parse_geom(geom_str) if geom_str else []
                    data["regions"].append({
                        "type":    rtype,
                        "id":      row.get("typed_id", "").strip(),
                        "name":    (row.get("name") or f"{rtype}-{row.get('typed_id','')}").strip(),
                        "polygon": [list(p) for p in polygon],
                        "color":   REGION_COLORS.get(rtype, "rgba(100,100,100,0.12)"),
                    })
            except Exception as e:
                logger.warning("get_site_data(%s) regions: %s", site_id, e)

        # ── Robots ──────────────────────────────────────────────────────────────
        robots_data = self._read_bytes(site_id, "app/gwm/robots.json")
        if robots_data:
            try:
                content = json.loads(robots_data.decode("utf-8", errors="replace"))
                agents = content.get("agents", []) if isinstance(content, dict) else []
                data["robots"] = [
                    {"id": int(a.get("robot_id", a.get("id", 0))), "name": a.get("name", "")}
                    for a in agents
                ]
            except Exception as e:
                logger.warning("get_site_data(%s) robots: %s", site_id, e)

        # ── Graph nodes + edges ───────────────────────────────────────────────
        maps_data = self._read_bytes(site_id, "app/gwm/maps.json")
        if maps_data:
            try:
                content = json.loads(maps_data.decode("utf-8", errors="replace"))
                maps = content.get("maps", []) if isinstance(content, dict) else []
                map_entry = next(
                    (
                        m for m in maps
                        if isinstance(m, dict) and str(m.get("name", "")).strip() == site_id
                    ),
                    None,
                )
                if map_entry is None:
                    map_entry = next((m for m in maps if isinstance(m, dict)), None)

                if isinstance(map_entry, dict):
                    # Build node table first so edges can be validated.
                    node_by_id: Dict[int, Dict[str, Any]] = {}
                    raw_nodes = map_entry.get("nodes", [])
                    if isinstance(raw_nodes, list):
                        for raw_node in raw_nodes:
                            if not isinstance(raw_node, dict):
                                continue
                            try:
                                node_id = int(raw_node.get("id"))
                            except (TypeError, ValueError):
                                continue
                            if node_id in node_by_id:
                                # Keep first occurrence for deterministic behavior.
                                continue

                            pos = raw_node.get("pos", {})
                            coords = pos.get("coordinates", []) if isinstance(pos, dict) else []
                            if not isinstance(coords, list) or len(coords) < 2:
                                continue

                            try:
                                x = float(coords[0])
                                y = float(coords[1])
                            except (TypeError, ValueError):
                                continue

                            if not (math.isfinite(x) and math.isfinite(y)):
                                continue

                            radius_raw = raw_node.get("radius", 0.0)
                            try:
                                radius = float(radius_raw)
                            except (TypeError, ValueError):
                                radius = 0.0
                            if not math.isfinite(radius):
                                radius = 0.0

                            meta_data = raw_node.get("meta_data", {})
                            node_by_id[node_id] = {
                                "id": node_id,
                                "x": x,
                                "y": y,
                                "parkable": bool(raw_node.get("parkable", False)),
                                "radius": radius,
                                "meta_kind": str(raw_node.get("meta_kind", "")),
                                "spin_mode": meta_data.get("spin_mode") if isinstance(meta_data, dict) else None,
                                "spin_turn": meta_data.get("spin_turn") if isinstance(meta_data, dict) else None,
                            }

                    data["nodes"] = sorted(node_by_id.values(), key=lambda n: n["id"])

                    raw_edges = map_entry.get("edges", [])
                    if isinstance(raw_edges, list) and node_by_id:
                        dedup: set[tuple[int, int, bool]] = set()
                        edges: List[Dict[str, Any]] = []
                        for idx, raw_edge in enumerate(raw_edges):
                            if not isinstance(raw_edge, dict):
                                continue
                            try:
                                n1 = int(raw_edge.get("node1"))
                                n2 = int(raw_edge.get("node2"))
                            except (TypeError, ValueError):
                                continue
                            if n1 not in node_by_id or n2 not in node_by_id:
                                continue

                            directed = bool(raw_edge.get("directed", False))
                            key = (n1, n2, directed) if directed else (min(n1, n2), max(n1, n2), directed)
                            if key in dedup:
                                continue
                            dedup.add(key)

                            edge_id = raw_edge.get("id")
                            try:
                                edge_id = int(edge_id)
                            except (TypeError, ValueError):
                                edge_id = idx + 1

                            edges.append({
                                "id": edge_id,
                                "node1": n1,
                                "node2": n2,
                                "directed": directed,
                                "speed_scale_estimate": str(raw_edge.get("speed_scale_estimate", "1")),
                            })

                        data["edges"] = edges
            except Exception as e:
                logger.warning("get_site_data(%s) maps.json: %s", site_id, e)

        return data

    def get_map_bounds(self, site_id: str) -> Optional[Dict[str, float]]:
        """Return world-coordinate bounding box of the site map."""
        meta = self.get_map_meta(site_id)
        if not meta["width"]:
            return None
        ox, oy = meta["origin"][0], meta["origin"][1]
        res = meta["resolution"]
        return {
            "x_min": ox,
            "y_min": oy,
            "x_max": ox + meta["width"] * res,
            "y_max": oy + meta["height"] * res,
        }

    def get_markers(self, site_id: str) -> Dict[str, Any]:
        """
        Parse config/param/markers.yaml and return AR marker poses.

        Each marker position is in world coordinates (metres, ROS map frame).
        Orientation is [roll, pitch, yaw] in degrees — the yaw component is
        the bearing the marker faces; we expose it in radians for the canvas.

        Returns::

            {
              "markers": [
                {"id": 46, "x": -4.45, "y": -28.65, "z": 0.0, "yaw": 0.0},
                ...
              ]
            }
        """
        if not _YAML_OK:
            logger.warning("get_markers(%s): pyyaml not available", site_id)
            return {"markers": []}

        yaml_data = self._read_bytes(site_id, "config/param/markers.yaml")
        if not yaml_data:
            logger.info("get_markers(%s): markers.yaml not found", site_id)
            return {"markers": []}

        try:
            content = yaml.safe_load(io.BytesIO(yaml_data))
            raw_markers = content.get("markers") if isinstance(content, dict) else None
            if not isinstance(raw_markers, dict):
                return {"markers": []}

            import math as _math
            result: List[Dict[str, Any]] = []
            for marker_id, pose in raw_markers.items():
                if not isinstance(pose, dict):
                    continue
                pos = pose.get("position", [0.0, 0.0, 0.0])
                ori = pose.get("orientation", [0.0, 0.0, 0.0])
                try:
                    x    = float(pos[0]) if len(pos) > 0 else 0.0
                    y    = float(pos[1]) if len(pos) > 1 else 0.0
                    z    = float(pos[2]) if len(pos) > 2 else 0.0
                    # orientation[2] = yaw in degrees
                    yaw_deg = float(ori[2]) if len(ori) > 2 else 0.0
                    yaw_rad = _math.radians(yaw_deg)
                    result.append({
                        "id":  int(marker_id),
                        "x":   x,
                        "y":   y,
                        "z":   z,
                        "yaw": yaw_rad,
                    })
                except (TypeError, ValueError, IndexError):
                    pass

            logger.info("get_markers(%s): %d markers loaded", site_id, len(result))
            return {"markers": result}
        except Exception as e:
            logger.error("get_markers(%s): %s", site_id, e)
            return {"markers": []}
