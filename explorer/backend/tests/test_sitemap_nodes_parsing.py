"""Tests for node/edge parsing from app/gwm/maps.json."""
from __future__ import annotations

import json
import os
import sys

# Make backend package importable from tests/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.sitemap.service import SiteMapService


def _write_maps_json(tmp_path, site_id: str, payload: dict) -> None:
    maps_file = tmp_path / site_id / "app" / "gwm" / "maps.json"
    maps_file.parent.mkdir(parents=True, exist_ok=True)
    maps_file.write_text(json.dumps(payload), encoding="utf-8")


def test_get_site_data_parses_nodes_edges_and_skips_invalid(tmp_path):
    site_id = "demo"
    _write_maps_json(
        tmp_path,
        site_id,
        {
            "maps": [
                {
                    "name": site_id,
                    "nodes": [
                        {
                            "id": 1,
                            "pos": {"coordinates": [1.0, 2.0]},
                            "parkable": True,
                            "radius": 1.5,
                            "meta_kind": "NODE_KIND_WAYPOINT",
                            "meta_data": {"spin_mode": 0, "spin_turn": 0},
                        },
                        {
                            "id": 1,
                            "pos": {"coordinates": [9.0, 9.0]},
                            "parkable": False,
                            "radius": 2.2,
                        },
                        {
                            "id": 2,
                            "pos": {"coordinates": ["nan", 5.0]},
                            "parkable": False,
                            "radius": 1.0,
                        },
                        {
                            "id": 3,
                            "pos": {"coordinates": [3.0, 4.0]},
                            "parkable": False,
                            "radius": "bad-value",
                        },
                        {"id": 4, "pos": {}},
                    ],
                    "edges": [
                        {"id": 10, "node1": 1, "node2": 3, "directed": False},
                        {"id": 20, "node1": 3, "node2": 1, "directed": False},
                        {"id": 11, "node1": 1, "node2": 3, "directed": True},
                        {"id": 12, "node1": 1, "node2": 3, "directed": True},
                        {"id": "bad", "node1": 3, "node2": 1, "directed": True},
                        {"id": 99, "node1": 1, "node2": 999, "directed": False},
                    ],
                }
            ]
        },
    )

    svc = SiteMapService(str(tmp_path))
    data = svc.get_site_data(site_id)

    assert [n["id"] for n in data["nodes"]] == [1, 3]
    assert data["nodes"][0]["x"] == 1.0
    assert data["nodes"][0]["y"] == 2.0
    assert data["nodes"][1]["radius"] == 0.0

    # Undirected duplicate is removed, directed duplicate is removed, and invalid endpoint is skipped.
    assert len(data["edges"]) == 3
    assert {(e["node1"], e["node2"], e["directed"]) for e in data["edges"]} == {
        (1, 3, False),
        (1, 3, True),
        (3, 1, True),
    }


def test_get_site_data_falls_back_to_first_map_when_name_missing(tmp_path):
    site_id = "demo"
    _write_maps_json(
        tmp_path,
        site_id,
        {
            "maps": [
                {
                    "name": "different-site",
                    "nodes": [
                        {
                            "id": 7,
                            "pos": {"coordinates": [7.5, -3.25]},
                            "parkable": True,
                            "radius": 0.4,
                        }
                    ],
                    "edges": [],
                }
            ]
        },
    )

    svc = SiteMapService(str(tmp_path))
    data = svc.get_site_data(site_id)

    assert [n["id"] for n in data["nodes"]] == [7]
    assert data["nodes"][0]["x"] == 7.5
    assert data["nodes"][0]["y"] == -3.25
