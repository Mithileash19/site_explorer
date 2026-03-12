"""
Tests for AR marker loading and API endpoint.

Covers:
  - Unit tests for SiteMapService.get_markers() with mocked _read_bytes
  - Integration tests against real markers.yaml files in sootballs_sites/
  - API smoke test for GET /api/v1/sitemap/{site_id}/markers
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, List
from unittest.mock import patch

import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.sitemap.service import SiteMapService

# ── Constants ──────────────────────────────────────────────────────────────────

SITES_ROOT = Path(__file__).parent.parent.parent / "sootballs_sites" / "sites"

# ── Helpers ────────────────────────────────────────────────────────────────────


def _svc(yaml_content: str | None) -> SiteMapService:
    """Return a SiteMapService whose _read_bytes is mocked to return *yaml_content*."""
    svc = SiteMapService("/fake/root")
    raw = yaml_content.encode() if isinstance(yaml_content, str) else yaml_content
    svc._read_bytes = lambda *_: raw  # type: ignore[method-assign]
    return svc


def _real_svc() -> SiteMapService:
    """Return a SiteMapService reading directly from the real sites directory."""
    return SiteMapService(str(SITES_ROOT))


def _raw_markers(site_id: str) -> Dict[int, Any]:
    """Parse markers.yaml directly (ground truth) and return {id: pose} dict."""
    yaml_path = SITES_ROOT / site_id / "config" / "param" / "markers.yaml"
    if not yaml_path.exists():
        return {}
    content = yaml.safe_load(yaml_path.read_bytes())
    raw = content.get("markers", {}) if isinstance(content, dict) else {}
    return {int(k): v for k, v in raw.items()} if isinstance(raw, dict) else {}


# ── Unit Tests ─────────────────────────────────────────────────────────────────


class TestGetMarkersUnit:

    # ── happy-path basics ──────────────────────────────────────────────────────

    def test_returns_correct_count(self):
        yaml_src = dedent("""\
            markers:
              10:
                position: [1.0, 2.0, 0.0]
                orientation: [0.0, 0.0, 0.0]
              20:
                position: [3.0, 4.0, 0.5]
                orientation: [0.0, 0.0, 90.0]
        """)
        result = _svc(yaml_src).get_markers("any_site")
        assert len(result["markers"]) == 2

    def test_response_has_markers_key(self):
        yaml_src = dedent("""\
            markers:
              1:
                position: [0.0, 0.0, 0.0]
                orientation: [0.0, 0.0, 0.0]
        """)
        result = _svc(yaml_src).get_markers("any_site")
        assert "markers" in result
        assert isinstance(result["markers"], list)

    def test_marker_fields_present(self):
        yaml_src = dedent("""\
            markers:
              7:
                position: [1.5, 2.5, 0.1]
                orientation: [0.0, 0.0, 45.0]
        """)
        marker = _svc(yaml_src).get_markers("any_site")["markers"][0]
        for field in ("id", "x", "y", "z", "yaw"):
            assert field in marker, f"field '{field}' missing"

    def test_marker_id_is_integer(self):
        yaml_src = dedent("""\
            markers:
              42:
                position: [0.0, 0.0, 0.0]
                orientation: [0.0, 0.0, 0.0]
        """)
        marker = _svc(yaml_src).get_markers("any_site")["markers"][0]
        assert marker["id"] == 42
        assert isinstance(marker["id"], int)

    def test_position_extracted_correctly(self):
        yaml_src = dedent("""\
            markers:
              5:
                position: [12.50, -3.75, 0.80]
                orientation: [0.0, 0.0, 0.0]
        """)
        m = _svc(yaml_src).get_markers("any_site")["markers"][0]
        assert m["x"] == pytest.approx(12.50)
        assert m["y"] == pytest.approx(-3.75)
        assert m["z"] == pytest.approx(0.80)

    # ── yaw conversion ─────────────────────────────────────────────────────────

    def test_zero_yaw_stays_zero(self):
        yaml_src = dedent("""\
            markers:
              1:
                position: [0.0, 0.0, 0.0]
                orientation: [0.0, 0.0, 0.0]
        """)
        m = _svc(yaml_src).get_markers("any_site")["markers"][0]
        assert m["yaw"] == pytest.approx(0.0)

    def test_90_degree_yaw_converted_to_radians(self):
        yaml_src = dedent("""\
            markers:
              1:
                position: [0.0, 0.0, 0.0]
                orientation: [0.0, 0.0, 90.0]
        """)
        m = _svc(yaml_src).get_markers("any_site")["markers"][0]
        assert m["yaw"] == pytest.approx(math.pi / 2)

    def test_180_degree_yaw_converted_to_radians(self):
        yaml_src = dedent("""\
            markers:
              1:
                position: [0.0, 0.0, 0.0]
                orientation: [0.0, 0.0, 180.0]
        """)
        m = _svc(yaml_src).get_markers("any_site")["markers"][0]
        assert m["yaw"] == pytest.approx(math.pi)

    def test_270_degree_yaw_converted_to_radians(self):
        yaml_src = dedent("""\
            markers:
              1:
                position: [0.0, 0.0, 0.0]
                orientation: [0.0, 0.0, 270.0]
        """)
        m = _svc(yaml_src).get_markers("any_site")["markers"][0]
        assert m["yaw"] == pytest.approx(3 * math.pi / 2)

    def test_negative_yaw_converted_correctly(self):
        yaml_src = dedent("""\
            markers:
              1:
                position: [0.0, 0.0, 0.0]
                orientation: [0.0, 0.0, -90.0]
        """)
        m = _svc(yaml_src).get_markers("any_site")["markers"][0]
        assert m["yaw"] == pytest.approx(-math.pi / 2)

    def test_yaw_uses_index_2_not_index_0(self):
        """Roll (index 0) must NOT be used as yaw."""
        yaml_src = dedent("""\
            markers:
              1:
                position: [0.0, 0.0, 0.0]
                orientation: [999.0, 0.0, 45.0]
        """)
        m = _svc(yaml_src).get_markers("any_site")["markers"][0]
        assert m["yaw"] == pytest.approx(math.radians(45.0))

    def test_full_rpy_orientation_only_yaw_extracted(self):
        """askhid001 marker 54 has orientation [90.0, 0.0, 270.0]."""
        yaml_src = dedent("""\
            markers:
              54:
                position: [-0.1, 10.47, 0.81]
                orientation: [90.0, 0.0, 270.0]
        """)
        m = _svc(yaml_src).get_markers("any_site")["markers"][0]
        assert m["yaw"] == pytest.approx(math.radians(270.0))

    # ── edge / error cases ─────────────────────────────────────────────────────

    def test_missing_yaml_returns_empty(self):
        result = _svc(None).get_markers("any_site")
        assert result == {"markers": []}

    def test_empty_bytes_returns_empty(self):
        result = _svc("").get_markers("any_site")
        assert result == {"markers": []}

    def test_yaml_without_markers_key_returns_empty(self):
        yaml_src = "configs:\n  use_ar_tags: true\n"
        result = _svc(yaml_src).get_markers("any_site")
        assert result == {"markers": []}

    def test_markers_as_list_not_dict_returns_empty(self):
        yaml_src = "markers:\n  - 1\n  - 2\n"
        result = _svc(yaml_src).get_markers("any_site")
        assert result == {"markers": []}

    def test_pose_missing_position_key_defaults_to_origin(self):
        """A marker with no position key is kept and defaults to (0, 0, 0)."""
        yaml_src = dedent("""\
            markers:
              1:
                orientation: [0.0, 0.0, 0.0]
              2:
                position: [5.0, 6.0, 0.0]
                orientation: [0.0, 0.0, 0.0]
        """)
        result = _svc(yaml_src).get_markers("any_site")
        ids = [m["id"] for m in result["markers"]]
        # Both markers returned; marker 1 defaults to origin
        assert 1 in ids
        assert 2 in ids
        m1 = next(m for m in result["markers"] if m["id"] == 1)
        assert m1["x"] == pytest.approx(0.0)
        assert m1["y"] == pytest.approx(0.0)
        assert m1["z"] == pytest.approx(0.0)

    def test_orientation_missing_defaults_yaw_to_zero(self):
        yaml_src = dedent("""\
            markers:
              3:
                position: [1.0, 1.0, 0.0]
        """)
        m = _svc(yaml_src).get_markers("any_site")["markers"][0]
        assert m["yaw"] == pytest.approx(0.0)

    def test_partial_orientation_defaults_yaw_to_zero(self):
        """1-element orientation list: no index 2 → yaw=0."""
        yaml_src = dedent("""\
            markers:
              3:
                position: [1.0, 1.0, 0.0]
                orientation: [90.0]
        """)
        m = _svc(yaml_src).get_markers("any_site")["markers"][0]
        assert m["yaw"] == pytest.approx(0.0)

    def test_multiple_markers_all_parsed(self):
        yaml_src = dedent("""\
            markers:
              0:
                position: [19.65, 21.61, 0.0]
                orientation: [0.0, 0.0, 0.0]
              1:
                position: [19.43, 32.48, 0.0]
                orientation: [0.0, 0.0, 0.0]
              2:
                position: [18.09, 39.99, 0.0]
                orientation: [0.0, 0.0, 0.0]
        """)
        result = _svc(yaml_src).get_markers("any_site")
        ids = {m["id"] for m in result["markers"]}
        assert ids == {0, 1, 2}

    def test_configs_section_does_not_interfere(self):
        """Real yamls have a 'configs' section before 'markers'."""
        yaml_src = dedent("""\
            configs:
              use_ar_tags: true
              marker_length: 0.176
            markers:
              5:
                position: [1.0, 2.0, 0.0]
                orientation: [0.0, 0.0, 0.0]
        """)
        result = _svc(yaml_src).get_markers("any_site")
        assert len(result["markers"]) == 1
        assert result["markers"][0]["id"] == 5

    def test_non_integer_marker_id_converted(self):
        """YAML keys are integers; conversion to int must succeed."""
        yaml_src = dedent("""\
            markers:
              100:
                position: [0.0, 0.0, 0.0]
                orientation: [0.0, 0.0, 0.0]
        """)
        m = _svc(yaml_src).get_markers("any_site")["markers"][0]
        assert isinstance(m["id"], int)
        assert m["id"] == 100

    def test_negative_coordinates_preserved(self):
        """seniru002 uses negative world coordinates."""
        yaml_src = dedent("""\
            markers:
              46:
                position: [-4.45, -28.65, 0.0]
                orientation: [0.0, 0.0, 0.0]
        """)
        m = _svc(yaml_src).get_markers("any_site")["markers"][0]
        assert m["x"] == pytest.approx(-4.45)
        assert m["y"] == pytest.approx(-28.65)


# ── Integration Tests against real files ──────────────────────────────────────


def _all_real_sites_with_markers():
    """Collect (site_id, yaml_path) pairs for all real sites."""
    if not SITES_ROOT.exists():
        return []
    pairs = []
    for site_dir in sorted(SITES_ROOT.iterdir()):
        if not site_dir.is_dir():
            continue
        yaml_path = site_dir / "config" / "param" / "markers.yaml"
        if yaml_path.exists():
            pairs.append(site_dir.name)
    return pairs


REAL_SITES = _all_real_sites_with_markers()


@pytest.mark.skipif(not SITES_ROOT.exists(), reason="sootballs_sites not present")
class TestGetMarkersIntegration:

    def test_all_sites_return_marker_list(self):
        """Every site with a markers.yaml must return a non-empty list."""
        svc = _real_svc()
        failures = []
        for site_id in REAL_SITES:
            result = svc.get_markers(site_id)
            if not isinstance(result.get("markers"), list) or len(result["markers"]) == 0:
                failures.append(f"{site_id}: empty or missing markers list")
        assert not failures, "\n".join(failures)

    def test_all_marker_ids_are_non_negative_integers(self):
        svc = _real_svc()
        failures = []
        for site_id in REAL_SITES:
            for m in svc.get_markers(site_id)["markers"]:
                if not isinstance(m["id"], int) or m["id"] < 0:
                    failures.append(f"{site_id}: id={m['id']!r}")
        assert not failures, "\n".join(failures)

    def test_all_positions_are_finite(self):
        svc = _real_svc()
        failures = []
        for site_id in REAL_SITES:
            for m in svc.get_markers(site_id)["markers"]:
                for field in ("x", "y", "z"):
                    v = m[field]
                    if not isinstance(v, float) or not math.isfinite(v):
                        failures.append(f"{site_id} id={m['id']}: {field}={v!r} not finite")
        assert not failures, "\n".join(failures)

    def test_all_yaws_are_finite_radians(self):
        svc = _real_svc()
        failures = []
        for site_id in REAL_SITES:
            for m in svc.get_markers(site_id)["markers"]:
                yaw = m["yaw"]
                if not isinstance(yaw, float) or not math.isfinite(yaw):
                    failures.append(f"{site_id} id={m['id']}: yaw={yaw!r} not finite")
        assert not failures, "\n".join(failures)

    def test_no_duplicate_ids_within_site(self):
        svc = _real_svc()
        failures = []
        for site_id in REAL_SITES:
            ids = [m["id"] for m in svc.get_markers(site_id)["markers"]]
            dups = {i for i in ids if ids.count(i) > 1}
            if dups:
                failures.append(f"{site_id}: duplicate IDs {sorted(dups)}")
        assert not failures, "\n".join(failures)

    def test_marker_count_matches_yaml(self):
        """Parsed count must equal the number of entries in the raw YAML dict."""
        svc = _real_svc()
        failures = []
        for site_id in REAL_SITES:
            expected = len(_raw_markers(site_id))
            actual = len(svc.get_markers(site_id)["markers"])
            if actual != expected:
                failures.append(f"{site_id}: expected {expected}, got {actual}")
        assert not failures, "\n".join(failures)

    def test_marker_ids_match_yaml(self):
        """IDs returned by the service must exactly match the YAML keys."""
        svc = _real_svc()
        failures = []
        for site_id in REAL_SITES:
            expected_ids = set(_raw_markers(site_id).keys())
            actual_ids = {m["id"] for m in svc.get_markers(site_id)["markers"]}
            if expected_ids != actual_ids:
                extra = actual_ids - expected_ids
                missing = expected_ids - actual_ids
                failures.append(
                    f"{site_id}: extra={extra or '-'} missing={missing or '-'}"
                )
        assert not failures, "\n".join(failures)

    def test_marker_positions_match_yaml(self):
        """x, y, z from the service must equal the raw YAML position values."""
        svc = _real_svc()
        failures = []
        for site_id in REAL_SITES:
            raw = _raw_markers(site_id)
            by_id = {m["id"]: m for m in svc.get_markers(site_id)["markers"]}
            for mid, pose in raw.items():
                m = by_id.get(mid)
                if m is None:
                    continue
                pos = pose.get("position", [])
                for idx, field in enumerate(("x", "y", "z")):
                    expected = float(pos[idx]) if len(pos) > idx else 0.0
                    if not math.isclose(m[field], expected, abs_tol=1e-9):
                        failures.append(
                            f"{site_id} id={mid}: {field} expected {expected} got {m[field]}"
                        )
        assert not failures, "\n".join(failures)

    def test_marker_yaw_degrees_to_radians(self):
        """yaw in the service response must equal math.radians(orientation[2])."""
        svc = _real_svc()
        failures = []
        for site_id in REAL_SITES:
            raw = _raw_markers(site_id)
            by_id = {m["id"]: m for m in svc.get_markers(site_id)["markers"]}
            for mid, pose in raw.items():
                m = by_id.get(mid)
                if m is None:
                    continue
                ori = pose.get("orientation", [])
                yaw_deg = float(ori[2]) if len(ori) > 2 else 0.0
                expected_rad = math.radians(yaw_deg)
                if not math.isclose(m["yaw"], expected_rad, abs_tol=1e-9):
                    failures.append(
                        f"{site_id} id={mid}: expected yaw={expected_rad:.6f} rad "
                        f"({yaw_deg}°) got {m['yaw']:.6f}"
                    )
        assert not failures, "\n".join(failures)

    @pytest.mark.parametrize("site_id", [
        "asksta001",
        "seniru002",
        "askhid001",
        "askhid002",
    ])
    def test_known_sites_spot_check(self, site_id: str):
        """Spot-check a handful of well-known sites by name."""
        if not (SITES_ROOT / site_id).exists():
            pytest.skip(f"{site_id} not present on disk")
        svc = _real_svc()
        result = svc.get_markers(site_id)
        assert len(result["markers"]) > 0, f"{site_id} returned no markers"

    def test_asksta001_specific_markers(self):
        """asksta001: verify three known markers match their YAML entries exactly."""
        site_id = "asksta001"
        if not (SITES_ROOT / site_id).exists():
            pytest.skip(f"{site_id} not on disk")

        svc = _real_svc()
        by_id = {m["id"]: m for m in svc.get_markers(site_id)["markers"]}

        # marker 0: position [19.65, 21.61, 0.0], orientation [0.0, 0.0, 0.0]
        m0 = by_id[0]
        assert m0["x"] == pytest.approx(19.65)
        assert m0["y"] == pytest.approx(21.61)
        assert m0["z"] == pytest.approx(0.0)
        assert m0["yaw"] == pytest.approx(0.0)

        # marker 5: position [5.87, 33.66, 0.0]
        m5 = by_id[5]
        assert m5["x"] == pytest.approx(5.87)
        assert m5["y"] == pytest.approx(33.66)

        # marker 48: position [14.24, 44.09, 0.0]
        m48 = by_id[48]
        assert m48["x"] == pytest.approx(14.24)
        assert m48["y"] == pytest.approx(44.09)

    def test_askhid001_270_degree_yaw(self):
        """askhid001 marker 54 has yaw=270° → π*3/2 rad."""
        site_id = "askhid001"
        if not (SITES_ROOT / site_id).exists():
            pytest.skip(f"{site_id} not on disk")

        svc = _real_svc()
        by_id = {m["id"]: m for m in svc.get_markers(site_id)["markers"]}
        m54 = by_id[54]
        assert m54["yaw"] == pytest.approx(math.radians(270.0))

    def test_site_without_markers_yaml_returns_empty(self):
        """If we point at a site with no markers.yaml, we get empty list."""
        svc = SiteMapService("/tmp")  # no markers.yaml there
        result = svc.get_markers("nosuchsite")
        assert result == {"markers": []}


# ── API Tests ─────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not SITES_ROOT.exists(), reason="sootballs_sites not present")
class TestMarkersAPIEndpoint:

    @pytest.fixture(autouse=True)
    def _client(self):
        from fastapi.testclient import TestClient
        from app.main import app
        self.client = TestClient(app)

    def test_markers_endpoint_returns_200(self):
        response = self.client.get("/api/v1/sitemap/asksta001/markers")
        assert response.status_code == 200

    def test_markers_endpoint_response_schema(self):
        response = self.client.get("/api/v1/sitemap/asksta001/markers")
        body = response.json()
        assert "markers" in body
        assert isinstance(body["markers"], list)
        assert len(body["markers"]) > 0

    def test_markers_endpoint_marker_fields(self):
        response = self.client.get("/api/v1/sitemap/asksta001/markers")
        for m in response.json()["markers"]:
            for field in ("id", "x", "y", "z", "yaw"):
                assert field in m, f"field '{field}' missing from marker {m}"

    def test_markers_endpoint_unknown_site_returns_empty(self):
        response = self.client.get("/api/v1/sitemap/nosuchsite_xyz/markers")
        # Should still be 200 with empty list (graceful fallback)
        assert response.status_code == 200
        body = response.json()
        assert body["markers"] == []

    def test_markers_endpoint_count_matches_yaml(self):
        site_id = "asksta001"
        response = self.client.get(f"/api/v1/sitemap/{site_id}/markers")
        expected = len(_raw_markers(site_id))
        actual = len(response.json()["markers"])
        assert actual == expected
