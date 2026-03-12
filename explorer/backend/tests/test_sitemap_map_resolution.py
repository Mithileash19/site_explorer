"""Tests for effective map resolution returned by sitemap map endpoint."""
from __future__ import annotations

import os
import sys

import pytest

# Make backend package importable from tests/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.routes import sitemap


class _DummySvc:
    def __init__(
        self,
        *,
        native_res: float,
        origin: list[float],
        served_w: int,
        served_h: int,
        native_size: tuple[int, int] | None,
        meta_w: int = 0,
        meta_h: int = 0,
    ) -> None:
        self._meta = {
            "resolution": native_res,
            "origin": origin,
            "width": meta_w,
            "height": meta_h,
        }
        self._img = {
            "width": served_w,
            "height": served_h,
            "b64": "data:image/png;base64,dummy",
        }
        self._native_size = native_size

    def get_map_meta(self, _site_id: str):
        return self._meta

    def get_map_image(self, _site_id: str, _dark_mode: bool):
        return self._img

    def get_native_map_size(self, _site_id: str):
        return self._native_size


def test_get_site_map_returns_effective_resolution_when_served_image_is_scaled(monkeypatch):
    # native map: 1000 px wide at 0.05 m/px, served image: 500 px wide
    # scale=0.5 => effective resolution must double to 0.10 m/served-px
    dummy = _DummySvc(
        native_res=0.05,
        origin=[-22.8, -16.4, 0.0],
        served_w=500,
        served_h=400,
        native_size=(1000, 800),
    )
    monkeypatch.setattr(sitemap, "_get_svc", lambda: dummy)

    result = sitemap.get_site_map("demo", dark_mode=True)

    assert result["resolution"] == pytest.approx(0.10)
    assert result["width"] == 500
    assert result["height"] == 400


def test_get_site_map_keeps_native_resolution_when_sizes_match(monkeypatch):
    dummy = _DummySvc(
        native_res=0.05,
        origin=[0.0, 0.0, 0.0],
        served_w=1000,
        served_h=800,
        native_size=(1000, 800),
    )
    monkeypatch.setattr(sitemap, "_get_svc", lambda: dummy)

    result = sitemap.get_site_map("demo", dark_mode=False)

    assert result["resolution"] == pytest.approx(0.05)


def test_get_site_map_falls_back_to_meta_dimensions_if_native_size_unavailable(monkeypatch):
    # If get_native_map_size is unavailable/None, use meta dimensions fallback.
    dummy = _DummySvc(
        native_res=0.05,
        origin=[0.0, 0.0, 0.0],
        served_w=500,
        served_h=400,
        native_size=None,
        meta_w=1000,
        meta_h=800,
    )
    monkeypatch.setattr(sitemap, "_get_svc", lambda: dummy)

    result = sitemap.get_site_map("demo", dark_mode=False)

    assert result["resolution"] == pytest.approx(0.10)
