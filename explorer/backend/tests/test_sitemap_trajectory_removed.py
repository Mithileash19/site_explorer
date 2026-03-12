"""Regression tests: trajectory sitemap APIs must remain removed."""
import os
import sys

# Make backend package importable from tests/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.main import app


def test_sitemap_upload_endpoint_removed_from_openapi():
    paths = app.openapi().get("paths", {})
    assert "/api/v1/sitemap/bags/upload" not in paths


def test_sitemap_trajectory_endpoint_removed_from_openapi():
    paths = app.openapi().get("paths", {})
    assert "/api/v1/sitemap/bags/{bag_name}/trajectory" not in paths


def test_sitemap_navgraph_endpoint_removed_from_openapi():
    paths = app.openapi().get("paths", {})
    assert "/api/v1/sitemap/{site_id}/navgraph" not in paths


def test_sitemap_bag_list_endpoint_removed_from_openapi():
    paths = app.openapi().get("paths", {})
    assert "/api/v1/sitemap/bags/list" not in paths


def test_sitemap_bag_delete_endpoint_removed_from_openapi():
    paths = app.openapi().get("paths", {})
    assert "/api/v1/sitemap/bags/{filename}" not in paths
