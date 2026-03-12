"""
Tests for GitRepoManager branch-cleanup logic.

All tests use unittest.mock to avoid touching any real git repo,
so they run identically in CI and local environments.
"""
from __future__ import annotations

import subprocess
import sys
import os
from typing import List
from unittest.mock import MagicMock, patch, call

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.sitemap.git_manager import GitRepoManager


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_git_result(stdout: str, returncode: int = 0) -> subprocess.CompletedProcess:
    result = MagicMock(spec=subprocess.CompletedProcess)
    result.returncode = returncode
    result.stdout = stdout.encode("utf-8")
    result.stderr = b""
    return result


def _mgr(remote_branches: List[str]) -> GitRepoManager:
    """Return a GitRepoManager whose `_git` is mocked to return *remote_branches*."""
    mgr = GitRepoManager("/fake/repo")
    branch_output = "\n".join(f"  origin/{b}" for b in remote_branches)
    mgr._git = MagicMock(return_value=_make_git_result(branch_output))  # type: ignore[method-assign]
    return mgr


# ── list_clean_branches ────────────────────────────────────────────────────────

class TestListCleanBranches:
    def test_keeps_main_and_site_branches(self):
        mgr = _mgr(["main", "mncyok001", "alcbrk001", "feat/my-feature"])
        result = mgr.list_clean_branches(["mncyok001", "alcbrk001"])
        assert result == ["alcbrk001", "main", "mncyok001"]

    def test_excludes_feature_branches(self):
        mgr = _mgr(["main", "feature/some-work", "hotfix-123", "mncyok001"])
        result = mgr.list_clean_branches(["mncyok001"])
        assert "feature/some-work" not in result
        assert "hotfix-123" not in result
        assert "main" in result
        assert "mncyok001" in result

    def test_returns_sorted(self):
        mgr = _mgr(["main", "zzzsite001", "aaabbb001"])
        result = mgr.list_clean_branches(["zzzsite001", "aaabbb001"])
        assert result == sorted(result)

    def test_empty_site_ids_only_keeps_main(self):
        mgr = _mgr(["main", "mncyok001", "ci-branch"])
        result = mgr.list_clean_branches([])
        assert result == ["main"]

    def test_no_matching_branches_returns_empty(self):
        mgr = _mgr(["feature/x", "ci/build"])
        result = mgr.list_clean_branches(["mncyok001"])
        assert result == []

    def test_site_not_on_remote_not_added(self):
        """A site ID that has no corresponding remote branch should not appear."""
        mgr = _mgr(["main", "alcbrk001"])
        result = mgr.list_clean_branches(["mncyok001", "alcbrk001"])
        assert "mncyok001" not in result   # has no remote branch
        assert "alcbrk001" in result


# ── get_branch_cleanup_plan ────────────────────────────────────────────────────

class TestGetBranchCleanupPlan:
    def test_separates_valid_and_invalid(self):
        mgr = _mgr(["main", "mncyok001", "feat/stuff", "hotfix"])
        plan = mgr.get_branch_cleanup_plan(["mncyok001"])
        assert "main"      in plan["valid_branches"]
        assert "mncyok001" in plan["valid_branches"]
        assert "feat/stuff" in plan["invalid_branches"]
        assert "hotfix"     in plan["invalid_branches"]

    def test_total_branches_is_correct(self):
        branches = ["main", "mncyok001", "alcbrk001", "ci-test"]
        mgr = _mgr(branches)
        plan = mgr.get_branch_cleanup_plan(["mncyok001", "alcbrk001"])
        assert plan["total_branches"] == len(branches)

    def test_sites_without_own_branch(self):
        mgr = _mgr(["main", "alcbrk001"])
        plan = mgr.get_branch_cleanup_plan(["mncyok001", "alcbrk001"])
        # mncyok001 has no remote branch → listed under sites_without_own_branch
        assert "mncyok001" in plan["sites_without_own_branch"]
        assert "alcbrk001" not in plan["sites_without_own_branch"]

    def test_all_clean_nothing_to_remove(self):
        mgr = _mgr(["main", "mncyok001"])
        plan = mgr.get_branch_cleanup_plan(["mncyok001"])
        assert plan["invalid_branches"] == []
        assert plan["valid_branches"] == ["main", "mncyok001"]

    def test_all_invalid_no_main_no_site(self):
        mgr = _mgr(["feat/a", "ci/b", "wip"])
        plan = mgr.get_branch_cleanup_plan(["mncyok001"])
        assert plan["valid_branches"] == []
        assert set(plan["invalid_branches"]) == {"feat/a", "ci/b", "wip"}

    def test_valid_branches_sorted(self):
        mgr = _mgr(["main", "zzzsite001", "aaabbb001"])
        plan = mgr.get_branch_cleanup_plan(["zzzsite001", "aaabbb001"])
        assert plan["valid_branches"] == sorted(plan["valid_branches"])


# ── prune_invalid_remote_refs ──────────────────────────────────────────────────

class TestPruneInvalidRemoteRefs:
    def _mgr_with_prune(self, remote_branches: List[str], prune_returncode: int = 0) -> GitRepoManager:
        """
        Returns a manager where:
        - `git branch -r`  → returns *remote_branches*
        - `git branch -rd` → returns *prune_returncode*
        """
        mgr = GitRepoManager("/fake/repo")
        branch_output = "\n".join(f"  origin/{b}" for b in remote_branches)

        def _fake_git(*args: str) -> subprocess.CompletedProcess:
            if args[0] == "branch" and "-r" in args:
                return _make_git_result(branch_output)
            if args[0] == "branch" and "-rd" in args:
                return _make_git_result("", prune_returncode)
            return _make_git_result("")

        mgr._git = _fake_git  # type: ignore[method-assign]
        return mgr

    def test_removes_invalid_branches(self):
        mgr = self._mgr_with_prune(["main", "mncyok001", "feat/junk"])
        result = mgr.prune_invalid_remote_refs(["mncyok001"])
        assert "feat/junk" in result["removed"]
        assert "main"      in result["kept"]
        assert "mncyok001" in result["kept"]

    def test_keeps_main_always(self):
        mgr = self._mgr_with_prune(["main", "ci-branch"])
        result = mgr.prune_invalid_remote_refs([])
        assert "main" in result["kept"]
        assert "ci-branch" in result["removed"]

    def test_nothing_to_remove_when_already_clean(self):
        mgr = self._mgr_with_prune(["main", "mncyok001"])
        result = mgr.prune_invalid_remote_refs(["mncyok001"])
        assert result["removed"] == []
        assert result["errors"]  == []
        assert set(result["kept"]) == {"main", "mncyok001"}

    def test_errors_recorded_when_git_fails(self):
        mgr = self._mgr_with_prune(["main", "feat/junk"], prune_returncode=1)
        result = mgr.prune_invalid_remote_refs([])
        assert "feat/junk" in result["errors"]
        assert "feat/junk" not in result["removed"]

    def test_cache_invalidated_after_prune(self):
        mgr = self._mgr_with_prune(["main", "mncyok001", "stale"])
        # Pre-populate cache
        mgr._site_branches = {"mncyok001"}
        mgr._cache_ts = 9e9
        mgr.prune_invalid_remote_refs(["mncyok001"])
        # Cache should have been cleared
        assert mgr._site_branches is None
        assert mgr._cache_ts == 0.0

    def test_does_not_affect_site_data(self):
        """prune only calls `git branch -rd`, never touches working tree commands."""
        mgr = self._mgr_with_prune(["main", "feat/junk"])
        called_commands: list = []

        def _spy(*args: str) -> subprocess.CompletedProcess:
            called_commands.append(args)
            branch_output = "  origin/main\n  origin/feat/junk"
            if args[0] == "branch" and "-r" in args:
                return _make_git_result(branch_output)
            return _make_git_result("", 0)

        mgr._git = _spy  # type: ignore[method-assign]
        mgr.prune_invalid_remote_refs([])

        for cmd in called_commands:
            assert "checkout" not in cmd, "prune must not run git checkout"
            assert "reset" not in cmd,    "prune must not run git reset"
            assert "clean" not in cmd,    "prune must not run git clean"


# ── API route integration (TestClient) ────────────────────────────────────────

class TestCleanupRoutes:
    """Light integration tests using FastAPI's TestClient."""

    def setup_method(self):
        from fastapi.testclient import TestClient
        from app.main import app
        self.client = TestClient(app)

    def _patch_git(self, remote_branches: List[str]):
        """Patch GitRepoManager._git for route-level tests."""
        import re as _re
        _SITE_RE = _re.compile(r"^[a-z]{3}[a-z0-9]{3}[0-9]{3}$")
        branch_output = "\n".join(f"  origin/{b}" for b in remote_branches)
        # ls-tree only exposes genuine site IDs (pattern-matching names)
        site_names = [b for b in remote_branches if _SITE_RE.match(b)]

        def _fake_git(self_ref, *args):  # bound-method signature
            if args[0] == "branch":
                return _make_git_result(branch_output)
            if args[0] == "ls-tree":
                sites = "\n".join(f"sites/{b}" for b in site_names)
                return _make_git_result(sites)
            return _make_git_result("")

        return patch.object(GitRepoManager, "_git", _fake_git)

    def test_cleanup_plan_endpoint_returns_200(self):
        with self._patch_git(["main", "mncyok001", "feat/junk"]):
            resp = self.client.get("/api/v1/sitemap/cleanup/plan")
        assert resp.status_code == 200

    def test_cleanup_plan_has_expected_keys(self):
        with self._patch_git(["main", "mncyok001", "feat/junk"]):
            body = self.client.get("/api/v1/sitemap/cleanup/plan").json()
        assert "valid_branches"           in body
        assert "invalid_branches"         in body
        assert "sites_without_own_branch" in body
        assert "total_branches"           in body

    def test_cleanup_plan_identifies_invalid(self):
        with self._patch_git(["main", "mncyok001", "feat/junk"]):
            body = self.client.get("/api/v1/sitemap/cleanup/plan").json()
        assert "feat/junk" in body["invalid_branches"]
        assert "main"      in body["valid_branches"]

    def test_cleanup_execute_endpoint_returns_200(self):
        with self._patch_git(["main", "mncyok001"]):
            resp = self.client.post("/api/v1/sitemap/cleanup")
        assert resp.status_code == 200

    def test_cleanup_execute_returns_removed_kept_errors(self):
        with self._patch_git(["main", "mncyok001"]):
            body = self.client.post("/api/v1/sitemap/cleanup").json()
        assert "removed" in body
        assert "kept"    in body
        assert "errors"  in body
