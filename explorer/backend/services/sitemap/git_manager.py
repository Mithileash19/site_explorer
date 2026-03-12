"""
services/sitemap/git_manager.py
────────────────────────────────
GitRepoManager — branch-aware, no-checkout git file reader for sootballs_sites.

All reads use `git show <ref>:sites/<site_id>/<rel_path>` so the working tree
is never touched. Branch list is fetched from remote and cached for 5 minutes.
"""
from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

from core.logging import get_logger

logger = get_logger(__name__)

# Site-ID branch name pattern  (e.g. mncyok001, alcbrk001, actmaa001)
_SITE_BRANCH_RE = re.compile(r"^[a-z]{3}[a-z0-9]{3}[0-9]{3}$")

# How long to keep the branch-list cache (seconds)
_CACHE_TTL = 300


class GitRepoManager:
    """
    Read-only, branch-aware access to the sootballs_sites repo.
    Uses ``git show`` to read file content from any ref without checking out.
    """

    def __init__(self, repo_root: str) -> None:
        self.repo_root = Path(repo_root).resolve()
        # Per-site branch overrides: site_id → full ref (e.g. "origin/mncyok001")
        self._overrides: Dict[str, str] = {}
        # Cached set of remote site-ID branch short names
        self._site_branches: Optional[Set[str]] = None
        self._cache_ts: float = 0.0
        # Throttle for git fetch
        self._last_fetch_ts: float = 0.0
        logger.info("GitRepoManager: repo_root=%s", self.repo_root)

    # ── Git subprocess ─────────────────────────────────────────────────────────

    def _git(self, *args: str) -> subprocess.CompletedProcess:
        """Run a git command in the repo root directory. Never raises."""
        return subprocess.run(
            ["git", *args],
            cwd=str(self.repo_root),
            capture_output=True,
        )

    # ── Remote sync ────────────────────────────────────────────────────────────

    def fetch(self, force: bool = False) -> None:
        """
        Run ``git fetch origin --prune`` to sync remote refs.
        Throttled: skipped if last fetch was < _CACHE_TTL seconds ago,
        unless *force* is True.
        """
        now = time.monotonic()
        if not force and (now - self._last_fetch_ts) < _CACHE_TTL:
            return
        logger.info("GitRepoManager: fetching origin…")
        result = self._git("fetch", "origin", "--prune")
        if result.returncode != 0:
            logger.warning("git fetch failed: %s", result.stderr.decode("utf-8", errors="replace"))
        else:
            self._last_fetch_ts = now
            # Invalidate branch cache
            self._site_branches = None
            self._cache_ts = 0.0
            logger.info("GitRepoManager: fetch complete.")

    # ── Branch discovery ───────────────────────────────────────────────────────

    def list_site_branches(self) -> Set[str]:
        """
        Return the set of short branch names that:
          - match the site-ID pattern (e.g. mncyok001)
          - exist as remote tracking refs (origin/<name>)

        Result is cached for up to _CACHE_TTL seconds.
        """
        now = time.monotonic()
        if self._site_branches is not None and (now - self._cache_ts) < _CACHE_TTL:
            return self._site_branches

        result = self._git("branch", "-r")
        branches: Set[str] = set()
        for line in result.stdout.decode("utf-8", errors="replace").splitlines():
            # Lines look like "  origin/mncyok001" or "  origin/HEAD -> origin/main"
            name = line.strip()
            if "->" in name:
                continue
            short = name.replace("origin/", "", 1)
            if _SITE_BRANCH_RE.match(short):
                branches.add(short)

        self._site_branches = branches
        self._cache_ts = now
        logger.debug("GitRepoManager: found %d site branches", len(branches))
        return branches

    def list_all_remote_branches(self) -> List[str]:
        """
        Return all short remote-tracking branch names for use in the
        override dropdown (excludes the HEAD pointer line).
        """
        result = self._git("branch", "-r")
        names: List[str] = []
        for line in result.stdout.decode("utf-8", errors="replace").splitlines():
            name = line.strip()
            if "->" in name:
                continue
            short = name.replace("origin/", "", 1)
            if short:
                names.append(short)
        return sorted(names)

    def resolve_branch(self, site_id: str) -> str:
        """
        Return the full git ref to use for the given site.

        Priority:
          1. Manual override (set via set_override)
          2. Site-specific remote branch (origin/<site_id>) if it exists
          3. origin/main as fallback
        """
        if site_id in self._overrides:
            return self._overrides[site_id]
        site_branches = self.list_site_branches()
        if site_id in site_branches:
            return f"origin/{site_id}"
        return "origin/main"

    def set_override(self, site_id: str, branch: str) -> None:
        """Override which branch to use for a specific site (in-memory only)."""
        ref = branch if branch.startswith("origin/") else f"origin/{branch}"
        self._overrides[site_id] = ref
        logger.info("GitRepoManager: override %s → %s", site_id, ref)

    def clear_override(self, site_id: str) -> None:
        """Remove branch override for a site (reverts to auto-detect)."""
        self._overrides.pop(site_id, None)
        logger.info("GitRepoManager: cleared override for %s", site_id)

    def is_override(self, site_id: str) -> bool:
        """Return True if a manual override is active for the given site."""
        return site_id in self._overrides

    # ── File reading ───────────────────────────────────────────────────────────

    def read_file(self, ref: str, site_id: str, rel_path: str) -> Optional[bytes]:
        """
        Read a file from a specific git ref without touching the working tree.

        Uses: ``git show <ref>:sites/<site_id>/<rel_path>``

        Returns raw bytes on success, None if the path does not exist on that ref.
        """
        git_path = f"sites/{site_id}/{rel_path}"
        result = self._git("show", f"{ref}:{git_path}")
        if result.returncode == 0:
            return result.stdout
        return None

    def read_file_for_site(self, site_id: str, rel_path: str) -> Optional[bytes]:
        """Convenience: auto-resolve the branch for *site_id* then read the file."""
        ref = self.resolve_branch(site_id)
        return self.read_file(ref, site_id, rel_path)

    # ── Commit metadata ────────────────────────────────────────────────────────

    def get_last_commit(self, ref: str) -> Dict[str, str]:
        """Return hash, subject, and ISO8601 author-date of the tip of *ref*."""
        result = self._git(
            "log", "-1",
            "--pretty=format:%H|%s|%aI",
            ref,
        )
        if result.returncode != 0:
            return {"hash": "", "message": "", "date": ""}
        parts = result.stdout.decode("utf-8", errors="replace").split("|", 2)
        return {
            "hash":    parts[0] if len(parts) > 0 else "",
            "message": parts[1] if len(parts) > 1 else "",
            "date":    parts[2].strip() if len(parts) > 2 else "",
        }

    # ── Site listing ───────────────────────────────────────────────────────────

    def list_sites_from_git(self) -> List[str]:
        """
        List site directories from ``origin/main:sites/`` via git ls-tree.
        Returns sorted list of site-ID strings.
        """
        result = self._git("ls-tree", "--name-only", "origin/main", "sites/")
        if result.returncode != 0:
            return []
        return sorted(
            name.removeprefix("sites/")
            for name in result.stdout.decode("utf-8", errors="replace").splitlines()
            if name.strip()
        )

    # ── Branch cleanup ─────────────────────────────────────────────────────────

    def list_clean_branches(self, site_ids: List[str]) -> List[str]:
        """
        Return the filtered branch list for the UI dropdown:
        only ``main`` and branches whose name is a known site ID.

        This is the read-only, safe version — nothing is deleted.
        """
        valid: Set[str] = {"main"} | set(site_ids)
        return sorted(b for b in self.list_all_remote_branches() if b in valid)

    def get_branch_cleanup_plan(self, site_ids: List[str]) -> dict:
        """
        Dry-run analysis of which remote-tracking refs are valid vs invalid.

        Valid means: branch name is ``main`` OR equals a known site ID.

        Returns a dict with:
          - ``valid_branches``         — branches to keep
          - ``invalid_branches``       — branches to remove (not main, not a site)
          - ``sites_without_own_branch`` — site IDs that have no dedicated branch
          - ``total_branches``         — total remote branches found
        """
        all_branches = self.list_all_remote_branches()
        valid_set: Set[str] = {"main"} | set(site_ids)
        valid   = [b for b in all_branches if b in valid_set]
        invalid = [b for b in all_branches if b not in valid_set]
        branch_set = set(all_branches)
        sites_without = [s for s in sorted(site_ids) if s not in branch_set]
        return {
            "valid_branches":            valid,
            "invalid_branches":          invalid,
            "sites_without_own_branch":  sites_without,
            "total_branches":            len(all_branches),
        }

    def prune_invalid_remote_refs(self, site_ids: List[str]) -> dict:
        """
        Delete **local** remote-tracking refs for any branch that is not
        ``main`` and not a known site ID.

        This is safe:
        - Only touches ``refs/remotes/origin/<name>`` (local copies).
        - Does NOT delete branches on the actual remote server.
        - ``git fetch origin --prune`` would do the same for deleted remote branches.
        - Site configs and working-tree files are completely unaffected.

        Returns ``{"removed": [...], "kept": [...], "errors": [...]}``.
        """
        plan = self.get_branch_cleanup_plan(site_ids)
        removed: List[str] = []
        errors:  List[str] = []

        for branch in plan["invalid_branches"]:
            res = self._git("branch", "-rd", f"origin/{branch}")
            if res.returncode == 0:
                removed.append(branch)
                logger.info("GitRepoManager: pruned local ref origin/%s", branch)
            else:
                err_msg = res.stderr.decode("utf-8", errors="replace").strip()
                logger.warning(
                    "GitRepoManager: failed to prune origin/%s — %s", branch, err_msg
                )
                errors.append(branch)

        # Invalidate branch cache so next call re-reads from git
        self._site_branches = None
        self._cache_ts = 0.0

        return {
            "removed": removed,
            "kept":    plan["valid_branches"],
            "errors":  errors,
        }
