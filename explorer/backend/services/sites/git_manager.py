"""
services/sites/git_manager.py
──────────────────────────────
GitSyncEngine: shallow-clone or pull a site data repository.
Source: site_commander/backend/git_manager.py
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List

from core.config import settings
from core.logging import get_logger

logger = get_logger(__name__)

try:
    from git import Repo, InvalidGitRepositoryError
    _GIT_OK = True
except ImportError:
    logger.warning("gitpython not installed — GitSyncEngine disabled.")
    _GIT_OK = False


class GitSyncEngine:
    """
    Syncs a remote sites repository locally.
    Safe for Docker volume mounts: empties directory contents, not the mount point.
    """

    def __init__(self, local_path: str = None) -> None:
        self.repo_url   = settings.repo_url
        self.local_path = local_path or settings.sites_root

    def sync(self) -> bool:
        if not _GIT_OK or not self.repo_url:
            logger.info("GitSyncEngine: skipping sync (repo_url=%s, gitpython=%s).", self.repo_url, _GIT_OK)
            return False
        try:
            if (Path(self.local_path) / ".git").exists():
                logger.info("GitSyncEngine: pulling %s", self.local_path)
                repo = Repo(self.local_path)
                repo.remotes.origin.pull()
            else:
                self._clean_and_clone()
            logger.info("GitSyncEngine: sync complete.")
            return True
        except Exception as e:
            logger.error("GitSyncEngine.sync failed: %s", e)
            return False

    def _clean_and_clone(self) -> None:
        path = Path(self.local_path)
        path.mkdir(parents=True, exist_ok=True)
        # Empty directory contents (safe for volume mounts)
        for child in path.iterdir():
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                import shutil
                shutil.rmtree(child)
        logger.info("GitSyncEngine: cloning %s → %s", self.repo_url, self.local_path)
        Repo.clone_from(self.repo_url, str(path), depth=1)

    def get_sites(self) -> List[str]:
        """Return site directories containing a navigation_map.yaml."""
        sites = []
        root  = Path(self.local_path)
        if not root.exists():
            return sites
        for d in root.iterdir():
            if d.is_dir() and (d / "navigation_map.yaml").exists():
                sites.append(d.name)
        return sorted(sites)
