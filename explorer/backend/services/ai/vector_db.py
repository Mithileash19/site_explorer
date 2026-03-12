"""
services/ai/vector_db.py
─────────────────────────
Persisted FAISS vector DB for historical incident similarity search.
Index and metadata survive restarts via filesystem persistence.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.config import settings
from core.logging import get_logger

logger = get_logger(__name__)

try:
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer
    _FAISS_OK = True
except ImportError:
    logger.warning("FAISS or sentence-transformers not installed — HistoricalMatcher in stub mode.")
    _FAISS_OK = False


class HistoricalMatcher:
    """
    Stores incident embeddings in FAISS and persists index + metadata to disk.
    """

    DIM = 384  # all-MiniLM-L6-v2 output dimension

    def __init__(self) -> None:
        self.faiss_path = Path(settings.faiss_path)
        self.meta_path  = Path(settings.metadata_path)
        self.metadata_store: List[Dict[str, Any]] = []

        if not _FAISS_OK:
            self.index  = None
            self.model  = None
            return

        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        # Load existing index or create fresh
        if self.faiss_path.exists() and self.meta_path.exists():
            try:
                self.index = faiss.read_index(str(self.faiss_path))
                with open(self.meta_path) as f:
                    self.metadata_store = json.load(f)
                logger.info("HistoricalMatcher: loaded %d incidents from disk.", len(self.metadata_store))
            except Exception as e:
                logger.error("HistoricalMatcher: failed to load persisted index (%s) — starting fresh.", e)
                self._init_fresh()
        else:
            self._init_fresh()

    def _init_fresh(self) -> None:
        self.index          = faiss.IndexFlatL2(self.DIM)
        self.metadata_store = []
        logger.info("HistoricalMatcher: initialised fresh empty index.")

    def _embed(self, text: str):
        vec = self.model.encode([text], convert_to_numpy=True).astype("float32")
        return vec

    def _persist(self) -> None:
        self.faiss_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.faiss_path))
        with open(self.meta_path, "w") as f:
            json.dump(self.metadata_store, f, indent=2)

    def ingest(
        self,
        text:       str,
        root_cause: str = "",
        fix:        str = "",
        title:      str = "",
    ) -> int:
        """Embed `text` and add to the index. Returns the new incident ID."""
        if not _FAISS_OK:
            return -1
        vec = self._embed(text)
        self.index.add(vec)
        incident_id = len(self.metadata_store)
        self.metadata_store.append({
            "id":         incident_id,
            "summary":    text[:500],
            "title":      title or text[:80],
            "root_cause": root_cause,
            "fix":        fix,
            "timestamp":  time.time(),
        })
        self._persist()
        logger.info("HistoricalMatcher: ingested incident #%d.", incident_id)
        return incident_id

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Return top-k similar incidents with similarity percentage."""
        if not _FAISS_OK or self.index is None or self.index.ntotal == 0:
            return []
        vec = self._embed(query)
        distances, indices = self.index.search(vec, min(k, self.index.ntotal))
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata_store):
                continue
            meta    = self.metadata_store[idx]
            # Convert L2 distance to a rough similarity percentage
            sim_pct = max(0.0, round(100.0 / (1.0 + dist), 1))
            results.append({**meta, "similarity_pct": sim_pct})
        return results

    def list_incidents(self) -> List[Dict[str, Any]]:
        """Return StoredIncident-shaped dicts for all stored incidents."""
        import datetime
        out = []
        for m in self.metadata_store:
            ts = m.get("timestamp", 0)
            try:
                created = datetime.datetime.utcfromtimestamp(ts).isoformat() + "Z"
            except Exception:
                created = ""
            summary = m.get("summary", "")
            out.append({
                "id":          m["id"],
                "title":       m.get("title") or summary[:80],
                "description": summary,
                "root_cause":  m.get("root_cause", ""),
                "resolution":  m.get("fix", ""),
                "created_at":  created,
            })
        return out

    @property
    def total(self) -> int:
        if self.index is None:
            return 0
        return self.index.ntotal
