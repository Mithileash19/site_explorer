"""
app/routes/health.py — Unified healthcheck endpoint.
"""
from fastapi import APIRouter
from core.config import settings

router = APIRouter()

# These are set by the app after singleton creation
_llm_service   = None
_matcher       = None
_site_manager  = None


def register_singletons(llm, matcher, site_mgr):
    global _llm_service, _matcher, _site_manager
    _llm_service  = llm
    _matcher      = matcher
    _site_manager = site_mgr


@router.get("/api/v1/health", tags=["system"])
def health():
    faiss_entries = _matcher.total if _matcher else 0
    site_count    = len(_site_manager.list_sites()) if _site_manager else 0
    return {
        "status":       "ok",
        "version":      "1.0.0",
        "model":        settings.ollama_model,
        "faiss_entries": faiss_entries,
        "sites_loaded": site_count,
    }
