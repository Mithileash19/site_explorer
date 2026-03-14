"""
app/main.py — Unified AMR Intelligence Platform — FastAPI application factory.
"""
from __future__ import annotations

import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from core.config import settings
from core.logging import get_logger
from core.middleware import RequestContextMiddleware

logger = get_logger(__name__)

# ── Create data directories ────────────────────────────────────────────────────
for _d in [settings.bag_upload_dir, settings.sites_root, str(Path(settings.faiss_path).parent)]:
    Path(_d).mkdir(parents=True, exist_ok=True)

# ── Initialise singletons ──────────────────────────────────────────────────────
from services.ai.llm_service import LLMService
from services.ai.vector_db import HistoricalMatcher
from services.ai.investigation_engine import InvestigationEngine
from services.sites.data_loader import SiteDataManager

llm_service  = LLMService()
matcher      = HistoricalMatcher()
site_manager = SiteDataManager(settings.sites_root)
inv_engine   = InvestigationEngine(llm=llm_service, matcher=matcher)

# ── Optional site sync on startup ────────────────────────────────────────────
if settings.site_sync_enabled:
    from services.sites.git_manager import GitSyncEngine
    GitSyncEngine().sync()

# ── Routers ───────────────────────────────────────────────────────────────────
from app.routes import health, sites, bags, investigation, slack_investigation, analyse
from app.routes import sitemap as sitemap_route
from app.routes import grafana as grafana_route
from app.routes import logs as logs_route
from services.ai.slack_investigation_service import SlackInvestigationService

health.register_singletons(llm_service, matcher, site_manager)
sites.register_singletons(site_manager)
bags.register_singletons(llm_service, site_manager)
investigation.register_singletons(inv_engine, llm_service)
slack_investigation.register_singletons(llm_service)
grafana_route.register_singletons()
analyse.register_singletons(llm_service, SlackInvestigationService(llm_service))
logs_route.register_singletons(grafana_route._svc)

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title      = "AMR Intelligence Platform",
    description= "Unified robotics fleet monitoring + AI-powered incident investigation.",
    version    = "1.0.0",
    docs_url   = "/docs",
    redoc_url  = "/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = settings.allowed_origins,
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)
app.add_middleware(RequestContextMiddleware)

app.include_router(health.router)
app.include_router(sites.router)
app.include_router(bags.router)
app.include_router(investigation.router)
app.include_router(slack_investigation.router)
app.include_router(sitemap_route.router)
app.include_router(grafana_route.router)
app.include_router(analyse.router)
app.include_router(logs_route.router)


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all: ensure every unhandled exception returns JSON (not the bare
    Starlette plain-text 'Internal Server Error' page that the browser renders
    as a white-screen with no context)."""
    logger.error(
        "Unhandled exception [%s %s]: %s: %s",
        request.method,
        request.url.path,
        type(exc).__name__,
        exc,
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={"detail": f"{type(exc).__name__}: {exc}"},
    )


logger.info(
    "AMR Intelligence Platform ready — model=%s, sites_root=%s, faiss=%d incidents",
    settings.ollama_model,
    settings.sites_root,
    matcher.total,
)

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host    = settings.host,
        port    = settings.port,
        reload  = True,
        log_level = settings.log_level.lower(),
    )
