"""
core/config.py — Unified application settings loaded from environment variables.
Covers both aiassist (LLM, bag analysis) and site_commander (maps, git sync) settings.
"""
import os
from dotenv import load_dotenv

_ENV_PATH = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(dotenv_path=_ENV_PATH, override=False)


class _Settings:
    # ── LLM (Ollama / OpenAI-compatible) ────────────────────────────────────
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    ollama_model:    str = os.getenv("OLLAMA_MODEL",    "qwen2.5-coder")
    openai_api_key:  str = os.getenv("OPENAI_API_KEY",  "")

    # ── Server ───────────────────────────────────────────────────────────────
    host:      str = os.getenv("HOST",      "0.0.0.0")
    port:      int = int(os.getenv("PORT",  "8000"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO").upper()

    # ── File storage ─────────────────────────────────────────────────────────
    bag_upload_dir: str = os.getenv("BAG_UPLOAD_DIR", "data/bags")
    sites_root:     str = os.getenv("SITES_ROOT",     "data/sites")

    # ── FAISS / Vector DB ────────────────────────────────────────────────────
    faiss_path:    str = os.getenv("FAISS_PATH",    "data/faiss.index")
    metadata_path: str = os.getenv("META_PATH",     "data/metadata.json")

    # ── Git sync ─────────────────────────────────────────────────────────────
    repo_url:         str  = os.getenv("REPO_URL",          "")
    site_sync_enabled: bool = os.getenv("SITE_SYNC_ENABLED", "false").lower() == "true"

    # ── External integrations ────────────────────────────────────────────────
    slack_bot_token: str = os.getenv("SLACK_BOT_TOKEN", "")
    grafana_api_key: str = os.getenv("GRAFANA_API_KEY", "")
    github_token:    str = os.getenv("GITHUB_TOKEN",    "")

    # ── Site Map (sootballs_sites repo) ─────────────────────────────────────────
    sootballs_sites_root: str = os.getenv(
        "SOOTBALLS_SITES_ROOT",
        os.path.join(os.path.dirname(__file__), "..", "..", "sootballs_sites", "sites"),
    )
    sootballs_repo_root: str = os.getenv(
        "SOOTBALLS_REPO_ROOT",
        os.path.join(os.path.dirname(__file__), "..", "..", "sootballs_sites"),
    )

    # ── CORS ─────────────────────────────────────────────────────────────────
    allowed_origins: list = os.getenv(
        "ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:3001"
    ).split(",")


settings = _Settings()
