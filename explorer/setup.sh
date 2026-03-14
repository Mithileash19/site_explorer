#!/usr/bin/env bash
# setup.sh — First-time setup for AMR Intelligence Platform
# Usage: bash setup.sh [--docker | --docker-dev | --local]
set -euo pipefail

MODE="${1:-}"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[✓]${NC} $*"; }
warn()  { echo -e "${YELLOW}[!]${NC} $*"; }
error() { echo -e "${RED}[✗]${NC} $*"; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Create .env if missing ─────────────────────────────────────────────────────
if [[ ! -f "backend/.env" ]]; then
  cp backend/.env.example backend/.env
  warn "Created backend/.env from .env.example — review & edit before starting."
fi

# ────────────────────────────────────────────────────────────────────────────
# Docker mode
# ────────────────────────────────────────────────────────────────────────────
if [[ "$MODE" == "--docker" ]]; then
  info "Building Docker images…"
  docker compose build --parallel

  info "Starting services (Ollama, backend, frontend, nginx)…"
  docker compose up -d

  info "Waiting for backend health check…"
  until docker compose exec backend curl -sf http://localhost:8000/api/v1/health > /dev/null; do
    sleep 3
    echo -n "."
  done
  echo ""

  # Pull default Ollama model
  MODEL=$(grep OLLAMA_MODEL backend/.env | cut -d= -f2 | xargs)
  MODEL="${MODEL:-qwen2.5-coder}"
  info "Pulling Ollama model: ${MODEL}"
  docker compose exec ollama ollama pull "${MODEL}" || warn "Model pull failed — run manually: docker compose exec ollama ollama pull ${MODEL}"

  info "Done! Platform is running at http://localhost"
  info "  Frontend : http://localhost:3000"
  info "  API      : http://localhost:8000/api/v1/health"
  info "  nginx    : http://localhost"
  exit 0
fi

# ────────────────────────────────────────────────────────────────────────────
# Docker dev mode (frontend hot reload)
# ────────────────────────────────────────────────────────────────────────────
if [[ "$MODE" == "--docker-dev" ]]; then
  info "Building Docker images for dev stack…"
  docker compose -f docker-compose.yml -f docker-compose.dev.yml build --parallel

  info "Starting dev services with frontend hot reload…"
  docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d --remove-orphans

  info "Waiting for backend health check…"
  until docker compose -f docker-compose.yml -f docker-compose.dev.yml exec backend curl -sf http://localhost:8000/api/v1/health > /dev/null; do
    sleep 3
    echo -n "."
  done
  echo ""

  MODEL=$(grep OLLAMA_MODEL backend/.env | cut -d= -f2 | xargs)
  MODEL="${MODEL:-qwen2.5-coder}"
  info "Pulling Ollama model: ${MODEL}"
  docker compose -f docker-compose.yml -f docker-compose.dev.yml exec ollama ollama pull "${MODEL}" || warn "Model pull failed — run manually: docker compose -f docker-compose.yml -f docker-compose.dev.yml exec ollama ollama pull ${MODEL}"

  info "Done! Dev platform is running with live frontend updates."
  info "  Frontend (hot reload): http://localhost:3000"
  info "  API                : http://localhost:8000/api/v1/health"
  info "  nginx              : http://localhost"
  exit 0
fi

# ────────────────────────────────────────────────────────────────────────────
# Local (virtualenv) mode
# ────────────────────────────────────────────────────────────────────────────
info "Setting up Python virtual environment…"
cd backend
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
info "Python dependencies installed."
deactivate
cd ..

info "Setting up Node.js dependencies…"
cd frontend
npm ci --prefer-offline --silent
info "Node.js dependencies installed."
cd ..

# Ensure data dirs exist
mkdir -p backend/data/bags backend/data/sites

echo ""
info "Setup complete."
echo ""
echo "  Start backend:   cd backend && source .venv/bin/activate && uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"
echo "  Start frontend:  cd frontend && npm run dev"
echo ""
echo "  Or use Docker:   bash setup.sh --docker"
echo "  Docker + hot reload: bash setup.sh --docker-dev"
