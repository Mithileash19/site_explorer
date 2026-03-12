# AMR Intelligence Platform — Complete Setup, Run, and Understanding Guide

This file explains how to set up and run the full platform end-to-end, and how each major part works.

## 1) What You Are Running

The platform has 4 runtime parts:

1. `backend` (FastAPI, Python): REST APIs, bag parsing, site-map services, AI orchestration.
2. `frontend` (Next.js, TypeScript): UI pages (dashboard, sitemap, investigation, etc.).
3. `ollama` (optional/local LLM): model inference endpoint for AI features.
4. `nginx` (Docker mode): reverse proxy for frontend + backend under one host.

Main runtime roots:

- `unified_platform/backend`
- `unified_platform/frontend`
- `unified_platform/sootballs_sites` (site map/config data)
- `unified_platform/data` (bags, FAISS, metadata)

## 2) Prerequisites

Local mode:

- Linux/macOS shell
- Python 3.10+ (3.11 recommended)
- Node.js LTS (18+; 20+ recommended)
- npm

Docker mode:

- Docker + Docker Compose plugin (`docker compose`)

## 3) Environment Configuration

Backend env file:

```bash
cd unified_platform
cp backend/.env.example backend/.env
```

Important variables in `backend/.env`:

- `OLLAMA_BASE_URL` (default `http://localhost:11434/v1`)
- `OLLAMA_MODEL` (default `qwen2.5-coder`)
- `OPENAI_API_KEY` (optional, if using OpenAI instead of Ollama)
- `BAG_UPLOAD_DIR` (default `data/bags`)
- `SITES_ROOT` (default `data/sites`)
- `FAISS_PATH` (default `data/faiss.index`)
- `META_PATH` (default `data/metadata.json`)
- `SOOTBALLS_REPO_ROOT` (for sitemap git-backed reads)
- `SOOTBALLS_SITES_ROOT` (site data root)

Frontend API base:

- `frontend/lib/api.ts` uses `NEXT_PUBLIC_API_URL`.
- Fallback is `http://localhost:8000`.

## 4) One-Command Setup

From repo root:

```bash
cd unified_platform
bash setup.sh
```

What this does:

- Creates `backend/.env` if missing.
- Creates backend venv and installs `backend/requirements.txt`.
- Installs frontend dependencies with `npm ci`.

## 5) Run the Full Stack

### Option A: Docker (recommended for complete stack)

```bash
cd unified_platform
bash setup.sh --docker
```

Services started:

- `amr_ollama`
- `amr_backend`
- `amr_frontend`
- `amr_nginx`

Open:

- `http://localhost` (nginx)
- `http://localhost:3000` (frontend direct)
- `http://localhost:8000/api/v1/health` (backend health)

Useful Docker commands:

```bash
docker compose ps
docker compose logs -f backend
docker compose logs -f frontend
docker compose logs -f ollama
docker compose down
```

### Option B: Local dev (two terminals)

Terminal 1 (backend):

```bash
cd unified_platform/backend
source .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Terminal 2 (frontend):

```bash
cd unified_platform/frontend
npm run dev
```

Open:

- `http://localhost:3000`
- Backend docs: `http://localhost:8000/docs`
- Backend health: `http://localhost:8000/api/v1/health`

## 6) Site Commander (Sitemap) Data Expectations

For each site under `sootballs_sites/sites/<site_id>/`, the sitemap flow reads:

- Map metadata: `config/maps/navigation_map.yaml` (or fallback variants)
- Map image: `config/maps/map.png` (or simulation fallback)
- Spots: `config/fixtures/spots.csv`
- Racks: `config/fixtures/rack_mapping.csv`
- Regions: fixture CSVs (via `SiteMapService` parser)
- Graph: `config/maps/graph.svg`

## 7) End-to-End Validation Checklist

Backend checks:

```bash
curl -s http://localhost:8000/api/v1/health
```

Sitemap checks:

1. Load a site in `/sitemap` page.
2. Confirm spots, racks, regions, nav graph, and markers render correctly.
3. Confirm search and pan-to interactions focus the expected objects.
4. Confirm branch selection/sync updates site-map content as expected.

## 8) Useful Dev Commands

Backend tests:

```bash
cd unified_platform/backend
source .venv/bin/activate
pytest -q
```

Frontend lint:

```bash
cd unified_platform/frontend
npm run lint
```

Backend only logs (Docker):

```bash
cd unified_platform
docker compose logs -f backend
```

## 9) Common Problems and Fixes

1. Error: `No module named pytest`
- Cause: using system Python, not backend venv.
- Fix: `source backend/.venv/bin/activate` then run tests.

2. Frontend loads but API fails
- Cause: `NEXT_PUBLIC_API_URL` or backend not reachable.
- Fix: verify `http://localhost:8000/api/v1/health` and env values.

3. Site map missing
- Cause: site files not found under configured roots.
- Fix: validate `SOOTBALLS_SITES_ROOT` and site folder structure.

## 10) Quick Start Summary

- Fastest full run: `bash setup.sh --docker`
- Fastest local dev: `bash setup.sh`, then run backend + frontend commands above.
- Use `/docs` and `/api/v1/health` to confirm backend is healthy.
- Use `/sitemap` to validate map and site data behavior visually.
