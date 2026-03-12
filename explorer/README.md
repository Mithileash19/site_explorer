# AMR Intelligence Platform

A unified platform combining **AI-powered ROS bag log analysis** with **fleet operations management** for Autonomous Mobile Robots (AMR).

## Features

| Page | Description |
|------|-------------|
| **Dashboard** | Fleet overview, site health, quick-action cards |
| **Fleet Map** | Interactive Plotly topology map with live fleet status |
| **Bag Analyzer** | Upload ROS1/ROS2 bags → timeline → log explorer → map diff |
| **Investigate** | SSE-streamed LLM incident investigation with FAISS similarity search |
| **Knowledge Base** | Browse, add, and ingest incidents (manual or Slack) |
| **AI Assistant** | Conversational streaming interface to the investigation pipeline |

## Architecture

```
unified_platform/
├── backend/               # Python 3.11 · FastAPI · Pydantic v2
│   ├── app/
│   │   ├── main.py        # App factory + singleton registration
│   │   └── routes/        # health · bags · sites · investigate · knowledge
│   ├── core/              # config · logging · middleware
│   ├── schemas/           # bag_analysis · investigation · site_data
│   └── services/
│       ├── ros/           # log_extractor · log_analyzer_engine · map_processor
│       ├── ai/            # llm_service · vector_db (FAISS) · investigation_engine
│       └── sites/         # data_loader · git_manager
├── frontend/              # Next.js 15 · React 19 · Tailwind v3
│   ├── app/               # App Router pages
│   ├── components/        # bags/ · fleet/ · investigation/ · layout/
│   └── lib/               # api.ts · types.ts
├── infrastructure/
│   └── nginx/nginx.conf   # Reverse proxy
├── docker-compose.yml
└── setup.sh
```

## Quick Start

### Docker (recommended)

```bash
cd unified_platform
bash setup.sh --docker
```

Open **http://localhost** in your browser.

### Local development

```bash
cd unified_platform
bash setup.sh          # installs Python venv + Node deps

# Terminal 1 — backend
cd backend && source .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 — frontend
cd frontend && npm run dev
```

Open **http://localhost:3000**.

## Environment variables

Copy `backend/.env.example` → `backend/.env` and edit:

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Ollama endpoint |
| `OLLAMA_MODEL` | `qwen2.5-coder` | Default local model |
| `OPENAI_API_KEY` | _(empty)_ | Set to use GPT-4o instead |
| `BAG_UPLOAD_DIR` | `data/bags` | Uploaded bag storage |
| `SITES_ROOT` | `data/sites` | Site data root |
| `FAISS_PATH` | `data/faiss.index` | Persisted FAISS index |
| `SLACK_BOT_TOKEN` | _(empty)_ | For Slack incident ingest |

## API Reference

All endpoints are under `/api/v1/`:

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | System health + model info |
| GET | `/sites` | List all sites |
| GET | `/sites/{id}/data` | Site topology (nodes + edges) |
| GET | `/sites/{id}/map` | Map image (occupancy PNG) |
| POST | `/bags/upload` | Upload ROS bag |
| GET | `/bags/timeline` | Log volume histogram |
| POST | `/bags/analyze` | Full AI log analysis |
| POST | `/bags/mapdiff` | Map change detection (IoU) |
| POST | `/investigate` | Full incident investigation |
| GET | `/investigate/stream` | SSE streaming investigation |
| GET | `/incidents` | List knowledge base |
| POST | `/ingest/manual` | Add incident manually |
| POST | `/ingest/slack` | Ingest from Slack thread |

## Technology Stack

- **Backend:** Python 3.11, FastAPI, Pydantic v2, rosbags, OpenAI SDK (Ollama-compat), FAISS, sentence-transformers, OpenCV, pandas, GitPython
- **Frontend:** Next.js 15, React 19, Tailwind CSS, Plotly.js, Axios, react-hook-form, react-markdown
- **Infrastructure:** Docker Compose, nginx, Ollama
