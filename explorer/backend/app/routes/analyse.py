"""Route for the combined Log + Slack AI analysis endpoint."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from core.logging import get_logger
from schemas.analyse import AnalyseRequest, AnalyseResponse

logger = get_logger(__name__)
router = APIRouter()

_llm_service = None
_slack_service = None


def register_singletons(llm_service, slack_service) -> None:
    global _llm_service, _slack_service
    _llm_service = llm_service
    _slack_service = slack_service


@router.post(
    "/api/v1/investigate/analyse",
    tags=["investigation"],
    response_model=AnalyseResponse,
)
def analyse_logs_and_slack(req: AnalyseRequest) -> AnalyseResponse:
    """Combined AI analysis of Grafana logs + optional Slack thread."""
    if _llm_service is None:
        raise HTTPException(503, "LLM service not available.")

    from services.ai.slack_investigation_service import (
        SlackInvestigationService,
        parse_slack_thread_url,
    )
    from core.config import settings
    import requests as http_requests

    MAX_IMAGES = 4
    ollama_host = settings.ollama_host.rstrip("/")
    text_model = settings.ollama_text_model
    vision_model = settings.ollama_vision_model

    # ── Fetch Slack thread if URL provided ──────────────────────────────────
    slack_text = ""
    slack_images: list[str] = []
    slack_msg_count = 0

    if req.slack_thread_url and _slack_service:
        try:
            from schemas.slack_investigation import SlackThreadInvestigationRequest

            ref = parse_slack_thread_url(req.slack_thread_url)
            messages, attachments = _slack_service._fetch_thread_messages(
                ref, include_bots=False, max_messages=200
            )
            slack_msg_count = len(messages)

            thread_lines: list[str] = []
            for msg in messages:
                line = f"[{msg.datetime}] {msg.user}: {msg.text}"
                for idx, block in enumerate(msg.log_blocks, 1):
                    line += f"\n  [Log block {idx}]\n{block[:2000]}"
                thread_lines.append(line)
            slack_text = "\n\n".join(thread_lines)

            for att in attachments:
                if att.filetype == "image" and att.b64_image:
                    slack_images.append(att.b64_image)

        except (ValueError, RuntimeError) as exc:
            logger.warning("Slack fetch skipped: %s", exc)
            slack_text = f"[Slack thread could not be fetched: {exc}]"
        except Exception as exc:
            logger.error("Slack fetch failed: %s", exc, exc_info=True)
            slack_text = f"[Slack thread error: {exc}]"

    # ── Build log summary ───────────────────────────────────────────────────
    log_lines: list[str] = []
    for entry in req.logs[:2000]:
        lvl = entry.level or entry.labels.get("detected_level", "")
        dep = entry.deployment or entry.labels.get("deployment_name", "")
        host = entry.hostname or entry.labels.get("hostname", "")
        log_lines.append(f"[{entry.timestamp_ms}] [{lvl}] [{host}/{dep}] {entry.message[:500]}")

    # ── Model selection ─────────────────────────────────────────────────────
    has_images = len(slack_images) > 0

    def _installed_models() -> list[str]:
        try:
            resp = http_requests.get(f"{ollama_host}/api/tags", timeout=5)
            resp.raise_for_status()
            return [m.get("name", "") for m in resp.json().get("models", [])]
        except Exception:
            return []

    installed = _installed_models()

    if has_images:
        vp = vision_model.split(":", 1)[0]
        if any(vp in n for n in installed):
            model = vision_model
        else:
            model = text_model
            has_images = False
    else:
        model = text_model

    tp = model.split(":", 1)[0]
    if not any(tp in n for n in installed):
        raise HTTPException(503, f"Model '{model}' not installed. Pull it with: ollama pull {model}")

    # ── Build prompt ────────────────────────────────────────────────────────
    # If no Slack data, use log-only analysis sections
    has_slack_data = bool(slack_text and slack_text.strip() and not slack_text.startswith("[Slack"))

    if has_slack_data:
        system = (
            "You are a senior SRE analysing operational logs and a Slack incident thread for "
            "a warehouse robotics team.\n"
            "Produce a detailed combined incident summary with these exact sections:\n\n"
            "## What Happened\n"
            "## Evidence from Logs\n"
            "## Evidence from Slack Thread\n"
            "## Root Cause\n"
            "## Resolution & Status\n"
            "## Action Items\n\n"
            "IMPORTANT: Do NOT mention or reference any user names, display names, or Slack handles. "
            "Write in an impersonal, role-neutral style. "
            "Quote exact log lines where useful. "
            "If uncertain, explicitly state uncertainty."
        )
    else:
        system = (
            "You are a senior SRE analysing operational logs for a warehouse robotics team.\n"
            "Produce a detailed log analysis with these exact sections:\n\n"
            "## What Happened\n"
            "## Key Log Events (with timestamps)\n"
            "## Error Pattern Analysis\n"
            "## Root Cause Assessment\n"
            "## Recommended Next Steps\n\n"
            "Quote exact log lines where useful including timestamps. "
            "If uncertain, explicitly state uncertainty."
        )

    user_content = f"ISSUE DESCRIPTION:\n{req.issue_description}\n\n"

    if req.site_id:
        user_content += f"SITE: {req.site_id}\n"
    if req.env:
        user_content += f"ENVIRONMENT: {req.env}\n"
    if req.hostname:
        user_content += f"HOSTNAME: {req.hostname}\n"
    if req.deployment:
        user_content += f"DEPLOYMENT: {req.deployment}\n"
    if req.time_from or req.time_to:
        user_content += f"TIME RANGE: {req.time_from or '?'} → {req.time_to or '?'}\n\n"

    if log_lines:
        user_content += f"OPERATIONAL LOGS ({len(log_lines)} entries)\n"
        user_content += "-" * 60 + "\n"
        user_content += "\n".join(log_lines[:1500])
        user_content += "\n\n"

    if slack_text:
        user_content += f"SLACK THREAD ({slack_msg_count} messages)\n"
        user_content += "-" * 60 + "\n"
        user_content += slack_text[:8000]
        user_content += "\n"

    if not log_lines and not slack_text:
        user_content += "[No logs or Slack data provided — analyse based on the issue description only.]\n"

    chat_messages: list[dict] = [{"role": "system", "content": system}]
    user_msg: dict = {"role": "user", "content": user_content}
    if has_images:
        user_msg["images"] = slack_images[:MAX_IMAGES]
    chat_messages.append(user_msg)

    # ── Call Ollama ─────────────────────────────────────────────────────────
    try:
        payload = {
            "model": model,
            "messages": chat_messages,
            "stream": False,
            "options": {"temperature": 0.2, "num_ctx": 32768},
        }
        resp = http_requests.post(
            f"{ollama_host}/api/chat", json=payload, timeout=240
        )
        resp.raise_for_status()
        summary = (resp.json().get("message") or {}).get("content", "").strip()
    except http_requests.exceptions.ConnectionError:
        raise HTTPException(503, f"Ollama is not running at {ollama_host}.")
    except Exception as exc:
        logger.error("Ollama chat error: %s", exc, exc_info=True)
        raise HTTPException(500, f"LLM analysis failed: {exc}")

    return AnalyseResponse(
        model_used=model,
        has_images=has_images,
        slack_messages=slack_msg_count,
        log_count=len(req.logs),
        summary=summary,
    )
