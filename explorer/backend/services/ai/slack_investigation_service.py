"""Slack thread reader + local Ollama summarization service."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from datetime import datetime, timezone
import io
import re
from typing import Dict, List, Tuple

import requests
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from core.config import resolve_slack_bot_token, settings
from core.logging import get_logger
from schemas.slack_investigation import (
    SlackLLMStatusResponse,
    SlackThreadAttachment,
    SlackThreadInvestigationRequest,
    SlackThreadInvestigationResponse,
    SlackThreadMessage,
)

logger = get_logger(__name__)

MAX_FILE_CHARS = 12_000
MAX_IMAGES = 4


@dataclass
class ParsedSlackThreadRef:
    workspace: str
    channel_id: str
    thread_ts: str


def _p_timestamp_to_ts(value: str) -> str:
    if not value.isdigit() or len(value) < 7:
        raise ValueError("Invalid Slack message timestamp format.")
    return f"{value[:-6]}.{value[-6:]}"


def parse_slack_thread_url(url: str) -> ParsedSlackThreadRef:
    """Parse Slack permalink: .../archives/<CHANNEL>/p<TIMESTAMP>."""
    match = re.search(r"https://([^.]+)\.slack\.com/archives/([A-Z0-9]+)/p(\d+)", url)
    if not match:
        raise ValueError("Slack thread URL is invalid. Expected .../archives/<CHANNEL>/p<TIMESTAMP>.")
    workspace, channel_id, p_ts = match.groups()
    return ParsedSlackThreadRef(
        workspace=workspace,
        channel_id=channel_id,
        thread_ts=_p_timestamp_to_ts(p_ts),
    )


def _extract_log_blocks(text: str) -> Tuple[str, List[str]]:
    blocks: List[str] = []
    triple = re.findall(r"```(.*?)```", text or "", re.DOTALL)
    blocks.extend(b.strip() for b in triple if b.strip())
    clean = re.sub(r"```.*?```", "[log block]", text or "", flags=re.DOTALL)

    single = re.findall(r"`([^`]{40,})`", clean)
    blocks.extend(b.strip() for b in single if b.strip())
    clean = re.sub(r"`[^`]{40,}`", "[log snippet]", clean)
    return clean.strip(), blocks


def _as_bullets(text: str) -> List[str]:
    out: List[str] = []
    for line in text.splitlines():
        cleaned = re.sub(r"^[-*\d.\s]+", "", line).strip()
        if cleaned:
            out.append(cleaned)
    return out


def _split_markdown_sections(markdown_text: str) -> Dict[str, str]:
    sections: Dict[str, List[str]] = {}
    current = ""
    for line in markdown_text.splitlines():
        heading_match = re.match(r"^##\s+(.+?)\s*$", line.strip())
        if heading_match:
            current = heading_match.group(1).strip().lower()
            sections.setdefault(current, [])
            continue
        if current:
            sections[current].append(line)
    return {k: "\n".join(v).strip() for k, v in sections.items()}


class SlackInvestigationService:
    def __init__(self, _llm_service=None) -> None:
        self.client: WebClient | None = None
        self._client_token = ""
        self._user_cache: Dict[str, str] = {}
        self.ollama_host = settings.ollama_host.rstrip("/")
        self.vision_model = settings.ollama_vision_model
        self.text_model = settings.ollama_text_model

    def _slack_token(self) -> str:
        token = resolve_slack_bot_token()
        if token:
            return token
        raise RuntimeError(
            "SLACK_BOT_TOKEN is not configured on the backend. "
            "Set SLACK_BOT_TOKEN (or SLACK_TOKEN) in backend/.env. "
            "If using Docker, recreate backend: docker compose up -d --force-recreate backend"
        )

    def _require_client(self) -> WebClient:
        token = self._slack_token()
        if self.client is None or self._client_token != token:
            self.client = WebClient(token=token)
            self._client_token = token
        return self.client

    def _slack_headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self._slack_token()}"}

    def _ollama_ping(self) -> bool:
        try:
            resp = requests.get(f"{self.ollama_host}/api/tags", timeout=3)
            return resp.status_code == 200
        except Exception:
            return False

    def _ollama_models(self) -> List[str]:
        try:
            resp = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
            resp.raise_for_status()
            return [m.get("name", "") for m in resp.json().get("models", [])]
        except Exception:
            return []

    def llm_status(self) -> SlackLLMStatusResponse:
        if not self._ollama_ping():
            return SlackLLMStatusResponse(
                status="offline",
                vision_model=self.vision_model,
                text_model=self.text_model,
                vision_ready=False,
                text_ready=False,
                installed=[],
                fix=f"Ollama is not running at {self.ollama_host}. Run: ollama serve",
            )

        installed = self._ollama_models()
        vision_prefix = self.vision_model.split(":", 1)[0]
        text_prefix = self.text_model.split(":", 1)[0]
        return SlackLLMStatusResponse(
            status="online",
            vision_model=self.vision_model,
            text_model=self.text_model,
            vision_ready=any(vision_prefix in name for name in installed),
            text_ready=any(text_prefix in name for name in installed),
            installed=installed,
        )

    def _resolve_user(self, user_id: str | None) -> str:
        if not user_id:
            return "unknown"
        if user_id in self._user_cache:
            return self._user_cache[user_id]
        if user_id.startswith("B"):
            self._user_cache[user_id] = f"bot:{user_id}"
            return self._user_cache[user_id]

        client = self._require_client()
        try:
            data = client.users_info(user=user_id)
            user = data.get("user", {})
            profile = user.get("profile", {})
            display = profile.get("display_name") or profile.get("real_name") or user.get("name") or user_id
        except SlackApiError:
            display = user_id
        self._user_cache[user_id] = display
        return display

    def _download_file(self, url: str) -> bytes:
        resp = requests.get(url, headers=self._slack_headers(), timeout=30, stream=True)
        if resp.status_code != 200:
            raise ValueError(f"HTTP {resp.status_code}")
        return resp.content

    def _proc_image(self, data: bytes, filename: str) -> SlackThreadAttachment:
        return SlackThreadAttachment(
            filename=filename,
            filetype="image",
            extracted=f"[Image: {filename}]",
            b64_image=base64.b64encode(data).decode(),
        )

    def _proc_pdf(self, data: bytes, filename: str) -> SlackThreadAttachment:
        try:
            import pdfplumber

            pages: List[str] = []
            with pdfplumber.open(io.BytesIO(data)) as pdf:
                for idx, page in enumerate(pdf.pages, 1):
                    text = (page.extract_text() or "").strip()
                    if text:
                        pages.append(f"[Page {idx}]\n{text}")
            content = "\n\n".join(pages) or "[No extractable text in PDF]"
            return SlackThreadAttachment(
                filename=filename,
                filetype="pdf",
                extracted=content[:MAX_FILE_CHARS],
            )
        except Exception as exc:
            logger.warning("PDF extraction failed for %s: %s", filename, exc)
            return SlackThreadAttachment(filename=filename, filetype="pdf", extracted=f"[PDF read error: {exc}]")

    def _proc_pptx(self, data: bytes, filename: str) -> SlackThreadAttachment:
        try:
            from pptx import Presentation

            prs = Presentation(io.BytesIO(data))
            slides: List[str] = []
            for idx, slide in enumerate(prs.slides, 1):
                lines: List[str] = []
                for shape in slide.shapes:
                    if getattr(shape, "has_text_frame", False):
                        for para in shape.text_frame.paragraphs:
                            line = " ".join(run.text for run in para.runs).strip()
                            if line:
                                lines.append(line)
                if lines:
                    slides.append(f"[Slide {idx}]\n" + "\n".join(lines))
            content = "\n\n".join(slides) or "[No text in presentation]"
            return SlackThreadAttachment(
                filename=filename,
                filetype="pptx",
                extracted=content[:MAX_FILE_CHARS],
            )
        except Exception as exc:
            logger.warning("PPTX extraction failed for %s: %s", filename, exc)
            return SlackThreadAttachment(filename=filename, filetype="pptx", extracted=f"[PPTX read error: {exc}]")

    def _proc_text(self, data: bytes, filename: str, filetype: str = "text") -> SlackThreadAttachment:
        content = data.decode("utf-8", errors="replace")
        return SlackThreadAttachment(
            filename=filename,
            filetype=filetype,
            extracted=content[:MAX_FILE_CHARS],
        )

    def _process_attachment(self, slack_file: Dict) -> SlackThreadAttachment | None:
        url = slack_file.get("url_private_download") or slack_file.get("url_private")
        if not url:
            return None

        filename = slack_file.get("name", "unknown")
        mimetype = slack_file.get("mimetype", "")
        lower_name = filename.lower()

        try:
            data = self._download_file(url)
        except Exception as exc:
            logger.warning("Attachment download failed for %s: %s", filename, exc)
            return SlackThreadAttachment(
                filename=filename,
                filetype="unknown",
                extracted=f"[Download failed: {exc}]",
            )

        if mimetype.startswith("image/"):
            return self._proc_image(data, filename)
        if mimetype == "application/pdf" or lower_name.endswith(".pdf"):
            return self._proc_pdf(data, filename)
        if "presentation" in mimetype or lower_name.endswith((".ppt", ".pptx")):
            return self._proc_pptx(data, filename)
        if mimetype.startswith("text/") or lower_name.endswith((".txt", ".log", ".yaml", ".yml", ".json", ".xml", ".csv")):
            filetype = "log" if lower_name.endswith(".log") else "text"
            return self._proc_text(data, filename, filetype)

        return SlackThreadAttachment(
            filename=filename,
            filetype="unknown",
            extracted=f"[Unsupported file type: {mimetype}]",
        )

    def _ensure_in_channel(self, client: WebClient, channel_id: str) -> None:
        """Join a public channel if the bot is not already a member."""
        try:
            channel_info = client.conversations_info(channel=channel_id)
            is_member = (channel_info.get("channel") or {}).get("is_member", False)
            if is_member:
                return
        except SlackApiError:
            pass  # proceed to join attempt

        try:
            client.conversations_join(channel=channel_id)
            logger.info("Auto-joined channel %s to fetch thread.", channel_id)
        except SlackApiError as exc:
            join_err = exc.response.get("error", "unknown")
            if join_err == "method_not_supported_for_channel_type":
                raise ValueError(
                    f"The bot is not in channel '{channel_id}' (private channel). "
                    "Invite the bot manually: /invite @<bot_name>"
                ) from exc
            if join_err == "is_archived":
                raise ValueError(
                    f"Channel '{channel_id}' is archived and cannot be joined."
                ) from exc
            raise ValueError(
                f"Could not join channel '{channel_id}' ({join_err}). "
                "Invite the bot to the channel first."
            ) from exc

    def _fetch_thread_messages(
        self,
        ref: ParsedSlackThreadRef,
        include_bots: bool,
        max_messages: int,
    ) -> Tuple[List[SlackThreadMessage], List[SlackThreadAttachment]]:
        client = self._require_client()
        cursor = None
        messages: List[SlackThreadMessage] = []
        attachments: List[SlackThreadAttachment] = []
        _auto_joined = False

        while len(messages) < max_messages:
            limit = min(200, max_messages - len(messages))
            try:
                resp = client.conversations_replies(
                    channel=ref.channel_id,
                    ts=ref.thread_ts,
                    limit=limit,
                    cursor=cursor,
                    inclusive=True,
                )
            except SlackApiError as exc:
                err = exc.response.get("error", "unknown_error")
                if err == "not_in_channel" and not _auto_joined:
                    self._ensure_in_channel(client, ref.channel_id)
                    _auto_joined = True
                    continue  # retry with the joined channel
                if err == "not_in_channel":
                    raise ValueError(
                        f"Bot is not in channel '{ref.channel_id}'. "
                        "Invite the bot to the channel first."
                    ) from exc
                if err == "channel_not_found":
                    raise ValueError("Channel not found. Check the Slack thread URL.") from exc
                if err == "thread_not_found":
                    raise ValueError("Thread not found. Check the Slack thread URL.") from exc
                if err == "missing_scope":
                    raise RuntimeError(
                        "Missing Slack API scopes. Ensure the bot token has "
                        "'channels:history', 'groups:history', and 'channels:join' scopes."
                    ) from exc
                if err == "invalid_auth":
                    raise RuntimeError(
                        "Invalid Slack token. Check SLACK_BOT_TOKEN in backend/.env and recreate the backend container."
                    ) from exc
                raise RuntimeError(f"Slack API error: {err}") from exc

            for raw in resp.get("messages", []):
                if not include_bots and (raw.get("subtype") == "bot_message" or raw.get("bot_id")):
                    continue

                raw_text = raw.get("text") or ""
                clean_text, log_blocks = _extract_log_blocks(raw_text)

                message_attachments: List[SlackThreadAttachment] = []
                for slack_file in raw.get("files", []) or []:
                    item = self._process_attachment(slack_file)
                    if item:
                        message_attachments.append(item)
                        attachments.append(item)

                if not clean_text and not log_blocks and not message_attachments:
                    continue

                ts = str(raw.get("ts", ""))
                try:
                    dt = datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
                except Exception:
                    dt = ts

                user_id = raw.get("user") or raw.get("bot_id")
                messages.append(
                    SlackThreadMessage(
                        ts=ts,
                        datetime=dt,
                        user=self._resolve_user(user_id),
                        text=clean_text,
                        log_blocks=log_blocks,
                        attachments=message_attachments,
                    )
                )
                if len(messages) >= max_messages:
                    break

            cursor = (resp.get("response_metadata") or {}).get("next_cursor") or None
            if not cursor:
                break

        return messages, attachments

    def _ollama_chat(self, chat_messages: List[Dict], model: str) -> str:
        payload = {
            "model": model,
            "messages": chat_messages,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_ctx": 32768,
            },
        }
        try:
            resp = requests.post(f"{self.ollama_host}/api/chat", json=payload, timeout=240)
            resp.raise_for_status()
            return (resp.json().get("message") or {}).get("content", "").strip()
        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(f"Ollama is not running at {self.ollama_host}. Start it with: ollama serve") from exc
        except Exception as exc:
            logger.error("Ollama chat error (model=%s): %s", model, exc)
            raise RuntimeError(f"Local LLM error: {exc}") from exc

    def _generate_summary(
        self,
        req: SlackThreadInvestigationRequest,
        messages: List[SlackThreadMessage],
        attachments: List[SlackThreadAttachment],
    ) -> Tuple[str, str, bool]:
        has_images = any(a.filetype == "image" and a.b64_image for a in attachments)
        model = self.vision_model if has_images else self.text_model

        # Honour per-request model override
        if req.model_override:
            model = req.model_override
            has_images = False  # skip image embedding when using a custom/text model

        # Fall back to text model when the chosen model is not installed
        installed = self._ollama_models()
        model_prefix = model.split(":", 1)[0]
        if not any(model_prefix in name for name in installed):
            if model == self.vision_model and any(
                self.text_model.split(":", 1)[0] in name for name in installed
            ):
                logger.warning(
                    "Vision model %s not installed, falling back to text model %s",
                    self.vision_model, self.text_model,
                )
                model = self.text_model
                has_images = False
            else:
                raise RuntimeError(
                    f"Model '{model}' is not installed in Ollama. "
                    f"Pull it with: ollama pull {model}"
                )

        system = (
            "You are a senior SRE analyzing a Slack incident thread for a warehouse robotics team. "
            "You will receive full thread conversation, extracted file contents, and pasted log blocks. "
            "Produce a detailed summary with these exact sections:\n\n"
            "## The Issue\n"
            "## The Investigation\n"
            "## Key Evidence\n"
            "## Root Cause\n"
            "## Resolution & Status\n"
            "## Action Items\n\n"
            "IMPORTANT: Do NOT mention or reference any user names, display names, or Slack handles anywhere in your "
            "summary. Write in an impersonal, role-neutral style (e.g. 'the team', 'the engineer', 'it was noted', "
            "'the investigation found'). "
            "Quote exact log lines where useful. "
            "If uncertain, explicitly state uncertainty instead of guessing."
        )

        thread_lines: List[str] = []
        for msg in messages:
            line = f"[{msg.datetime}] {msg.user}: {msg.text}"
            for idx, block in enumerate(msg.log_blocks, 1):
                line += f"\n  [Log block {idx}]\n{block[:2000]}"
            for attachment in msg.attachments:
                line += f"\n  [Attachment: {attachment.filename} ({attachment.filetype})]"
            thread_lines.append(line)

        attachment_sections: List[str] = []
        for attachment in attachments:
            if attachment.filetype == "image":
                attachment_sections.append(f"=== IMAGE: {attachment.filename} ===\n[See visual content above]")
            else:
                attachment_sections.append(
                    f"=== {attachment.filetype.upper()}: {attachment.filename} ===\n{attachment.extracted}"
                )

        prompt = (
            f"SLACK THREAD ({len(messages)} messages)\n"
            f"Site: {req.site_id or 'N/A'}\n"
            f"Description: {req.description}\n\n"
            + "\n\n".join(thread_lines)
        )
        if attachment_sections:
            prompt += "\n\n" + ("-" * 60) + "\nATTACHMENTS\n" + ("-" * 60) + "\n\n"
            prompt += "\n\n".join(attachment_sections)
        if req.custom_prompt:
            prompt += f"\n\nSPECIAL FOCUS: {req.custom_prompt}"

        chat: List[Dict] = [{"role": "system", "content": system}]
        user_message: Dict = {"role": "user", "content": prompt}
        if has_images:
            user_message["images"] = [
                a.b64_image for a in attachments if a.filetype == "image" and a.b64_image
            ][:MAX_IMAGES]
        chat.append(user_message)

        summary = self._ollama_chat(chat, model)
        return summary, model, has_images

    def _infer_risk(self, summary: str) -> str:
        lowered = summary.lower()
        if any(token in lowered for token in ["sev1", "critical", "safety", "production down", "data loss"]):
            return "high"
        if any(token in lowered for token in ["degraded", "intermittent", "warning", "retry"]):
            return "medium"
        return "low"

    def investigate(self, req: SlackThreadInvestigationRequest) -> SlackThreadInvestigationResponse:
        if not self._ollama_ping():
            raise RuntimeError(
                f"Ollama is not running at {self.ollama_host}. Run: ollama serve"
            )

        ref = parse_slack_thread_url(req.slack_thread_url)
        messages, attachments = self._fetch_thread_messages(ref, req.include_bots, req.max_messages)
        if not messages:
            raise RuntimeError("Thread is empty or inaccessible with current token/scopes.")

        summary, model_used, has_images = self._generate_summary(req, messages, attachments)
        sections = _split_markdown_sections(summary)

        issue = sections.get("the issue", "").strip()
        root_cause = sections.get("root cause", "").strip()
        resolution = sections.get("resolution & status", "").strip()
        key_evidence = sections.get("key evidence", "").strip()
        action_items = sections.get("action items", "").strip()

        thread_summary = "\n\n".join(
            block for block in [issue, root_cause, resolution] if block
        ).strip() or summary[:1200]

        findings = _as_bullets(key_evidence) or [
            "Review raw analysis for detailed evidence extracted from messages and files."
        ]
        actions = _as_bullets(action_items) or [
            "No explicit action items detected; assign owners to follow up on unresolved findings."
        ]

        participants = sorted({msg.user for msg in messages if msg.user})

        return SlackThreadInvestigationResponse(
            workspace=ref.workspace,
            channel_id=ref.channel_id,
            thread_ts=ref.thread_ts,
            message_count=len(messages),
            attachment_count=len(attachments),
            has_images=has_images,
            model_used=model_used,
            participants=participants,
            thread_summary=thread_summary,
            key_findings=findings,
            recommended_actions=actions,
            risk_level=self._infer_risk(summary),
            timeline=messages,
            attachments=attachments,
            raw_analysis=summary,
        )
