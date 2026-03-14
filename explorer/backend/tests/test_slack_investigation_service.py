"""Unit tests for Slack investigation parsing and model-selection helpers."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from schemas.slack_investigation import (
    SlackThreadAttachment,
    SlackThreadInvestigationRequest,
    SlackThreadMessage,
)
from services.ai.slack_investigation_service import (
    SlackInvestigationService,
    _extract_log_blocks,
    parse_slack_thread_url,
)


def test_parse_slack_thread_url_success() -> None:
    ref = parse_slack_thread_url("https://example.slack.com/archives/C123ABC45/p1772691175223000")
    assert ref.workspace == "example"
    assert ref.channel_id == "C123ABC45"
    assert ref.thread_ts == "1772691175.223000"


def test_parse_slack_thread_url_rejects_invalid_url() -> None:
    with pytest.raises(ValueError):
        parse_slack_thread_url("https://example.slack.com/archives/C123ABC45")


def test_extract_log_blocks_from_triple_and_inline() -> None:
    clean, blocks = _extract_log_blocks(
        "Issue observed.```ERROR stack trace line 1\nline 2```and `inline long log payload with more than forty chars`"
    )
    assert "[log block]" in clean
    assert "[log snippet]" in clean
    assert len(blocks) == 2
    assert "ERROR stack trace" in blocks[0]


def test_generate_summary_selects_vision_model_when_images_present(monkeypatch) -> None:
    svc = SlackInvestigationService()

    req = SlackThreadInvestigationRequest(
        slack_thread_url="https://example.slack.com/archives/C123ABC45/p1772691175223000",
        description="Robot stopped near dock",
        max_messages=200,
    )
    messages = [
        SlackThreadMessage(
            ts="1772691175.223000",
            datetime="2026-03-13 10:00 UTC",
            user="alice",
            text="Robot fault observed",
        )
    ]
    attachments = [
        SlackThreadAttachment(
            filename="fault.png",
            filetype="image",
            extracted="[Image: fault.png]",
            b64_image="ZmFrZQ==",
        )
    ]

    monkeypatch.setattr(svc, "_ollama_chat", lambda _messages, _model: "## The Issue\nX")
    monkeypatch.setattr(svc, "_ollama_models", lambda: ["qwen2.5:7b", "llama3.2-vision:11b"])

    _summary, model, has_images = svc._generate_summary(req, messages, attachments)
    assert has_images is True
    assert model == svc.vision_model


def test_slack_headers_accepts_alias_token(monkeypatch) -> None:
    monkeypatch.setenv("SLACK_BOT_TOKEN", "")
    monkeypatch.setenv("SLACK_TOKEN", '"xoxb-alias-token"')

    svc = SlackInvestigationService()
    headers = svc._slack_headers()

    assert headers["Authorization"] == "Bearer xoxb-alias-token"


# ── _ensure_in_channel ─────────────────────────────────────────────────────────

class _FakeResponse:
    """Minimal fake for SlackResponse used in tests."""
    def __init__(self, data: dict) -> None:
        self._data = data

    def get(self, key, default=None):
        return self._data.get(key, default)


def _make_slack_api_error(error_code: str):
    from slack_sdk.errors import SlackApiError
    resp = _FakeResponse({"ok": False, "error": error_code})
    return SlackApiError(message=error_code, response=resp)


def test_ensure_in_channel_skips_join_when_already_member(monkeypatch) -> None:
    svc = SlackInvestigationService()
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-fake")

    join_called = []

    class FakeClient:
        def conversations_info(self, channel):
            return _FakeResponse({"channel": {"is_member": True}})

        def conversations_join(self, channel):
            join_called.append(channel)

    svc._ensure_in_channel(FakeClient(), "C123")  # type: ignore
    assert not join_called, "Should not call join when already a member"


def test_ensure_in_channel_joins_public_channel_when_not_member(monkeypatch) -> None:
    svc = SlackInvestigationService()
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-fake")

    join_called = []

    class FakeClient:
        def conversations_info(self, channel):
            return _FakeResponse({"channel": {"is_member": False}})

        def conversations_join(self, channel):
            join_called.append(channel)

    svc._ensure_in_channel(FakeClient(), "C456")  # type: ignore
    assert join_called == ["C456"]


def test_ensure_in_channel_raises_value_error_for_private_channel(monkeypatch) -> None:
    svc = SlackInvestigationService()
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-fake")

    class FakeClient:
        def conversations_info(self, channel):
            return _FakeResponse({"channel": {"is_member": False}})

        def conversations_join(self, channel):
            raise _make_slack_api_error("method_not_supported_for_channel_type")

    with pytest.raises(ValueError, match="private channel"):
        svc._ensure_in_channel(FakeClient(), "G789")  # type: ignore


def test_fetch_thread_messages_auto_joins_and_retries(monkeypatch) -> None:
    """When conversations_replies returns not_in_channel, the service should
    auto-join and retry successfully."""
    import os
    from services.ai.slack_investigation_service import ParsedSlackThreadRef

    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-fake")
    svc = SlackInvestigationService()

    calls = {"replies": 0, "join_attempted": False}

    class FakeClient:
        def conversations_info(self, channel):
            return _FakeResponse({"channel": {"is_member": False}})

        def conversations_join(self, channel):
            calls["join_attempted"] = True

        def conversations_replies(self, channel, ts, limit, cursor, inclusive):
            calls["replies"] += 1
            if calls["replies"] == 1:
                raise _make_slack_api_error("not_in_channel")
            # Second call succeeds with one message
            return _FakeResponse({
                "messages": [
                    {
                        "ts": "1000000001.000000",
                        "user": "U123",
                        "text": "robot fault detected",
                    }
                ],
                "response_metadata": {},
            })

        def users_info(self, user):
            return _FakeResponse({
                "user": {"profile": {"display_name": "alice"}, "name": "alice"}
            })

    svc.client = FakeClient()  # type: ignore
    svc._client_token = "xoxb-fake"

    ref = ParsedSlackThreadRef(
        workspace="example",
        channel_id="C123",
        thread_ts="1000000001.000000",
    )
    messages, attachments = svc._fetch_thread_messages(ref, include_bots=False, max_messages=50)

    assert calls["join_attempted"] is True
    assert len(messages) == 1
    assert messages[0].text == "robot fault detected"


def test_fetch_thread_messages_raises_for_missing_scope(monkeypatch) -> None:
    import os
    from services.ai.slack_investigation_service import ParsedSlackThreadRef

    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-fake")
    svc = SlackInvestigationService()

    class FakeClient:
        def conversations_replies(self, **kwargs):
            raise _make_slack_api_error("missing_scope")

    svc.client = FakeClient()  # type: ignore
    svc._client_token = "xoxb-fake"

    ref = ParsedSlackThreadRef(workspace="example", channel_id="C123", thread_ts="1.0")
    with pytest.raises(RuntimeError, match="Missing Slack API scopes"):
        svc._fetch_thread_messages(ref, include_bots=False, max_messages=50)


def test_fetch_thread_messages_raises_for_invalid_auth(monkeypatch) -> None:
    import os
    from services.ai.slack_investigation_service import ParsedSlackThreadRef

    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-fake")
    svc = SlackInvestigationService()

    class FakeClient:
        def conversations_replies(self, **kwargs):
            raise _make_slack_api_error("invalid_auth")

    svc.client = FakeClient()  # type: ignore
    svc._client_token = "xoxb-fake"

    ref = ParsedSlackThreadRef(workspace="example", channel_id="C123", thread_ts="1.0")
    with pytest.raises(RuntimeError, match="Invalid Slack token"):
        svc._fetch_thread_messages(ref, include_bots=False, max_messages=50)
