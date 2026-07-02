"""
Slack Bot integration for RAG App.

Listens for @mentions or direct messages in configured Slack channels,
runs the RAG pipeline (SFDC + PDF), and posts the answer back.

Required env vars:
  SLACK_BOT_TOKEN      - xoxb-... bot token
  SLACK_SIGNING_SECRET - from Slack App > Basic Information
  SLACK_CHANNELS       - comma-separated channel names to listen in
                         e.g. "aie-eng-helpdesk,hpe-private-cloud-ai"
  SLACK_SOURCE_MODE    - "sfdc", "pdf", or "both" (default: "both")
  SLACK_PRODUCT_LINE   - product line filter for SFDC (default: "")
"""

import hashlib
import hmac
import logging
import time

from app.config import Config

logger = logging.getLogger(__name__)


def verify_slack_signature(request) -> bool:
    """Verify the request came from Slack using signing secret."""
    signing_secret = Config.SLACK_SIGNING_SECRET
    if not signing_secret:
        logger.warning("[Slack] No signing secret configured — skipping verification")
        return True

    timestamp = request.headers.get("X-Slack-Request-Timestamp", "")
    slack_signature = request.headers.get("X-Slack-Signature", "")

    # Reject stale requests (replay attack prevention)
    try:
        if abs(time.time() - float(timestamp)) > 300:
            logger.warning("[Slack] Stale request timestamp")
            return False
    except ValueError:
        return False

    sig_basestring = f"v0:{timestamp}:{request.get_data(as_text=True)}"
    computed = "v0=" + hmac.new(
        signing_secret.encode("utf-8"),
        sig_basestring.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    return hmac.compare_digest(computed, slack_signature)


def get_allowed_channels() -> set:
    """Return the set of channel names the bot is allowed to respond in."""
    raw = Config.SLACK_CHANNELS or ""
    return {ch.strip().lstrip("#") for ch in raw.split(",") if ch.strip()}


def handle_slack_event(payload: dict, app) -> None:
    """
    Process a Slack event payload. Runs the RAG query and posts the answer
    back to the channel. Called from the route handler.

    :param payload: Parsed JSON body from Slack Events API
    :param app:     Flask app instance (for app_context)
    """
    event = payload.get("event", {})
    event_type = event.get("type", "")

    # Only handle app_mention and regular messages (not bot messages)
    if event_type not in ("app_mention", "message"):
        return
    if event.get("bot_id") or event.get("subtype"):
        return  # ignore bot messages and message edits/deletes

    channel = event.get("channel", "")
    channel_name = event.get("channel_name") or _resolve_channel_name(channel, app)
    text = (event.get("text") or "").strip()
    thread_ts = event.get("thread_ts") or event.get("ts")

    # Strip bot mention prefix (<@UXXXXXX> ...) from message text
    import re
    text = re.sub(r"<@[A-Z0-9]+>", "", text).strip()

    if not text:
        return

    # Channel filter
    allowed = get_allowed_channels()
    if allowed and channel_name not in allowed:
        logger.debug("[Slack] Ignoring message in channel '%s' (not in allowed list)", channel_name)
        return

    logger.info("[Slack] Question in #%s: %s", channel_name, text)

    # Run RAG query
    source_mode = Config.SLACK_SOURCE_MODE
    product_line = Config.SLACK_PRODUCT_LINE
    answer = _run_rag_query(text, source_mode, product_line, app)

    # Post answer back to Slack
    _post_message(channel, answer, thread_ts, app)


def _run_rag_query(question: str, source_mode: str, product_line: str, app) -> str:
    """Run the RAG pipeline and return an answer string."""
    try:
        import requests as req_lib
        # Call our own /ask endpoint internally
        with app.test_request_context():
            from flask import current_app as capp
            from app.utils import retrieve_relevant_chunks_pg, llm_answer
            from app.config import Config as Cfg
            from app.routes.rag_routes import _format_sfdc_articles

            chunks = []
            sfdc_articles = []

            if source_mode in ("pdf", "both"):
                with app.app_context():
                    chunks.extend(retrieve_relevant_chunks_pg(question, top_k=Cfg.MAX_RESULTS))

            sfdc_client = app.sfdc_client if hasattr(app, "sfdc_client") else None
            if source_mode in ("sfdc", "both") and sfdc_client:
                try:
                    from app.services.sfdc_knowledge import fetch_articles_for_query, articles_to_chunks
                    sfdc_articles = fetch_articles_for_query(
                        sfdc_client, question,
                        limit=Cfg.SFDC_SEARCH_LIMIT,
                        product_line=product_line,
                    )
                    if sfdc_articles:
                        chunks.extend([c["text"] for c in articles_to_chunks(sfdc_articles)])
                except Exception as e:
                    logger.error("[Slack RAG] SFDC fetch failed: %s", e)

            if source_mode == "sfdc":
                return _format_sfdc_articles(sfdc_articles, question)

            with app.app_context():
                return llm_answer("".join(chunks), question)

    except Exception as e:
        logger.error("[Slack RAG] Query failed: %s", e)
        return f"Sorry, I encountered an error processing your question: {e}"


def _post_message(channel: str, text: str, thread_ts: str, app) -> None:
    """Post a message to a Slack channel using the Bot Token."""
    import requests

    token = Config.SLACK_BOT_TOKEN
    if not token:
        logger.error("[Slack] No SLACK_BOT_TOKEN configured")
        return

    # Slack message limit is 3000 chars per block
    max_len = 2900
    if len(text) > max_len:
        text = text[:max_len] + "\n...(truncated)"

    payload = {
        "channel": channel,
        "text": text,
        "thread_ts": thread_ts,
        "mrkdwn": True,
    }

    try:
        resp = requests.post(
            "https://slack.com/api/chat.postMessage",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json=payload,
            timeout=10,
        )
        result = resp.json()
        if not result.get("ok"):
            logger.error("[Slack] chat.postMessage failed: %s", result.get("error"))
        else:
            logger.info("[Slack] Reply posted to channel %s", channel)
    except Exception as e:
        logger.error("[Slack] Failed to post message: %s", e)


def _resolve_channel_name(channel_id: str, app) -> str:
    """Resolve a Slack channel ID to its name via API."""
    import requests

    token = Config.SLACK_BOT_TOKEN
    if not token or not channel_id:
        return channel_id

    try:
        resp = requests.get(
            "https://slack.com/api/conversations.info",
            headers={"Authorization": f"Bearer {token}"},
            params={"channel": channel_id},
            timeout=5,
        )
        data = resp.json()
        if data.get("ok"):
            return data.get("channel", {}).get("name", channel_id)
    except Exception:
        pass
    return channel_id
