"""
Slack Events API routes.
Endpoint: POST /slack/events
"""

import logging
import threading

from flask import Blueprint, request, jsonify, current_app
from app.services.slack_service import verify_slack_signature, handle_slack_event

logger = logging.getLogger(__name__)

slack_bp = Blueprint("slack", __name__, url_prefix="/slack")


@slack_bp.route("/events", methods=["POST"])
def slack_events():
    """
    Receives Slack Events API payloads.
    Handles:
      - url_verification challenge (Slack setup handshake)
      - app_mention  — bot is @mentioned in a channel
      - message      — message posted in a channel the bot is in
    """
    # 1. Verify Slack signature
    if not verify_slack_signature(request):
        logger.warning("[Slack] Invalid signature — rejected")
        return jsonify({"error": "invalid signature"}), 403

    payload = request.get_json(force=True) or {}

    # 2. URL verification handshake (one-time during Slack App setup)
    if payload.get("type") == "url_verification":
        return jsonify({"challenge": payload.get("challenge")}), 200

    # 3. Event callback — process asynchronously so Slack gets 200 quickly
    if payload.get("type") == "event_callback":
        app = current_app._get_current_object()
        thread = threading.Thread(
            target=handle_slack_event,
            args=(payload, app),
            daemon=True,
        )
        thread.start()

    # Always return 200 immediately to Slack (within 3s timeout)
    return jsonify({"ok": True}), 200


@slack_bp.route("/channels", methods=["GET"])
def list_channels():
    """List configured Slack channels the bot is listening to."""
    from app.services.slack_service import get_allowed_channels
    from app.config import Config

    channels = sorted(get_allowed_channels())
    return jsonify({
        "configured_channels": channels,
        "source_mode": Config.SLACK_SOURCE_MODE,
        "product_line": Config.SLACK_PRODUCT_LINE,
        "bot_configured": bool(Config.SLACK_BOT_TOKEN),
    }), 200
