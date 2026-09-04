"""Structured logging, error alerting and service metrics."""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.request

import db

LOG = logging.getLogger("electripredict")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

ALERT_WEBHOOK_URL = os.getenv("ALERT_WEBHOOK_URL", "").strip()
STARTED_AT = time.time()


def _post_alert(title: str, detail: str) -> None:
    if not ALERT_WEBHOOK_URL:
        return
    body = json.dumps({"text": f":rotating_light: {title}\n{detail[:1500]}"}).encode()
    req = urllib.request.Request(
        ALERT_WEBHOOK_URL, data=body, headers={"Content-Type": "application/json"}
    )
    try:
        urllib.request.urlopen(req, timeout=10).close()
    except Exception as exc:  # alerting must never break the request path
        LOG.warning("alert webhook failed: %s", exc)


def event(level: str, name: str, detail: str = "") -> None:
    LOG.log(logging.ERROR if level == "error" else logging.INFO, "%s %s", name, detail)
    try:
        db.log_event(level, name, detail)
    except Exception as exc:
        LOG.warning("event persist failed: %s", exc)
    if level == "error":
        threading.Thread(target=_post_alert, args=(name, detail), daemon=True).start()


def snapshot() -> dict[str, object]:
    c = db.counts()
    total = max(1, c["succeeded"] + c["failed"])
    return {
        "uptimeSeconds": round(time.time() - STARTED_AT, 1),
        "alerting": bool(ALERT_WEBHOOK_URL),
        "counts": c,
        "successRate": round(c["succeeded"] / total, 4),
        "recentEvents": db.list_events(30),
    }
