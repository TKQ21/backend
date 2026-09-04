"""API-key authentication + in-process rate limiting.

Env:
  ML_API_KEYS       comma separated keys. If unset, auth is DISABLED (open API).
  ML_RATE_LIMIT     requests per window per client (default 60)
  ML_RATE_WINDOW    window seconds (default 60)
  ML_TRAIN_LIMIT    training jobs per hour per client (default 5)
"""
from __future__ import annotations

import os
import threading
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request

API_KEYS = {k.strip() for k in os.getenv("ML_API_KEYS", "").split(",") if k.strip()}
AUTH_ENABLED = bool(API_KEYS)
RATE_LIMIT = int(os.getenv("ML_RATE_LIMIT", "60"))
RATE_WINDOW = int(os.getenv("ML_RATE_WINDOW", "60"))
TRAIN_LIMIT = int(os.getenv("ML_TRAIN_LIMIT", "5"))
TRAIN_WINDOW = 3600

_hits: dict[str, deque[float]] = defaultdict(deque)
_trains: dict[str, deque[float]] = defaultdict(deque)
_lock = threading.Lock()


def client_id(req: Request) -> str:
    key = req.headers.get("x-api-key")
    if key:
        return f"key:{key[:8]}"
    fwd = req.headers.get("x-forwarded-for", "")
    ip = fwd.split(",")[0].strip() or (req.client.host if req.client else "unknown")
    return f"ip:{ip}"


def _hit(bucket: dict[str, deque[float]], cid: str, limit: int, window: int) -> int:
    """Returns remaining allowance, raises 429 when exhausted."""
    now = time.time()
    with _lock:
        dq = bucket[cid]
        while dq and now - dq[0] > window:
            dq.popleft()
        if len(dq) >= limit:
            retry = int(window - (now - dq[0])) + 1
            raise HTTPException(
                429,
                detail=f"Rate limit exceeded ({limit} per {window}s). Retry in {retry}s.",
                headers={"Retry-After": str(retry)},
            )
        dq.append(now)
        return limit - len(dq)


def authenticate(req: Request) -> str:
    """FastAPI dependency: verifies the API key and applies the general rate limit."""
    if AUTH_ENABLED:
        key = req.headers.get("x-api-key") or ""
        auth = req.headers.get("authorization", "")
        if not key and auth.lower().startswith("bearer "):
            key = auth[7:].strip()
        if key not in API_KEYS:
            raise HTTPException(401, "Invalid or missing API key (send it as x-api-key).")
    cid = client_id(req)
    _hit(_hits, cid, RATE_LIMIT, RATE_WINDOW)
    return cid


def limit_training(cid: str) -> None:
    _hit(_trains, cid, TRAIN_LIMIT, TRAIN_WINDOW)


def config() -> dict[str, object]:
    return {
        "authEnabled": AUTH_ENABLED,
        "rateLimit": RATE_LIMIT,
        "rateWindowSeconds": RATE_WINDOW,
        "trainingLimitPerHour": TRAIN_LIMIT,
    }
