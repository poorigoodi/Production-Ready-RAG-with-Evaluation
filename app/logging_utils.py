"""Structured logging helpers."""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import contextmanager
from typing import Any, Iterator


def setup_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level.upper())
        return
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    root.addHandler(handler)
    root.setLevel(level.upper())


def new_request_id() -> str:
    return str(uuid.uuid4())


def log_event(
    logger: logging.Logger,
    *,
    request_id: str,
    operation: str,
    duration_ms: float | None = None,
    status: str = "ok",
    error_category: str | None = None,
    **extra: Any,
) -> None:
    payload = {
        "request_id": request_id,
        "operation": operation,
        "duration_ms": None if duration_ms is None else round(duration_ms, 2),
        "status": status,
        "error_category": error_category,
        **extra,
    }
    # Avoid logging full document contents; truncate free-text extras.
    for key in ("text", "content", "question", "answer"):
        if key in payload and isinstance(payload[key], str) and len(payload[key]) > 120:
            payload[key] = payload[key][:120] + "…"
    message = json.dumps(payload, ensure_ascii=False, default=str)
    if status == "ok":
        logger.info(message)
    else:
        logger.error(message)


@contextmanager
def timed_operation(
    logger: logging.Logger,
    *,
    request_id: str,
    operation: str,
    **extra: Any,
) -> Iterator[dict[str, Any]]:
    start = time.perf_counter()
    meta: dict[str, Any] = {"status": "ok", "error_category": None}
    try:
        yield meta
    except Exception as exc:
        meta["status"] = "error"
        meta["error_category"] = type(exc).__name__
        duration_ms = (time.perf_counter() - start) * 1000
        log_event(
            logger,
            request_id=request_id,
            operation=operation,
            duration_ms=duration_ms,
            status="error",
            error_category=meta["error_category"],
            **extra,
        )
        raise
    else:
        duration_ms = (time.perf_counter() - start) * 1000
        meta["duration_ms"] = duration_ms
        log_event(
            logger,
            request_id=request_id,
            operation=operation,
            duration_ms=duration_ms,
            status=meta["status"],
            error_category=meta["error_category"],
            **extra,
        )
