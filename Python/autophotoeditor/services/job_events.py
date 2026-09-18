"""Job lifecycle events shared by API routes and processing workers."""

from __future__ import annotations

import logging
import shutil
import tempfile
import threading
import uuid
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from typing import Any, Optional

from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder

from autophotoeditor.core.events import JobEvents

CURRENT_JOB: ContextVar[Optional[str]] = ContextVar("current_api_job", default=None)
JOB_DEPTH: ContextVar[int] = ContextVar("api_job_depth", default=0)
CURRENT_CANCEL_TOKEN: ContextVar[Optional[threading.Event]] = ContextVar(
    "current_api_cancel_token", default=None,
)
EVENTS = JobEvents()
_CANCELLATION_TOKENS: dict[str, threading.Event] = {}
_CANCELLATION_LOCK = threading.Lock()


def validate_job_id(job_id: str) -> str:
    try:
        return str(uuid.UUID(str(job_id)))
    except (ValueError, AttributeError, TypeError) as exc:
        raise HTTPException(status_code=400, detail="job_id must be a valid UUID") from exc


class JobLogHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        job_id = CURRENT_JOB.get()
        if job_id:
            EVENTS.publish(job_id, {
                "type": "log",
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            })


def configure_logging() -> None:
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    if not any(isinstance(handler, JobLogHandler) for handler in root_logger.handlers):
        root_logger.addHandler(JobLogHandler())


def cancellation_token(job_id: str) -> threading.Event:
    with _CANCELLATION_LOCK:
        return _CANCELLATION_TOKENS.setdefault(job_id, threading.Event())


def cancel_job(job_id: str) -> bool:
    with _CANCELLATION_LOCK:
        token = _CANCELLATION_TOKENS.get(job_id)
    if token is None:
        return False
    token.set()
    return True


def check_cancelled() -> None:
    token = CURRENT_CANCEL_TOKEN.get()
    if token is not None and token.is_set():
        raise JobCancelled("Job cancelled")


def publish_progress(stage: str, percent: Optional[float] = None, **extra: Any) -> None:
    job_id = CURRENT_JOB.get()
    if not job_id:
        return
    check_cancelled()
    event: dict[str, Any] = {
        "type": "progress", "job_id": job_id, "stage": stage, **extra,
    }
    if percent is not None:
        event["percent"] = max(0, min(100, percent))
    EVENTS.publish(job_id, event)


class JobCancelled(Exception):
    """Raised when a job cancellation token is set."""


def job_operation(name: str):
    def decorator(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            job_id = validate_job_id(kwargs.get("job_id") or str(uuid.uuid4()))
            kwargs["job_id"] = job_id
            cancel_token = CURRENT_CANCEL_TOKEN.get() or cancellation_token(job_id)
            job_context = CURRENT_JOB.set(job_id)
            depth = JOB_DEPTH.get()
            depth_token = JOB_DEPTH.set(depth + 1)
            cancel_context = CURRENT_CANCEL_TOKEN.set(cancel_token)
            EVENTS.publish(job_id, {
                "type": "started", "operation": name, "job_id": job_id,
                "stage": "starting", "percent": 0,
                "nested": depth > 0,
            })
            try:
                check_cancelled()
                result = function(*args, **kwargs)
                if isinstance(result, dict):
                    result.setdefault("job_id", job_id)
                EVENTS.publish(job_id, {
                    "type": "completed", "operation": name, "job_id": job_id,
                    "stage": "finished", "percent": 100,
                    "result": jsonable_encoder(result) if isinstance(result, dict) else None,
                    "final": depth == 0,
                })
                return result
            except JobCancelled as exc:
                EVENTS.publish(job_id, {
                    "type": "cancelled", "operation": name, "job_id": job_id,
                    "stage": "cancelled", "message": str(exc),
                    "final": depth == 0,
                })
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            except Exception as exc:
                cancelled = cancel_token.is_set()
                EVENTS.publish(job_id, {
                    "type": "cancelled" if cancelled else "error",
                    "operation": name, "job_id": job_id,
                    "stage": "cancelled" if cancelled else "failed",
                    "message": "Job cancelled" if cancelled else str(exc),
                    "final": depth == 0,
                })
                if cancelled:
                    raise HTTPException(status_code=409, detail="Job cancelled") from exc
                raise
            finally:
                if depth == 0:
                    shutil.rmtree(
                        Path(tempfile.gettempdir()) / "autophotoeditor" / job_id,
                        ignore_errors=True,
                    )
                JOB_DEPTH.reset(depth_token)
                CURRENT_CANCEL_TOKEN.reset(cancel_context)
                CURRENT_JOB.reset(job_context)
        return wrapped
    return decorator
