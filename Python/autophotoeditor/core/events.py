"""Thread-safe job event history and async subscriptions."""

from __future__ import annotations

import asyncio
import threading
from typing import Any


class JobEvents:
    """Thread-safe job history used by the HTTP polling endpoint."""

    def __init__(self) -> None:
        self._listeners: dict[str, list[tuple[asyncio.AbstractEventLoop, asyncio.Queue]]] = {}
        self._history: dict[str, list[dict[str, Any]]] = {}
        self._lock = threading.Lock()

    def publish(self, job_id: str, event: dict[str, Any]) -> None:
        with self._lock:
            history = self._history.setdefault(job_id, [])
            history.append(event)
            del history[:-500]
            if len(self._history) > 1000:
                oldest_job = next(iter(self._history))
                if oldest_job != job_id and oldest_job not in self._listeners:
                    self._history.pop(oldest_job, None)
            listeners = list(self._listeners.get(job_id, []))
        for loop, queue in listeners:
            try:
                loop.call_soon_threadsafe(self._enqueue, queue, event)
            except RuntimeError:
                continue

    @staticmethod
    def _enqueue(queue: asyncio.Queue, event: dict[str, Any]) -> None:
        try:
            queue.put_nowait(event)
        except asyncio.QueueFull:
            try:
                queue.get_nowait()
                queue.put_nowait(event)
            except (asyncio.QueueEmpty, asyncio.QueueFull):
                pass

    def subscribe(self, job_id: str, loop: asyncio.AbstractEventLoop) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        with self._lock:
            history = list(self._history.get(job_id, []))
            self._listeners.setdefault(job_id, []).append((loop, queue))
        for event in history:
            queue.put_nowait(event)
        return queue

    def unsubscribe(self, job_id: str, queue: asyncio.Queue) -> None:
        with self._lock:
            listeners = self._listeners.get(job_id, [])
            self._listeners[job_id] = [item for item in listeners if item[1] is not queue]
            if not self._listeners[job_id]:
                self._listeners.pop(job_id, None)

    def snapshot(self, job_id: str) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._history.get(job_id, []))
