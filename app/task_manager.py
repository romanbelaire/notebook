from __future__ import annotations

"""Lightweight background task manager using ThreadPoolExecutor.

Designed for Streamlit: schedules long-running CPU/IO jobs so the UI thread is
not blocked.  Keeps a global registry of tasks so status can be queried across
reruns.
"""

from concurrent.futures import ThreadPoolExecutor, Future
from typing import Callable, Any, Dict
import uuid
import threading

# -----------------------------------------------------------------------------
# Executor configured with a small pool – adjust depending on workload.
# -----------------------------------------------------------------------------
_executor = ThreadPoolExecutor(max_workers=2)
_tasks: Dict[str, Future] = {}
_task_locks: Dict[str, threading.Lock] = {}

def submit(func: Callable[..., Any], *args: Any, **kwargs: Any) -> str:
    """Submit *func* to run in the background. Returns task_id."""
    task_id = str(uuid.uuid4())
    fut = _executor.submit(func, *args, **kwargs)
    _tasks[task_id] = fut
    _task_locks[task_id] = threading.Lock()
    return task_id


def status(task_id: str) -> str:
    """Return simple status string for *task_id*: pending/running/done/error."""
    fut = _tasks.get(task_id)
    if fut is None:
        return "unknown"
    if fut.running():
        return "running"
    if fut.done():
        return "error" if fut.exception() else "done"
    return "pending"


def result(task_id: str):
    fut = _tasks.get(task_id)
    if fut is None:
        raise KeyError(task_id)
    return fut.result()


def exception(task_id: str):
    fut = _tasks.get(task_id)
    if fut is None:
        raise KeyError(task_id)
    return fut.exception() 