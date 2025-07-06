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
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="task_manager")
_tasks: Dict[str, Future] = {}
_task_locks: Dict[str, threading.Lock] = {}
_task_progress: Dict[str, Dict[str, Any]] = {}  # Track progress for each task

def get_active_task_count() -> int:
    """Return the number of currently running tasks."""
    return sum(1 for fut in _tasks.values() if fut.running())

def submit(func: Callable[..., Any], *args: Any, **kwargs: Any) -> str:
    """Submit *func* to run in the background. Returns task_id."""
    # Check if we have too many active tasks
    active_count = get_active_task_count()
    if active_count >= 5:  # Limit concurrent tasks
        raise RuntimeError(f"Too many active tasks ({active_count}). Please wait for some to complete.")
    
    task_id = str(uuid.uuid4())
    
    # Create a wrapper that injects the task_id and handles exceptions properly
    def wrapper():
        try:
            # If the function expects a task_id parameter, provide it
            try:
                import inspect
                sig = inspect.signature(func)
                if 'task_id' in sig.parameters:
                    kwargs['task_id'] = task_id
            except Exception:
                # If inspect fails, continue without task_id injection
                pass
            
            return func(*args, **kwargs)
        except Exception as e:
            # Ensure exceptions are properly captured and don't leak to the executor
            import traceback
            print(f"Task {task_id} failed with exception: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            raise  # Re-raise so it's captured by Future.exception()
    
    fut = _executor.submit(wrapper)
    _tasks[task_id] = fut
    _task_locks[task_id] = threading.Lock()
    return task_id


def status(task_id: str) -> str:
    """Return simple status string for *task_id*: pending/running/done/error."""
    # Periodic cleanup
    if len(_tasks) > 10:  # Clean up when we have many tasks
        cleanup_completed_tasks()
    
    fut = _tasks.get(task_id)
    if fut is None:
        # Also clean up progress for unknown tasks
        if task_id in _task_progress:
            del _task_progress[task_id]
        return "unknown"
    
    try:
        if fut.running():
            return "running"
        if fut.done():
            return "error" if fut.exception() else "done"
        return "pending"
    except Exception as e:
        print(f"Error checking task status for {task_id}: {e}")
        return "error"


def result(task_id: str):
    fut = _tasks.get(task_id)
    if fut is None:
        raise KeyError(task_id)
    try:
        return fut.result()
    except Exception as e:
        print(f"Error getting result for task {task_id}: {e}")
        raise


def exception(task_id: str):
    fut = _tasks.get(task_id)
    if fut is None:
        raise KeyError(task_id)
    try:
        return fut.exception()
    except Exception as e:
        print(f"Error getting exception for task {task_id}: {e}")
        return e


def set_progress(task_id: str, current: int, total: int, message: str = ""):
    """Update progress for a task."""
    try:
        # Use the task lock if available for thread safety
        lock = _task_locks.get(task_id)
        if lock:
            with lock:
                _task_progress[task_id] = {
                    "current": current,
                    "total": total,
                    "message": message,
                    "percentage": (current / total * 100) if total > 0 else 0
                }
        else:
            # No lock available, update directly (less safe but functional)
            _task_progress[task_id] = {
                "current": current,
                "total": total,
                "message": message,
                "percentage": (current / total * 100) if total > 0 else 0
            }
    except Exception as e:
        print(f"Error setting progress for task {task_id}: {e}")


def get_progress(task_id: str) -> Dict[str, Any]:
    """Get progress information for a task."""
    return _task_progress.get(task_id, {})


def cleanup_completed_tasks():
    """Clean up completed tasks to prevent memory leaks."""
    completed_task_ids = []
    
    for task_id, fut in _tasks.items():
        if fut.done():
            completed_task_ids.append(task_id)
    
    for task_id in completed_task_ids:
        # Keep the task for a bit longer but clean up its data
        if task_id in _task_locks:
            del _task_locks[task_id]
        # Keep progress data until explicitly cleaned by status endpoint


def shutdown():
    """Shutdown the executor gracefully."""
    try:
        _executor.shutdown(wait=True, cancel_futures=True)
    except Exception as e:
        print(f"Error during executor shutdown: {e}")


# Auto-cleanup completed tasks periodically
import atexit
atexit.register(shutdown) 