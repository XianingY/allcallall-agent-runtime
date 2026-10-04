from __future__ import annotations

import threading

from fastapi import FastAPI

from ..async_tool_queue import QueuedTask, ToolQueueWorker, get_default_tool_queue
from ..config import config as runtime_config
from ..tool_bridge import GoToolBridge
from .routes import router


def _tool_queue_executor(task: QueuedTask) -> None:
    """Execute one queued write proposal via the Go backend (shared token auth)."""
    bridge = GoToolBridge()
    bridge.execute_write_tool(
        organization_id=int(task.payload.get("organization_id", 0)),
        user_id=int(task.payload.get("user_id", 0)),
        tool_name=task.tool_name,
        tool_input=task.payload,
    )


def create_app() -> FastAPI:
    """Create the FastAPI application and attach runtime-owned routes."""
    application = FastAPI(title="AllCallAll Agent Runtime", version="0.1.0")
    application.include_router(router)

    if runtime_config.enable_tool_queue:
        worker = ToolQueueWorker(get_default_tool_queue(), _tool_queue_executor)
        worker_thread = threading.Thread(
            target=worker.run,
            name="agent-tool-queue-worker",
            daemon=True,
        )
        worker_thread.start()

    return application

