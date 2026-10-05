from __future__ import annotations

import concurrent.futures
import logging
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from ..admission import AdmissionController
from ..async_tool_queue import QueuedTask, ToolQueueWorker, get_default_tool_queue
from ..config import config as runtime_config
from ..config import effective_max_active_runs
from ..harness import set_invoke_executor, shutdown_invoke_executor
from ..tool_bridge import GoToolBridge
from .routes import router

logger = logging.getLogger(__name__)


def _tool_queue_executor(task: QueuedTask) -> None:
    """Execute one queued write proposal via the Go backend (shared token auth)."""
    bridge = GoToolBridge()
    bridge.execute_write_tool(
        organization_id=int(task.payload.get("organization_id", 0)),
        user_id=int(task.payload.get("user_id", 0)),
        tool_name=task.tool_name,
        tool_input=task.payload,
    )


@asynccontextmanager
async def _lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Manage admission controller and executor lifecycle."""
    effective_active = effective_max_active_runs(runtime_config)
    application.state.admission = AdmissionController(
        max_active=effective_active,
        max_queued=runtime_config.max_queued_runs,
        max_queue_wait_seconds=runtime_config.max_queue_wait_seconds,
    )
    # Single authoritative executor, sized from effective_max_active_runs and
    # injected into the harness module so all code paths share one pool.
    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=effective_active,
        thread_name_prefix="agent-harness-invoke",
    )
    set_invoke_executor(executor)
    logger.info(
        "agent runtime starting: configured_active=%d effective_active=%d "
        "queue_limit=%d provider=%s checkpoint_pool_size=%d",
        runtime_config.max_active_runs,
        effective_active,
        runtime_config.max_queued_runs,
        runtime_config.provider,
        runtime_config.checkpoint_mysql_pool_size,
    )
    yield
    shutdown_invoke_executor(wait=False)


def create_app() -> FastAPI:
    """Create the FastAPI application and attach runtime-owned routes."""
    application = FastAPI(title="AllCallAll Agent Runtime", version="0.1.0", lifespan=_lifespan)
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
