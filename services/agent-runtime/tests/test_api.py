from __future__ import annotations

import asyncio
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from allcallall_agent_runtime.admission import AdmissionController
from allcallall_agent_runtime.api.app import _lifespan, create_app
from allcallall_agent_runtime.harness import AllCallAllAgentHarness, get_harness
from allcallall_agent_runtime.harness import shutdown_invoke_executor
from allcallall_agent_runtime.main import app
from allcallall_agent_runtime.models import AgentRunRequest, WorkflowResponse


EXPECTED_ROUTES = {
    ("GET", "/docs"),
    ("HEAD", "/docs"),
    ("GET", "/docs/oauth2-redirect"),
    ("HEAD", "/docs/oauth2-redirect"),
    ("GET", "/health"),
    ("GET", "/metrics"),
    ("GET", "/openapi.json"),
    ("HEAD", "/openapi.json"),
    ("GET", "/ready"),
    ("GET", "/redoc"),
    ("HEAD", "/redoc"),
    ("GET", "/v1/capabilities"),
    ("POST", "/v1/agents/react/run"),
    ("GET", "/v1/skills"),
    ("GET", "/v1/tool-queue/metrics"),
    ("GET", "/v1/tool-queue/status"),
    ("GET", "/v1/workflows"),
    ("POST", "/v1/workflows/meeting-brief/run"),
    ("POST", "/v1/workflows/{preset}/run"),
}


def route_pairs(application: FastAPI) -> set[tuple[str, str]]:
    pairs = {
        (method.upper(), path)
        for path, operations in application.openapi()["paths"].items()
        for method in operations
    }
    pairs.update(
        {
            ("GET", "/docs"),
            ("HEAD", "/docs"),
            ("GET", "/docs/oauth2-redirect"),
            ("HEAD", "/docs/oauth2-redirect"),
            ("GET", "/openapi.json"),
            ("HEAD", "/openapi.json"),
            ("GET", "/redoc"),
            ("HEAD", "/redoc"),
        }
    )
    return pairs


def test_public_imports_remain_available() -> None:
    assert app.title == "AllCallAll Agent Runtime"
    assert callable(get_harness)
    assert AllCallAllAgentHarness.name == "allcallall_v1"
    assert AgentRunRequest.__name__ == "MeetingBriefRequest"
    assert WorkflowResponse.__name__ == "MeetingBriefResponse"


def test_app_factory_preserves_routes_and_open_health_endpoint() -> None:
    created_app = create_app()

    assert route_pairs(app) == EXPECTED_ROUTES
    assert route_pairs(created_app) == EXPECTED_ROUTES
    assert TestClient(created_app).get("/health").json() == {
        "status": "ok",
        "runtime": "python_langgraph",
    }


def test_app_lifespan_creates_admission_controller() -> None:
    """The lifespan context manager creates an admission controller on app.state."""
    created_app = create_app()
    with TestClient(created_app):
        assert hasattr(created_app.state, "admission")
        assert isinstance(created_app.state.admission, AdmissionController)
        assert created_app.state.admission.max_active > 0


def test_app_lifespan_injects_executor_into_harness() -> None:
    """The lifespan injects the executor into the harness module."""
    import concurrent.futures
    from allcallall_agent_runtime.orchestration.harness import _get_invoke_executor

    created_app = create_app()
    with TestClient(created_app):
        executor = _get_invoke_executor()
        assert isinstance(executor, concurrent.futures.ThreadPoolExecutor)
        # The executor should be sized from effective_max_active_runs
        assert executor._max_workers > 0

    # After lifespan teardown, the executor should be shut down
    # (a new lazy one would be created on next access, but the injected one is gone)


def test_app_lifespan_cleans_executor_when_client_startup_fails() -> None:
    """A startup failure after executor creation must not leak the executor."""
    created_app = create_app()

    async def enter_lifespan() -> None:
        async with _lifespan(created_app):
            pass

    try:
        with (
            patch("allcallall_agent_runtime.api.app.build_runtime_clients", side_effect=RuntimeError("startup failed")),
            patch(
                "allcallall_agent_runtime.api.app.shutdown_invoke_executor",
                wraps=shutdown_invoke_executor,
            ) as shutdown_spy,
        ):
            with pytest.raises(RuntimeError, match="startup failed"):
                asyncio.run(enter_lifespan())
            shutdown_spy.assert_called_once_with(wait=False)
    finally:
        shutdown_invoke_executor(wait=False)
