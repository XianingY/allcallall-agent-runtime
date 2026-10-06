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
from allcallall_agent_runtime.models import AgentRunRequest, ToolProposal, WorkflowRequest, WorkflowResponse


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
    ("POST", "/v1/agents/react/resume"),
    ("GET", "/v1/skills"),
    ("GET", "/v1/tool-queue/metrics"),
    ("GET", "/v1/tool-queue/status"),
    ("GET", "/v1/workflows"),
    ("POST", "/v1/workflows/meeting-brief/run"),
    ("POST", "/v1/workflows/{preset}/resume"),
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


def test_app_lifespan_rejects_tool_queue_in_multi_replica_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created_app = create_app()

    async def enter_lifespan() -> None:
        async with _lifespan(created_app):
            pass

    monkeypatch.setattr(
        "allcallall_agent_runtime.api.app.runtime_config.enable_tool_queue",
        True,
    )

    try:
        with pytest.raises(ValueError, match="PY_AGENT_DEPLOYMENT_MODE=single_process"):
            asyncio.run(enter_lifespan())
    finally:
        shutdown_invoke_executor(wait=False)


def test_ready_and_capabilities_report_effective_deployment_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    default = TestClient(app).get("/v1/capabilities").json()
    assert default["tool_queue"]["enabled"] is False
    assert default["tool_queue"]["mode"] == "go_outbox"

    monkeypatch.setattr(
        "allcallall_agent_runtime.api.routes.runtime_config.deployment_mode",
        "single_process",
        raising=False,
    )

    client = TestClient(app)

    assert client.get("/ready").json()["deployment_mode"] == "single_process"
    capabilities = client.get("/v1/capabilities").json()
    assert capabilities["deployment_mode"] == "single_process"
    assert capabilities["tool_queue"]["mode"] == "async_after_approval"


def test_workflow_response_still_returns_approved_proposals_in_multi_replica_mode() -> None:
    class ProposalHarness:
        def run_workflow(self, request: WorkflowRequest) -> WorkflowResponse:
            return WorkflowResponse(
                status="requires_action",
                proposed_tool_calls=[
                    ToolProposal(
                        tool_name="write_conversation_message",
                        arguments={"body": "approved"},
                        idempotency_key="go-owns-this-write",
                        approval_required=True,
                    )
                ],
            )

    created_app = create_app()
    with patch("allcallall_agent_runtime.api.routes.get_harness", return_value=ProposalHarness()):
        response = TestClient(created_app).post(
            "/v1/workflows/risk_review/run",
            json={
                "organization_id": 1,
                "user_id": 2,
                "conversation_id": 3,
                "workflow_run_id": 4,
                "goal": "Generate an approved write",
            },
        )

    assert response.status_code == 200
    proposals = response.json()["proposed_tool_calls"]
    assert len(proposals) == 1
    assert proposals[0]["tool_name"] == "write_conversation_message"
    assert proposals[0]["approval_required"] is True
    assert proposals[0]["idempotency_key"] == "go-owns-this-write"
