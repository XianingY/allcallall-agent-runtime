from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from allcallall_agent_runtime.api.app import create_app
from allcallall_agent_runtime.harness import AllCallAllAgentHarness, get_harness
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
