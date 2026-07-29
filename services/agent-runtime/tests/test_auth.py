"""Tests for inbound API bearer-token authentication (P0#1)."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import allcallall_agent_runtime.config as cfg
from allcallall_agent_runtime.api_auth import require_auth, reset_auth_warning
from allcallall_agent_runtime.main import app

VALID_BODY = {
    "organization_id": 1,
    "user_id": 2,
    "conversation_id": 3,
    "workflow_run_id": 4,
    "goal": "test",
}


@pytest.fixture
def with_token() -> Iterator[str]:
    reset_auth_warning()
    previous = cfg.config.api_token
    cfg.config.api_token = "secret-token"
    yield "secret-token"
    cfg.config.api_token = previous


@pytest.fixture
def without_token() -> Iterator[None]:
    reset_auth_warning()
    previous = cfg.config.api_token
    cfg.config.api_token = ""
    yield
    cfg.config.api_token = previous


def test_require_auth_accepts_correct_token(with_token: str) -> None:
    # No exception is raised when the bearer token matches.
    require_auth(authorization=f"Bearer {with_token}")


def test_require_auth_rejects_missing_header(with_token: str) -> None:
    with pytest.raises(HTTPException) as exc:
        require_auth(authorization=None)
    assert exc.value.status_code == 401
    assert exc.value.headers is not None
    assert "WWW-Authenticate" in exc.value.headers


def test_require_auth_rejects_wrong_token(with_token: str) -> None:
    with pytest.raises(HTTPException) as exc:
        require_auth(authorization="Bearer wrong")
    assert exc.value.status_code == 401


def test_require_auth_rejects_malformed_header(with_token: str) -> None:
    with pytest.raises(HTTPException) as exc:
        require_auth(authorization="Token abc")
    assert exc.value.status_code == 401


def test_require_auth_passthrough_when_unconfigured(without_token: None) -> None:
    # Backward-compatible default: no token configured -> request allowed.
    require_auth(authorization=None)
    require_auth(authorization="Bearer anything")


def test_run_endpoint_rejects_missing_token(with_token: str) -> None:
    client = TestClient(app)
    resp = client.post("/v1/workflows/meeting-brief/run", json=VALID_BODY)
    assert resp.status_code == 401


def test_run_endpoint_rejects_wrong_token(with_token: str) -> None:
    client = TestClient(app)
    resp = client.post(
        "/v1/workflows/meeting-brief/run",
        json=VALID_BODY,
        headers={"Authorization": "Bearer nope"},
    )
    assert resp.status_code == 401


def test_run_endpoint_accepts_correct_token(with_token: str) -> None:
    client = TestClient(app)
    resp = client.post(
        "/v1/workflows/meeting-brief/run",
        json=VALID_BODY,
        headers={"Authorization": f"Bearer {with_token}"},
    )
    # Auth accepted: control reaches the harness (not a 401 rejection).
    assert resp.status_code != 401


def test_run_endpoint_open_when_token_unset(without_token: None) -> None:
    client = TestClient(app)
    resp = client.post("/v1/workflows/meeting-brief/run", json=VALID_BODY)
    assert resp.status_code != 401


def test_health_and_ready_remain_open(with_token: str) -> None:
    client = TestClient(app)
    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code == 200
