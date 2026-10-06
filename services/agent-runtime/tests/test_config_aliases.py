from __future__ import annotations

import pytest

from allcallall_agent_runtime.config import AgentRuntimeConfig


def test_helm_shared_cancellation_grace_variable_is_honored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PY_AGENT_RUNTIME_CANCELLATION_GRACE_SEC", "30")

    assert AgentRuntimeConfig(_env_file=None).cancellation_grace_seconds == 30.0


def test_python_local_cancellation_grace_variable_remains_supported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PY_AGENT_CANCELLATION_GRACE_SECONDS", "7")

    assert AgentRuntimeConfig(_env_file=None).cancellation_grace_seconds == 7.0
