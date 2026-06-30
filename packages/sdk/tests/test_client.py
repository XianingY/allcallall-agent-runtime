from __future__ import annotations

import httpx
import pytest

from allcallall_agent_runtime_sdk import AgentRuntimeClient, RAGRuntimeClient


def test_agent_client_decodes_workflow_response(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(self: httpx.Client, url: str, json: dict[str, object]) -> httpx.Response:
        assert url.endswith("/v1/workflows/meeting_brief/run")
        return httpx.Response(200, json={"status": "ready", "summary": "ok"})

    monkeypatch.setattr(httpx.Client, "post", fake_post)
    client = AgentRuntimeClient("http://runtime")
    response = client.run_meeting_brief(
        {"organization_id": 1, "user_id": 2, "conversation_id": 3, "goal": "brief"}
    )
    assert response.status == "ready"
    assert response.summary == "ok"


def test_rag_client_decodes_query_response(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(self: httpx.Client, url: str, json: dict[str, object]) -> httpx.Response:
        assert url.endswith("/v1/retrieval/query")
        return httpx.Response(200, json={"query": json["query"], "chunks": [], "count": 0})

    monkeypatch.setattr(httpx.Client, "post", fake_post)
    client = RAGRuntimeClient("http://rag")
    response = client.query({"query": "risk", "top_k": 3})
    assert response.query == "risk"
    assert response.count == 0
