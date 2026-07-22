from __future__ import annotations

from typing import Any

from allcallall_agent_runtime.helpers import (
    build_graph_expansion,
    citations_from_chunks,
    contains_any,
    dedupe_citations,
    estimate_retrieval_confidence,
    evaluate_context_sufficiency,
    first_non_empty,
    infer_relation,
    normalize_workflow_preset,
    request_with_runtime_context,
    runtime_subject_id,
    summarize_observation,
    tokenize_route_terms,
    tool_allowed,
    top_snippets,
    unique_strings,
)
from allcallall_agent_runtime.models import (
    Citation,
    ContextChunk,
    EvidencePack,
    WorkflowRequest,
)


def _request(preset: str = "meeting_brief", **kw: Any) -> WorkflowRequest:
    base: dict[str, Any] = dict(organization_id=1, user_id=2, conversation_id=3, workflow_run_id=9, goal="g")
    base.update(kw)
    return WorkflowRequest(preset=preset, **base)


def _chunk(source_type: str, source_id: str = "1", snippet: str = "snip", chunk_id: str = "") -> ContextChunk:
    return ContextChunk(
        chunk_id=chunk_id, source_type=source_type, source_id=source_id, snippet=snippet, score=5
    )


# --- pure string utilities -------------------------------------------------
def test_contains_any_is_case_insensitive() -> None:
    assert contains_any("We have a RISK here", ("risk", "blocker"))
    assert not contains_any("all clear", ("risk", "blocker"))


def test_unique_strings_normalizes_and_dedupes() -> None:
    assert unique_strings(["a", " a ", "b", "a", ""]) == ["a", "b"]


def test_first_non_empty() -> None:
    assert first_non_empty(["", "  ", "x", "y"]) == "x"
    assert first_non_empty(["", ""]) == ""


# --- citation helpers ------------------------------------------------------
def test_dedupe_citations_by_source() -> None:
    c1 = Citation(source_type="knowledge", source_id="1", snippet="s")
    c2 = Citation(source_type="knowledge", source_id="1", snippet="different")
    c3 = Citation(source_type="knowledge", source_id="2", snippet="s")
    out = dedupe_citations([c1, c2, c3])
    assert len(out) == 2
    assert {c.source_id for c in out} == {"1", "2"}


def test_citations_from_chunks_skips_incomplete() -> None:
    good = _chunk("knowledge", "1", "useful snippet")
    no_snippet = ContextChunk(source_type="knowledge", source_id="2", snippet="")
    no_id = ContextChunk(source_type="knowledge", source_id="", snippet="x")
    out = citations_from_chunks([good, no_snippet, no_id])
    assert len(out) == 1
    assert out[0].source_id == "1"


def test_top_snippets_respects_limit() -> None:
    chunks = [_chunk("knowledge", str(i), f"snip {i}") for i in range(5)]
    assert top_snippets(chunks, 3) == ["snip 0", "snip 1", "snip 2"]


def test_summarize_observation() -> None:
    assert summarize_observation("query_recent_meetings", []) == "recent meeting metadata inspected"
    chunks = [_chunk("knowledge", "1", "hello world")]
    assert "hello world" in summarize_observation("query_knowledge_chunks", chunks)


# --- workflow / request helpers --------------------------------------------
def test_normalize_workflow_preset_alias_and_default() -> None:
    assert normalize_workflow_preset("follow_up") == "follow_up_planner"
    assert normalize_workflow_preset("") == "meeting_brief"
    assert normalize_workflow_preset("risk_review") == "risk_review"


def test_tool_allowed_respects_policy() -> None:
    req = _request(tool_policy={"read_tools": ["a", "b"]})
    assert tool_allowed(req, "a")
    assert not tool_allowed(req, "c")
    assert tool_allowed(_request(), "anything")


def test_runtime_subject_id_prefers_agent_run() -> None:
    assert runtime_subject_id(_request(agent_run_id=7)) == "agent:7"
    assert runtime_subject_id(_request()) == "workflow:9"


def test_request_with_runtime_context_overrides_chunks() -> None:
    req = _request()
    state = {"request": req.model_dump(), "reranked_context_chunks": [_chunk("knowledge", "9")]}
    updated = request_with_runtime_context(state)
    assert len(updated.context_chunks) == 1
    assert updated.context_chunks[0].source_id == "9"


# --- retrieval confidence / sufficiency ------------------------------------
def test_estimate_retrieval_confidence_grows_with_relevant_sources() -> None:
    base = _chunk("knowledge", "1")
    low = estimate_retrieval_confidence(_request("meeting_brief"), [base])
    high = estimate_retrieval_confidence(
        _request("meeting_brief"),
        [_chunk("meeting_transcript", "1"), _chunk("knowledge", "2"), _chunk("memory", "3")],
    )
    assert low < high <= 1.0


def test_estimate_retrieval_confidence_penalizes_missing_source() -> None:
    no_transcript = estimate_retrieval_confidence(_request("meeting_brief"), [_chunk("knowledge", "1")])
    with_transcript = estimate_retrieval_confidence(_request("meeting_brief"), [_chunk("meeting_transcript", "1")])
    assert with_transcript > no_transcript


def test_evaluate_context_sufficiency_risk_requires_evidence() -> None:
    empty = EvidencePack(route_intent="risk", confidence=0.9, source_types=[], citations=[])
    insufficient = evaluate_context_sufficiency(_request(), empty)
    assert insufficient.sufficient is False
    assert "risk evidence" in insufficient.missing_info

    enough = EvidencePack(
        route_intent="risk",
        confidence=0.9,
        source_types=["meeting_transcript", "conversation"],
        citations=[Citation(source_type="meeting_transcript", source_id="1", snippet="s")],
    )
    assert evaluate_context_sufficiency(_request(), enough).sufficient is True


def test_evaluate_context_sufficiency_consult_requires_knowledge() -> None:
    pack = EvidencePack(route_intent="consult", confidence=0.9, source_types=["meeting_transcript"], citations=[])
    result = evaluate_context_sufficiency(_request(), pack)
    assert result.sufficient is False
    assert "knowledge citation" in result.missing_info


# --- graph expansion / relation --------------------------------------------
def test_infer_relation_from_keywords() -> None:
    assert infer_relation("this task requires approval") == "requires"
    assert infer_relation("it blocks the release") == "blocks"
    assert infer_relation("plain text") == ""


def test_tokenize_route_terms_dedupes_and_filters() -> None:
    terms = tokenize_route_terms("Risk risk the the approval")
    assert terms == ["risk", "the", "approval"]


def test_build_graph_expansion_creates_edges() -> None:
    chunk = _chunk("knowledge", "1", "the launch requires approval and blocks the release")
    expansion = build_graph_expansion("launch approval", [chunk])
    assert expansion.enabled is True
    assert len(expansion.edges) >= 1
    assert all(e.relation for e in expansion.edges)
