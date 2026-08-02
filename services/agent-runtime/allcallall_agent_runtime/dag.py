"""DAG definition and compilation for the agent runtime graph."""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph

from .config import config as app_config
from .state import GraphState
from .nodes import (
    approval_gate,
    collect_context,
    critic_check,
    decompose,
    finalize,
    memory_agent,
    propose_tools,
    quality_check,
    reflect_and_plan_memory,
    retrieval_planner,
    retrieval_loop,
    retrieve_context,
    rerank_context,
    risk_analyst,
    safety_check,
    searcher,
    synthesize,
)
from .nodes.check import route_quality
from .nodes.retrieval import build_evidence_pack, grounding_check, merge, sufficiency_gate
from .nodes.role_router import next_role_after, route_roles

# Possible targets of the dynamic role-router conditional edges. Every router
# edge may resolve to any role node or straight to ``merge`` (when no further
# role is scheduled), so all five must appear in each edge's path map.
_ROLE_TARGETS: list[str] = ["searcher", "memory_agent", "synthesize", "risk_analyst", "merge"]


def _route_first(state: GraphState) -> str:
    return next_role_after(state, None)


def _route_after_searcher(state: GraphState) -> str:
    return next_role_after(state, "searcher")


def _route_after_memory_agent(state: GraphState) -> str:
    return next_role_after(state, "memory_agent")


def _route_after_synthesize(state: GraphState) -> str:
    return next_role_after(state, "synthesize")


def _route_after_risk_analyst(state: GraphState) -> str:
    return next_role_after(state, "risk_analyst")


def build_workflow_graph(checkpointer: BaseCheckpointSaver[Any] | None = None) -> Any:
    """Build and compile the LangGraph workflow graph.

    When ``checkpointer`` is provided (e.g. the MySQL ``CheckpointSaver``), the
    compiled graph gains durable, resumable checkpoints keyed by thread id.

    The role chain (``decompose -> searcher -> memory_agent -> synthesize ->
    risk_analyst -> merge``) is static by default. When
    ``app_config.enable_role_router`` is True, a ``role_router`` node with
    conditional edges is inserted so clearly-redundant roles (e.g. the risk
    analyst for ``context_qa``) can be skipped while still reaching ``merge``.
    """
    graph = StateGraph(GraphState)
    graph.add_node("collect_context", collect_context)
    graph.add_node("retrieval_planner", retrieval_planner)
    graph.add_node("retrieval_loop", retrieval_loop)
    graph.add_node("retrieve_context", retrieve_context)
    graph.add_node("rerank_context", rerank_context)
    graph.add_node("evidence_pack", build_evidence_pack)
    graph.add_node("sufficiency_gate", sufficiency_gate)
    graph.add_node("decompose", decompose)
    graph.add_node("searcher", searcher)
    graph.add_node("memory_agent", memory_agent)
    graph.add_node("synthesize", synthesize)
    graph.add_node("risk_analyst", risk_analyst)
    graph.add_node("merge", merge)
    graph.add_node("grounding_check", grounding_check)
    graph.add_node("memory_reflection", reflect_and_plan_memory)
    graph.add_node("propose_tools", propose_tools)
    graph.add_node("critic_check", critic_check)
    graph.add_node("quality_check", quality_check)
    graph.add_node("safety_check", safety_check)
    graph.add_node("approval_gate", approval_gate)
    graph.add_node("finalize", finalize)
    graph.add_node("role_router", route_roles)
    graph.set_entry_point("collect_context")
    graph.add_edge("collect_context", "retrieval_planner")
    graph.add_edge("retrieval_planner", "retrieval_loop")
    graph.add_edge("retrieval_loop", "retrieve_context")
    graph.add_edge("retrieve_context", "rerank_context")
    graph.add_edge("rerank_context", "evidence_pack")
    graph.add_edge("evidence_pack", "sufficiency_gate")

    if app_config.enable_role_router:
        # Dynamic chain: route roles based on the computed allocation.
        graph.add_edge("sufficiency_gate", "decompose")
        graph.add_edge("decompose", "role_router")
        graph.add_conditional_edges("role_router", _route_first, _ROLE_TARGETS)
        graph.add_conditional_edges("searcher", _route_after_searcher, _ROLE_TARGETS)
        graph.add_conditional_edges("memory_agent", _route_after_memory_agent, _ROLE_TARGETS)
        graph.add_conditional_edges("synthesize", _route_after_synthesize, _ROLE_TARGETS)
        graph.add_conditional_edges("risk_analyst", _route_after_risk_analyst, _ROLE_TARGETS)
    else:
        # Static chain (legacy behavior): every role runs in fixed order.
        graph.add_edge("sufficiency_gate", "decompose")
        graph.add_edge("decompose", "searcher")
        graph.add_edge("searcher", "memory_agent")
        graph.add_edge("memory_agent", "synthesize")
        graph.add_edge("synthesize", "risk_analyst")
        graph.add_edge("risk_analyst", "merge")

    graph.add_edge("merge", "grounding_check")
    graph.add_edge("grounding_check", "memory_reflection")
    graph.add_edge("memory_reflection", "propose_tools")
    graph.add_edge("propose_tools", "critic_check")
    graph.add_edge("critic_check", "quality_check")
    # L1 quality gate: loop back to synthesize on "revise", else advance to L2.
    graph.add_conditional_edges(
        "quality_check",
        route_quality,
        {"revise": "synthesize", "safety": "safety_check"},
    )
    # L2 safety gate: always advance to the approval gate (which handles
    # both normal approval and human escalation).
    graph.add_edge("safety_check", "approval_gate")
    graph.add_edge("approval_gate", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=checkpointer)
