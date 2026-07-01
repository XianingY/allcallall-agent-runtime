"""Context collection and retrieval planning nodes."""

from __future__ import annotations


from ..config import config as app_config
from ..models import (
    AgenticRAGConfig,
    GraphExpansion,
    IntentRoute,
    RetrievalPlan,
    RetrievalPlanStep,
    TraceEvent,
    WorkflowRequest,
)
from ..prompts import prompt_version_for
from ..helpers import (
    READ_TOOL_CONTEXT_CHUNKS,
    READ_TOOL_KNOWLEDGE_CHUNKS,
    READ_TOOL_MEETING_TRANSCRIPTS,
    WORKFLOW_CONTEXT_QA,
    build_graph_expansion,
    route_request_intent,
    tool_allowed,
    unique_strings,
)
from ..state import GraphState


def collect_context(state: GraphState) -> GraphState:
    """Collect and validate the incoming request context."""
    request = state["request"]
    trace = state.get("trace_events", [])
    prompt_version = prompt_version_for(request)
    trace.append(TraceEvent(event="graph.node.started", node="collect_context", status="running"))
    trace.append(
        TraceEvent(
            event="graph.node.completed",
            node="collect_context",
            status="completed",
            metadata={
                "preset": request.preset,
                "messages": len(request.messages),
                "notes": len(request.notes),
                "meeting_transcripts": len(request.meeting_transcripts),
                "context_chunks": len(request.context_chunks),
                "prompt_version": prompt_version,
            },
        )
    )
    return {"trace_events": trace, "prompt_version": prompt_version}


def retrieval_planner(state: GraphState) -> GraphState:
    """Plan retrieval steps based on workflow preset and configuration."""
    request = state["request"]
    trace = state.get("trace_events", [])
    config = resolve_agentic_rag_config(request.agentic_rag)
    enabled = agentic_rag_enabled(config)
    route = route_request_intent(request)
    graph = build_graph_expansion(request.goal, request.context_chunks)
    plan = build_retrieval_plan(request, config, enabled, route, graph)
    trace.append(TraceEvent(event="graph.node.started", node="retrieval_planner", status="running"))
    trace.append(
        TraceEvent(
            event="intent.route",
            node="retrieval_planner",
            status="completed",
            metadata=route.model_dump(),
        )
    )
    if graph.enabled:
        trace.append(
            TraceEvent(
                event="rag.graph_expand",
                node="retrieval_planner",
                status="completed",
                metadata={
                    "expanded_terms": graph.expanded_terms,
                    "edges": [edge.model_dump() for edge in graph.edges],
                },
            )
        )
    trace.append(
        TraceEvent(
            event="rag.plan",
            node="retrieval_planner",
            status="completed",
            metadata={
                "enabled": plan.enabled,
                "max_steps": plan.max_steps,
                "min_confidence": plan.min_confidence,
                "steps": [step.model_dump() for step in plan.steps],
            },
        )
    )
    trace.append(TraceEvent(event="graph.node.completed", node="retrieval_planner", status="completed"))
    return {
        "trace_events": trace,
        "agentic_rag_enabled": enabled,
        "intent_route": route,
        "graph_expansion": graph,
        "retrieval_plan": plan,
    }


def resolve_agentic_rag_config(config: AgenticRAGConfig) -> AgenticRAGConfig:
    """Resolve AgenticRAG configuration with defaults."""
    enabled = config.enabled or app_config.enable_agentic_rag
    max_steps = config.max_steps
    if max_steps <= 0:
        max_steps = app_config.rag_max_retrieval_steps
    max_steps = max(1, min(max_steps, 3))
    min_confidence = config.min_confidence
    if min_confidence <= 0:
        min_confidence = app_config.rag_min_confidence
    min_confidence = max(0.1, min(min_confidence, 1.0))
    allowed = [item for item in config.allowed_source_types if item.strip()]
    if not allowed:
        allowed = [
            "meeting_transcript",
            "knowledge",
            "conversation",
            "message",
            "note",
            "followup",
            "memory",
            "contact_profile",
        ]
    return config.model_copy(
        update={
            "enabled": enabled,
            "max_steps": max_steps,
            "min_confidence": min_confidence,
            "allowed_source_types": allowed,
        }
    )


def agentic_rag_enabled(config: AgenticRAGConfig) -> bool:
    """Check if agentic RAG is enabled."""
    return config.enabled


def build_retrieval_plan(
    request: WorkflowRequest,
    config: AgenticRAGConfig,
    enabled: bool,
    route: IntentRoute | None = None,
    graph: GraphExpansion | None = None,
) -> RetrievalPlan:
    """Build a retrieval plan based on workflow preset and configuration."""
    route = route or route_request_intent(request)
    graph = graph or GraphExpansion()
    if not enabled:
        return RetrievalPlan(
            enabled=False,
            max_steps=config.max_steps,
            min_confidence=config.min_confidence,
            intent_route=route,
            graph_expansion=graph,
        )
    candidates: list[RetrievalPlanStep] = []
    goal = request.goal.strip()
    expanded = " ".join(graph.expanded_terms[:6])
    route_sources = [item for item in route.required_source_types if item in config.allowed_source_types]
    if route.intent == "consult" or request.preset == WORKFLOW_CONTEXT_QA:
        candidates.append(
            RetrievalPlanStep(
                step=1,
                query=join_query(goal, expanded, "policy knowledge checklist evidence"),
                source_scope="knowledge",
                tool_name=READ_TOOL_KNOWLEDGE_CHUNKS,
                rationale="Consult intent should first inspect organization knowledge.",
                strategy=route.retrieval_strategy,
                expanded_terms=graph.expanded_terms,
            )
        )
        candidates.append(
            RetrievalPlanStep(
                step=2,
                query=join_query(goal, "meeting transcript conversation evidence", expanded),
                source_scope="all",
                tool_name=READ_TOOL_CONTEXT_CHUNKS,
                rationale="Refine with transcript and conversation evidence if knowledge is insufficient.",
                strategy="adaptive",
                expanded_terms=graph.expanded_terms,
            )
        )
    elif route.intent == "risk":
        candidates.append(
            RetrievalPlanStep(
                step=1,
                query=join_query(goal, "risk blocker approval deadline budget security legal"),
                source_scope="meeting_transcript",
                tool_name=READ_TOOL_MEETING_TRANSCRIPTS,
                rationale="Risk intent should ground claims in meeting transcript segments first.",
                strategy=route.retrieval_strategy,
                expanded_terms=graph.expanded_terms,
            )
        )
        candidates.append(
            RetrievalPlanStep(
                step=2,
                query=join_query(goal, expanded, "risk policy knowledge approval guardrail"),
                source_scope="knowledge",
                tool_name=READ_TOOL_KNOWLEDGE_CHUNKS,
                rationale="Supplement risks with policy or knowledge evidence.",
                strategy="graph_augmented" if graph.enabled else "adaptive",
                expanded_terms=graph.expanded_terms,
            )
        )
    else:
        candidates.append(
            RetrievalPlanStep(
                step=1,
                query=join_query(goal, "conversation notes transcript knowledge memory"),
                source_scope="all",
                tool_name=READ_TOOL_CONTEXT_CHUNKS,
                rationale="Chat/general tasks should inspect scoped conversation and memory context first.",
                strategy=route.retrieval_strategy,
                expanded_terms=graph.expanded_terms,
            )
        )
        candidates.append(
            RetrievalPlanStep(
                step=2,
                query=join_query(goal, expanded, "related knowledge policy context"),
                source_scope="knowledge",
                tool_name=READ_TOOL_KNOWLEDGE_CHUNKS,
                rationale="Retrieve related knowledge when the transcript alone does not cover policy context.",
                strategy="graph_augmented" if graph.enabled else "adaptive",
                expanded_terms=graph.expanded_terms,
            )
        )
    candidates.append(
        RetrievalPlanStep(
            step=len(candidates) + 1,
            query=join_query(goal, "conversation notes follow ups memory", " ".join(route_sources)),
            source_scope="all",
            tool_name=READ_TOOL_CONTEXT_CHUNKS,
            rationale="Final bounded fallback over all scoped conversation context.",
            strategy="multi_hop" if route.intent == "risk" else "adaptive",
            expanded_terms=graph.expanded_terms,
        )
    )
    steps: list[RetrievalPlanStep] = []
    for step in candidates:
        if step.source_scope != "all" and step.source_scope not in config.allowed_source_types:
            continue
        if not tool_allowed(request, step.tool_name):
            fallback = step.model_copy(update={"tool_name": READ_TOOL_CONTEXT_CHUNKS, "source_scope": "all"})
            if tool_allowed(request, fallback.tool_name):
                steps.append(fallback.model_copy(update={"step": len(steps) + 1}))
            continue
        steps.append(step.model_copy(update={"step": len(steps) + 1}))
        if len(steps) >= config.max_steps:
            break
    if not steps and tool_allowed(request, READ_TOOL_CONTEXT_CHUNKS):
        steps.append(
            RetrievalPlanStep(
                step=1,
                query=goal,
                source_scope="all",
                tool_name=READ_TOOL_CONTEXT_CHUNKS,
                rationale="Fallback to scoped context retrieval.",
            )
        )
    return RetrievalPlan(
        enabled=True,
        max_steps=config.max_steps,
        min_confidence=config.min_confidence,
        steps=steps,
        intent_route=route,
        graph_expansion=graph,
    )


def join_query(*parts: str) -> str:
    return " ".join(unique_strings([part for part in parts if part.strip()]))
