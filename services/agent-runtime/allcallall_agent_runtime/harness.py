"""Agent Runtime Harness for request normalization, graph execution, and loop projection."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from .config import config as app_config
from .dag import build_workflow_graph
from .helpers import SUPPORTED_WORKFLOWS, normalize_workflow_preset
from .models import (
    AgentHarnessMetadata,
    AgentRunRequest,
    AgentRunResponse,
    ContextSufficiency,
    CriticResult,
    EvidencePack,
    GraphExpansion,
    IntentRoute,
    LoopBudget,
    LoopSpec,
    LoopStep,
    LoopStopReason,
    LoopTrace,
    MemoryReflection,
    MeetingBriefRequest,
    MeetingBriefResponse,
    RetrievalPlan,
    RouteDecision,
    RiskAssessment,
    RoleResult,
    TraceEvent,
    WorkflowRequest,
    WorkflowResponse,
)
from .prompts import prompt_version_for
from .providers import ProviderError, create_provider
from .tool_bridge import GoToolBridge


class AllCallAllAgentHarness:
    """Run Agent workflows with consistent contracts, trace, and eval projection."""

    name = "allcallall_v1"
    graph_name = "supervisor_workflow_with_bounded_loops"

    def run_meeting_brief(self, request: MeetingBriefRequest) -> MeetingBriefResponse:
        return self.run_workflow(request.model_copy(update={"preset": "meeting_brief"}))

    def run_react_agent(self, request: AgentRunRequest) -> AgentRunResponse:
        return self.run_workflow(request.model_copy(update={"preset": "react_general"}))

    def run_workflow(self, request: WorkflowRequest) -> WorkflowResponse:
        preset = normalize_workflow_preset(request.preset)
        if preset not in SUPPORTED_WORKFLOWS:
            request = request.model_copy(update={"preset": preset})
            return self._failure_response(
                request,
                provider_name=app_config.provider or "rules",
                error=f"unsupported workflow preset: {request.preset}",
                trace=[],
            )

        request = request.model_copy(update={"preset": preset})
        provider_name = app_config.provider or "rules"
        try:
            provider = create_provider()
            provider_name = provider.name
            result = build_workflow_graph().invoke(
                {
                    "request": request,
                    "provider": provider,
                    "tool_bridge": GoToolBridge(),
                    "trace_events": [],
                    "role_results": [],
                }
            )
        except ProviderError as exc:
            return self._failure_response(
                request,
                provider_name=provider_name,
                error=f"{exc.kind}: {exc}",
                trace=[
                    TraceEvent(
                        event="provider.error",
                        node="provider",
                        status="failed",
                        metadata={"kind": exc.kind, "retryable": exc.retryable},
                    )
                ],
            )

        return self._response_from_graph_result(request, provider_name, result)

    def _failure_response(
        self,
        request: WorkflowRequest,
        provider_name: str,
        error: str,
        trace: list[TraceEvent],
    ) -> WorkflowResponse:
        prompt_version = prompt_version_for(request)
        return WorkflowResponse(
            status="failed",
            provider=provider_name,
            error=error,
            prompt_version=prompt_version,
            trace_events=trace,
            harness=self._harness_metadata(request, prompt_version),
            route_decision=self._route_decision(request, IntentRoute()),
            critic_result=CriticResult(
                passed=False,
                issues=[error],
                budget_respected=True,
                write_proposal_safe=True,
                grounding_passed=False,
                context_sufficient=False,
            ),
            stop_reason="runtime_error",
        )

    def _response_from_graph_result(
        self,
        request: WorkflowRequest,
        provider_name: str,
        result: dict[str, Any],
    ) -> WorkflowResponse:
        proposed = result.get("proposed_tool_calls", [])
        status = "requires_action" if proposed else "ready"
        trace_events = result.get("trace_events", [])
        role_results = result.get("role_results", [])
        intent_route = result.get("intent_route", IntentRoute())
        context_sufficiency = result.get("context_sufficiency", ContextSufficiency())
        evidence_pack = result.get("evidence_pack", EvidencePack())
        grounding = result.get("grounding_check_result", {})
        prompt_version = result.get("prompt_version", prompt_version_for(request))
        loop_traces = self._loop_traces(request, role_results)
        budget = self._aggregate_budget(loop_traces, proposed)
        critic_result = result.get("critic_result") or self._critic_result(
            context_sufficiency,
            grounding,
            evidence_pack,
            proposed,
            loop_traces,
        )
        stop_reason = self._stop_reason(status, context_sufficiency, critic_result, loop_traces)

        return WorkflowResponse(
            status=status,
            provider=provider_name,
            summary=result.get("summary", ""),
            action_items=result.get("action_items", []),
            next_step=result.get("next_step", ""),
            risk_flags=result.get("risk_flags", []),
            citations=result.get("citations", []),
            role_results=role_results,
            trace_events=trace_events,
            proposed_tool_calls=proposed,
            prompt_version=prompt_version,
            grounding_check_result=grounding,
            retrieval_plan=result.get("retrieval_plan") or RetrievalPlan(),
            retrieval_attempts=result.get("retrieval_attempts", []),
            evidence_pack=evidence_pack,
            context_sufficiency=context_sufficiency,
            intent_route=intent_route,
            route_decision=self._route_decision(request, intent_route),
            critic_result=critic_result,
            harness=self._harness_metadata(request, prompt_version),
            loop_traces=loop_traces,
            stop_reason=stop_reason,
            budget=budget,
            graph_expansion=result.get("graph_expansion", GraphExpansion()),
            memory_reflection=result.get("memory_reflection", MemoryReflection()),
            risk_assessment=result.get("risk_assessment", RiskAssessment()),
        )

    def _harness_metadata(self, request: WorkflowRequest, prompt_version: str) -> AgentHarnessMetadata:
        modalities = ["text"]
        if request.meeting_transcripts:
            modalities.append("audio_transcript")
        for attachment in request.attachments:
            if attachment.modality == "image":
                modalities.append("image_metadata")
            elif attachment.modality == "audio":
                modalities.append("audio_transcript")
            elif attachment.modality == "video":
                modalities.append("video_transcript")
            else:
                modalities.append(attachment.modality)
        return AgentHarnessMetadata(
            name=self.name,
            graph_name=self.graph_name,
            prompt_version=prompt_version,
            input_modalities=sorted(set(modalities)),
        )

    def _route_decision(self, request: WorkflowRequest, intent_route: IntentRoute) -> RouteDecision:
        route = "CHAT"
        if request.preset == "meeting_brief":
            route = "MEETING_RECAP"
        elif request.preset == "follow_up_planner":
            route = "FOLLOW_UP"
        elif intent_route.intent == "risk" or request.preset == "risk_review":
            route = "RISK"
        elif intent_route.intent == "consult" or request.preset == "context_qa":
            route = "CONSULT"
        return RouteDecision(
            route=route,
            intent=intent_route.intent,
            target_workflow=request.preset or intent_route.target_workflow,
            confidence=intent_route.confidence,
            rationale=intent_route.rationale,
            retrieval_strategy=intent_route.retrieval_strategy,
        )

    def _loop_traces(self, request: WorkflowRequest, role_results: list[RoleResult]) -> list[LoopTrace]:
        traces: list[LoopTrace] = []
        for result in role_results:
            role_events = result.react_trace
            if not role_events:
                continue
            events_by_iteration: dict[int, list[TraceEvent]] = defaultdict(list)
            for event in role_events:
                iteration = event.iteration or int(event.metadata.get("iteration", 0) or 0)
                if iteration:
                    events_by_iteration[iteration].append(event)
            max_steps = self._role_max_steps(request, result.role)
            steps: list[LoopStep] = []
            for iteration in sorted(events_by_iteration):
                events = events_by_iteration[iteration]
                observation_event = self._last_event(events, "react.observe") or self._last_event(events, "tool.result")
                tool_event = self._last_event(events, "tool.call") or observation_event
                failed = any(event.status == "failed" for event in events)
                stop_reason: LoopStopReason = "tool_error" if failed else "completed"
                if iteration >= max_steps and not failed:
                    stop_reason = "max_iterations"
                citation_ids = [item.chunk_id or item.source_id for item in result.citations if item.chunk_id or item.source_id]
                confidence = min(1.0, 0.35 + (0.15 * len(citation_ids)))
                steps.append(
                    LoopStep(
                        iteration=iteration,
                        role=result.role,
                        thought_summary=(observation_event.thought if observation_event else "")[:240],
                        selected_skill=tool_event.tool_name if tool_event else "",
                        input_schema=tool_event.tool_input if tool_event else {},
                        observation=(observation_event.observation if observation_event else "")[:600],
                        citation_ids=sorted(set(citation_ids)),
                        confidence=confidence,
                        stop_reason=stop_reason,
                        budget_used=LoopBudget(
                            max_steps=max_steps,
                            used_steps=iteration,
                            read_tool_calls=len([event for event in events if event.event == "tool.call"]),
                        ),
                    )
                )
            loop_stop = self._loop_stop_reason(max_steps, steps)
            traces.append(
                LoopTrace(
                    role=result.role,
                    spec=LoopSpec(
                        role=result.role,
                        objective=self._role_objective(request, result.role),
                        max_steps=max_steps,
                        allowed_tools=sorted(
                            {
                                event.tool_name
                                for event in role_events
                                if event.tool_name and event.event in {"tool.call", "react.observe"}
                            }
                        ),
                        stop_conditions=["confidence_reached", "max_iterations", "tool_error"],
                    ),
                    steps=steps,
                    stop_reason=loop_stop,
                    completed=loop_stop != "tool_error",
                    budget=LoopBudget(
                        max_steps=max_steps,
                        used_steps=len(steps),
                        read_tool_calls=len(
                            [event for event in role_events if event.event == "tool.call" and event.tool_name]
                        ),
                    ),
                )
            )
        return traces

    def _role_max_steps(self, request: WorkflowRequest, role: str) -> int:
        defaults = {
            "searcher": 3,
            "risk_analyst": 2,
            "risk_guardian": 2,
            "memory_agent": 1,
            "follow_up_planner": 2,
        }
        return max(1, min(request.max_iterations.get(role, defaults.get(role, 1)), app_config.loop_max_steps))

    def _role_objective(self, request: WorkflowRequest, role: str) -> str:
        if role == "risk_analyst":
            return "Inspect retrieved evidence for approval, blocker, timeline, or policy risk."
        if role == "memory_agent":
            return "Summarize durable memory and decide whether reflection should be proposed."
        if role == "follow_up_planner":
            return "Extract owner-bound action items and propose follow-up writes through approval."
        return f"Collect evidence for preset={request.preset} goal={request.goal[:120]}"

    def _loop_stop_reason(self, max_steps: int, steps: list[LoopStep]) -> LoopStopReason:
        if not steps:
            return "no_tool_needed"
        if any(step.stop_reason == "tool_error" for step in steps):
            return "tool_error"
        if len(steps) >= max_steps:
            return "max_iterations"
        if any(step.confidence >= 0.6 for step in steps):
            return "confidence_reached"
        return "completed"

    def _aggregate_budget(self, loop_traces: list[LoopTrace], proposed: list[Any]) -> LoopBudget:
        return LoopBudget(
            max_steps=sum(loop.budget.max_steps for loop in loop_traces),
            used_steps=sum(loop.budget.used_steps for loop in loop_traces),
            read_tool_calls=sum(loop.budget.read_tool_calls for loop in loop_traces),
            write_tool_proposals=len(proposed),
        )

    def _critic_result(
        self,
        sufficiency: ContextSufficiency,
        grounding: dict[str, Any],
        evidence_pack: EvidencePack,
        proposed: list[Any],
        loop_traces: list[LoopTrace],
    ) -> CriticResult:
        issues: list[str] = []
        grounding_passed = bool(grounding.get("grounded", True))
        if not grounding_passed:
            issues.append("grounding_failed")
        budget_respected = all(loop.budget.used_steps <= loop.budget.max_steps for loop in loop_traces)
        if not budget_respected:
            issues.append("loop_budget_exceeded")
        write_safe = all(getattr(item, "approval_required", False) for item in proposed)
        if not write_safe:
            issues.append("unsafe_write_proposal")
        if not sufficiency.sufficient:
            issues.append("insufficient_context_guarded")
        return CriticResult(
            passed=grounding_passed and budget_respected and write_safe,
            issues=issues,
            citation_coverage=evidence_pack.coverage,
            budget_respected=budget_respected,
            write_proposal_safe=write_safe,
            grounding_passed=grounding_passed,
            context_sufficient=sufficiency.sufficient,
        )

    def _stop_reason(
        self,
        status: str,
        sufficiency: ContextSufficiency,
        critic_result: CriticResult,
        loop_traces: list[LoopTrace],
    ) -> str:
        if not sufficiency.sufficient:
            return "insufficient_context"
        if not critic_result.grounding_passed:
            return "grounding_failed"
        if status == "requires_action":
            return "approval_required"
        if any(loop.stop_reason == "max_iterations" for loop in loop_traces):
            return "max_iterations"
        return "completed"

    @staticmethod
    def _last_event(events: list[TraceEvent], event_name: str) -> TraceEvent | None:
        for event in reversed(events):
            if event.event == event_name:
                return event
        return None
