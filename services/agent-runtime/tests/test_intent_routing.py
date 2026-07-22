from __future__ import annotations

from allcallall_agent_runtime.helpers import route_request_intent
from allcallall_agent_runtime.models import ConversationMessage, WorkflowRequest


def _request(
    goal: str = "g",
    preset: str = "meeting_brief",
    messages: list[ConversationMessage] | None = None,
    notes: list[object] | None = None,
) -> WorkflowRequest:
    return WorkflowRequest(
        organization_id=1,
        user_id=2,
        conversation_id=3,
        workflow_run_id=9,
        preset=preset,
        goal=goal,
        messages=messages or [],
        notes=notes or [],
    )


def test_default_goal_routes_to_chat() -> None:
    route = route_request_intent(_request(goal="summarize the thread"))
    assert route.intent == "chat"


def test_risk_review_preset_routes_to_risk() -> None:
    route = route_request_intent(_request(preset="risk_review", goal="review"))
    assert route.intent == "risk"


def test_context_qa_preset_routes_to_consult() -> None:
    route = route_request_intent(_request(preset="context_qa", goal="q"))
    assert route.intent == "consult"


def test_message_body_consult_keyword_routes_to_consult() -> None:
    route = route_request_intent(
        _request(messages=[ConversationMessage(body="what is the policy on refunds?")])
    )
    assert route.intent == "consult"


def test_message_body_risk_keyword_routes_to_risk() -> None:
    route = route_request_intent(
        _request(messages=[ConversationMessage(body="we have a deadline risk and need approval")])
    )
    assert route.intent == "risk"


def test_goal_only_risk_keyword_still_routes_to_risk() -> None:
    route = route_request_intent(_request(goal="assess the approval risk"))
    assert route.intent == "risk"
