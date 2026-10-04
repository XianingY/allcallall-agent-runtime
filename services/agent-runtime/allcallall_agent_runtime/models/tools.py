from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ToolProposal(BaseModel):
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""
    idempotency_key: str = ""
    approval_required: bool = True
    execution_mode: Literal["proposal_only", "async_after_approval"] = "async_after_approval"
    queue_name: str = "agent_writebacks"
    priority: Literal["low", "normal", "high"] = "normal"
    max_attempts: int = 3
    rate_limit_key: str = ""
    dead_letter_queue: str = "agent_writebacks_dead_letter"

