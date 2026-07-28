"""Tool layer abstraction: decouples the harness from concrete tool execution.

The workflow nodes only ever call ``state["tool_bridge"].execute_read_tool(...)``.
By routing bridge construction through a :class:`ToolLayer`, the harness can be
wired against the real :class:`GoToolBridge` in production or a deterministic
stub in tests, without touching any node code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .models import WorkflowRequest
from .tool_bridge import GoToolBridge, ToolObservation


@runtime_checkable
class ToolBridgeLike(Protocol):
    """Structural interface every tool bridge (real or stub) must satisfy."""

    def configured(self) -> bool:
        ...

    def execute_read_tool(
        self,
        request: WorkflowRequest,
        tool_name: str,
        tool_input: dict[str, Any],
    ) -> ToolObservation | None:
        ...


@runtime_checkable
class ToolLayer(Protocol):
    """Produces the tool bridge a workflow run should use."""

    def build(self) -> ToolBridgeLike:
        ...


class GoToolBridgeLayer:
    """Production tool layer: constructs the real Go backend bridge."""

    def build(self) -> GoToolBridge:
        return GoToolBridge()


@dataclass
class StubGoToolBridge:
    """Deterministic read-tool stand-in for tests and offline evaluation."""

    observations: dict[tuple[str, str], ToolObservation] = field(default_factory=dict)
    default: ToolObservation | None = None

    def configured(self) -> bool:
        return True

    def execute_read_tool(
        self,
        request: WorkflowRequest,
        tool_name: str,
        tool_input: dict[str, Any],
    ) -> ToolObservation | None:
        key = (tool_name, json.dumps(tool_input, sort_keys=True, default=str))
        return self.observations.get(key, self.default)


@dataclass
class StubToolLayer:
    """Tool layer that hands every run the same deterministic stub bridge."""

    observations: dict[tuple[str, str], ToolObservation] = field(default_factory=dict)
    default: ToolObservation | None = None

    def build(self) -> StubGoToolBridge:
        return StubGoToolBridge(observations=self.observations, default=self.default)
