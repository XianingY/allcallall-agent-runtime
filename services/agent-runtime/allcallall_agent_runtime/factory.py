"""Composition root: wires the three decoupled harness layers together.

This is the only place that knows how to turn configuration into a fully
constructed :class:`AllCallAllAgentHarness`. Production calls
:func:`build_agent_harness` with no arguments; tests pass explicit
``checkpoint_store`` / ``tool_layer`` / ``provider`` overrides to isolate layers.
"""

from __future__ import annotations

from typing import Any

from .checkpoint.store import (
    CheckpointStore,
    MemoryCheckpointStore,
    MySQLCheckpointStore,
    NullCheckpointStore,
    SQLiteCheckpointStore,
)
from .config import config as _default_config
from .harness import AllCallAllAgentHarness
from .providers.base import LLMProvider
from .tool_layer import GoToolBridgeLayer, ToolLayer


def build_checkpoint_store(cfg: Any = _default_config) -> CheckpointStore:
    """Select a checkpoint backend from configuration (decoupled from harness)."""
    store = (cfg.checkpoint_store or "").strip().lower()
    if store == "mysql" or (not store and cfg.checkpoint_mysql_enabled):
        return MySQLCheckpointStore(cfg.checkpoint_mysql_dsn)
    if store == "sqlite":
        return SQLiteCheckpointStore(cfg.checkpoint_sqlite_path or ":memory:")
    if store == "memory":
        return MemoryCheckpointStore()
    return NullCheckpointStore()


def build_tool_layer(cfg: Any = _default_config) -> ToolLayer:
    """Select the tool execution layer (production always uses the Go bridge)."""
    return GoToolBridgeLayer()


def build_agent_harness(
    *,
    checkpoint_store: CheckpointStore | None = None,
    tool_layer: ToolLayer | None = None,
    provider: LLMProvider | None = None,
    cfg: Any = _default_config,
) -> AllCallAllAgentHarness:
    return AllCallAllAgentHarness(
        checkpoint_store=checkpoint_store or build_checkpoint_store(cfg),
        tool_layer=tool_layer or build_tool_layer(cfg),
        provider=provider,
    )
