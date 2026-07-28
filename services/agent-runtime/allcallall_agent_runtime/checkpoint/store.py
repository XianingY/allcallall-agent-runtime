"""Checkpoint store abstraction: decouples the harness from any database.

`AllCallAllAgentHarness` depends only on the :class:`CheckpointStore` protocol,
never on a concrete backend. Swapping MySQL for SQLite or pure in-memory state
is a one-line configuration change and requires no harness edits, which is what
lets each layer evolve and be tested independently.
"""

from __future__ import annotations

from typing import Protocol

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver

from .mysql import MySQLCheckpointSaver
from .sqlite_saver import SQLiteCheckpointSaver


# Re-export langgraph's in-memory saver under the project's naming convention.
MemoryCheckpointSaver = MemorySaver


class CheckpointStore(Protocol):
    """Produces the LangGraph checkpointer the harness should use."""

    kind: str

    def make_checkpointer(self) -> BaseCheckpointSaver | None:
        """Return a checkpointer, or ``None`` when durability is disabled."""
        ...


class NullCheckpointStore:
    """No durability — matches the previous default (``checkpointer=None``)."""

    kind = "none"

    def make_checkpointer(self) -> BaseCheckpointSaver | None:
        return None


class MySQLCheckpointStore:
    """Durable MySQL checkpoints via the existing transactional saver."""

    kind = "mysql"

    def __init__(self, dsn: str) -> None:
        if not dsn.strip():
            raise ValueError("PY_AGENT_CHECKPOINT_MYSQL_DSN is required when MySQL checkpoints are enabled")
        self._dsn = dsn
        self._checkpointer: BaseCheckpointSaver | None = None

    def make_checkpointer(self) -> BaseCheckpointSaver | None:
        if self._checkpointer is None:
            self._checkpointer = MySQLCheckpointSaver(self._dsn)
        return self._checkpointer


class SQLiteCheckpointStore:
    """Zero-dependency SQLite checkpoints for local/dev and reproducible tests."""

    kind = "sqlite"

    def __init__(self, path: str = ":memory:") -> None:
        self._path = path
        self._checkpointer: BaseCheckpointSaver | None = None

    def make_checkpointer(self) -> BaseCheckpointSaver | None:
        if self._checkpointer is None:
            self._checkpointer = SQLiteCheckpointSaver(self._path)
        return self._checkpointer


class MemoryCheckpointStore:
    """Pure in-memory checkpoints (no external process), built on langgraph's saver."""

    kind = "memory"

    def __init__(self) -> None:
        self._checkpointer: BaseCheckpointSaver | None = None

    def make_checkpointer(self) -> BaseCheckpointSaver | None:
        if self._checkpointer is None:
            self._checkpointer = MemoryCheckpointSaver()
        return self._checkpointer
