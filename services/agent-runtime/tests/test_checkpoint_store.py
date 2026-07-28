"""Checkpoint store decoupling tests.

Verifies that the harness can be backed by MySQL, SQLite, pure in-memory, or no
durability at all through a single ``CheckpointStore`` protocol, and that the
SQLite saver (zero-dependency) actually persists and reloads checkpoints.
"""

from __future__ import annotations

import operator
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from allcallall_agent_runtime.checkpoint.store import (
    CheckpointStore,
    MemoryCheckpointStore,
    NullCheckpointStore,
    SQLiteCheckpointStore,
)


class _State(TypedDict):
    x: Annotated[list[str], operator.add]


def _node(state: _State) -> dict[str, list[str]]:
    return {"x": ["run"]}


def _compile(checkpointer: BaseCheckpointSaver[Any] | None) -> Any:
    graph = StateGraph(_State)
    graph.add_node("n", _node)
    graph.add_edge(START, "n")
    graph.add_edge("n", END)
    return graph.compile(checkpointer=checkpointer)


def _roundtrip(store: CheckpointStore) -> None:
    app = _compile(store.make_checkpointer())
    cfg: RunnableConfig = {"configurable": {"thread_id": "t1"}}
    app.invoke({"x": []}, cfg)
    saver = store.make_checkpointer()
    assert saver is not None
    tup = saver.get_tuple(cfg)
    assert tup is not None, "checkpoint was not persisted"
    assert tup.checkpoint["channel_values"]["x"] == ["run"]
    assert len(list(saver.list(cfg))) >= 1


def test_null_store_has_no_checkpointer() -> None:
    assert NullCheckpointStore().make_checkpointer() is None


def test_memory_store_roundtrip() -> None:
    _roundtrip(MemoryCheckpointStore())


def test_sqlite_store_roundtrip() -> None:
    _roundtrip(SQLiteCheckpointStore())


def test_sqlite_store_file_path_roundtrip(tmp_path: Path) -> None:
    db = tmp_path / "checkpoints.sqlite"
    _roundtrip(SQLiteCheckpointStore(str(db)))


def test_store_kinds() -> None:
    assert NullCheckpointStore().kind == "none"
    assert MemoryCheckpointStore().kind == "memory"
    assert SQLiteCheckpointStore().kind == "sqlite"
    assert SQLiteCheckpointStore("/tmp/x.sqlite").kind == "sqlite"
