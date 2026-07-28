"""SQLite-backed LangGraph checkpoint saver (zero-dependency, reproducible).

Shares the three-table schema used by the MySQL backend but with SQLite
syntax, so the Agent Runtime can run durable, resumable workflows in a
reproducible test environment or a local deployment without standing up a
MySQL server.
"""

from __future__ import annotations

import sqlite3
import threading
from typing import Any, Iterator, Sequence, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    ChannelVersions,
    get_checkpoint_id,
    get_checkpoint_metadata,
)
import anyio

from .mysql import checkpoint_safe_config, optional_int


class SQLiteCheckpointSaver(BaseCheckpointSaver):
    """Single-connection SQLite checkpoint saver.

    A single shared connection is required because every ``sqlite3.connect(":memory:")``
    produces a distinct in-memory database; reusing one connection keeps the
    ``:memory:`` store alive across the many ``put``/``get_tuple`` calls a single
    graph invocation makes.
    """

    def __init__(self, path: str = ":memory:") -> None:
        super().__init__()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS langgraph_checkpoint_threads (
                    thread_id TEXT NOT NULL,
                    checkpoint_ns TEXT NOT NULL DEFAULT '',
                    current_version INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT,
                    PRIMARY KEY (thread_id, checkpoint_ns)
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS langgraph_checkpoints (
                    thread_id TEXT NOT NULL,
                    checkpoint_ns TEXT NOT NULL DEFAULT '',
                    checkpoint_id TEXT NOT NULL,
                    parent_checkpoint_id TEXT,
                    execution_id TEXT,
                    workflow_run_id INTEGER,
                    agent_run_id INTEGER,
                    version INTEGER NOT NULL DEFAULT 0,
                    checkpoint_type TEXT,
                    checkpoint_blob BLOB,
                    metadata_type TEXT,
                    metadata_blob BLOB,
                    created_at TEXT,
                    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS langgraph_checkpoint_writes (
                    thread_id TEXT NOT NULL,
                    checkpoint_ns TEXT NOT NULL DEFAULT '',
                    checkpoint_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    task_path TEXT NOT NULL DEFAULT '',
                    write_index INTEGER NOT NULL,
                    channel TEXT NOT NULL,
                    value_type TEXT,
                    value_blob BLOB,
                    created_at TEXT,
                    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, task_path, write_index)
                )
                """
            )

    # ------------------------------------------------------------------ read
    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        configurable = config["configurable"]
        thread_id = str(configurable["thread_id"])
        checkpoint_ns = str(configurable.get("checkpoint_ns", ""))
        checkpoint_id = get_checkpoint_id(config)
        query = (
            "SELECT thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, version, "
            "checkpoint_type, checkpoint_blob, metadata_type, metadata_blob "
            "FROM langgraph_checkpoints WHERE thread_id = ? AND checkpoint_ns = ?"
        )
        params: list[Any] = [thread_id, checkpoint_ns]
        if checkpoint_id:
            query += " AND checkpoint_id = ?"
            params.append(checkpoint_id)
        else:
            query += " ORDER BY version DESC, checkpoint_id DESC LIMIT 1"
        with self._lock, self._conn:
            row = self._conn.execute(query, params).fetchone()
            if row is None:
                return None
            write_rows = self._conn.execute(
                "SELECT task_id, task_path, write_index, channel, value_type, value_blob "
                "FROM langgraph_checkpoint_writes WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?",
                (thread_id, checkpoint_ns, row["checkpoint_id"]),
            ).fetchall()
        saved_config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": str(row["checkpoint_id"]),
                "checkpoint_version": int(row["version"]),
            }
        }
        parent_id = str(row["parent_checkpoint_id"] or "")
        parent_config: RunnableConfig | None = None
        if parent_id:
            parent_config = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": parent_id,
                }
            }
        return CheckpointTuple(
            config=saved_config,
            checkpoint=cast(
                Checkpoint,
                self.serde.loads_typed((str(row["checkpoint_type"]), bytes(row["checkpoint_blob"]))),
            ),
            metadata=cast(
                CheckpointMetadata,
                self.serde.loads_typed((str(row["metadata_type"]), bytes(row["metadata_blob"]))),
            ),
            parent_config=parent_config,
            pending_writes=[
                (
                    str(item["task_id"]),
                    str(item["channel"]),
                    self.serde.loads_typed((str(item["value_type"]), bytes(item["value_blob"]))),
                )
                for item in write_rows
            ],
        )

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        requested_thread_id = ""
        requested_ns = ""
        if config is not None:
            configurable = config["configurable"]
            requested_thread_id = str(configurable.get("thread_id", ""))
            requested_ns = str(configurable.get("checkpoint_ns", ""))
        before_id = get_checkpoint_id(before) if before is not None else None
        query = (
            "SELECT checkpoint_id FROM langgraph_checkpoints "
            "WHERE thread_id = ? AND checkpoint_ns = ?"
        )
        params: list[Any] = [requested_thread_id, requested_ns]
        if before_id:
            query += " AND checkpoint_id < ?"
            params.append(before_id)
        query += " ORDER BY version DESC, checkpoint_id DESC"
        if limit is not None:
            query += f" LIMIT {int(limit)}"
        with self._lock, self._conn:
            rows = self._conn.execute(query, params).fetchall()
        for row in rows:
            item = self.get_tuple(
                {
                    "configurable": {
                        "thread_id": requested_thread_id,
                        "checkpoint_ns": requested_ns,
                        "checkpoint_id": str(row["checkpoint_id"]),
                    }
                }
            )
            if item is None:
                continue
            if filter and not self._metadata_matches(item.metadata, filter):
                continue
            yield item

    # ----------------------------------------------------------------- write
    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        del new_versions
        configurable = config["configurable"]
        thread_id = str(configurable["thread_id"])
        checkpoint_ns = str(configurable.get("checkpoint_ns", ""))
        checkpoint_id = str(checkpoint["id"])
        parent_checkpoint_id = str(configurable.get("checkpoint_id", ""))
        checkpoint_type, checkpoint_blob = self.serde.dumps_typed(
            self._drop_unserializable_channels(checkpoint)
        )
        metadata_type, metadata_blob = self.serde.dumps_typed(
            get_checkpoint_metadata(checkpoint_safe_config(config), metadata)
        )
        execution_id = str(configurable.get("execution_id", ""))
        workflow_run_id = optional_int(configurable.get("workflow_run_id"))
        agent_run_id = optional_int(configurable.get("agent_run_id"))
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO langgraph_checkpoint_threads (thread_id, checkpoint_ns, current_version, updated_at)
                VALUES (?, ?, 1, CURRENT_TIMESTAMP)
                ON CONFLICT(thread_id, checkpoint_ns) DO UPDATE SET
                    current_version = current_version + 1, updated_at = CURRENT_TIMESTAMP
                """,
                (thread_id, checkpoint_ns),
            )
            version_row = self._conn.execute(
                "SELECT current_version FROM langgraph_checkpoint_threads WHERE thread_id = ? AND checkpoint_ns = ?",
                (thread_id, checkpoint_ns),
            ).fetchone()
            version = int(version_row["current_version"])
            self._conn.execute(
                """
                INSERT INTO langgraph_checkpoints (
                    thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, execution_id,
                    workflow_run_id, agent_run_id, version, checkpoint_type, checkpoint_blob,
                    metadata_type, metadata_blob, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(thread_id, checkpoint_ns, checkpoint_id) DO UPDATE SET
                    parent_checkpoint_id = excluded.parent_checkpoint_id,
                    checkpoint_type = excluded.checkpoint_type,
                    checkpoint_blob = excluded.checkpoint_blob,
                    metadata_type = excluded.metadata_type,
                    metadata_blob = excluded.metadata_blob,
                    version = excluded.version
                """,
                (
                    thread_id,
                    checkpoint_ns,
                    checkpoint_id,
                    parent_checkpoint_id,
                    execution_id,
                    workflow_run_id,
                    agent_run_id,
                    version,
                    str(checkpoint_type),
                    bytes(checkpoint_blob),
                    str(metadata_type),
                    bytes(metadata_blob),
                ),
            )
            self._conn.commit()
        return {
            "configurable": {
                **configurable,
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
                "checkpoint_version": version,
            }
        }

    def _drop_unserializable_channels(self, checkpoint: Checkpoint) -> Checkpoint:
        """Replace channels that cannot be serialized (e.g. live ``provider`` /
        ``tool_bridge`` objects injected per-run) with ``None`` before
        persistence. They are re-injected by the harness on every invoke, so
        dropping them from the durable checkpoint is safe and avoids forcing
        every transient dependency into the serde."""
        channel_values = dict(checkpoint.get("channel_values", {}))
        dirty = False
        for key, value in list(channel_values.items()):
            try:
                self.serde.dumps_typed({"__probe__": value})
            except Exception:
                channel_values[key] = None
                dirty = True
        if not dirty:
            return checkpoint
        return {**checkpoint, "channel_values": channel_values}

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        configurable = config["configurable"]
        thread_id = str(configurable["thread_id"])
        checkpoint_ns = str(configurable.get("checkpoint_ns", ""))
        checkpoint_id = str(configurable["checkpoint_id"])
        records: list[tuple[Any, ...]] = []
        for index, (channel, value) in enumerate(writes):
            try:
                value_type, value_blob = self.serde.dumps_typed(value)
            except Exception:
                # Non-serializable transient (e.g. the per-run provider/tool_bridge
                # object captured as a write) is dropped; the harness re-injects it.
                value_type, value_blob = self.serde.dumps_typed(None)
            records.append(
                (
                    thread_id,
                    checkpoint_ns,
                    checkpoint_id,
                    task_id,
                    task_path,
                    WRITES_IDX_MAP.get(channel, index),
                    channel,
                    str(value_type),
                    bytes(value_blob),
                )
            )
        if not records:
            return
        with self._lock, self._conn:
            self._conn.executemany(
                """
                INSERT INTO langgraph_checkpoint_writes (
                    thread_id, checkpoint_ns, checkpoint_id, task_id, task_path,
                    write_index, channel, value_type, value_blob, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(thread_id, checkpoint_ns, checkpoint_id, task_id, task_path, write_index) DO UPDATE SET
                    channel = excluded.channel, value_type = excluded.value_type, value_blob = excluded.value_blob
                """,
                records,
            )
            self._conn.commit()

    def delete_thread(self, thread_id: str) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM langgraph_checkpoint_writes WHERE thread_id = ?", (thread_id,))
            self._conn.execute("DELETE FROM langgraph_checkpoints WHERE thread_id = ?", (thread_id,))
            self._conn.execute("DELETE FROM langgraph_checkpoint_threads WHERE thread_id = ?", (thread_id,))
            self._conn.commit()

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _metadata_matches(metadata: CheckpointMetadata, expected: dict[str, Any]) -> bool:
        return all(metadata.get(key) == value for key, value in expected.items())

    # ------------------------------------------------------------- async API
    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        return await anyio.to_thread.run_sync(self.get_tuple, config)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ):
        items = await anyio.to_thread.run_sync(
            lambda: list(self.list(config, filter=filter, before=before, limit=limit))
        )
        for item in items:
            yield item

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return await anyio.to_thread.run_sync(self.put, config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        await anyio.to_thread.run_sync(self.put_writes, config, writes, task_id, task_path)

    async def adelete_thread(self, thread_id: str) -> None:
        await anyio.to_thread.run_sync(self.delete_thread, thread_id)
