from __future__ import annotations

import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
import pymysql
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata

from allcallall_agent_runtime.checkpoint import (
    CheckpointExecutionBusy,
    CheckpointVersionConflict,
    MySQLCheckpointSaver,
)
from allcallall_agent_runtime.checkpoint.mysql import mysql_connection_factory

MYSQL_DSN = os.getenv("PY_AGENT_TEST_MYSQL_DSN", "").strip()
pytestmark = pytest.mark.skipif(not MYSQL_DSN, reason="PY_AGENT_TEST_MYSQL_DSN is not configured")


def _checkpoint(checkpoint_id: str) -> Checkpoint:
    return {
        "v": 1,
        "id": checkpoint_id,
        "ts": "2024-01-01T00:00:00+00:00",
        "channel_values": {"summary": "ok"},
        "channel_versions": {"summary": checkpoint_id},
        "versions_seen": {},
        "updated_channels": [],
    }


def _metadata() -> CheckpointMetadata:
    return {"source": "input", "step": 1, "parents": {}}


def _config(thread_id: str, execution_id: str, run_id: int, checkpoint_id: str = "") -> RunnableConfig:
    return {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": "",
            "checkpoint_id": checkpoint_id,
            "execution_id": execution_id,
            "workflow_run_id": run_id,
        }
    }


def test_mysql_execution_lock_serializes_one_thread() -> None:
    saver = MySQLCheckpointSaver(MYSQL_DSN)
    thread_id = f"workflow:lock:{uuid.uuid4()}"

    def contend() -> None:
        with saver.execution_lock(thread_id, "", timeout_seconds=0):
            pass

    with ThreadPoolExecutor(max_workers=1) as executor:
        with saver.execution_lock(thread_id, ""):
            future = executor.submit(contend)
            with pytest.raises(CheckpointExecutionBusy):
                future.result()

    with saver.execution_lock(thread_id, "", timeout_seconds=0):
        pass


def test_mysql_checkpoint_roundtrip_via_saver() -> None:
    saver = MySQLCheckpointSaver(MYSQL_DSN)
    run_id = uuid.uuid4().int % 1_000_000_000 + 1_000_000_000
    thread_id = f"workflow:{run_id}"
    execution_id = f"{thread_id}:attempt:1"
    config = _config(thread_id, execution_id, run_id)
    try:
        with saver.checkpoint_transaction(thread_id, execution_id):
            saver.put(config, _checkpoint("cp-1"), _metadata(), {"summary": "cp-1"})
            saver.put_writes(config, [("summary", "ok")], "task-1")

        execution_config = saver.find_execution_config(thread_id, "", execution_id)
        assert execution_config is not None
        checkpoint = saver.get_tuple(execution_config)
        assert checkpoint is not None
        assert checkpoint.config["configurable"]["checkpoint_version"] > 0

        items = list(saver.list({"configurable": {"thread_id": thread_id}}))
        assert items

        snapshot = saver.get_tuple({"configurable": {"thread_id": thread_id}})
        assert snapshot is not None
        assert snapshot.config["configurable"]["checkpoint_version"] > 0
    finally:
        saver.delete_thread(thread_id)


def test_mysql_checkpoint_transaction_rolls_back_partial_flush() -> None:
    saver = MySQLCheckpointSaver(MYSQL_DSN)
    run_id = uuid.uuid4().int % 1_000_000_000 + 2_000_000_000
    thread_id = f"workflow:{run_id}"
    execution_id = f"{thread_id}:atomic-failure"
    config = _config(thread_id, execution_id, run_id)
    trigger_name = "codex_fail_langgraph_write"
    try:
        drop_write_trigger(trigger_name)
        with pytest.raises(pymysql.MySQLError):
            with saver.checkpoint_transaction(thread_id, execution_id):
                saver.put(config, _checkpoint("cp-1"), _metadata(), {"summary": "cp-1"})
                saver.put_writes(config, [("summary", "ok")], "task-1")
                create_write_trigger(trigger_name)
        assert checkpoint_row_counts(thread_id) == (0, 0, 0)
    finally:
        drop_write_trigger(trigger_name)
        saver.delete_thread(thread_id)


@pytest.mark.anyio
async def test_mysql_async_checkpoint_transaction_commits_once() -> None:
    saver = MySQLCheckpointSaver(MYSQL_DSN)
    run_id = uuid.uuid4().int % 1_000_000_000 + 3_000_000_000
    thread_id = f"workflow:{run_id}"
    execution_id = f"{thread_id}:atomic-async"
    config = _config(thread_id, execution_id, run_id)
    try:
        async with saver.acheckpoint_transaction(thread_id, execution_id):
            await saver.aput(config, _checkpoint("cp-1"), _metadata(), {"summary": "cp-1"})
            await saver.aput_writes(config, [("summary", "ok")], "task-1")
        thread_count, checkpoint_count, write_count = checkpoint_row_counts(thread_id)
        assert thread_count == 1
        assert checkpoint_count > 0
        assert write_count > 0
    finally:
        await saver.adelete_thread(thread_id)


def test_mysql_checkpoint_transaction_detects_stale_namespace() -> None:
    run_id = uuid.uuid4().int % 1_000_000_000 + 5_000_000_000
    thread_id = f"workflow:{run_id}"
    ready_to_commit = Barrier(2)

    def contend(suffix: str) -> Exception | None:
        saver = MySQLCheckpointSaver(MYSQL_DSN)
        execution_id = f"{thread_id}:{suffix}"
        config = _config(thread_id, execution_id, run_id)
        try:
            with saver.checkpoint_transaction(thread_id, execution_id):
                saver.put(config, _checkpoint(f"cp-{suffix}"), _metadata(), {"summary": f"cp-{suffix}"})
                saver.put_writes(config, [("summary", f"cp-{suffix}")], "task-1")
                ready_to_commit.wait(timeout=10)
        except Exception as exc:
            return exc
        return None

    saver = MySQLCheckpointSaver(MYSQL_DSN)
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            failures = list(executor.map(contend, ("stale-a", "stale-b")))
        assert sum(failure is None for failure in failures) == 1
        assert sum(isinstance(failure, CheckpointVersionConflict) for failure in failures) == 1
        thread_count, checkpoint_count, write_count = checkpoint_row_counts(thread_id)
        assert thread_count == 1
        assert checkpoint_count > 0
        assert write_count > 0
    finally:
        saver.delete_thread(thread_id)


def checkpoint_row_counts(thread_id: str) -> tuple[int, int, int]:
    with mysql_connection_factory(MYSQL_DSN)() as connection, connection.cursor() as cursor:
        counts: list[int] = []
        for table in (
            "langgraph_checkpoint_threads",
            "langgraph_checkpoints",
            "langgraph_checkpoint_writes",
        ):
            cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE thread_id = %s", (thread_id,))
            row = cursor.fetchone()
            assert row is not None
            counts.append(int(row[0]))
    return counts[0], counts[1], counts[2]


def create_write_trigger(trigger_name: str) -> None:
    with mysql_connection_factory(MYSQL_DSN)() as connection, connection.cursor() as cursor:
        cursor.execute(
            f"""
            CREATE TRIGGER {trigger_name}
            BEFORE INSERT ON langgraph_checkpoint_writes
            FOR EACH ROW
            SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'checkpoint write fault injection'
            """
        )
        connection.commit()


def drop_write_trigger(trigger_name: str) -> None:
    with mysql_connection_factory(MYSQL_DSN)() as connection, connection.cursor() as cursor:
        cursor.execute(f"DROP TRIGGER IF EXISTS {trigger_name}")
        connection.commit()
