from __future__ import annotations

from pymysql.connections import Connection


def initialize_checkpoint_schema(connection: Connection) -> None:
    """Create the idempotent LangGraph checkpoint schema."""
    with connection.cursor() as cursor:
        # VARCHAR(150) keeps composite keys within MySQL's utf8mb4 index limit.
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS langgraph_checkpoint_threads (
                thread_id VARCHAR(150) NOT NULL,
                checkpoint_ns VARCHAR(150) NOT NULL DEFAULT '',
                current_version BIGINT NOT NULL DEFAULT 0,
                updated_at DATETIME(6),
                PRIMARY KEY (thread_id, checkpoint_ns)
            ) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin ROW_FORMAT=DYNAMIC
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS langgraph_checkpoints (
                thread_id VARCHAR(150) NOT NULL,
                checkpoint_ns VARCHAR(150) NOT NULL DEFAULT '',
                checkpoint_id VARCHAR(150) NOT NULL,
                parent_checkpoint_id VARCHAR(150),
                execution_id VARCHAR(150),
                workflow_run_id BIGINT,
                agent_run_id BIGINT,
                version BIGINT NOT NULL DEFAULT 0,
                checkpoint_type VARCHAR(150),
                checkpoint_blob LONGBLOB,
                metadata_type VARCHAR(150),
                metadata_blob LONGBLOB,
                created_at DATETIME(6),
                PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
            ) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin ROW_FORMAT=DYNAMIC
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS langgraph_checkpoint_writes (
                thread_id VARCHAR(150) NOT NULL,
                checkpoint_ns VARCHAR(150) NOT NULL DEFAULT '',
                checkpoint_id VARCHAR(150) NOT NULL,
                task_id VARCHAR(150) NOT NULL,
                task_path VARCHAR(150) NOT NULL DEFAULT '',
                write_index INT NOT NULL,
                channel VARCHAR(150) NOT NULL,
                value_type VARCHAR(150),
                value_blob LONGBLOB,
                created_at DATETIME(6),
                PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, task_path, write_index)
            ) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin ROW_FORMAT=DYNAMIC
            """
        )
        connection.commit()
