from __future__ import annotations

import queue
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from threading import Lock
from typing import cast
from urllib.parse import unquote, urlparse

import pymysql
from pymysql.connections import Connection


ConnectionFactory = Callable[[], Connection]


def mysql_connection_factory(dsn: str) -> ConnectionFactory:
    parsed = urlparse(dsn)
    if parsed.scheme not in {"mysql", "mysql+pymysql"} or not parsed.hostname or not parsed.path.strip("/"):
        raise ValueError("checkpoint MySQL DSN must be mysql://user:password@host:3306/database")

    def connect() -> Connection:
        return pymysql.connect(
            host=cast(str, parsed.hostname),
            port=parsed.port or 3306,
            user=unquote(parsed.username or ""),
            password=unquote(parsed.password or ""),
            database=parsed.path.strip("/"),
            charset="utf8mb4",
            autocommit=False,
        )

    return connect


class MySQLConnectionPool:
    """Small bounded pool for reusable checkpoint database connections."""

    def __init__(self, connection_factory: ConnectionFactory, pool_size: int) -> None:
        self._connection_factory = connection_factory
        self._pool_size = max(1, int(pool_size))
        self._free_connections: queue.Queue[Connection] = queue.Queue(maxsize=self._pool_size)
        self._created_connections = 0
        self._pool_lock = Lock()

    @contextmanager
    def connection(self) -> Iterator[Connection]:
        connection = self._acquire()
        try:
            yield connection
        except BaseException:
            self._retire_and_replenish(connection)
            raise
        else:
            self._release(connection)

    def _acquire(self) -> Connection:
        try:
            return self._free_connections.get_nowait()
        except queue.Empty:
            with self._pool_lock:
                if self._created_connections < self._pool_size:
                    self._created_connections += 1
                    return self._connection_factory()
            return self._free_connections.get()

    def _release(self, connection: Connection) -> None:
        try:
            connection.rollback()
        except Exception:
            self._retire_and_replenish(connection)
            return
        try:
            self._free_connections.put_nowait(connection)
        except queue.Full:  # pragma: no cover - defensive
            self._retire_and_replenish(connection)

    def _retire_and_replenish(self, connection: Connection) -> None:
        try:
            connection.close()
        except Exception:
            pass
        with self._pool_lock:
            self._created_connections -= 1
            try:
                fresh = self._connection_factory()
                self._created_connections += 1
                self._free_connections.put_nowait(fresh)
            except Exception:
                # A later acquire retries growth when the database is available.
                pass
