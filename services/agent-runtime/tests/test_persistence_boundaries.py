from __future__ import annotations

from allcallall_agent_runtime.checkpoint.mysql import mysql_connection_factory
from allcallall_agent_runtime.persistence.mysql_pool import (
    mysql_connection_factory as canonical_mysql_connection_factory,
)


def test_legacy_mysql_factory_import_points_to_persistence_boundary() -> None:
    assert mysql_connection_factory is canonical_mysql_connection_factory
