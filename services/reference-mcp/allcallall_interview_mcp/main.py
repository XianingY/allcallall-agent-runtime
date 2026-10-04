"""Compatibility exports for :mod:`allcallall_reference_mcp.main`."""

from allcallall_reference_mcp.main import (
    BearerAuthMiddleware,
    DB_PATH,
    TOKEN_FILE,
    app,
    create_support_ticket,
    get_ticket,
    health,
    lookup_policy,
    mcp,
    metrics,
)

__all__ = [
    "BearerAuthMiddleware",
    "DB_PATH",
    "TOKEN_FILE",
    "app",
    "create_support_ticket",
    "get_ticket",
    "health",
    "lookup_policy",
    "mcp",
    "metrics",
]
