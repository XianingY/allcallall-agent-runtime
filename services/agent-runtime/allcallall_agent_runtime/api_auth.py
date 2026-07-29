"""Request authentication for the production workflow run endpoints.

The runtime already authenticates *outbound* calls to the Go backend via the
shared ``AGENT_RUNTIME_TOOL_BRIDGE_TOKEN``. Inbound ``/v1/*`` run endpoints,
however, were previously unauthenticated. This module closes that gap with a
FastAPI dependency, :func:`require_auth`, that validates an
``Authorization: Bearer <token>`` header.

Safe default (backward compatible): if ``PY_AGENT_API_TOKEN`` is not set the
runtime does **not** refuse requests — instead it logs a one-time warning and
lets traffic through. This prevents a configuration change from breaking
existing deployments that rely on a perimeter/firewall for access. Once an
operator sets the token, every run endpoint enforces it and returns ``401`` for
missing or mismatched credentials.
"""

from __future__ import annotations

import hmac
import logging
from typing import Any

from fastapi import Header, HTTPException, status

from .config import config

logger = logging.getLogger(__name__)

# Module-level guard so the "token not configured" warning is emitted once per
# process rather than on every request. Exposed for tests to reset.
_warned_missing_token = False


def reset_auth_warning() -> None:
    """Reset the one-time warning guard (test helper)."""
    global _warned_missing_token
    _warned_missing_token = False


def require_auth(authorization: str | None = Header(default=None)) -> None:
    """FastAPI dependency enforcing a bearer-token on protected run endpoints.

    Raises ``HTTPException(401)`` when a token is configured but the request is
    missing a valid ``Authorization: Bearer <token>`` header. When no token is
    configured the request is allowed (with a one-time warning), preserving
    backward compatibility for deployments that have not yet set
    ``PY_AGENT_API_TOKEN``.
    """
    expected = config.api_token
    if not expected:
        global _warned_missing_token
        if not _warned_missing_token:
            _warned_missing_token = True
            logger.warning(
                "PY_AGENT_API_TOKEN is not set; /v1 run endpoints are unauthenticated. "
                "Set PY_AGENT_API_TOKEN to require a bearer token on workflow run endpoints."
            )
        return

    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    parts: list[str] = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer" or not parts[1]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not hmac.compare_digest(parts[1], expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )


def auth_scheme() -> dict[str, Any]:
    """Return an OpenAPI security scheme fragment documenting the bearer auth."""
    return {
        "type": "http",
        "scheme": "bearer",
        "description": "Bearer token from PY_AGENT_API_TOKEN (optional; if unset, endpoints are open).",
    }
