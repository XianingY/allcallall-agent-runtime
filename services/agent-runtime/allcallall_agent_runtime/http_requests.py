"""Shared outbound HTTP request helpers for the Agent Runtime."""

from __future__ import annotations

from http.cookiejar import CookieJar, DefaultCookiePolicy
from typing import Any

import httpx

from .config import AgentRuntimeConfig


class _RejectAllCookiesPolicy(DefaultCookiePolicy):
    """Prevent shared clients from retaining tenant-specific cookie state."""

    def set_ok(self, cookie: object, request: object) -> bool:
        return False


def _cookieless_jar() -> CookieJar:
    return CookieJar(policy=_RejectAllCookiesPolicy())


def build_http_client(config: AgentRuntimeConfig) -> httpx.Client:
    """Create a bounded, cookieless, process-lifetime HTTP client."""
    return httpx.Client(
        cookies=_cookieless_jar(),
        limits=httpx.Limits(
            max_connections=config.http_max_connections,
            max_keepalive_connections=config.http_max_keepalive_connections,
            keepalive_expiry=config.http_keepalive_expiry_sec,
        ),
        timeout=httpx.Timeout(
            connect=config.http_connect_timeout_sec,
            read=config.http_read_timeout_sec,
            write=config.http_write_timeout_sec,
            pool=config.http_pool_timeout_sec,
        ),
    )


def post_json_without_cookies(
    client: httpx.Client,
    url: str,
    *,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout_sec: float,
) -> httpx.Response:
    """POST JSON without replaying cookies from the client's shared cookie jar."""
    client.cookies.clear()
    request = client.build_request(
        "POST",
        url,
        json=payload,
        headers=headers,
        timeout=timeout_sec,
    )
    request.headers.pop("Cookie", None)
    try:
        return client.send(request)
    finally:
        client.cookies.clear()
