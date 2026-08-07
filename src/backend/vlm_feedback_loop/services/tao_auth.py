# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TAO bearer-token resolver with JWT auto-exchange.

The Blueprint supports two forms for ``TAO_API_KEY``:

1. An **NGC Personal API Key** (``nvapi-...``). FTMS rejects this as a
   Bearer token directly — it expects a JWT obtained by exchanging the
   NGC key via ``POST /api/v2/login``. This module detects the NGC form,
   calls :func:`services.tao_client.login_tao` once, caches the JWT
   in-memory, and uses it for all subsequent TAO calls until a 401 forces
   a re-exchange.

2. A **pre-exchanged JWT**. Operators who generate the JWT out-of-band
   (e.g., via the TAO UI or a curl login) can store the JWT directly in
   ``TAO_API_KEY`` — this module passes it through unchanged.

The cache is process-wide and keyed on (``TAO_API_BASE_URL``,
``TAO_ORG_NAME``, hash-of-NGC-key) so multiple deployments in the same
process would not cross-pollinate. :func:`invalidate_tao_bearer` is
callable from any TAO service that observes a 401.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

import httpx

from vlm_feedback_loop.config import Settings
from vlm_feedback_loop.services.http_client import HttpResult, resilient_request
from vlm_feedback_loop.services.nim_client import NIM_DEFAULT_HEADERS
from vlm_feedback_loop.services.runtime_secrets import get_effective_secret

logger = logging.getLogger("vlm_feedback_loop.services.tao_auth")

# Any ``TAO_API_KEY`` with this prefix is treated as an NGC Personal API
# Key requiring a /login exchange before use as a Bearer token.
_NGC_KEY_PREFIX = "nvapi-"


class TaoAuthError(RuntimeError):
    """Raised when TAO authentication cannot be established.

    Covers both unconfigured credentials (``TAO_API_KEY`` missing) and
    failed ``/login`` exchanges (NGC key rejected, endpoint unreachable,
    network error). Subclasses :class:`RuntimeError` so existing callers
    that catch ``RuntimeError`` continue to work; new callers MAY catch
    this class specifically to render structured "TAO unreachable"
    responses instead of propagating a 500 to the UI.
    """


# In-memory JWT cache: cache key (base_url|org|key-hash) → resolved JWT.
_token_cache: dict[str, str] = {}


def tao_base_url(settings: Settings) -> str:
    """Return ``TAO_API_BASE_URL`` normalized (no trailing slash).

    The single place TAO services derive the FTMS base URL from, so
    endpoint builders never repeat the ``rstrip("/")`` dance.
    """
    return (settings.TAO_API_BASE_URL or "").rstrip("/")


def _cache_key_for(settings: Settings) -> str:
    """Build the cache key used to look up JWTs.

    The key includes a SHA-256 hash of the NGC key (never the raw key)
    so log output and error messages can safely reference the cache
    without leaking credentials.
    """
    raw_key = get_effective_secret("TAO_API_KEY", settings) or ""
    key_hash = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:12]
    base = (settings.TAO_API_BASE_URL or "").rstrip("/")
    org = settings.TAO_ORG_NAME or ""
    return f"{base}|{org}|{key_hash}"


def _looks_like_ngc_key(value: str | None) -> bool:
    """Return True when ``TAO_API_KEY`` is the NGC Personal API Key form."""
    return bool(value) and value.startswith(_NGC_KEY_PREFIX)


async def get_tao_bearer(settings: Settings) -> str:
    """Return the Bearer token string for TAO API calls.

    When ``TAO_API_KEY`` is an NGC key (``nvapi-...``), exchange it for a
    JWT via :func:`services.tao_client.login_tao` on first call and cache
    the result. Subsequent calls in the same process return the cached
    JWT. Non-NGC values pass through unchanged (pre-exchanged JWT form).

    Raises :class:`TaoAuthError` when ``TAO_API_KEY`` is unset or the
    login exchange fails — callers treat this as an auth failure and
    surface a structured TAO-unreachable error. ``TaoAuthError`` is a
    subclass of ``RuntimeError`` for backward compatibility.
    """
    raw = get_effective_secret("TAO_API_KEY", settings)
    if not raw:
        raise TaoAuthError("TAO_API_KEY is not configured")

    # Backward-compat: pre-exchanged JWT passes through. (JWTs do not
    # start with ``nvapi-`` — they are ``header.payload.signature``.)
    if not _looks_like_ngc_key(raw):
        return raw

    cache_key = _cache_key_for(settings)
    cached = _token_cache.get(cache_key)
    if cached is not None:
        return cached

    # First-time exchange for this (base_url, org, key) tuple.
    # Local import avoids circularity with ``tao_client`` which may
    # itself import this module.
    from vlm_feedback_loop.services.tao_client import login_tao

    logger.info(
        "Exchanging NGC key for TAO JWT (cache_key=%s)",
        cache_key,
    )

    result = await login_tao(
        settings.TAO_API_BASE_URL or "",
        raw,
        settings.TAO_ORG_NAME or "",
    )

    if not result.get("success") or not result.get("token"):
        error = result.get("error") or "unknown error"
        raise TaoAuthError(f"TAO /login exchange failed: {error}")

    token = str(result["token"])
    _token_cache[cache_key] = token
    return token


async def tao_auth_headers(settings: Settings) -> dict[str, str]:
    """Build Authorization + default headers for every TAO call.

    Awaits ``get_tao_bearer`` which handles the NGC → JWT exchange on
    first use and caches the JWT for the remainder of the process.
    """
    return {
        **NIM_DEFAULT_HEADERS,
        "Authorization": f"Bearer {await get_tao_bearer(settings)}",
    }


async def tao_preflight(
    settings: Settings,
) -> tuple[dict[str, str] | None, str | None]:
    """Config-completeness check + auth-header build for an outbound TAO call.

    The single shared prologue for every TAO request: verifies the
    endpoint trio (``TAO_API_BASE_URL`` / ``TAO_API_KEY`` /
    ``TAO_ORG_NAME``) is configured, then resolves the Bearer headers via
    :func:`tao_auth_headers`.

    Returns ``(headers, None)`` when TAO is reachable-in-principle, or
    ``(None, error)`` where ``error`` names the first missing config key
    (``"TAO_API_BASE_URL is not configured"``, …) or the auth failure
    (``"TAO authentication failed: …"``). Callers wrap the error string in
    their own response shapes.
    """
    if not settings.TAO_API_BASE_URL:
        return None, "TAO_API_BASE_URL is not configured"
    if not get_effective_secret("TAO_API_KEY", settings):
        return None, "TAO_API_KEY is not configured"
    if not settings.TAO_ORG_NAME:
        return None, "TAO_ORG_NAME is not configured"
    try:
        return await tao_auth_headers(settings), None
    except RuntimeError as exc:
        return None, f"TAO authentication failed: {exc}"


def invalidate_tao_bearer(settings: Settings) -> None:
    """Drop the cached JWT for this settings tuple.

    Called by TAO services when a request returns 401 so the next call
    re-runs the /login exchange.
    """
    cache_key = _cache_key_for(settings)
    if cache_key in _token_cache:
        del _token_cache[cache_key]
        logger.info("Invalidated cached TAO JWT (cache_key=%s)", cache_key)


async def retry_once_on_401(
    result: HttpResult,
    *,
    method: str,
    url: str,
    settings: Settings,
    deadline_s: float,
    max_retries: int,
    json_body: dict[str, Any] | None = None,
    _transport: httpx.AsyncBaseTransport | None = None,
    reraise_auth_error: bool = False,
) -> HttpResult:
    """Refresh the cached TAO JWT and retry once when *result* is a 401.

    The TAO JWT is cached process-wide with no TTL. A request that meets
    an expired token would otherwise fail on every attempt and wedge
    long-running callers (the polling loop, provisioning) permanently on
    401. The single shared implementation of the single-401-retry
    contract used by ``tao_job_service`` and
    ``tao_base_experiment_provisioning_service``.

    Returns the original result unchanged when it is not a 401. When
    re-authentication itself fails: with ``reraise_auth_error=False``
    (the ``tao_job_service`` contract) the original 401 result is
    returned so the caller's existing error handling still fires; with
    ``reraise_auth_error=True`` (the provisioning contract) the auth
    error propagates.
    """
    if result.status_code != 401:
        return result
    invalidate_tao_bearer(settings)
    try:
        headers = await tao_auth_headers(settings)
    except RuntimeError:
        if reraise_auth_error:
            raise
        return result
    return await resilient_request(
        method,
        url,
        deadline_s=deadline_s,
        max_retries=max_retries,
        headers=headers,
        json_body=json_body,
        _transport=_transport,
    )
