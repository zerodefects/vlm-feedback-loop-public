# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deployment-scoped Secrets API.

``POST /v1/secrets:set`` lets the SME paste a key in the NIM
Configuration UI and have it take effect on the next NIM call without
a backend restart.

Deployment-scoped, not project-scoped — mirrors
``POST /v1/nim/test_connection``. The endpoint is gated by
``ALLOW_UI_SECRET_PERSIST`` for the disk-write path; the runtime
override path is always available because the same security posture
already covers the connection-test endpoint (transient key in request
memory).

Audit-event emission: structured ``info``-level logs. The
project-scoped AuditEvent table doesn't fit this
deployment-scoped operation, and the JSON logger writes to
``{project_dir}/logs`` (and stdout) on every run — sufficient for
forensic audit. The redaction filter suppresses key values.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException

from vlm_feedback_loop.config import Settings
from vlm_feedback_loop.routers.projects import get_current_settings
from vlm_feedback_loop.schemas.secrets import SecretSetRequest, SecretSetResponse
from vlm_feedback_loop.services import runtime_secrets

logger = logging.getLogger("vlm_feedback_loop.routers.secrets")

secrets_router = APIRouter(tags=["secrets"])


@secrets_router.post("/secrets:set", response_model=SecretSetResponse)
async def set_secret(
    body: SecretSetRequest,
    settings: Settings = Depends(get_current_settings),
) -> SecretSetResponse:
    """Apply a deployment secret at runtime, optionally persisting to .env.

    Flow:
    1. Always installs the runtime override — applies immediately to the
       next NIM call (embedding worker spawn, Teacher proposal, local
       NIM image pull). The SME can rely on this taking effect even when
       they don't choose to persist.
    2. When ``persist=true`` AND ``ALLOW_UI_SECRET_PERSIST`` is true,
       additionally writes the line to ``~/.vlm_feedback_loop/.env``,
       reloads Settings, and retains the same runtime override so queued
       work with an older Settings snapshot still sees the new value.
    3. When ``persist=true`` AND ``ALLOW_UI_SECRET_PERSIST`` is false,
       returns 403 ``ui_secret_persist_disabled`` — the runtime
       application step is NOT executed, since the SME's explicit
       choice was "persist or nothing." If they only want session-only,
       they should re-call with ``persist=false``.

    Audit log lines: ``secret_runtime_set`` (always) and
    ``secret_persisted`` (on successful persist). Key VALUES are
    redacted to length-only.
    """
    name = body.name
    allow_persist = settings.ALLOW_UI_SECRET_PERSIST

    if body.persist and not allow_persist:
        # Refuse before mutating any state so the SME's "all or nothing"
        # intent is preserved.
        raise HTTPException(
            status_code=403,
            detail="ui_secret_persist_disabled",
        )

    # 1) Runtime override — applies immediately.
    runtime_secrets.set_runtime_secret(name, body.value)
    logger.info(
        "secret_runtime_set name=%s value_length=%d persist_requested=%s",
        name,
        len(body.value),
        body.persist,
    )

    env_path_str: str | None = None
    persisted = False

    if body.persist:
        try:
            # Disk read/write/chmod/mkdir + a Settings reload — run it off the
            # event loop so a slow filesystem doesn't stall every other
            # request served by this single-process backend.
            written_path = await asyncio.to_thread(
                runtime_secrets.persist_secret_to_env, name, body.value
            )
        except OSError as exc:
            # Disk write failed (permissions, full disk, etc.). The
            # runtime override is still installed, so the SME's session
            # works; we surface the disk failure as a 500 with a clear
            # detail so they can decide whether to retry or accept
            # session-only.
            logger.error(
                "secret_persist_failed name=%s error=%s; runtime override stays",
                name,
                exc,
            )
            raise HTTPException(
                status_code=500,
                detail=f"secret_persist_io_error: {exc}",
            ) from exc
        env_path_str = str(written_path)
        persisted = True
        logger.info(
            "secret_persisted name=%s env_path=%s value_length=%d",
            name,
            env_path_str,
            len(body.value),
        )

    return SecretSetResponse(
        effective=True,
        persisted=persisted,
        env_path=env_path_str,
        allow_persist=allow_persist,
    )
