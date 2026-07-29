# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Request / response schemas for the deployment-scoped Secrets API.

Backs ``POST /v1/secrets:set``, which applies a secret to the running
process and can optionally persist it to the deployment ``.env``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

#: Names the endpoint accepts. Mirrors
#: :data:`services.runtime_secrets.ALLOWED_SECRETS`; the duplication is
#: deliberate — Pydantic Literal can't reference a runtime frozenset, and
#: keeping the API surface explicit here makes accidental drift visible
#: in code review.
SecretName = Literal["NVIDIA_API_KEY", "NGC_API_KEY", "TAO_API_KEY"]


class SecretSetRequest(BaseModel):
    """Body of ``POST /v1/secrets:set``.

    ``persist=false`` (the default) writes to the in-process override
    layer only; ``persist=true`` additionally writes the line to
    ``~/.vlm_feedback_loop/.env`` (subject to the deployment-level
    ``ALLOW_UI_SECRET_PERSIST`` flag — 403 when disabled).

    The ``value`` field is held in request memory only; it is never
    logged and is never returned in the response body.
    """

    model_config = ConfigDict(extra="forbid")

    name: SecretName
    value: str = Field(min_length=1)
    persist: bool = False

    @field_validator("value")
    @classmethod
    def _reject_control_characters(cls, v: str) -> str:
        # A newline in the value injects extra KEY=VALUE lines into the
        # persisted .env, smuggling settings past the name allowlist; any
        # control character also corrupts the auth header it feeds. Real
        # credentials carry none — reject at the API boundary (422).
        if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in v):
            raise ValueError("value must not contain control characters")
        return v


class SecretSetResponse(BaseModel):
    """Response from ``POST /v1/secrets:set``.

    Reports what actually happened — ``effective=true`` confirms the
    override is installed for this process; ``persisted=true`` confirms
    the disk write succeeded (and the override was cleared in favor of
    the reloaded Settings value). When ``persist`` was requested but
    blocked by ``ALLOW_UI_SECRET_PERSIST=false`` the endpoint returns
    403 instead of an OK response with ``persisted=false``, so this
    field is only ever ``false`` when the SME chose session-only.
    """

    effective: bool
    persisted: bool
    env_path: str | None = None
    allow_persist: bool
