# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SQLAlchemy declarative bases, UUID4/timestamp helpers, and event hooks.

Two separate DeclarativeBase subclasses keep project-scoped and
deployment-scoped table metadata separate — they live in different
SQLite databases and must never share a metadata registry.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import String, event
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# ── Helpers ──────────────────────────────────────────────────────────────────


def generate_uuid4() -> str:
    """Return a UUID4 in canonical lowercase hyphenated format."""
    return str(uuid.uuid4())


def utc_now() -> str:
    """Return current UTC time as ISO 8601 with Z suffix."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── Column factories ────────────────────────────────────────────────────────


def uuid_pk() -> Mapped[str]:
    """Primary-key column with auto-generated UUID4."""
    return mapped_column(String(36), primary_key=True, default=generate_uuid4)


def created_at_col() -> Mapped[str]:
    """Server-set creation timestamp. Value forced by before_insert hook."""
    return mapped_column(String(24), nullable=False, default=utc_now)


def updated_at_col() -> Mapped[str]:
    """Server-set update timestamp. Value forced by before_insert/update hooks."""
    return mapped_column(String(24), nullable=False, default=utc_now, onupdate=utc_now)


# ── Declarative bases ───────────────────────────────────────────────────────


class ProjectBase(DeclarativeBase):
    """Base for all project-scoped tables (stored in per-project project.db)."""

    pass


class DeploymentBase(DeclarativeBase):
    """Base for deployment-scoped tables (stored in workspace deployment.db)."""

    pass


# ── Event hooks — guarantee server-set timestamps ────────────────────────────
# These fire regardless of what values the caller provides on the ORM object,
# ensuring client-provided timestamp values are ignored.


def _before_insert(
    mapper: Any,  # noqa: ARG001 — SQLAlchemy event hook signature
    connection: Any,  # noqa: ARG001 — SQLAlchemy event hook signature
    target: Any,
) -> None:
    now = utc_now()
    if hasattr(target, "created_at"):
        target.created_at = now
    if hasattr(target, "updated_at"):
        target.updated_at = now


def _before_update(
    mapper: Any,  # noqa: ARG001 — SQLAlchemy event hook signature
    connection: Any,  # noqa: ARG001 — SQLAlchemy event hook signature
    target: Any,
) -> None:
    if hasattr(target, "updated_at"):
        target.updated_at = utc_now()


event.listen(ProjectBase, "before_insert", _before_insert, propagate=True)
event.listen(ProjectBase, "before_update", _before_update, propagate=True)
event.listen(DeploymentBase, "before_insert", _before_insert, propagate=True)
event.listen(DeploymentBase, "before_update", _before_update, propagate=True)
