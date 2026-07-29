# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Alembic environment for programmatic migration execution.

This env.py is NOT driven by ``alembic.ini``.  Instead, the migration
runner in ``db/engine.py`` sets ``config.attributes["connection"]`` to
an active SQLAlchemy connection and calls ``command.upgrade()``.
"""

from alembic import context

import vlm_feedback_loop.db.models  # noqa: F401  # pyright: ignore[reportUnusedImport] — populates SQLAlchemy metadata for Alembic autogeneration
from vlm_feedback_loop.db.base import ProjectBase

target_metadata = ProjectBase.metadata


def run_migrations_online() -> None:
    """Run migrations using a connection supplied via config attributes."""
    connectable = context.config.attributes.get("connection")
    if connectable is None:
        raise RuntimeError(
            "Alembic connection not set. "
            "Use db.engine._run_migrations() instead of the Alembic CLI."
        )

    context.configure(connection=connectable, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


run_migrations_online()
