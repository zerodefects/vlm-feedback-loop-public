# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the SQLite engine factory's pool-hardening guarantees.

Covers the stale-snapshot scenario: an external raw-``sqlite3`` writer
(e.g. a seeding script run while the backend is up) commits rows into the
project DB *after* the backend has opened its engine and warmed the pool.
Without the hardening applied in ``_create_engine`` (``pool_pre_ping``,
``pool_recycle``, and the ``checkout`` rollback event), pooled connections
can retain a pre-write read snapshot and ``session.get`` silently returns
``None`` for rows that exist on disk.
"""

from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.orm import Session

from vlm_feedback_loop.db.engine import _create_engine, open_project_db
from vlm_feedback_loop.db.models.model_config import ModelConfig


def _insert_model_config_via_raw_sqlite(db_path: Path, model_config_id: str) -> None:
    """Write a row the way an external raw-sqlite3 seeder does (WAL journal
    mode, DEFERRED isolation, single commit)."""
    conn = sqlite3.connect(str(db_path), timeout=30, isolation_level="DEFERRED")
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            """INSERT INTO model_configs(
                model_config_id, project_id, endpoint_id, model_name,
                context_window_tokens, eligible_roles, supports_image_input,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                model_config_id,
                "test-project",
                "test-endpoint",
                "nvidia/cosmos-reason2-8b",
                256000,
                '["teacher"]',
                1,
                "2026-04-22T19:00:00Z",
            ),
        )
        conn.commit()
    finally:
        conn.close()


class TestExternalWriterVisibility:
    """Writes from a separate raw-sqlite3 process must be visible to fresh
    SQLAlchemy sessions on a previously-warmed engine."""

    def test_session_get_sees_external_insert_after_pool_warmup(self, tmp_path):
        """After the pool is warm, a subsequent raw-sqlite3 insert MUST be
        visible via ``session.get`` on a new Session bound to the same engine.
        """
        proj = tmp_path / "projects" / str(uuid.uuid4())
        proj.mkdir(parents=True)
        engine = open_project_db(proj)

        # Warm the pool: run a no-op query so a pooled connection exists.
        with Session(engine) as s:
            s.query(ModelConfig).all()

        # External writer commits a new row outside SQLAlchemy.
        mid = str(uuid.uuid4())
        _insert_model_config_via_raw_sqlite(proj / "project.db", mid)

        # Fresh Session must see it via session.get (identity-map path) AND
        # via raw SELECT (connection-level path).
        with Session(engine) as s:
            mc = s.get(ModelConfig, mid)
            assert mc is not None, (
                "session.get returned None for externally-inserted row"
            )
            assert mc.model_name == "nvidia/cosmos-reason2-8b"

            scalar = s.execute(
                text("SELECT model_name FROM model_configs WHERE model_config_id=:id"),
                {"id": mid},
            ).scalar_one_or_none()
            assert scalar == "nvidia/cosmos-reason2-8b"

        engine.dispose()

    def test_session_get_sees_external_insert_without_pool_warmup(self, tmp_path):
        """Sanity check: first-use Session also sees external writes."""
        proj = tmp_path / "projects" / str(uuid.uuid4())
        proj.mkdir(parents=True)
        engine = open_project_db(proj)

        mid = str(uuid.uuid4())
        _insert_model_config_via_raw_sqlite(proj / "project.db", mid)

        with Session(engine) as s:
            mc = s.get(ModelConfig, mid)
            assert mc is not None
            assert mc.model_name == "nvidia/cosmos-reason2-8b"

        engine.dispose()


class TestPoolConfiguration:
    """Lock in the engine factory's pool-hardening defaults so they can't
    regress silently."""

    def test_engine_has_pool_pre_ping(self, tmp_path):
        db_path = tmp_path / "probe.db"
        db_path.touch()
        engine = _create_engine(db_path)
        try:
            # SQLAlchemy stores pre-ping as an attribute on the pool.
            assert getattr(engine.pool, "_pre_ping", False) is True
        finally:
            engine.dispose()

    def test_engine_has_pool_recycle(self, tmp_path):
        db_path = tmp_path / "probe.db"
        db_path.touch()
        engine = _create_engine(db_path)
        try:
            # Pool recycle bounds the maximum age of any pooled connection
            # — preventing unbounded WAL-snapshot age on long-running services.
            assert engine.pool._recycle == 3600
        finally:
            engine.dispose()
