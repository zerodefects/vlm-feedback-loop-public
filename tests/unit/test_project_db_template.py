# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canaries for the template-project-DB fast path (tests/unit/conftest.py).

The suite replaces the fresh-DB Alembic replay with a file copy of a
once-per-worker template. Two things must never silently drift:

1. the template must be byte-equivalent in schema AND data to a true
   migration run — a future migration that seeds nondeterministic values
   (uuid4 / wall-clock rows) on a fresh DB would fail here loudly instead
   of quietly giving template projects different content than replayed
   ones;
2. the fast path must actually be engaged — a refactor that renames the
   patched hook would silently revert the whole suite to slow chain
   replays with no test failure.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from conftest import ProjectDbTemplate
from vlm_feedback_loop.db.engine import open_project_db


def _schema_dump(db_path: Path) -> list[tuple]:
    with closing(sqlite3.connect(db_path)) as conn:
        return sorted(
            conn.execute(
                "SELECT type, name, tbl_name, sql FROM sqlite_master"
            ).fetchall()
        )


def _data_dump(db_path: Path) -> dict[str, list[tuple]]:
    """Every row of every table (including alembic_version), sorted."""
    with closing(sqlite3.connect(db_path)) as conn:
        tables = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        ]
        return {
            t: sorted(conn.execute(f'SELECT * FROM "{t}"').fetchall(), key=repr)
            for t in tables
        }


@pytest.mark.real_migrations
def test_template_equals_migration_run(
    tmp_path: Path, project_db_template: ProjectDbTemplate
) -> None:
    """Template DB equals a real migration run in schema and table contents."""
    template_path = project_db_template.ensure_built()

    replay_dir = tmp_path / "replayed"
    open_project_db(replay_dir).dispose()  # real runner — marker disables the patch
    replay_path = replay_dir / "project.db"

    assert _schema_dump(template_path) == _schema_dump(replay_path)
    assert _data_dump(template_path) == _data_dump(replay_path)


@pytest.mark.skipif(
    bool(os.environ.get("VLM_TEST_REAL_MIGRATIONS")),
    reason="fast path deliberately disabled via VLM_TEST_REAL_MIGRATIONS",
)
def test_fast_path_is_engaged(
    tmp_path: Path, project_db_template: ProjectDbTemplate
) -> None:
    """A fresh project DB is produced by template copy, not chain replay."""
    before = project_db_template.fast_copies
    engine = open_project_db(tmp_path / "fresh")
    try:
        assert project_db_template.fast_copies == before + 1
        # And the copy is a usable, fully-migrated database.
        with engine.connect() as conn:
            head = conn.exec_driver_sql(
                "SELECT version_num FROM alembic_version"
            ).scalar()
        assert head == project_db_template.head_rev
    finally:
        engine.dispose()
