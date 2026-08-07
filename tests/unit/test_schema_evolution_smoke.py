"""Behavior checks for the schema-evolution live smoke's DB evidence path."""

from __future__ import annotations

import sqlite3
import sys
from contextlib import closing
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from schema_evolution_smoke import _db_query, _require_project_db  # noqa: E402


def test_db_query_uses_api_returned_project_database(tmp_path: Path) -> None:
    """Direct evidence follows project_dir instead of a reconstructed workspace."""

    project_db = tmp_path / "custom-workspace" / "projects" / "p-1" / "project.db"
    project_db.parent.mkdir(parents=True)
    with closing(sqlite3.connect(project_db)) as connection:
        connection.execute("CREATE TABLE marker (value TEXT NOT NULL)")
        connection.execute("INSERT INTO marker VALUES ('api-returned-path')")
        connection.commit()

    assert _db_query(project_db, "SELECT value FROM marker") == [
        {"value": "api-returned-path"}
    ]


def test_missing_api_returned_db_fails_before_sqlite_can_create_it(
    tmp_path: Path,
) -> None:
    """A container-only path cannot create a misleading empty local DB."""

    project_dir = tmp_path / "not-mounted-project"
    project_dir.mkdir()

    with pytest.raises(RuntimeError, match="local-source backend"):
        _require_project_db(project_dir)

    assert not (project_dir / "project.db").exists()
