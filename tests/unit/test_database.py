# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the database layer — engine setup, migrations, and ID/timestamp helpers."""

from __future__ import annotations

import contextlib
import re
import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path

import pytest
from alembic.autogenerate import compare_metadata
from alembic.runtime.migration import MigrationContext
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from vlm_feedback_loop.db.base import ProjectBase, generate_uuid4, utc_now
from vlm_feedback_loop.db.engine import (
    DatabaseCorruptionError,
    DatabaseMigrationError,
    open_project_db,
)
from vlm_feedback_loop.model_catalog_constants import (
    EMBEDDING_NIM_GPU_MIN_GB,
    EMBEDDING_NIM_IMAGE,
)

UUID4_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
ISO8601_Z_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

EXPECTED_PROJECT_TABLES = {
    "alembic_version",
    "audit_events",
    "clip_embeddings",
    "dataset_exports",
    "examples",
    "guidances",
    "labels",
    "local_nim_deployments",
    "model_configs",
    "nim_endpoints",
    "operation_records",
    "pools",
    "projects",
    "run_records",
    "student_models",
    "tao_jobs",
    "training_suites",
}


# ── WAL mode enabled ──────────────────────────────────────────────────


class TestDatabasePragmas:
    def test_wal_mode_enabled(self, project_engine):
        """PRAGMA journal_mode returns 'wal'."""
        with project_engine.connect() as conn:
            result = conn.execute(text("PRAGMA journal_mode")).scalar()
            assert result == "wal"

    def test_busy_timeout_configured(self, project_engine):
        """PRAGMA busy_timeout >= 5000."""
        with project_engine.connect() as conn:
            result = conn.execute(text("PRAGMA busy_timeout")).scalar()
            assert result >= 5000

    def test_foreign_keys_enabled_on_every_runtime_connection(self, project_engine):
        """Every pooled project connection enforces the repaired schema."""
        with project_engine.connect() as first, project_engine.connect() as second:
            assert first.execute(text("PRAGMA foreign_keys")).scalar() == 1
            assert second.execute(text("PRAGMA foreign_keys")).scalar() == 1

    @pytest.mark.parametrize("reported_state", [(0,), None])
    def test_foreign_key_hook_fails_closed_when_sqlite_refuses_enforcement(
        self, reported_state
    ):
        """A connection is rejected unless SQLite confirms enforcement."""
        from vlm_feedback_loop.db.engine import _enable_sqlite_foreign_keys

        class RefusingCursor:
            closed = False

            def execute(self, statement):
                assert statement in {
                    "PRAGMA foreign_keys=ON",
                    "PRAGMA foreign_keys",
                }
                return self

            def fetchone(self):
                return reported_state

            def close(self):
                self.closed = True

        cursor = RefusingCursor()

        class RefusingConnection:
            def cursor(self):
                return cursor

        with pytest.raises(sqlite3.DatabaseError, match="refused"):
            _enable_sqlite_foreign_keys(RefusingConnection(), None)
        assert cursor.closed is True

    def test_embedding_foreign_key_rejects_mismatch_and_cascades(self, project_engine):
        """Embedding ownership is exact and deleting its Example removes it."""
        with project_engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO examples "
                    "(example_key, project_id, storage_ref, ingested_at, "
                    "source_metadata, state) VALUES "
                    "('fk-example', 'project-a', '/tmp/fk.png', "
                    "'2026-07-26T00:00:00Z', '{}', 'Unlabeled')"
                )
            )
            conn.execute(
                text(
                    "INSERT INTO clip_embeddings "
                    "(project_id, example_key, embedding_provider, "
                    "clip_embedding_model_id, clip_embedding_dim, "
                    "vector_blob_f32, created_at, updated_at) VALUES "
                    "('project-a', 'fk-example', 'hosted_nvclip', 'model', 2, "
                    ":vector, '2026-07-26T00:00:00Z', "
                    "'2026-07-26T00:00:00Z')"
                ),
                {"vector": b"\x00" * 8},
            )

        with pytest.raises(IntegrityError):
            with project_engine.begin() as conn:
                conn.execute(
                    text(
                        "INSERT INTO clip_embeddings "
                        "(project_id, example_key, embedding_provider, "
                        "clip_embedding_model_id, clip_embedding_dim, "
                        "vector_blob_f32, created_at, updated_at) VALUES "
                        "('project-b', 'fk-example', 'hosted_nvclip', 'model', "
                        "2, :vector, '2026-07-26T00:00:00Z', "
                        "'2026-07-26T00:00:00Z')"
                    ),
                    {"vector": b"\x01" * 8},
                )

        with project_engine.begin() as conn:
            conn.execute(text("DELETE FROM examples WHERE example_key = 'fk-example'"))

        with project_engine.connect() as conn:
            assert (
                conn.execute(
                    text(
                        "SELECT count(*) FROM clip_embeddings "
                        "WHERE example_key = 'fk-example'"
                    )
                ).scalar()
                == 0
            )


# ── quick_check catches corruption ─────────────────────────────────────


class TestQuickCheck:
    def test_corrupt_db_raises(self, tmp_project_dir):
        """Corrupt database file → DatabaseCorruptionError with path."""
        db_path = tmp_project_dir / "project.db"
        db_path.write_bytes(b"this is not a valid sqlite database file at all")

        with pytest.raises(DatabaseCorruptionError):
            open_project_db(tmp_project_dir)

    def test_foreign_key_violation_raises_when_reopening_current_head(
        self, tmp_project_dir
    ):
        """A current-revision DB cannot bypass the post-migration FK check."""
        engine = open_project_db(tmp_project_dir)
        engine.dispose()

        db_path = tmp_project_dir / "project.db"
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "INSERT INTO clip_embeddings "
                "(project_id, example_key, embedding_provider, "
                "clip_embedding_model_id, clip_embedding_dim, "
                "vector_blob_f32, created_at, updated_at) VALUES "
                "('project-a', 'missing', 'hosted_nvclip', 'model', 2, ?, "
                "'2026-07-26T00:00:00Z', '2026-07-26T00:00:00Z')",
                (b"\x00" * 8,),
            )
            conn.commit()

        with pytest.raises(DatabaseCorruptionError, match="Foreign-key integrity"):
            open_project_db(tmp_project_dir)


# ── Alembic manages migrations; all tables created ────────────────────


@pytest.mark.real_migrations
class TestAlembicMigrations:
    def test_all_project_tables_exist(self, project_engine):
        """All 16 project tables plus alembic_version exist after open."""
        with project_engine.connect() as conn:
            tables = set(
                conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
                .scalars()
                .all()
            )
        assert EXPECTED_PROJECT_TABLES.issubset(tables)

    def test_fresh_database_starts_at_public_v1(self, project_engine):
        """A fresh project database is stamped at the public v1 baseline."""
        with project_engine.connect() as conn:
            rev = conn.execute(text("SELECT version_num FROM alembic_version")).scalar()
        assert rev == "v1_0001"

    def test_public_v1_schema_matches_orm_metadata(self, project_engine):
        """Model schema changes require a corresponding post-v1 migration."""
        with project_engine.connect() as conn:
            context = MigrationContext.configure(
                conn,
                opts={"compare_type": True},
            )
            differences = compare_metadata(context, ProjectBase.metadata)

        assert differences == []

    def test_tao_chain_columns_present(self, project_engine):
        """The TAO chain columns exist on model_configs and dataset_exports."""
        with project_engine.connect() as conn:
            mc_cols = {
                row[1] for row in conn.execute(text("PRAGMA table_info(model_configs)"))
            }
            de_cols = {
                row[1]
                for row in conn.execute(text("PRAGMA table_info(dataset_exports)"))
            }
        assert "tao_base_experiment_id" in mc_cols
        assert "tao_base_experiment_pull_status" in mc_cols
        assert "dataset_upload_ref" in de_cols
        assert "dataset_upload_uri" in de_cols

    def test_tao_chain_columns_nullable_and_default_null(self, project_engine):
        """Fresh DB has all four new columns nullable with no default."""
        with project_engine.connect() as conn:
            mc_info = {
                row[1]: row
                for row in conn.execute(text("PRAGMA table_info(model_configs)"))
            }
            de_info = {
                row[1]: row
                for row in conn.execute(text("PRAGMA table_info(dataset_exports)"))
            }
        # PRAGMA table_info returns (cid, name, type, notnull, dflt_value, pk)
        for col_name, info in [
            ("tao_base_experiment_id", mc_info["tao_base_experiment_id"]),
            (
                "tao_base_experiment_pull_status",
                mc_info["tao_base_experiment_pull_status"],
            ),
            ("dataset_upload_ref", de_info["dataset_upload_ref"]),
            ("dataset_upload_uri", de_info["dataset_upload_uri"]),
        ]:
            notnull = info[3]
            dflt = info[4]
            assert notnull == 0, f"{col_name} should be nullable"
            assert dflt is None, f"{col_name} should have no default"


# ── Backup before migration ───────────────────────────────────────────


@pytest.mark.real_migrations
class TestMigrationBackup:
    def test_backup_created_when_existing_db_needs_migration(self, tmp_project_dir):
        """Backup is created before applying migrations to an existing DB.

        Verified via the migration failure test: the failure test confirms
        that a backup was created before the failing migration attempt
        (the error message includes the backup path). This test verifies
        the backup function directly.
        """
        from vlm_feedback_loop.db.engine import _backup_database

        db_path = tmp_project_dir / "project.db"
        with contextlib.closing(sqlite3.connect(db_path)) as conn:
            conn.execute("CREATE TABLE backup_probe (value TEXT NOT NULL)")
            conn.execute("INSERT INTO backup_probe VALUES ('original')")
            conn.commit()

        backup_path = _backup_database(db_path)
        assert backup_path.exists()
        assert "project.db.backup." in backup_path.name
        with contextlib.closing(sqlite3.connect(backup_path)) as conn:
            assert conn.execute("PRAGMA quick_check").fetchone() == ("ok",)
            assert conn.execute("SELECT value FROM backup_probe").fetchone() == (
                "original",
            )

    def test_backup_includes_committed_wal_frames_with_reader_pin(
        self, tmp_project_dir
    ):
        """A busy WAL checkpoint cannot make the recovery snapshot stale."""
        from vlm_feedback_loop.db.engine import _backup_database

        db_path = tmp_project_dir / "project.db"
        writer = sqlite3.connect(db_path)
        reader = sqlite3.connect(db_path)
        try:
            assert writer.execute("PRAGMA journal_mode=WAL").fetchone() == ("wal",)
            writer.execute("CREATE TABLE backup_probe (value TEXT NOT NULL)")
            writer.execute("INSERT INTO backup_probe VALUES ('main-file')")
            writer.commit()
            assert writer.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone() == (
                0,
                0,
                0,
            )

            # Put one committed frame in the WAL and pin it with an old read
            # snapshot before committing the value the backup must preserve.
            writer.execute("INSERT INTO backup_probe VALUES ('reader-anchor')")
            writer.commit()
            reader.execute("BEGIN")
            assert reader.execute("SELECT count(*) FROM backup_probe").fetchone() == (
                2,
            )
            writer.execute("INSERT INTO backup_probe VALUES ('committed-in-wal')")
            writer.commit()

            checkpoint = writer.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
            assert checkpoint is not None and checkpoint[0] == 1

            backup_path = _backup_database(db_path)
            with contextlib.closing(sqlite3.connect(backup_path)) as backup:
                assert backup.execute("PRAGMA quick_check").fetchone() == ("ok",)
                assert backup.execute(
                    "SELECT value FROM backup_probe ORDER BY rowid"
                ).fetchall() == [
                    ("main-file",),
                    ("reader-anchor",),
                    ("committed-in-wal",),
                ]
        finally:
            reader.rollback()
            reader.close()
            writer.close()

    def test_same_timestamp_backups_preserve_both_recovery_points(
        self, tmp_project_dir, monkeypatch
    ):
        """Rapid migration retries never overwrite the first backup."""
        import vlm_feedback_loop.db.engine as engine_module

        class FixedDateTime:
            @classmethod
            def now(cls, tz):
                return datetime(2026, 7, 25, 12, 0, tzinfo=tz)

        monkeypatch.setattr(engine_module, "datetime", FixedDateTime)

        db_path = tmp_project_dir / "project.db"
        with contextlib.closing(sqlite3.connect(db_path)) as conn:
            conn.execute("CREATE TABLE backup_probe (value TEXT NOT NULL)")
            conn.execute("INSERT INTO backup_probe VALUES ('original')")
            conn.commit()

        first_backup = engine_module._backup_database(db_path)
        with contextlib.closing(sqlite3.connect(db_path)) as conn:
            conn.execute("UPDATE backup_probe SET value = 'partial-after-failure'")
            conn.commit()
        second_backup = engine_module._backup_database(db_path)

        assert first_backup != second_backup
        assert first_backup.exists()
        assert second_backup.exists()
        with contextlib.closing(sqlite3.connect(first_backup)) as first:
            assert first.execute("SELECT value FROM backup_probe").fetchone() == (
                "original",
            )
        with contextlib.closing(sqlite3.connect(second_backup)) as second:
            assert second.execute("SELECT value FROM backup_probe").fetchone() == (
                "partial-after-failure",
            )
        assert not list(tmp_project_dir.glob(".project.db.backup.*"))

    def test_failed_backup_publishes_no_recovery_file(self, tmp_project_dir):
        """A failed SQLite copy leaves neither a public backup nor a temp file."""
        from vlm_feedback_loop.db.engine import _backup_database

        db_path = tmp_project_dir / "project.db"
        db_path.write_bytes(b"not a sqlite database")

        with pytest.raises(sqlite3.DatabaseError):
            _backup_database(db_path)

        assert not list(tmp_project_dir.glob("project.db.backup.*"))
        assert not list(tmp_project_dir.glob(".project.db.backup.*"))

    def test_invalid_snapshot_is_not_published(self, tmp_project_dir):
        """A copy that succeeds but fails quick_check is not a recovery point."""
        from vlm_feedback_loop.db.engine import _backup_database

        db_path = tmp_project_dir / "project.db"
        with contextlib.closing(sqlite3.connect(db_path)) as conn:
            conn.execute("CREATE TABLE backup_probe (value TEXT NOT NULL)")
            conn.commit()
            conn.execute("PRAGMA writable_schema=ON")
            conn.execute(
                "UPDATE sqlite_schema SET rootpage = 2147483647 "
                "WHERE type = 'table' AND name = 'backup_probe'"
            )
            conn.commit()

        with pytest.raises(DatabaseCorruptionError, match="Backup integrity"):
            _backup_database(db_path)

        assert not list(tmp_project_dir.glob("project.db.backup.*"))
        assert not list(tmp_project_dir.glob(".project.db.backup.*"))

    def test_publication_failure_cleans_validated_snapshot(
        self, tmp_project_dir, monkeypatch
    ):
        """Publication happens after validation and cleans every temporary file."""
        import vlm_feedback_loop.db.engine as engine_module

        db_path = tmp_project_dir / "project.db"
        with contextlib.closing(sqlite3.connect(db_path)) as conn:
            assert conn.execute("PRAGMA journal_mode=WAL").fetchone() == ("wal",)
            conn.execute("CREATE TABLE backup_probe (value TEXT NOT NULL)")
            conn.execute("INSERT INTO backup_probe VALUES ('committed')")
            conn.commit()

        inspected_snapshot = False

        def _inspect_then_fail(source: str | Path, _destination: str | Path) -> None:
            nonlocal inspected_snapshot
            with contextlib.closing(sqlite3.connect(source)) as snapshot:
                assert snapshot.execute("PRAGMA quick_check").fetchone() == ("ok",)
                assert snapshot.execute(
                    "SELECT value FROM backup_probe"
                ).fetchone() == ("committed",)
            inspected_snapshot = True
            raise OSError("simulated publication failure")

        monkeypatch.setattr(engine_module.os, "link", _inspect_then_fail)

        with pytest.raises(OSError, match="publication failure"):
            engine_module._backup_database(db_path)

        assert inspected_snapshot
        assert not list(tmp_project_dir.glob("project.db.backup.*"))
        assert not list(tmp_project_dir.glob(".project.db.backup.*"))

    def test_no_backup_on_fresh_db(self, tmp_project_dir):
        """Fresh database (no prior revision) does not create a backup."""
        open_project_db(tmp_project_dir)
        backups = list(tmp_project_dir.glob("project.db.backup.*"))
        assert len(backups) == 0


# ── Migration failure halts with backup path ──────────────────────────


@pytest.mark.real_migrations
class TestMigrationFailure:
    def test_migration_failure_raises_with_backup(self, tmp_project_dir, monkeypatch):
        """Failed migration preserves each reported pre-attempt recovery point."""
        import vlm_feedback_loop.db.engine as engine_module

        class FixedDateTime:
            @classmethod
            def now(cls, tz):
                return datetime(2026, 7, 25, 12, 0, tzinfo=tz)

        monkeypatch.setattr(engine_module, "datetime", FixedDateTime)
        db_path = tmp_project_dir / "project.db"

        # Start from the supported v1 baseline, then model a future head.
        open_project_db(tmp_project_dir).dispose()
        with contextlib.closing(sqlite3.connect(db_path)) as conn:
            conn.execute("CREATE TABLE recovery_probe (value TEXT NOT NULL)")
            conn.execute("INSERT INTO recovery_probe VALUES ('original')")
            conn.commit()
        monkeypatch.setattr(
            engine_module.ScriptDirectory,
            "get_current_head",
            lambda _script: "v1_0002",
        )

        # Monkeypatch alembic upgrade to raise
        from alembic import command as alembic_command

        migration_attempt = 0

        def _fake_upgrade(cfg, _revision):
            nonlocal migration_attempt
            migration_attempt += 1
            connection = cfg.attributes["connection"]
            connection.exec_driver_sql(
                "UPDATE recovery_probe SET value = "
                f"'changed-by-migration-{migration_attempt}'"
            )
            connection.commit()
            raise RuntimeError("Simulated migration failure")

        monkeypatch.setattr(alembic_command, "upgrade", _fake_upgrade)

        with pytest.raises(DatabaseMigrationError, match="backup") as first_error:
            open_project_db(tmp_project_dir)
        first_backups = set(tmp_project_dir.glob("project.db.backup.*"))
        assert len(first_backups) == 1
        first_backup = first_backups.pop()
        assert str(first_backup) in str(first_error.value)
        with contextlib.closing(sqlite3.connect(first_backup)) as backup:
            assert backup.execute("SELECT value FROM recovery_probe").fetchone() == (
                "original",
            )

        # Model an immediate retry after a failed migration changed the live
        # file. The original recovery point must remain intact.
        with contextlib.closing(sqlite3.connect(db_path)) as conn:
            conn.execute("UPDATE recovery_probe SET value = 'partial-after-failure'")
            conn.commit()

        with pytest.raises(DatabaseMigrationError, match="backup") as second_error:
            open_project_db(tmp_project_dir)
        all_backups = set(tmp_project_dir.glob("project.db.backup.*"))
        assert len(all_backups) == 2
        assert first_backup in all_backups
        second_backup = (all_backups - {first_backup}).pop()
        assert str(second_backup) in str(second_error.value)
        with contextlib.closing(sqlite3.connect(first_backup)) as backup:
            assert backup.execute("SELECT value FROM recovery_probe").fetchone() == (
                "original",
            )
        with contextlib.closing(sqlite3.connect(second_backup)) as backup:
            assert backup.execute("SELECT value FROM recovery_probe").fetchone() == (
                "partial-after-failure",
            )

    def test_pre_v1_database_is_rejected_with_fresh_workspace_guidance(
        self,
        tmp_project_dir,
    ):
        """Private development revisions cannot masquerade as public v1."""

        db_path = tmp_project_dir / "project.db"
        with contextlib.closing(sqlite3.connect(db_path)) as conn:
            conn.execute(
                "CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)"
            )
            conn.execute("INSERT INTO alembic_version VALUES ('061')")
            conn.execute("CREATE TABLE recovery_probe (value TEXT NOT NULL)")
            conn.execute("INSERT INTO recovery_probe VALUES ('unchanged')")
            conn.commit()

        with pytest.raises(
            DatabaseMigrationError,
            match="pre-v1 development databases are intentionally unsupported",
        ):
            open_project_db(tmp_project_dir)

        with contextlib.closing(sqlite3.connect(db_path)) as conn:
            assert conn.execute("SELECT value FROM recovery_probe").fetchone() == (
                "unchanged",
            )
        assert not list(tmp_project_dir.glob("project.db.backup.*"))


# ── Write discipline (busy_timeout enforced) ──────────────────────────


class TestWriteDiscipline:
    def test_busy_timeout_enforced(self, tmp_project_dir):
        """A write that hits a concurrent writer's lock waits it out and
        succeeds (busy_timeout) instead of failing instantly.

        With ``busy_timeout=0`` the engine write would raise ``database is
        locked`` the moment it hits the raw connection's ``BEGIN
        IMMEDIATE``; with the configured 30s timeout it waits ~1s for the
        commit and completes. The lock window is kept short — the point is
        the wait-vs-instant-failure distinction, not the timeout's length.
        """
        engine = open_project_db(tmp_project_dir)
        db_path = str(tmp_project_dir / "project.db")
        lock_held = threading.Event()
        errors: list[Exception] = []

        def hold_lock():
            """Hold a write lock briefly using raw sqlite3."""
            raw = sqlite3.connect(db_path)
            try:
                raw.execute("BEGIN IMMEDIATE")
                lock_held.set()
                time.sleep(1.0)
                raw.execute("COMMIT")
            finally:
                raw.close()

        def try_write():
            """Write while the lock is definitely held."""
            assert lock_held.wait(timeout=10), "lock holder never started"
            try:
                with engine.connect() as conn:
                    conn.execute(
                        text(
                            "INSERT INTO audit_events "
                            "(audit_event_id, project_id, event_type, event_data, created_at) "
                            "VALUES ('test-id', 'test-proj', 'test', '{}', '2026-01-01T00:00:00Z')"
                        )
                    )
                    conn.commit()
            except Exception as e:  # noqa: BLE001 — collected and asserted below
                errors.append(e)

        t1 = threading.Thread(target=hold_lock)
        t2 = threading.Thread(target=try_write)
        t1.start()
        t2.start()
        t1.join(timeout=15)
        t2.join(timeout=15)

        assert errors == [], f"write should have waited out the lock: {errors}"
        with engine.connect() as conn:
            count = conn.execute(
                text("SELECT COUNT(*) FROM audit_events WHERE audit_event_id='test-id'")
            ).scalar()
        assert count == 1, "the blocked write must have landed after the wait"


# ── EmbeddingDeploymentConfig singleton seeded ─────────────────────────


class TestEmbeddingDeploymentConfig:
    def test_singleton_seeded_with_defaults(self, deployment_engine):
        """Exactly one record with all required seeded defaults."""
        from vlm_feedback_loop.db.deployment_models import EmbeddingDeploymentConfig

        with Session(deployment_engine) as session:
            configs = session.query(EmbeddingDeploymentConfig).all()
            assert len(configs) == 1

            c = configs[0]
            assert c.model_name == "nvidia/llama-nemotron-embed-vl-1b-v2"
            assert c.embedding_dim == 2048
            assert c.nim_container_image == EMBEDDING_NIM_IMAGE
            assert c.preferred_host_port == 8001
            assert c.gpu_memory_minimum_gb == EMBEDDING_NIM_GPU_MIN_GB
            assert c.provider == "none"

    def test_idempotent_init(self, tmp_workspace):
        """Calling init_deployment_db twice does not create duplicates."""
        from vlm_feedback_loop.db.deployment_models import EmbeddingDeploymentConfig
        from vlm_feedback_loop.db.engine import init_deployment_db

        init_deployment_db(tmp_workspace)
        engine = init_deployment_db(tmp_workspace)

        with Session(engine) as session:
            assert session.query(EmbeddingDeploymentConfig).count() == 1

    def test_init_upgrades_shipped_embedding_runtime(self, tmp_workspace):
        """A deployed 1.x default moves to the tested 2.0 runtime and floor."""
        from vlm_feedback_loop.db.deployment_models import EmbeddingDeploymentConfig
        from vlm_feedback_loop.db.engine import init_deployment_db

        engine = init_deployment_db(tmp_workspace)
        with Session(engine) as session:
            config = session.query(EmbeddingDeploymentConfig).one()
            config.nim_container_image = (
                "nvcr.io/nim/nvidia/llama-nemotron-embed-vl-1b-v2:1.12.0"
            )
            config.gpu_memory_minimum_gb = 24
            session.commit()
        engine.dispose()

        upgraded = init_deployment_db(tmp_workspace)
        with Session(upgraded) as session:
            config = session.query(EmbeddingDeploymentConfig).one()
            assert config.nim_container_image == EMBEDDING_NIM_IMAGE
            assert config.gpu_memory_minimum_gb == EMBEDDING_NIM_GPU_MIN_GB


# ── TAODeploymentConfig singleton seeded ─────────────────────────────────────


class TestTAODeploymentConfig:
    """init_deployment_db seeds a singleton TAODeploymentConfig."""

    def test_singleton_seeded_with_not_bootstrapped_status(self, deployment_engine):
        """Exactly one record with bootstrap_status='not_bootstrapped'."""
        from vlm_feedback_loop.db.deployment_models import TAODeploymentConfig

        with Session(deployment_engine) as session:
            configs = session.query(TAODeploymentConfig).all()
            assert len(configs) == 1

            c = configs[0]
            assert c.bootstrap_status == "not_bootstrapped"
            assert c.tao_workspace_id is None
            assert c.tao_workspace_name is None
            assert c.tao_workspace_cloud_type is None
            assert c.tao_workspace_bucket is None
            assert c.tao_workspace_s3_endpoint_url_internal is None
            assert c.tao_workspace_s3_endpoint_url_external is None
            assert c.tao_workspace_s3_access_key_ref is None
            assert c.tao_workspace_s3_secret_key_ref is None
            assert c.bootstrap_last_run_at is None
            assert c.bootstrap_error_ref is None
            # UUID4 primary key populated
            assert c.tao_deployment_config_id is not None
            # Server-set timestamps
            assert c.created_at is not None
            assert c.updated_at is not None

    def test_idempotent_init_does_not_duplicate_tao_config(self, tmp_workspace):
        """init_deployment_db is additive — repeated calls keep one TAO row."""
        from vlm_feedback_loop.db.deployment_models import TAODeploymentConfig
        from vlm_feedback_loop.db.engine import init_deployment_db

        init_deployment_db(tmp_workspace)
        engine = init_deployment_db(tmp_workspace)

        with Session(engine) as session:
            assert session.query(TAODeploymentConfig).count() == 1


# ── UUID4 canonical format ────────────────────────────────────────────


class TestUUID4Format:
    def test_uuid4_regex(self):
        """generate_uuid4() matches the canonical lowercase hyphenated UUID4 format."""
        for _ in range(100):
            uid = generate_uuid4()
            assert UUID4_RE.match(uid), f"Invalid UUID4: {uid}"


# ── UTC ISO 8601 with Z suffix ───────────────────────────────────────


class TestTimestampFormat:
    def test_utc_iso8601_with_z(self):
        """utc_now() produces ISO 8601 with Z suffix, no offset."""
        ts = utc_now()
        assert ISO8601_Z_RE.match(ts), f"Invalid timestamp: {ts}"
        assert "+" not in ts
        assert ts.endswith("Z")


# ── created_at/updated_at server-set ──────────────────────────────────


class TestServerSetTimestamps:
    def test_project_timestamps_server_set(self, project_engine):
        """Client-provided created_at/updated_at are overridden."""
        from vlm_feedback_loop.db.models.project import Project

        with Session(project_engine) as session:
            proj = Project(
                name="test",
                project_dir="/tmp/test",
                created_at="1999-01-01T00:00:00Z",
                updated_at="1999-01-01T00:00:00Z",
            )
            session.add(proj)
            session.commit()
            session.refresh(proj)

            # Server-set hooks should have overridden the client values
            assert proj.created_at != "1999-01-01T00:00:00Z"
            assert proj.updated_at != "1999-01-01T00:00:00Z"
            assert ISO8601_Z_RE.match(proj.created_at)
            assert ISO8601_Z_RE.match(proj.updated_at)

    def test_deployment_timestamps_server_set(self, deployment_engine):
        """EmbeddingDeploymentConfig timestamps are server-set."""
        from vlm_feedback_loop.db.deployment_models import EmbeddingDeploymentConfig

        with Session(deployment_engine) as session:
            c = session.query(EmbeddingDeploymentConfig).first()
            assert ISO8601_Z_RE.match(c.created_at)
            assert ISO8601_Z_RE.match(c.updated_at)
