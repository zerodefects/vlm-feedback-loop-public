# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Database engine factory, PRAGMA configuration, integrity checks,
Alembic migration runner, and backup logic.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sqlite3
import tempfile
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Never

from alembic import command
from alembic.config import Config as AlembicConfig
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from alembic.util import CommandError
from sqlalchemy import Engine, create_engine, event, text
from sqlalchemy.engine import Connection
from sqlalchemy.orm import Session

# Import models/__init__ so all models register with ProjectBase.metadata
import vlm_feedback_loop.db.models  # noqa: F401  # pyright: ignore[reportUnusedImport]
from vlm_feedback_loop.db.base import DeploymentBase, generate_uuid4
from vlm_feedback_loop.db.deployment_models import (
    EmbeddingDeploymentConfig,
    TAODeploymentConfig,
)
from vlm_feedback_loop.model_catalog_constants import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL_ID,
    EMBEDDING_NIM_GPU_MIN_GB,
    EMBEDDING_NIM_IMAGE,
)

logger = logging.getLogger("vlm_feedback_loop.db")

# Legacy image pins seeded by earlier releases. ``1.13.0`` exists only for the
# text-only sibling image, never the VL variant; ``1.12.0`` was the last shipped
# VL pin before the 2.0 runtime. Frozen historical values stay literal here
# (NOT in model_catalog_constants) because their purpose is matching rows on
# disk, not naming the canonical image.
_LEGACY_BAD_EMBEDDING_NIM_IMAGE = (
    "nvcr.io/nim/nvidia/llama-nemotron-embed-vl-1b-v2:1.13.0"
)
_LEGACY_EMBEDDING_NIM_IMAGES = {
    _LEGACY_BAD_EMBEDDING_NIM_IMAGE,
    "nvcr.io/nim/nvidia/llama-nemotron-embed-vl-1b-v2:1.12.0",
}

_MIGRATIONS_DIR = Path(__file__).parent.parent / "migrations"
_CANONICAL_ALEMBIC_VERSION_DDL = (
    "CREATE TABLE alembic_version ( version_num VARCHAR(32) NOT NULL, "
    "CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num) )"
)

_deployment_engine_cache: dict[Path, Engine] = {}
_deployment_engine_cache_lock = threading.Lock()


# ── Exceptions ───────────────────────────────────────────────────────────────


class DatabaseCorruptionError(Exception):
    """Raised when SQLite structural or referential integrity checks fail."""

    pass


class DatabaseMigrationError(Exception):
    """Raised when an Alembic migration fails."""

    pass


# ── PRAGMA setup ─────────────────────────────────────────────────────────────


def _set_sqlite_pragmas(
    dbapi_conn: Any,
    connection_record: Any,  # noqa: ARG001 — SQLAlchemy event hook signature
) -> None:
    """Set per-connection SQLite PRAGMAs.

    These three are connection-local session state: SQLite resets them to
    their defaults on every fresh DBAPI connection, so they must be (re)set
    on each ``connect`` rather than once per engine.

      * ``busy_timeout=30000`` — when another writer (the backend itself or
        an external harness process holding a WAL write lock) is mid-commit,
        wait up to 30s for the lock instead of failing instantly. With the
        default of 0, concurrent cross-process writers surface immediately
        as ``sqlite3.OperationalError: disk I/O error`` / ``database is
        locked`` 500s. 30s comfortably covers a WAL checkpoint + fsync under
        contention.
      * ``synchronous=NORMAL`` — the SQLite-recommended durability level for
        WAL mode. Safe against application crashes; a checkpoint is only at
        risk on OS/power loss, which is the right trade-off for this tool and
        materially reduces fsync contention between concurrent writers.
      * ``wal_autocheckpoint=1000`` — checkpoint the WAL back into the main
        db roughly every 1000 pages, bounding WAL growth under sustained
        concurrent writes.

    WAL *journal mode* is deliberately NOT set here. It is a persistent
    property of the database file itself — setting it once at engine creation
    (see ``_ensure_wal_mode``) is sufficient, and re-confirming it on every
    checkout creates a race: it briefly exclusive-locks the ``-shm`` file, and
    under a burst of concurrent reads a competing writer can surface as a
    transient ``sqlite3.OperationalError: disk I/O error``
    (``SQLITE_IOERR_SHMMAP``) that propagates as a 500 from an
    otherwise-healthy endpoint.
    """
    cursor = dbapi_conn.cursor()
    try:
        cursor.execute("PRAGMA busy_timeout=30000")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA wal_autocheckpoint=1000")
    except BaseException:
        # SQLAlchemy does not yet own a DBAPI connection when a ``connect``
        # listener fails, so it cannot return or close that connection.
        with contextlib.suppress(Exception):
            cursor.close()
        with contextlib.suppress(Exception):
            dbapi_conn.close()
        raise
    else:
        cursor.close()


def _enable_sqlite_foreign_keys(
    dbapi_conn: Any,
    connection_record: Any,  # noqa: ARG001 — SQLAlchemy event hook signature
) -> None:
    """Enable and verify FK enforcement on an application runtime connection."""
    cursor = dbapi_conn.cursor()
    try:
        cursor.execute("PRAGMA foreign_keys=ON")
        row = cursor.execute("PRAGMA foreign_keys").fetchone()
        if row is None or row[0] != 1:
            raise sqlite3.DatabaseError(
                "SQLite refused to enable foreign-key enforcement"
            )
    except BaseException:
        # As above, a raising ``connect`` listener still owns the connection.
        with contextlib.suppress(Exception):
            cursor.close()
        with contextlib.suppress(Exception):
            dbapi_conn.close()
        raise
    else:
        cursor.close()


def _ensure_wal_mode(engine: Engine) -> None:
    """Ensure the database file is in WAL journal mode.

    Persistent on-disk setting — runs once at engine creation, not on
    every connection checkout. Tolerates a transient I/O error with a
    single retry so a brief filesystem hiccup during startup does not
    abort engine initialization.
    """
    for attempt in (1, 2):
        try:
            with engine.connect() as conn:
                conn.exec_driver_sql("PRAGMA journal_mode=WAL")
                conn.commit()
            return
        except Exception:
            if attempt == 2:
                raise


def _end_implicit_transaction_on_checkout(
    dbapi_conn: Any,
    connection_record: Any,  # noqa: ARG001 — SQLAlchemy event hook signature
    connection_proxy: Any,  # noqa: ARG001 — SQLAlchemy event hook signature
) -> None:
    """Terminate any lingering implicit read transaction on pool checkout.

    In a long-running backend, a pooled SQLite connection's implicit read
    transaction can outlive a single Session. When an external writer (e.g.
    the inspector seeder at ``scripts/seed_vlm_projects.py``) commits rows
    via a separate ``sqlite3`` connection, those rows are invisible to any
    pooled connection still sitting on the pre-write snapshot — ``session.get``
    silently returns ``None`` for rows that exist on disk.

    Explicitly rolling back the DBAPI connection at checkout ends any such
    snapshot and forces the next SELECT to read the latest committed WAL
    state. Rollback is a no-op if no transaction is open.
    """
    with contextlib.suppress(Exception):
        dbapi_conn.rollback()


def _create_engine(db_path: Path, *, enforce_foreign_keys: bool = False) -> Engine:
    """Create a SQLAlchemy engine with SQLite PRAGMAs attached.

    Migrations run with foreign keys disabled because SQLite table-rebuild
    revisions may need to replace related tables. ``open_project_db`` creates
    a fresh runtime engine with enforcement enabled only after migrations and
    integrity checks pass.

    Pool hardening (long-running backend):
      * ``pool_pre_ping=True`` — verifies connection health on checkout.
      * ``pool_recycle=3600`` — recycles connections every hour, bounding
        the maximum age of any pooled SQLite read snapshot.
      * ``checkout`` event issues an explicit ``rollback()`` on every
        DBAPI connection checkout so external-writer commits (inspector
        seeder, direct sqlite3 tooling) are visible to the next Session.
    """
    engine = create_engine(
        f"sqlite:///{db_path}",
        echo=False,
        pool_pre_ping=True,
        pool_recycle=3600,
    )
    event.listen(engine, "connect", _set_sqlite_pragmas)
    if enforce_foreign_keys:
        event.listen(engine, "connect", _enable_sqlite_foreign_keys)
    event.listen(engine, "checkout", _end_implicit_transaction_on_checkout)
    return engine


# ── Integrity check ──────────────────────────────────────────────────────────


def _run_quick_check(engine: Engine, db_path: Path) -> None:
    """Run ``PRAGMA quick_check``; raise on failure."""
    with engine.connect() as conn:
        result = conn.execute(text("PRAGMA quick_check")).scalar()
        if result != "ok":
            raise DatabaseCorruptionError(
                f"Database integrity check failed for {db_path}. "
                f"Restore from a backup file (project.db.backup.*)."
            )


def _run_foreign_key_check(engine: Engine, db_path: Path) -> None:
    """Reject a migrated database with invalid FK structure or rows."""
    try:
        with engine.connect() as conn:
            violations = conn.exec_driver_sql("PRAGMA foreign_key_check").fetchall()
    except Exception as exc:
        raise DatabaseCorruptionError(
            f"Foreign-key integrity check failed for {db_path}. "
            f"Restore from a backup file (project.db.backup.*). Error: {exc}"
        ) from exc
    if violations:
        raise DatabaseCorruptionError(
            f"Foreign-key integrity check found {len(violations)} violation(s) "
            f"in {db_path}. Restore from a backup file (project.db.backup.*)."
        )


# ── Backup ───────────────────────────────────────────────────────────────────


def _backup_database(db_path: Path) -> Path:
    """Create and validate a consistent SQLite snapshot before migration.

    SQLite's online backup API reads committed WAL frames as well as the main
    file, so a concurrent reader cannot make this recovery point silently
    stale. The completed snapshot is published with an exclusive hard link:
    no partial file is exposed as a backup and an existing backup is never
    replaced.
    """
    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%S.%fZ")
    backup_path = db_path.parent / (
        f"{db_path.name}.backup.{ts}.{uuid.uuid4().hex[:12]}"
    )
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{db_path.name}.backup.",
        suffix=".tmp",
        dir=db_path.parent,
    )
    os.close(fd)
    temp_path = Path(temp_name)

    try:
        with (
            contextlib.closing(sqlite3.connect(db_path, timeout=30.0)) as source,
            contextlib.closing(sqlite3.connect(temp_path, timeout=30.0)) as destination,
        ):
            source.backup(destination)
            try:
                result = destination.execute("PRAGMA quick_check").fetchone()
            except sqlite3.DatabaseError as exc:
                raise DatabaseCorruptionError(
                    f"Backup integrity check failed for {db_path}."
                ) from exc
            if result is None or result[0] != "ok":
                raise DatabaseCorruptionError(
                    f"Backup integrity check failed for {db_path}."
                )

        # Hard-link publication is atomic and refuses to replace an existing
        # path. Both files are in the project directory, so they share a
        # filesystem; unlinking the temporary name leaves the validated inode.
        os.link(temp_path, backup_path)
    finally:
        for suffix in ("", "-wal", "-shm", "-journal"):
            Path(f"{temp_path}{suffix}").unlink(missing_ok=True)

    logger.info("Database backed up to %s", backup_path)
    return backup_path


# ── Alembic migration runner ────────────────────────────────────────────────


def _get_alembic_config(connection: Connection) -> AlembicConfig:
    """Build an Alembic Config wired to the given connection."""
    cfg = AlembicConfig()
    cfg.set_main_option("script_location", str(_MIGRATIONS_DIR))
    cfg.attributes["connection"] = connection
    return cfg


def _refuse_unknown_revision_database(db_path: Path) -> Never:
    """Back up and refuse a database whose migration provenance is unknown."""
    backup_path = _backup_database(db_path)
    raise DatabaseMigrationError(
        "Refusing to migrate database with missing or invalid Alembic revision "
        f"state at {db_path}. Backup at {backup_path}. "
        "No migrations were applied. "
        "Determine the correct database revision before retrying."
    )


def _validate_migration_revision_state(
    connection: Connection,
    db_path: Path,
) -> None:
    """Allow only a known revision or a provably fresh SQLite schema.

    WAL initialization makes even a new SQLite file nonempty, so file size is
    not evidence of prior application state. An empty database and Alembic's
    canonical empty version table are the only retryable fresh states.
    """
    user_objects = {
        (row[0], row[1])
        for row in connection.exec_driver_sql(
            "SELECT type, name FROM sqlite_schema WHERE name NOT GLOB 'sqlite_*'"
        )
    }
    if not user_objects:
        return

    version_table = ("table", "alembic_version")
    if version_table not in user_objects:
        _refuse_unknown_revision_database(db_path)

    schema_sql = connection.exec_driver_sql(
        "SELECT sql FROM sqlite_schema "
        "WHERE type = 'table' AND name = 'alembic_version'"
    ).scalar_one_or_none()
    canonical_version_table = (
        isinstance(schema_sql, str)
        and " ".join(schema_sql.split()) == _CANONICAL_ALEMBIC_VERSION_DDL
    )
    if not canonical_version_table:
        _refuse_unknown_revision_database(db_path)

    revisions = connection.exec_driver_sql(
        "SELECT version_num FROM alembic_version LIMIT 2"
    ).fetchall()
    if len(revisions) > 1:
        _refuse_unknown_revision_database(db_path)
    if not revisions:
        if user_objects != {version_table}:
            _refuse_unknown_revision_database(db_path)
        return
    if not isinstance(revisions[0][0], str) or not revisions[0][0]:
        _refuse_unknown_revision_database(db_path)


def _run_migrations(engine: Engine, db_path: Path) -> None:
    """Detect and apply pending Alembic migrations with backup."""
    with engine.connect() as conn:
        _validate_migration_revision_state(conn, db_path)

        ctx = MigrationContext.configure(conn)
        current_rev = ctx.get_current_revision()

        cfg = _get_alembic_config(conn)
        script = ScriptDirectory.from_config(cfg)
        head_rev = script.get_current_head()

        if current_rev == head_rev:
            return  # up to date

        if current_rev is not None:
            try:
                script.get_revision(current_rev)
            except CommandError as exc:
                raise DatabaseMigrationError(
                    f"Unsupported database revision {current_rev!r} at {db_path}. "
                    "This build starts at the public v1 database baseline; "
                    "pre-v1 development databases are intentionally unsupported. "
                    "Create a fresh workspace and project."
                ) from exc

        # Backup before migration (only if the database has a known revision).
        backup_path = _backup_database(db_path) if current_rev is not None else None

        try:
            command.upgrade(cfg, "head")
            conn.commit()
            logger.info(
                "Migrations applied: %s → %s for %s",
                current_rev,
                head_rev,
                db_path,
            )
        except Exception as exc:
            with contextlib.suppress(Exception):
                conn.rollback()
            msg = f"Migration failed for {db_path}."
            if backup_path:
                msg += f" Backup at {backup_path}."
            msg += f" Error: {exc}"
            raise DatabaseMigrationError(msg) from exc


# ── Public API ───────────────────────────────────────────────────────────────


def open_project_db(project_dir: str | Path) -> Engine:
    """Open (or create) a per-project SQLite database.

    Sequence:
      1. Create engine with WAL + busy_timeout
      2. PRAGMA quick_check — fail on corruption
      3. Apply pending Alembic migrations (with backup)
      4. PRAGMA foreign_key_check
      5. Return a fresh runtime engine with foreign keys enforced

    Raises:
        DatabaseCorruptionError: if quick_check fails
        DatabaseMigrationError: if a migration fails
    """
    project_dir = Path(project_dir)
    # Honor the "open OR create" contract: a missing project_dir would
    # otherwise make the connect below fail and surface as a misleading
    # DatabaseCorruptionError ("file may be corrupt") rather than simply
    # creating the directory for a fresh project.db.
    project_dir.mkdir(parents=True, exist_ok=True)
    db_path = project_dir / "project.db"

    migration_engine = _create_engine(db_path)

    try:
        # Touch the database so the file exists before quick_check.
        # If the file is corrupt, the PRAGMA or SELECT will fail here.
        try:
            with migration_engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                conn.commit()
            # WAL journal mode is persistent on disk — set it once per engine
            # here, not on every connection checkout (see _set_sqlite_pragmas
            # docstring). Done inside the corruption try/except so a garbage
            # file on disk surfaces as DatabaseCorruptionError, not a raw
            # sqlite3.DatabaseError.
            _ensure_wal_mode(migration_engine)
        except Exception as exc:
            raise DatabaseCorruptionError(
                f"Cannot open database at {db_path}. "
                f"The file may be corrupt. Error: {exc}"
            ) from exc

        _run_quick_check(migration_engine, db_path)
        _run_migrations(migration_engine, db_path)
        _run_foreign_key_check(migration_engine, db_path)
    finally:
        migration_engine.dispose()

    return _create_engine(db_path, enforce_foreign_keys=True)


def init_deployment_db(workspace_root: str | Path) -> Engine:
    """Open (or create) the deployment-scoped database and seed defaults.

    The deployment database stores the EmbeddingDeploymentConfig and
    TAODeploymentConfig singletons.

    NOTE: deployment.db is intentionally NOT Alembic-managed — its schema is
    created idempotently via ``metadata.create_all`` and is treated as FROZEN.
    ``create_all`` only adds MISSING tables; it never adds a column to an
    existing table. So a new column on a deployment model would silently never
    reach existing deployment DBs. The hand-rolled data self-heals below
    (embedding model_name / image-tag corrections) exist precisely because
    there is no migration path here. If the deployment schema ever needs to
    evolve, give it its own Alembic branch rather than adding columns to these
    models.
    """
    workspace_path = Path(workspace_root).expanduser().resolve()
    workspace_path.mkdir(parents=True, exist_ok=True)
    db_path = workspace_path / "deployment.db"

    # Deployment state is process-scoped and shared by several services.
    # Serialize first-open plus the idempotent self-heal pass so concurrent
    # startup/request callers cannot create duplicate singleton rows.
    with _deployment_engine_cache_lock:
        engine = _deployment_engine_cache.get(workspace_path)
        created_engine = engine is None
        if engine is None:
            engine = _create_engine(db_path)

        try:
            if created_engine:
                _ensure_wal_mode(engine)
            DeploymentBase.metadata.create_all(engine)

            # Seed singletons if empty.
            with Session(engine) as session:
                existing = session.query(EmbeddingDeploymentConfig).first()
                if existing is None:
                    config = EmbeddingDeploymentConfig(
                        embedding_deployment_config_id=generate_uuid4(),
                        provider="none",
                        model_name=EMBEDDING_MODEL_ID,
                        embedding_dim=EMBEDDING_DIM,
                        endpoint_url=None,
                        nim_container_image=EMBEDDING_NIM_IMAGE,
                        preferred_host_port=8001,
                        # The smallest GPU SKUs validated for this model by
                        # the NIM 2.0.0 support matrix have 24 GB.
                        gpu_memory_minimum_gb=EMBEDDING_NIM_GPU_MIN_GB,
                        gpu_assignment=None,
                    )
                    session.add(config)
                    session.commit()
                    logger.info("Seeded EmbeddingDeploymentConfig in %s", db_path)
                elif existing.model_name in (
                    "nvidia/nvclip",
                    "nvidia/nvclip-vit-h-14",
                ):
                    # One-time idempotent migration from legacy NV-CLIP defaults.
                    previous_name = existing.model_name
                    existing.model_name = EMBEDDING_MODEL_ID
                    existing.embedding_dim = EMBEDDING_DIM
                    existing.nim_container_image = EMBEDDING_NIM_IMAGE
                    existing.gpu_memory_minimum_gb = EMBEDDING_NIM_GPU_MIN_GB
                    session.commit()
                    logger.info(
                        "Migrated legacy EmbeddingDeploymentConfig.model_name "
                        "'%s' -> '%s' in %s",
                        previous_name,
                        EMBEDDING_MODEL_ID,
                        db_path,
                    )
                elif (
                    existing.model_name == EMBEDDING_MODEL_ID
                    and existing.nim_container_image in _LEGACY_EMBEDDING_NIM_IMAGES
                ):
                    # Idempotent data migration from the shipped 1.x runtime.
                    previous_image = existing.nim_container_image
                    existing.nim_container_image = EMBEDDING_NIM_IMAGE
                    existing.gpu_memory_minimum_gb = EMBEDDING_NIM_GPU_MIN_GB
                    session.commit()
                    logger.info(
                        "Patched EmbeddingDeploymentConfig.nim_container_image "
                        "%s -> %s and GPU floor -> %s GB in %s",
                        previous_image,
                        EMBEDDING_NIM_IMAGE,
                        EMBEDDING_NIM_GPU_MIN_GB,
                        db_path,
                    )
                elif (
                    existing.model_name == EMBEDDING_MODEL_ID
                    and existing.nim_container_image == EMBEDDING_NIM_IMAGE
                    and existing.gpu_memory_minimum_gb < EMBEDDING_NIM_GPU_MIN_GB
                ):
                    previous_floor = existing.gpu_memory_minimum_gb
                    existing.gpu_memory_minimum_gb = EMBEDDING_NIM_GPU_MIN_GB
                    session.commit()
                    logger.info(
                        "Raised EmbeddingDeploymentConfig GPU floor %s -> %s GB in %s",
                        previous_floor,
                        EMBEDDING_NIM_GPU_MIN_GB,
                        db_path,
                    )

                existing_tao = session.query(TAODeploymentConfig).first()
                if existing_tao is None:
                    tao_config = TAODeploymentConfig(
                        tao_deployment_config_id=generate_uuid4(),
                        tao_workspace_id=None,
                        tao_workspace_name=None,
                        tao_workspace_cloud_type=None,
                        tao_workspace_bucket=None,
                        tao_workspace_s3_endpoint_url_internal=None,
                        tao_workspace_s3_endpoint_url_external=None,
                        tao_workspace_s3_access_key_ref=None,
                        tao_workspace_s3_secret_key_ref=None,
                        bootstrap_status="not_bootstrapped",
                        bootstrap_last_run_at=None,
                        bootstrap_error_ref=None,
                    )
                    session.add(tao_config)
                    session.commit()
                    logger.info("Seeded TAODeploymentConfig in %s", db_path)
        except BaseException:
            if created_engine:
                engine.dispose()
            raise

        if created_engine:
            _deployment_engine_cache[workspace_path] = engine
        return engine


def close_deployment_db_resources() -> None:
    """Dispose every process-scoped deployment database engine."""
    with _deployment_engine_cache_lock:
        engines = list(_deployment_engine_cache.values())
        _deployment_engine_cache.clear()

    for engine in engines:
        engine.dispose()
