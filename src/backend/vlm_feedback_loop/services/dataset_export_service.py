# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dataset export service.

Implements:
  - Cosmos-RL TAO-native dataset export: annotations.json + images/ in
    .tar.gz archives.
  - Three export field modes: all, aux_and_core, core_only.
  - Label selection by intent and tier: training (non-pool Verified),
    evaluation (Test Pool Verified), auto_labeled, combined.
  - Schema-invalid manifest for batch labeling runs.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import tarfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from vlm_feedback_loop.config import Settings
from vlm_feedback_loop.db.base import generate_uuid4, utc_now
from vlm_feedback_loop.db.models.dataset_export import DatasetExport
from vlm_feedback_loop.db.models.example import Example
from vlm_feedback_loop.db.models.guidance import Guidance
from vlm_feedback_loop.db.models.label import Label
from vlm_feedback_loop.db.models.operation import OperationRecord
from vlm_feedback_loop.db.models.project import Project
from vlm_feedback_loop.db.models.run import RunRecord
from vlm_feedback_loop.db.models.tao_job import TAOJob
from vlm_feedback_loop.services.authorized_file import (
    OpenedRegularFile,
    open_authorized_image,
    open_regular_file_beneath,
)
from vlm_feedback_loop.services.background import background_manager
from vlm_feedback_loop.services.hashing import sha256_file
from vlm_feedback_loop.services.pagination import (
    after_position,
    decode_cursor,
    encode_cursor,
)
from vlm_feedback_loop.services.project_service import (
    get_project_engine,
    project_dir_path,
    projects_root,
)
from vlm_feedback_loop.services.prompt_service import (
    render_training_conversation_prompt,
)
from vlm_feedback_loop.services.schema_core import (
    place_rationale_last,
    validate_and_derive,
)
from vlm_feedback_loop.services.sse import sse_manager

logger = logging.getLogger("vlm_feedback_loop.dataset_export_service")


@dataclass(frozen=True)
class DatasetExportArchive:
    """A verified project-contained archive ready for HTTP streaming."""

    opened_file: OpenedRegularFile
    checksum_sha256: str

    @property
    def path(self) -> Path:
        return self.opened_file.canonical_path


# ── Create ──────────────────────────────────────────────────────────────────


def create_dataset_export(
    project_id: str,
    *,
    dataset_intent: str,
    label_tier_filter: str = "verified_only",
    export_field_mode: str = "all",
    batch_label_run_id: str | None = None,
    selection_filters: dict[str, Any] | None = None,
    settings: Settings,
) -> dict[str, Any] | str:
    """Build a Cosmos-RL format dataset export archive.

    Returns a response dict on success or an error string for HTTP mapping.

    Synchronous variant: the caller blocks until the archive is fully
    built. The API endpoint uses :func:`start_dataset_export`
    (create-record-then-background-build) instead; the training-suite
    service calls this once per export in a worker thread, so each
    export commits in its own short-lived session.

    Thin wrapper: opens a session, delegates to
    :func:`persist_dataset_export_in_session`, commits, and returns.
    """
    engine = get_project_engine(project_id, settings.WORKSPACE_ROOT)
    if engine is None:
        return f"not found: Project {project_id}"

    with Session(engine) as session:
        result = persist_dataset_export_in_session(
            session,
            project_id,
            dataset_intent=dataset_intent,
            label_tier_filter=label_tier_filter,
            export_field_mode=export_field_mode,
            batch_label_run_id=batch_label_run_id,
            selection_filters=selection_filters,
            settings=settings,
        )
        if isinstance(result, str):
            # Error path — nothing staged, nothing to commit.
            return result
        session.commit()
        return result


def persist_dataset_export_in_session(
    session: Session,
    project_id: str,
    *,
    dataset_intent: str,
    label_tier_filter: str = "verified_only",
    export_field_mode: str = "all",
    batch_label_run_id: str | None = None,
    selection_filters: dict[str, Any] | None = None,
    settings: Settings,
) -> dict[str, Any] | str:
    """Session-scoped synchronous variant of :func:`create_dataset_export`.

    The caller owns the session lifecycle (BEGIN/COMMIT/ROLLBACK). This
    function only:
      1. Reads the Project + Guidance + Label rows,
      2. Builds the ``.tar.gz`` archive + manifest on disk — CPU + disk
         I/O proportional to total image bytes (no network). At product
         scale this takes minutes: async callers MUST run it in a worker
         thread (``asyncio.to_thread``) and MUST NOT hold a write
         transaction open across it (SQLite write discipline),
      3. Stages a new DatasetExport row (``status="completed"``) on
         ``session``.

    The training-suite service runs this in a worker thread with its own
    short-lived session; the standalone API endpoint uses
    :func:`start_dataset_export` (background build with a running row)
    instead.

    Archive files are written before the session insert. If the outer
    transaction rolls back, the tarball is orphaned on disk but the DB
    invariant is preserved — export artifacts are only reachable through
    committed DatasetExport rows.

    ``settings.WORKSPACE_ROOT`` determines export-archive placement through the
    canonical ``{workspace_root}/projects/{project_id}/exports`` layout. The
    stored ``Project.project_dir`` is metadata captured at create-time and can
    be stale after moving a workspace.

    Returns a response dict on success or an error string. On error, no
    rows are staged on the session; the caller may roll back its
    transaction or continue with unrelated work.
    """
    prepared = _prepare_dataset_export(
        session,
        project_id,
        dataset_intent=dataset_intent,
        label_tier_filter=label_tier_filter,
        export_field_mode=export_field_mode,
        batch_label_run_id=batch_label_run_id,
        selection_filters=selection_filters,
        settings=settings,
    )
    if isinstance(prepared, str):
        return prepared

    try:
        artifact_refs, manifest_ref = _build_export_artifacts(prepared, settings)
    except OSError as exc:
        _cleanup_export_artifacts(prepared.exports_dir, prepared.export_id)
        return f"validation: dataset export artifacts could not be built: {exc}"

    example_count = len(prepared.annotations)
    record = DatasetExport(
        dataset_export_id=prepared.export_id,
        project_id=project_id,
        dataset_intent=dataset_intent,
        export_field_mode=export_field_mode,
        guidance_id=prepared.guidance_id,
        label_tier_filter=label_tier_filter,
        selection_definition_snapshot=prepared.selection_snapshot,
        status="completed",
        started_at=prepared.created_at,
        completed_at=utc_now(),
        progress={"images_written": example_count, "images_total": example_count},
        artifact_refs=artifact_refs,
        manifest_ref=manifest_ref,
        example_count=example_count,
    )
    session.add(record)
    # Flush so the caller sees the row during the same transaction. If the
    # insert cannot be staged, no database owner can reach the generated
    # files, so remove them before propagating the transaction failure.
    try:
        session.flush()
    except Exception:
        _cleanup_export_artifacts(prepared.exports_dir, prepared.export_id)
        raise

    return {
        "dataset_export_id": prepared.export_id,
        "dataset_intent": dataset_intent,
        "export_field_mode": export_field_mode,
        "label_tier_filter": label_tier_filter,
        "guidance_id": prepared.guidance_id,
        "example_count": example_count,
        "artifact_refs": artifact_refs,
        "manifest_ref": manifest_ref,
        "created_at": prepared.created_at,
    }


# ── Selection phase ─────────────────────────────────────────────────────────


@dataclass
class _PreparedExport:
    """Selection-phase output: everything the archive build needs.

    Produced by :func:`_prepare_dataset_export` under a short DB read;
    consumed by :func:`_build_export_artifacts` without touching the
    database, so the (potentially multi-GB) build can run outside any
    transaction — and, on the API path, off the event loop.
    """

    export_id: str
    project_id: str
    created_at: str
    dataset_intent: str
    label_tier_filter: str
    export_field_mode: str
    guidance_id: str
    selection_snapshot: dict[str, Any]
    annotations: list[dict[str, Any]]
    example_refs: dict[str, str]
    skipped_missing_images: list[str]
    exports_dir: Path


def _prepare_dataset_export(
    session: Session,
    project_id: str,
    *,
    dataset_intent: str,
    label_tier_filter: str,
    export_field_mode: str,
    batch_label_run_id: str | None,
    selection_filters: dict[str, Any] | None,
    settings: Settings,
) -> _PreparedExport | str:
    """Selection phase: validate, select labels, build the annotations.

    DB reads and in-memory work only — no disk writes, no rows staged.
    Returns an error string on validation failure.
    """
    if batch_label_run_id == "":
        return "validation: batch_label_run_id must be non-empty when provided"

    if (
        dataset_intent in ("evaluation", "testing")
        and label_tier_filter != "verified_only"
    ):
        return (
            f"validation: {dataset_intent} dataset exports require "
            "label_tier_filter='verified_only'; Auto-Labeled predictions "
            "are not ground truth"
        )

    project = session.query(Project).filter_by(project_id=project_id).first()
    if project is None:
        return f"not found: Project {project_id}"
    if not project.active_guidance_id:
        return "No active Guidance configured"

    # ── Load Guidance ──────────────────────────────────────────────────
    guidance_id_filter = (
        selection_filters.get("guidance_id") if selection_filters else None
    ) or project.active_guidance_id

    guidance = (
        session.query(Guidance)
        .filter_by(
            project_id=project_id,
            guidance_id=guidance_id_filter,
        )
        .first()
    )
    if guidance is None:
        return f"not found: Guidance {guidance_id_filter}"

    envelope = guidance.schema or {}
    fields = envelope.get("fields", [])
    generation_order: list[str] = envelope.get("generation_order", [])
    derived_json_schema: dict[str, Any] = envelope.get("derived_json_schema", {})
    if not derived_json_schema and fields:
        validation = validate_and_derive(fields)
        if validation.save_allowed and validation.derived_json_schema:
            derived_json_schema = validation.derived_json_schema
    description = guidance.description or ""
    rules = guidance.rules or ""
    g_id = guidance.guidance_id

    # ── Select labels ──────────────────────────────────────────────────
    labels_data = _select_labels(
        session,
        project_id=project_id,
        guidance_id=g_id,
        dataset_intent=dataset_intent,
        label_tier_filter=label_tier_filter,
        batch_label_run_id=batch_label_run_id,
    )

    # ── Load storage_refs ──────────────────────────────────────────────
    example_keys = [ld["example_key"] for ld in labels_data]
    example_refs: dict[str, str] = {}
    if example_keys:
        rows = list(
            session.execute(
                select(Example.example_key, Example.storage_ref).where(
                    Example.project_id == project_id,
                    Example.example_key.in_(example_keys),
                )
            ).all()
        )
        example_refs = {r[0]: r[1] for r in rows}

    project_dir = str(project_dir_path(settings.WORKSPACE_ROOT, project_id))

    # ── Build annotations ──────────────────────────────────────────────
    annotations = _build_annotations(
        labels_data=labels_data,
        example_refs=example_refs,
        generation_order=generation_order,
        fields=fields,
        derived_json_schema=derived_json_schema,
        description=description,
        rules=rules,
        export_field_mode=export_field_mode,
    )

    # Images are stored by reference and can have moved since ingest. An
    # annotation whose image cannot be added to the tar must not ship —
    # TAO's loader would fail (or silently train short) on annotations
    # referencing images absent from the archive. Exclude them from BOTH
    # the annotations and the tar so every artifact (sidecar, tar,
    # manifest counts) stays self-consistent, and record what was skipped.
    skipped_missing_images: list[str] = []
    exportable: list[dict[str, Any]] = []
    for sample in annotations:
        ref = example_refs.get(sample["id"])
        if not ref:
            skipped_missing_images.append(sample["id"])
            continue
        try:
            with open_authorized_image(ref, settings):
                pass
        except FileNotFoundError:
            skipped_missing_images.append(sample["id"])
            continue
        except PermissionError as exc:
            return (
                f"validation: image for example {sample['id']} is not accessible "
                f"under the current IMAGE_ROOT policy: {exc}"
            )
        exportable.append(sample)
    if skipped_missing_images:
        logger.warning(
            "Dataset export: skipping %d example(s) whose image file is "
            "missing at its storage_ref (first few: %s)",
            len(skipped_missing_images),
            ", ".join(skipped_missing_images[:5]),
        )
    if annotations and not exportable:
        # Labels matched the selection but EVERY image is gone (typical
        # after a workspace migration that didn't carry the image tree).
        # A zero-sample export would upload cleanly and then fail — or
        # silently no-op — inside TAO training, so fail fast here instead.
        return (
            f"validation: none of the {len(annotations)} selected examples "
            "has its image file at its recorded path — images are stored "
            "by reference and appear to have moved. Re-ingest or remap "
            "paths (examples:remap_paths) before exporting."
        )
    annotations = exportable

    selection_snapshot: dict[str, Any] = {
        "dataset_intent": dataset_intent,
        "label_tier_filter": label_tier_filter,
        "export_field_mode": export_field_mode,
        "batch_label_run_id": batch_label_run_id,
        "guidance_id": g_id,
    }
    if selection_filters:
        selection_snapshot["selection_filters"] = selection_filters

    return _PreparedExport(
        export_id=generate_uuid4(),
        project_id=project_id,
        created_at=utc_now(),
        dataset_intent=dataset_intent,
        label_tier_filter=label_tier_filter,
        export_field_mode=export_field_mode,
        guidance_id=g_id,
        selection_snapshot=selection_snapshot,
        annotations=annotations,
        example_refs=example_refs,
        skipped_missing_images=skipped_missing_images,
        exports_dir=Path(project_dir) / "exports",
    )


# ── Build phase ─────────────────────────────────────────────────────────────


def _export_artifact_paths(
    exports_dir: Path, export_id: str
) -> tuple[Path, Path, Path]:
    """(archive, sidecar annotations.json, manifest) paths for an export.

    The single naming convention: the builder writes these paths and
    startup recovery deletes them for interrupted exports.
    """
    return (
        exports_dir / f"{export_id}.tar.gz",
        exports_dir / f"{export_id}_annotations.json",
        exports_dir / f"{export_id}_manifest.json",
    )


def _cleanup_export_artifacts(exports_dir: Path, export_id: str) -> None:
    """Best-effort removal of an export's (possibly partial) artifact files."""
    for path in _export_artifact_paths(exports_dir, export_id):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not delete export artifact %s", path, exc_info=True)


def cleanup_dataset_export_artifacts(
    project_id: str,
    dataset_export_id: str,
    *,
    settings: Settings,
) -> None:
    """Remove artifacts for an export whose database insert was rolled back."""
    exports_dir = project_dir_path(settings.WORKSPACE_ROOT, project_id) / "exports"
    _cleanup_export_artifacts(exports_dir, dataset_export_id)


def _build_export_artifacts(
    prepared: _PreparedExport,
    settings: Settings,
    progress_cb: Callable[[int, int], None] | None = None,
) -> tuple[dict[str, str], str]:
    """Build phase: write sidecar + tar.gz + checksum + manifest to disk.

    Pure disk I/O over the prepared selection — no DB access, so it is
    safe to run inside a caller-owned transaction (training suite) or in
    a worker thread (API path). Returns ``(artifact_refs, manifest_ref)``.
    """
    prepared.exports_dir.mkdir(parents=True, exist_ok=True)
    tar_path, annotations_json_path, manifest_path = _export_artifact_paths(
        prepared.exports_dir, prepared.export_id
    )
    # Sidecar annotations.json mirrors the JSON inside the tar (our
    # archive bundles annotations.json and images/ into one tar). Cosmos-RL's
    # ``tao_sft_example.py`` opens ``annotation_path`` as a JSON file
    # via ``json.load(open(...))`` — pointing the JSON at the extracted
    # directory raises FileNotFoundError — so TAO needs a SEPARATE URL
    # for the annotations to download as a standalone file. The sidecar
    # is uploaded alongside the tar.gz under the same dataset_export_id
    # S3 prefix.
    annotations_json_path.write_text(json.dumps(prepared.annotations))

    _build_tar_archive(
        tar_path,
        prepared.annotations,
        prepared.example_refs,
        settings,
        progress_cb=progress_cb,
    )

    checksum = sha256_file(tar_path)

    manifest = {
        "dataset_export_id": prepared.export_id,
        **prepared.selection_snapshot,
        "example_count": len(prepared.annotations),
        "example_keys": [s["id"] for s in prepared.annotations],
        "skipped_missing_images": prepared.skipped_missing_images,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    artifact_refs = {
        "archive_path": str(tar_path),
        "annotations_path": str(annotations_json_path),
        "checksum_sha256": checksum,
    }
    return artifact_refs, str(manifest_path)


# ── Background build (API path) ─────────────────────────────────────────────


async def start_dataset_export(
    project_id: str,
    *,
    dataset_intent: str,
    label_tier_filter: str = "verified_only",
    export_field_mode: str = "all",
    batch_label_run_id: str | None = None,
    selection_filters: dict[str, Any] | None = None,
    settings: Settings,
) -> dict[str, Any] | str:
    """Create a DatasetExport record and build its archive in the background.

    The API-endpoint variant of :func:`create_dataset_export`: selection
    and validation run synchronously (cheap DB reads), the record is
    committed with ``status="running"`` and null ``artifact_refs``, and
    the multi-GB tar build runs as a background task — the HTTP request
    returns immediately instead of blocking for minutes and timing out at
    the proxy. Consumers poll ``GET .../dataset_exports/{id}`` or follow
    the ``export_progress`` / ``export_completed`` / ``export_failed``
    SSE events.

    Returns the running-record response dict, or an error string for
    HTTP mapping (validation failures surface before the record exists).
    """
    engine = get_project_engine(project_id, settings.WORKSPACE_ROOT)
    if engine is None:
        return f"not found: Project {project_id}"

    with Session(engine) as session:
        # One archive build at a time per project: each build is a
        # multi-GB CPU/disk job, and an accidental re-POST after
        # navigation would silently double it.
        running = (
            session.query(DatasetExport.dataset_export_id)
            .filter_by(project_id=project_id, status="running")
            .first()
        )
        if running is not None:
            return (
                f"conflict: dataset export {running[0]} is already building "
                "for this project — it reports progress on its record; wait "
                "for it to finish before starting another."
            )
        prepared = _prepare_dataset_export(
            session,
            project_id,
            dataset_intent=dataset_intent,
            label_tier_filter=label_tier_filter,
            export_field_mode=export_field_mode,
            batch_label_run_id=batch_label_run_id,
            selection_filters=selection_filters,
            settings=settings,
        )
        if isinstance(prepared, str):
            return prepared

        example_count = len(prepared.annotations)
        started_at = utc_now()
        progress = {"images_written": 0, "images_total": example_count}
        record = DatasetExport(
            dataset_export_id=prepared.export_id,
            project_id=project_id,
            dataset_intent=dataset_intent,
            export_field_mode=export_field_mode,
            guidance_id=prepared.guidance_id,
            label_tier_filter=label_tier_filter,
            selection_definition_snapshot=prepared.selection_snapshot,
            status="running",
            started_at=started_at,
            progress=progress,
            artifact_refs=None,
            manifest_ref=None,
            example_count=example_count,
        )
        session.add(record)
        session.commit()
        # Post-commit attribute access refreshes from the committed row,
        # so the response matches what GET .../dataset_exports/{id} would
        # return (the response model drops the keys it does not declare).
        response = _export_to_dict(record)

    background_manager.register(
        task_id=f"dataset-export-{prepared.export_id}",
        coro=_execute_dataset_export(prepared, settings),
    )

    return response


async def _execute_dataset_export(
    prepared: _PreparedExport, settings: Settings
) -> None:
    """Background coroutine: build the archive, then finalize the record.

    The build runs in a worker thread (a multi-GB tar + sha256 is blocking
    disk I/O). Progress is persisted at ~1% cadence and mirrored as
    ``export_progress`` SSE hints; terminal transitions emit
    ``export_completed`` / ``export_failed``. On failure, partial artifact
    files are deleted. A crash or restart between dispatch and the
    terminal update leaves the row ``running`` —
    :func:`recover_dataset_exports` marks it failed on the next startup.
    """
    project_id = prepared.project_id
    export_id = prepared.export_id
    engine = get_project_engine(project_id, settings.WORKSPACE_ROOT)
    if engine is None:
        return  # project gone; nothing to do

    loop = asyncio.get_running_loop()

    def _on_progress(written: int, total: int) -> None:
        # Runs on the builder thread: persist progress (the REST
        # reconciliation source), then hand the SSE hint to the event
        # loop. Progress bookkeeping must never fail the build itself.
        try:
            with Session(engine) as session:
                row = session.get(DatasetExport, export_id)
                if row is not None:
                    row.progress = {
                        "images_written": written,
                        "images_total": total,
                    }
                    session.commit()
            asyncio.run_coroutine_threadsafe(
                sse_manager.emit(
                    project_id,
                    "export_progress",
                    {
                        "dataset_export_id": export_id,
                        "images_written": written,
                        "images_total": total,
                    },
                ),
                loop,
            )
        except Exception:
            logger.warning(
                "Dataset export %s: progress update failed",
                export_id,
                exc_info=True,
            )

    try:
        artifact_refs, manifest_ref = await asyncio.to_thread(
            _build_export_artifacts,
            prepared,
            settings,
            _on_progress,
        )
    except Exception as exc:
        _cleanup_export_artifacts(prepared.exports_dir, export_id)
        reason = f"{type(exc).__name__}: {exc}"[:500]
        with Session(engine) as session:
            row = session.get(DatasetExport, export_id)
            if row is not None:
                row.status = "failed"
                row.status_reason = reason
                row.completed_at = utc_now()
                session.commit()
        logger.error("Dataset export %s failed: %s", export_id, reason)
        await sse_manager.emit(
            project_id,
            "export_failed",
            {"dataset_export_id": export_id, "status_reason": reason},
        )
        return

    example_count = len(prepared.annotations)
    with Session(engine) as session:
        row = session.get(DatasetExport, export_id)
        if row is None:
            # Record deleted mid-build; artifacts are orphaned on disk but
            # unreachable — same invariant as a rolled-back sync export.
            return
        row.status = "completed"
        row.completed_at = utc_now()
        row.artifact_refs = artifact_refs
        row.manifest_ref = manifest_ref
        row.progress = {
            "images_written": example_count,
            "images_total": example_count,
        }
        session.commit()
    await sse_manager.emit(
        project_id,
        "export_completed",
        {
            "dataset_export_id": export_id,
            "example_count": example_count,
            "artifact_refs": artifact_refs,
        },
    )


def recover_dataset_exports(settings: Settings) -> None:
    """Fail exports interrupted by a restart (startup recovery contract).

    A backend restart kills the in-process archive build, which would
    leave the row ``running`` forever. Resuming a half-written tar.gz is
    not supported: partial artifact files are deleted by path convention
    and the record is marked failed so the SME simply retries. Mirrors
    ``recover_ingest_tasks``: walks the workspace, skips ``.archived``
    projects, per-project try/except so one corrupt DB cannot block
    startup.
    """
    projects_dir = projects_root(settings.WORKSPACE_ROOT)
    if not projects_dir.exists():
        return

    for entry in sorted(projects_dir.iterdir()):
        if not entry.is_dir() or not (entry / "project.db").exists():
            continue
        if (entry / ".archived").exists():
            continue

        project_id = entry.name
        try:
            engine = get_project_engine(project_id, settings.WORKSPACE_ROOT)
            if engine is None:
                continue
            with Session(engine) as session:
                running = (
                    session.query(DatasetExport)
                    .filter_by(project_id=project_id, status="running")
                    .all()
                )
                if not running:
                    continue
                now = utc_now()
                exports_dir = (
                    project_dir_path(settings.WORKSPACE_ROOT, project_id) / "exports"
                )
                for row in running:
                    _cleanup_export_artifacts(exports_dir, row.dataset_export_id)
                    row.status = "failed"
                    row.status_reason = "backend_restart_interrupted"
                    row.completed_at = now
                    logger.info(
                        "Recovery: dataset export %s → failed",
                        row.dataset_export_id,
                    )
                session.commit()
        except Exception as exc:
            logger.warning(
                "Skipping dataset-export recovery for project %s (%s: %s)",
                project_id,
                type(exc).__name__,
                str(exc) or "(no message)",
            )
            continue


# ── Label Selection ────────────────────────────────────────────────────────


def _select_labels(
    session: Session,
    *,
    project_id: str,
    guidance_id: str,
    dataset_intent: str,
    label_tier_filter: str,
    batch_label_run_id: str | None,
) -> list[dict[str, Any]]:
    """Select labels for export based on intent and tier filter.

    Returns a list of dicts with example_key and label_json.
    """
    results: list[dict[str, Any]] = []

    if label_tier_filter in ("verified_only", "combined"):
        # Verified labels under current Guidance
        stmt = select(Label.example_key, Label.label_json).where(
            Label.project_id == project_id,
            Label.label_status == "verified",
            Label.guidance_id == guidance_id,
        )

        if dataset_intent in ("evaluation", "testing"):
            # Test Pool members only
            stmt = stmt.where(Label.pool_assignment == "test_pool")
        else:
            # Training: non-pool (exclude Test Pool)
            stmt = stmt.where(Label.pool_assignment.is_(None))

        stmt = stmt.order_by(Label.example_key.asc())
        rows = list(session.execute(stmt).all())
        results.extend({"example_key": r[0], "label_json": r[1]} for r in rows)

    if label_tier_filter in ("auto_labeled_only", "combined"):
        # Auto-Labeled labels under current Guidance
        stmt = (
            select(Label.example_key, Label.label_json)
            .join(
                Example,
                (Example.project_id == Label.project_id)
                & (Example.example_key == Label.example_key),
            )
            .where(
                Label.project_id == project_id,
                Label.label_status == "auto_labeled",
                Label.guidance_id == guidance_id,
                Example.state == "Auto-Labeled",
            )
            .order_by(Label.example_key.asc())
        )
        if batch_label_run_id is not None:
            stmt = stmt.where(Label.batch_label_run_id == batch_label_run_id)

        rows = list(session.execute(stmt).all())
        results.extend({"example_key": r[0], "label_json": r[1]} for r in rows)

    # Deduplicate by example_key (combined may overlap if an example
    # was Auto-Labeled then promoted to Verified — keep Verified).
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for item in results:
        if item["example_key"] not in seen:
            seen.add(item["example_key"])
            deduped.append(item)
    return deduped


# ── Annotations Builder ────────────────────────────────────────────────────


def _build_annotations(
    *,
    labels_data: list[dict[str, Any]],
    example_refs: dict[str, str],
    generation_order: list[str],
    fields: list[dict[str, Any]],
    derived_json_schema: dict[str, Any],
    description: str,
    rules: str,
    export_field_mode: str,
) -> list[dict[str, Any]]:
    """Build the Cosmos-RL annotations.json content.

    Each sample: {id, images, conversations} where conversations uses
    from/value (not role/content). The human turn carries the rendered
    §6 serving prompt (zero-shot, self-contained — §9.3.2) so trained
    Students see the same prompt distribution at serve time; training and
    evaluation exports render it identically (§9.7.6 mirror rule). The compact
    prompt-visible schema carries the task contract; verbose Guidance
    Description/Rules prose is intentionally omitted, matching Teacher serve.
    """
    prompt_schema, prompt_generation_order = place_rationale_last(
        derived_json_schema,
        generation_order,
    )
    human_value = render_training_conversation_prompt(description, rules, prompt_schema)

    annotations: list[dict[str, Any]] = []
    for item in labels_data:
        example_key = item["example_key"]
        label_json = item["label_json"]

        # Determine image filename from storage_ref extension
        storage_ref = example_refs.get(example_key, "")
        ext = Path(storage_ref).suffix if storage_ref else ".jpg"
        image_path = f"images/{example_key}{ext}"

        # Serialize in the same rationale-last order the serving prompt asks
        # the Student to produce, avoiding train/serve output-order skew.
        gpt_value = _serialize_label_for_export(
            label_json,
            prompt_generation_order,
            fields,
            export_field_mode,
        )

        sample = {
            "id": example_key,
            "images": [image_path],
            "conversations": [
                {"from": "human", "value": human_value},
                {"from": "gpt", "value": gpt_value},
            ],
        }
        annotations.append(sample)

    return annotations


def _serialize_label_for_export(
    label_json: dict[str, Any],
    generation_order: list[str],
    fields: list[dict[str, Any]],
    export_field_mode: str,
) -> str:
    """Serialize a label to a JSON string with prescribed field ordering.

    The gpt turn value is always a text string containing valid JSON,
    NOT a nested JSON object.

    Field inclusion per export_field_mode:
      - all: other Aux + Core + rationale_note (serving-prompt order)
      - aux_and_core: Aux (excluding rationale_note) + Core
      - core_only: Core only
    """
    # Build a lookup of field_name → role
    field_roles = {f["field_name"]: f.get("role", "core") for f in fields}

    # Determine which field names to include
    included_names: set[str] = set()
    for fname in generation_order:
        role = field_roles.get(fname, "core")
        if export_field_mode == "all":
            included_names.add(fname)
        elif export_field_mode == "aux_and_core" and fname != "rationale_note":
            # Exclude rationale_note, include everything else
            included_names.add(fname)
        elif export_field_mode == "core_only" and role == "core":
            included_names.add(fname)

    # Build ordered dict following generation_order
    ordered: dict[str, Any] = {}
    for fname in generation_order:
        if fname in included_names and fname in label_json:
            ordered[fname] = label_json[fname]

    return json.dumps(ordered, separators=(",", ":"), ensure_ascii=False)


# ── Archive Builder ────────────────────────────────────────────────────────


def _build_tar_archive(
    tar_path: Path,
    annotations: list[dict[str, Any]],
    example_refs: dict[str, str],
    settings: Settings,
    progress_cb: Callable[[int, int], None] | None = None,
) -> None:
    """Create a .tar.gz archive with annotations.json + images/.

    ``progress_cb(images_written, images_total)`` (when provided) is
    invoked at ~1% cadence so the background path can surface progress.

    Uses tar.gz format (TAO/Cosmos-RL dataset packaging convention).

    Tar contents are flat: ``annotations.json`` + ``images/{key}.ext``
    at the archive root. TAO's ``_extract_images`` (in
    ``cloud_handlers/utils.py``) auto-detects the depth of ``images/``
    and strips that many components on extraction, so any internal
    namespace would be flattened anyway — the resulting layout under
    the destination is always ``{dest}/images/`` + ``{dest}/annotations.json``.

    TAO has a path-substitution mismatch (it sets
    ``media_path = local_path.replace(".tar.gz", "")`` but extracts at
    ``os.path.dirname(local_path)``); we compensate by
    pointing ``media_path`` at the parent directory URL rather than
    the .tar.gz URL — see ``apply_dataset_binding`` /
    ``training_suite_service``. With that fix, the simple flat tar
    layout below is correct.
    """
    with tarfile.open(tar_path, "w:gz") as tf:
        # Write annotations.json
        ann_bytes = json.dumps(annotations, indent=2, ensure_ascii=False).encode(
            "utf-8"
        )
        info = tarfile.TarInfo(name="annotations.json")
        info.size = len(ann_bytes)
        tf.addfile(info, io.BytesIO(ann_bytes))

        # Write images. The caller pre-filters annotations to images that
        # exist on disk, so no exists() re-stat here — only a guard for the
        # (tiny) window where a file vanishes between that check and tf.add,
        # which must be loud: a silently absent image would desynchronize
        # the tar from the annotations.json written above.
        total = len(annotations)
        progress_step = max(1, total // 100)
        for written, sample in enumerate(annotations, start=1):
            example_key = sample["id"]
            image_archive_path = sample["images"][0]
            storage_ref = example_refs.get(example_key)
            if not storage_ref:
                raise FileNotFoundError(
                    f"Dataset export image reference is missing for {example_key}"
                )
            try:
                with open_authorized_image(storage_ref, settings) as opened:
                    info = tarfile.TarInfo(name=image_archive_path)
                    info.size = opened.stat_result.st_size
                    with opened.open_binary() as source:
                        tf.addfile(info, source)
            except OSError as exc:
                logger.warning(
                    "Dataset export: image for %s became unavailable or "
                    "unauthorized before archiving (%s)",
                    example_key,
                    exc,
                )
                raise
            if progress_cb is not None and (
                written % progress_step == 0 or written == total
            ):
                progress_cb(written, total)


# ── Get / List ──────────────────────────────────────────────────────────────


def resolve_test_pool_dataset_sha(
    session: Session, dataset_export_ids: list[str]
) -> str | None:
    """Archive SHA-256 of a Test Pool export.

    Reproducibility provenance: the deployment handoff, the Compare screen's
    suite provenance, and the Student-baseline evaluation all report the same
    checksum, so they must resolve it identically. Training suites use
    ``dataset_intent="evaluation"`` for their held-out export; the public
    export API also permits the equivalent ``testing`` intent.
    """
    for export_id in dataset_export_ids:
        export = session.get(DatasetExport, export_id)
        if export is None:
            continue
        if export.dataset_intent in {"evaluation", "testing"}:
            refs = export.artifact_refs or {}
            sha = refs.get("checksum_sha256")
            if isinstance(sha, str) and sha:
                return sha
            return None
    return None


def resolve_paired_test_pool_dataset_sha(
    session: Session,
    *,
    artifact_parent_tao_job_id: str,
    fallback_export_ids: list[str] | None = None,
) -> str | None:
    """Resolve the held-out export paired with a Student artifact job.

    ``StudentModel.dataset_export_ids`` intentionally records training data
    only. The held-out export belongs to the evaluate job whose
    ``parent_tao_job_id`` is the train job for a baseline Student or the
    quantize job for a quantized Student. Following that relationship keeps
    production handoff provenance truthful without overloading the Student's
    training-data lineage.
    """
    evaluate_job = session.scalars(
        select(TAOJob)
        .where(
            TAOJob.parent_tao_job_id == artifact_parent_tao_job_id,
            TAOJob.action == "evaluate",
        )
        .order_by(TAOJob.chain_sequence.asc(), TAOJob.created_at.asc())
    ).first()
    if evaluate_job is not None:
        sha = resolve_test_pool_dataset_sha(
            session, list(evaluate_job.dataset_export_ids or [])
        )
        if sha is not None:
            return sha
    return resolve_test_pool_dataset_sha(session, list(fallback_export_ids or []))


def get_dataset_export(
    project_id: str,
    dataset_export_id: str,
    settings: Settings,
) -> dict[str, Any] | str:
    """Load a single DatasetExport record."""
    engine = get_project_engine(project_id, settings.WORKSPACE_ROOT)
    if engine is None:
        return f"not found: Project {project_id}"

    with Session(engine) as session:
        record = (
            session.query(DatasetExport)
            .filter_by(
                dataset_export_id=dataset_export_id,
                project_id=project_id,
            )
            .first()
        )
        if record is None:
            return f"not found: Dataset export {dataset_export_id}"
        return _export_to_dict(record)


def get_dataset_export_archive(
    project_id: str,
    dataset_export_id: str,
    settings: Settings,
) -> DatasetExportArchive | str:
    """Resolve a completed export archive without trusting its stored path.

    The returned descriptor is bound to the authorized inode and owned by the
    HTTP response, so a later pathname replacement cannot change the bytes.
    """

    engine = get_project_engine(project_id, settings.WORKSPACE_ROOT)
    if engine is None:
        return f"not found: Project {project_id}"

    with Session(engine) as session:
        record = (
            session.query(DatasetExport)
            .filter_by(
                dataset_export_id=dataset_export_id,
                project_id=project_id,
            )
            .first()
        )
        if record is None:
            return f"not found: Dataset export {dataset_export_id}"
        if record.status != "completed":
            return (
                f"conflict: Dataset export {dataset_export_id} is "
                f"{record.status}, not completed"
            )
        refs = record.artifact_refs or {}
        archive_ref = refs.get("archive_path")
        checksum = refs.get("checksum_sha256")

    if not isinstance(archive_ref, str) or not archive_ref:
        return f"conflict: Dataset export {dataset_export_id} has no archive"
    if not isinstance(checksum, str) or not checksum:
        return f"conflict: Dataset export {dataset_export_id} has no checksum"

    archive_path = Path(archive_ref)
    export_root = project_dir_path(settings.WORKSPACE_ROOT, project_id) / "exports"
    try:
        opened = open_regular_file_beneath(archive_path, export_root)
    except FileNotFoundError:
        return f"not found: Dataset export archive {dataset_export_id}"
    except PermissionError:
        return (
            "validation: Dataset export archive reference is outside or cannot "
            "be safely opened beneath the project exports directory"
        )

    return DatasetExportArchive(
        opened_file=opened,
        checksum_sha256=checksum,
    )


def list_dataset_exports(
    project_id: str,
    *,
    dataset_intent_filter: str | None = None,
    cursor: str | None = None,
    limit: int = 20,
    settings: Settings,
) -> tuple[list[dict[str, Any]], str | None]:
    """List dataset exports with cursor pagination, newest-first."""
    engine = get_project_engine(project_id, settings.WORKSPACE_ROOT)
    if engine is None:
        return [], None

    with Session(engine) as session:
        stmt = (
            select(DatasetExport)
            .where(DatasetExport.project_id == project_id)
            .order_by(
                DatasetExport.created_at.desc(), DatasetExport.dataset_export_id.desc()
            )
        )
        if dataset_intent_filter:
            stmt = stmt.where(DatasetExport.dataset_intent == dataset_intent_filter)
        if cursor:
            cur_ts, cur_id = decode_cursor(cursor)
            stmt = stmt.where(
                after_position(
                    DatasetExport.created_at,
                    DatasetExport.dataset_export_id,
                    cur_ts,
                    cur_id,
                )
            )
        stmt = stmt.limit(limit + 1)

        rows = list(session.execute(stmt).scalars().all())

    next_cursor = None
    if len(rows) > limit:
        rows = rows[:limit]
        next_cursor = encode_cursor(rows[-1].created_at, rows[-1].dataset_export_id)

    return [_export_to_dict(r) for r in rows], next_cursor


def _export_to_dict(record: DatasetExport) -> dict[str, Any]:
    """Convert a DatasetExport record to an API response dict."""
    return {
        "dataset_export_id": record.dataset_export_id,
        "project_id": record.project_id,
        "dataset_intent": record.dataset_intent,
        "export_field_mode": record.export_field_mode,
        "guidance_id": record.guidance_id,
        "label_tier_filter": record.label_tier_filter,
        "selection_definition_snapshot": record.selection_definition_snapshot,
        "example_count": record.example_count,
        "status": record.status,
        "status_reason": record.status_reason,
        "progress": record.progress,
        "started_at": record.started_at,
        "completed_at": record.completed_at,
        "artifact_refs": record.artifact_refs,
        "manifest_ref": record.manifest_ref,
        "created_at": record.created_at,
    }


# ── Schema-Invalid Manifest ─────────────────────────────────────────────────


def get_schema_invalid_manifest(
    project_id: str,
    run_id: str,
    settings: Settings,
) -> dict[str, Any] | str:
    """Return a manifest of schema-invalid examples from a batch run."""
    engine = get_project_engine(project_id, settings.WORKSPACE_ROOT)
    if engine is None:
        return f"not found: Project {project_id}"

    with Session(engine) as session:
        run = (
            session.query(RunRecord)
            .filter_by(
                run_id=run_id,
                project_id=project_id,
            )
            .first()
        )
        if run is None:
            return f"not found: Batch label run {run_id}"
        if run.run_type != "batch_label_run":
            return f"not found: Not a batch label run: {run_id}"

        invalid_ops = (
            session.query(
                OperationRecord.example_key,
                OperationRecord.validation_errors_core,
                OperationRecord.inference_invocation_id,
            )
            .filter(
                OperationRecord.batch_label_run_id == run_id,
                OperationRecord.schema_valid_core == False,  # noqa: E712
            )
            .order_by(OperationRecord.example_key.asc())
            .all()
        )

        examples: list[dict[str, Any]] = [
            {
                "example_key": op[0],
                "validation_errors_core": op[1] or [],
                "inference_invocation_id": op[2],
            }
            for op in invalid_ops
        ]

    return {
        "batch_label_run_id": run_id,
        "schema_invalid_examples": examples,
        "total_count": len(examples),
    }
