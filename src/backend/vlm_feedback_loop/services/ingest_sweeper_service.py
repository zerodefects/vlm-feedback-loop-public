# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Background pHash sweep for ingested examples.

The :ingest endpoint creates skeleton ``Example`` rows with ``phash=None``
and returns 202 in ~1s; this module sweeps ``Example WHERE phash IS NULL``
in multi-pass batches, computes the DCT-based 64-bit pHash via
``services.phash.compute_phash_from_path``, and writes the value back in
short transactions. SSE ``ingest_progress`` / ``ingest_completed`` events
keep the UI informed.

Structure mirrors ``clip_embedding_service.py``'s worker + recovery
sections exactly so the two workers share the same lifecycle,
restart-recovery, and dedup patterns. Only the per-row compute and the
queue marker
differ: this worker reads ``Example WHERE phash IS NULL`` instead of
``Example WHERE example_key NOT IN ClipEmbedding``.

Per-row error tolerance: ``compute_phash_from_path`` already returns
``None`` on PIL/IO failure. The sweeper logs a warning, adds
the key to ``attempted_keys`` so it does not re-appear in the next pass,
and continues. A bad file leaves the row at ``phash=None`` permanently; the review
selector's pHash-diverse scoring already skips ``phash=NULL`` rows, so
such a row is simply not used as a diversity signal there (it can still
be embedded and picked via CLIP-diversity, and is always labelable).
"""

from __future__ import annotations

import asyncio
import logging

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from vlm_feedback_loop.config import Settings
from vlm_feedback_loop.db.models.example import Example
from vlm_feedback_loop.services.background import (
    background_manager,
    run_in_low_priority_thread,
)
from vlm_feedback_loop.services.phash import compute_phash_from_path
from vlm_feedback_loop.services.project_db_locks import get_project_write_lock
from vlm_feedback_loop.services.project_service import (
    get_project_engine,
    projects_root,
)
from vlm_feedback_loop.services.sse import sse_manager

logger = logging.getLogger("vlm_feedback_loop.ingest_sweeper")

# Per-pass pHash batch size. Each batch computes this many pHashes
# (sequentially, off the event loop), opens one short write transaction,
# and emits one ``ingest_progress`` SSE event. Kept small so the first
# pHashes — and the first progress update — land after ~10 images instead
# of ~50, so early review-diversity signal appears quickly.
INGEST_SWEEP_BATCH_SIZE = 10

# pHash-specific inter-batch pause — a brief yield that releases the SQLite
# write lock (so a concurrent ingest POST can INSERT without hitting the
# busy_timeout) and lets the event loop service interactive requests
# between batches. Small on purpose: the decode is now cheap (draft-mode
# reduced-resolution JPEG decode in ``services.phash``), so the sweep no
# longer needs to be throttled to keep off the CPU — throttling only
# delayed reaching the point where the SME can move forward. Decoupled from
# the embedding worker's cadence (``clip_embedding_service.INTER_BATCH_SLEEP_S``).
INGEST_SWEEP_INTER_BATCH_SLEEP_S = 0.02


# ── Background ingest sweeper worker ────────────────────────────────────────


async def _ingest_worker(
    project_id: str,
    workspace_root: str,
    settings: Settings,
) -> None:
    """Background coroutine that backfills pHash for ingested examples.

    Multi-pass: after each sweep completes, re-queries the project DB for
    rows ingested while the previous sweep was running. Loops until a
    sweep finds nothing pending. Mirrors the
    ``_embedding_worker`` multi-pass shape so the two workers run
    concurrently without coupling — pHash and CLIP write disjoint columns
    on disjoint signals.
    """
    engine = get_project_engine(project_id, workspace_root)
    if engine is None:
        logger.warning("Ingest sweeper: project %s not found", project_id)
        return

    cumulative_processed = 0
    cumulative_total = 0
    pass_index = 0

    # Keys we have already attempted in this worker invocation. Rows that
    # failed pHash compute (unreadable file, decode error, etc.) end up
    # here so they do not re-surface in the rescan and cause an infinite
    # loop. On worker restart they get a fresh attempt automatically.
    attempted_keys: set[str] = set()

    # Shared per-project write lock — serializes against the :ingest
    # endpoint + CLIP embedding worker so we never time out at the
    # SQLite busy_timeout under load.
    write_lock = get_project_write_lock(project_id)

    while True:
        if background_manager.is_shutting_down():
            break

        # Re-query each pass: catches rows ingested while the previous
        # pass was running (same dispatch-race guard as the CLIP worker).
        with Session(engine) as session:
            rows = session.execute(
                select(Example.example_key, Example.storage_ref).where(
                    Example.project_id == project_id,
                    Example.phash.is_(None),
                )
            ).all()

        remaining: list[tuple[str, str]] = [
            (k, sr) for k, sr in rows if k not in attempted_keys
        ]
        if not remaining:
            if pass_index == 0:
                logger.info("Ingest sweeper: nothing pending for %s", project_id)
            break

        pass_index += 1
        pass_total = len(remaining)
        cumulative_total += pass_total
        if pass_index == 1:
            logger.info(
                "Ingest sweeper: %d rows to pHash for project %s",
                pass_total,
                project_id,
            )
        else:
            logger.info(
                "Ingest sweeper: pass %d picked up %d more rows for %s "
                "(ingested while previous pass ran)",
                pass_index,
                pass_total,
                project_id,
            )

        batches: list[list[tuple[str, str]]] = [
            remaining[i : i + INGEST_SWEEP_BATCH_SIZE]
            for i in range(0, pass_total, INGEST_SWEEP_BATCH_SIZE)
        ]

        processed_in_pass = 0
        for batch in batches:
            if background_manager.is_shutting_down():
                break

            # Mark these keys attempted before any I/O so a per-row
            # failure cannot resurface in the next pass.
            attempted_keys.update(k for k, _ in batch)

            # Compute pHash for each row off the event loop. The helper
            # already handles file-open + decode + null-on-failure,
            # so the worker never has to swallow PIL exceptions
            # directly.
            computed: list[tuple[str, str | None]] = []
            for key, storage_ref in batch:
                phash_value = await run_in_low_priority_thread(
                    compute_phash_from_path,
                    storage_ref,
                    settings,
                )
                computed.append((key, phash_value))

            # One short write transaction per batch. UPDATE only
            # the rows whose pHash succeeded; failures stay at phash=NULL
            # and the review selector ignores them. Held inside the
            # per-project write lock so neither the :ingest endpoint
            # nor the CLIP worker is in flight against the same DB.
            async with write_lock:
                with Session(engine) as session:
                    for key, phash_value in computed:
                        if phash_value is None:
                            continue
                        ex = session.execute(
                            select(Example).where(
                                Example.project_id == project_id,
                                Example.example_key == key,
                            )
                        ).scalar_one_or_none()
                        if ex is not None:
                            ex.phash = phash_value
                    session.commit()

            processed_in_pass += len(batch)
            await sse_manager.emit(
                project_id,
                "ingest_progress",
                {
                    "processed": processed_in_pass,
                    "total": pass_total,
                    "pass_index": pass_index,
                },
            )
            if INGEST_SWEEP_INTER_BATCH_SLEEP_S > 0:
                await asyncio.sleep(INGEST_SWEEP_INTER_BATCH_SLEEP_S)

        cumulative_processed += processed_in_pass
        # Loop top will rescan for newly-arrived rows.

    # Completion event (single emit covering all passes). Suppressed on
    # shutdown so a crash-mid-sweep does not lie to the UI; the next
    # backend startup's ``recover_ingest_tasks`` re-runs the sweep.
    if not background_manager.is_shutting_down():
        await sse_manager.emit(
            project_id,
            "ingest_completed",
            {"processed": cumulative_processed, "total": cumulative_total},
        )
        if pass_index == 0:
            logger.info(
                "Ingest sweeper: completed 0/0 for project %s (nothing pending)",
                project_id,
            )
        elif pass_index == 1:
            logger.info(
                "Ingest sweeper: completed %d/%d for project %s",
                cumulative_processed,
                cumulative_total,
                project_id,
            )
        else:
            logger.info(
                "Ingest sweeper: completed %d/%d for project %s across %d passes "
                "(caught rows ingested mid-run)",
                cumulative_processed,
                cumulative_total,
                project_id,
                pass_index,
            )


# ── Trigger and recovery ────────────────────────────────────────────────────


def trigger_ingest_processing(
    project_id: str,
    workspace_root: str,
    settings: Settings,
) -> None:
    """Trigger the background pHash sweeper for a project.

    Non-blocking — returns immediately. Deduplicates by task ID so
    repeated ingest POSTs within the same sweep do not spawn parallel
    workers (the existing worker's multi-pass rescan catches the
    newly-arrived rows on its next pass).
    """
    background_manager.try_register(
        f"ingest-sweep-{project_id}",
        _ingest_worker(project_id, workspace_root, settings),
        no_loop_warning=(
            f"Ingest sweeper NOT scheduled for {project_id}: caller has no "
            "running event loop. Async route handler required; sweeper will "
            "recover on next backend restart."
        ),
    )


async def recover_ingest_tasks(settings: Settings) -> None:
    """Recover incomplete pHash sweeps on startup (restart contract).

    Scans all active projects under ``WORKSPACE_ROOT/projects/`` and
    triggers the sweeper for any project with at least one row whose
    ``phash IS NULL``. Archived projects (marker file ``.archived``) are
    skipped. Per-project try/except so a single corrupt DB does not
    prevent backend startup (same guard as the CLIP recovery).
    """
    workspace_root = settings.WORKSPACE_ROOT
    projects_dir = projects_root(workspace_root)
    if not projects_dir.exists():
        return

    for entry in sorted(projects_dir.iterdir()):
        if not entry.is_dir():
            continue
        db_path = entry / "project.db"
        if not db_path.exists():
            continue
        # Archived projects do not need ingest recovery — the row set is
        # frozen while a project is paused.
        if (entry / ".archived").exists():
            continue

        project_id = entry.name
        try:
            engine = get_project_engine(project_id, workspace_root)
            if engine is None:
                continue

            with Session(engine) as session:
                null_count = (
                    session.execute(
                        select(func.count())
                        .select_from(Example)
                        .where(
                            Example.project_id == project_id,
                            Example.phash.is_(None),
                        )
                    ).scalar()
                    or 0
                )

            if null_count > 0:
                logger.info(
                    "Recovery: %d rows with null phash in project %s, triggering sweeper",
                    null_count,
                    project_id,
                )
                trigger_ingest_processing(project_id, workspace_root, settings)
        except Exception as exc:
            logger.warning(
                "Skipping ingest sweeper recovery for project %s (%s: %s)",
                project_id,
                type(exc).__name__,
                exc or "(no message)",
            )
            continue
