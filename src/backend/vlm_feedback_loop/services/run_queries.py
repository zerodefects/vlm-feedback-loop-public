# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared RunRecord lookup and list pagination for the run services.

evaluation_service and batch_label_service open their get/cancel/resume
entry points with the same guard: load the run, verify project ownership
and run_type, and signal failure as a "not found: ..." string (mapped to
404 by services.errors.map_service_error). Their list endpoints share the
same cursor-pagination stanza (:func:`list_runs_page`).
"""

from __future__ import annotations

from typing import Any, Literal, cast

from sqlalchemy import select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.orm import Session

from vlm_feedback_loop.db.models.run import RunRecord
from vlm_feedback_loop.services.pagination import (
    after_position,
    decode_cursor,
    encode_cursor,
)


def find_run(
    session: Session,
    project_id: str,
    run_id: str,
    *,
    run_type: str,
) -> RunRecord | str:
    """Load ``run_id`` in ``project_id``, requiring ``run_type``.

    Returns the session-attached RunRecord, or a "not found: ..." error
    string preserving each service's established wording.
    """
    noun = run_type.replace("_", " ")
    run = (
        session.query(RunRecord).filter_by(run_id=run_id, project_id=project_id).first()
    )
    if run is None:
        return f"not found: {noun.capitalize()} {run_id}"
    if run.run_type != run_type:
        article = "an" if noun[0] in "aeiou" else "a"
        return f"not found: Not {article} {noun}: {run_id}"
    return run


def list_runs_page(
    session: Session,
    *,
    project_id: str,
    run_type: str,
    status_filter: str | None,
    cursor: str | None,
    limit: int,
    basis: Literal["gate", "benchmark"] | None = None,
) -> tuple[list[RunRecord], str | None]:
    """Newest-first cursor page of ``run_type`` runs in ``project_id``.

    Filter by run_type (and optional status), order ``(created_at, run_id)``
    descending, over-fetch by one row to detect a next page, and encode the
    next cursor from the last returned row. Returns ``(rows, next_cursor)``;
    callers convert rows to their response dicts.

    ``basis`` scopes evaluation runs by provenance: ``"gate"`` keeps only
    gate-basis Teacher runs (``student_model_config_id IS NULL``),
    ``"benchmark"`` only Student benchmark runs (§9.5.2). None applies no
    provenance filter (all runs; batch callers).
    """
    stmt = (
        select(RunRecord)
        .where(
            RunRecord.project_id == project_id,
            RunRecord.run_type == run_type,
        )
        .order_by(RunRecord.created_at.desc(), RunRecord.run_id.desc())
    )
    if status_filter:
        stmt = stmt.where(RunRecord.status == status_filter)
    if basis == "gate":
        stmt = stmt.where(RunRecord.student_model_config_id.is_(None))
    elif basis == "benchmark":
        stmt = stmt.where(RunRecord.student_model_config_id.is_not(None))
    if cursor:
        cur_ts, cur_id = decode_cursor(cursor)
        stmt = stmt.where(
            after_position(RunRecord.created_at, RunRecord.run_id, cur_ts, cur_id)
        )
    stmt = stmt.limit(limit + 1)

    rows = list(session.execute(stmt).scalars().all())
    next_cursor: str | None = None
    if len(rows) > limit:
        rows = rows[:limit]
        next_cursor = encode_cursor(rows[-1].created_at, rows[-1].run_id)
    return rows, next_cursor


def update_run_if_not_terminal(
    session: Session,
    run_id: str,
    values: dict[str, Any],
    *,
    terminal_statuses: frozenset[str] | None = None,
    only_status: str | None = None,
) -> bool:
    """Apply ``values`` to a run only if it is not terminal, atomically.

    A conditional UPDATE is the transaction's first write, so it takes the
    SQLite write lock and evaluates the not-terminal predicate in the same
    step. A SELECT guard followed by a separate write does not: the read
    can see a live status and then commit behind a concurrent transaction
    that terminalized the run in between (schema evolution failing a run
    whose labels it is wiping), resurrecting the row. Returns True when the
    row was live and updated. Exactly one of ``terminal_statuses`` (match
    any status outside the set) and ``only_status`` (match a single
    expected status, used for the canceling → canceled transition) must be
    given.
    """
    if terminal_statuses is not None and only_status is None:
        status_clause = RunRecord.status.not_in(terminal_statuses)
    elif only_status is not None and terminal_statuses is None:
        status_clause = RunRecord.status == only_status
    else:
        raise ValueError("pass exactly one of terminal_statuses / only_status")
    result = cast(
        "CursorResult[Any]",
        session.execute(
            update(RunRecord)
            .where(RunRecord.run_id == run_id, status_clause)
            .values(**values)
        ),
    )
    return result.rowcount > 0
