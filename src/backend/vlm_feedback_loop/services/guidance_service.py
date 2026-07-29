# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guidance CRUD service.

Guidance records are immutable — each create call produces a new version.
``version_number`` is 1-based, monotonically increasing within a project,
and assigned by the backend.
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from vlm_feedback_loop.config import Settings
from vlm_feedback_loop.db.base import generate_uuid4
from vlm_feedback_loop.db.models.guidance import Guidance
from vlm_feedback_loop.db.models.label import Label
from vlm_feedback_loop.db.models.project import Project
from vlm_feedback_loop.services import schema_core
from vlm_feedback_loop.services.icl_service import count_icl_eligible_edits
from vlm_feedback_loop.services.project_service import get_project_engine

logger = logging.getLogger("vlm_feedback_loop.services.guidance")


# ── Public API ──────────────────────────────────────────────────────────────


def create_guidance(
    project_id: str,
    description: str,
    schema_fields: list[dict[str, Any]],
    rules: str,
    workspace_root: str,
) -> Guidance | str | None:
    """Create a new immutable Guidance version.

    Returns:
        Guidance: on success (expunged ORM object).
        str: on validation error (formatted issue messages).
        None: if the project does not exist.
    """
    engine = get_project_engine(project_id, workspace_root)
    if engine is None:
        return None

    # Validate and derive (same function as validate_draft)
    result = schema_core.validate_and_derive(schema_fields)

    if not result.save_allowed:
        messages = "; ".join(
            f"[{i.code}] {i.message}" for i in result.issues if i.severity == "error"
        )
        return messages

    # Build the schema envelope stored in the JSON column (loop-invariant).
    envelope: dict[str, Any] = {
        "fields": result.processed_fields,
        "derived_json_schema": result.derived_json_schema,
        "generation_order": result.generation_order,
        "schema_hash": result.schema_hash,
    }

    # Assign next version_number (1-based, monotonically increasing).
    # The read-max + insert is a race: two concurrent creates (sync routes run
    # on a thread pool) could read the same max and insert duplicate versions.
    # The UNIQUE(project_id, version_number) constraint makes
    # that a loser-fails IntegrityError; retry with a fresh max so the loser
    # simply takes the next number.
    max_attempts = 5
    for attempt in range(max_attempts):
        with Session(engine) as session:
            max_ver = session.execute(
                select(func.max(Guidance.version_number)).where(
                    Guidance.project_id == project_id
                )
            ).scalar()
            version_number = (max_ver or 0) + 1

            guidance = Guidance(
                guidance_id=generate_uuid4(),
                project_id=project_id,
                version_number=version_number,
                description=description,
                schema=envelope,
                rules=rules,
            )
            session.add(guidance)
            try:
                session.commit()
            except IntegrityError:
                session.rollback()
                if attempt == max_attempts - 1:
                    raise
                logger.info(
                    "Guidance version %d for project %s collided; retrying",
                    version_number,
                    project_id,
                )
                continue
            session.refresh(guidance)
            session.expunge(guidance)

            logger.info(
                "Created guidance %s v%d for project %s",
                guidance.guidance_id,
                version_number,
                project_id,
            )
            return guidance

    # Unreachable: the loop returns on success and re-raises on the last
    # attempt's IntegrityError. Present so the function has no implicit None
    # fall-through path.
    raise RuntimeError("create_guidance exhausted version-assignment retries")


def get_guidance(
    project_id: str,
    guidance_id: str,
    workspace_root: str,
) -> Guidance | None:
    """Retrieve a single Guidance version by ID. Returns None if not found."""
    engine = get_project_engine(project_id, workspace_root)
    if engine is None:
        return None

    with Session(engine) as session:
        guidance = session.execute(
            select(Guidance).where(
                Guidance.project_id == project_id,
                Guidance.guidance_id == guidance_id,
            )
        ).scalar_one_or_none()
        if guidance is not None:
            session.expunge(guidance)
        return guidance


def list_guidances(
    project_id: str,
    workspace_root: str,
    cursor: str | None = None,
    limit: int = 50,
) -> tuple[list[Guidance], str | None]:
    """List Guidance versions newest-first with cursor pagination.

    Order: ``version_number DESC`` (monotonically increasing → newest = highest).
    Cursor: ``guidance_id`` of the last item on the previous page.

    Returns (items, next_cursor).
    """
    engine = get_project_engine(project_id, workspace_root)
    if engine is None:
        return [], None

    with Session(engine) as session:
        # If cursor provided, look up its version_number to paginate
        cursor_version: int | None = None
        if cursor:
            cursor_version = session.execute(
                select(Guidance.version_number).where(
                    Guidance.project_id == project_id,
                    Guidance.guidance_id == cursor,
                )
            ).scalar_one_or_none()

        stmt = (
            select(Guidance)
            .where(Guidance.project_id == project_id)
            .order_by(Guidance.version_number.desc())
        )

        if cursor_version is not None:
            stmt = stmt.where(Guidance.version_number < cursor_version)

        # Fetch limit+1 to detect next page
        all_items = list(session.execute(stmt.limit(limit + 1)).scalars().all())

        page = all_items[:limit]
        next_cursor = page[-1].guidance_id if len(all_items) > limit else None

        for g in page:
            session.expunge(g)

        return page, next_cursor


def validate_draft(
    project_id: str,
    description: str,  # noqa: ARG001 — part of the draft request contract; description is prompt content, not a validation input
    schema_fields: list[dict[str, Any]],
    rules: str,  # noqa: ARG001 — part of the draft request contract; rules is prompt content, not a validation input
    workspace_root: str,
) -> schema_core.ValidationResult | None:
    """Validate a draft Guidance without persisting.

    Uses the same ``validate_and_derive`` function as ``create_guidance``
    (one canonical derivation, used by both preview and save).

    Returns None if the project does not exist.
    """
    engine = get_project_engine(project_id, workspace_root)
    if engine is None:
        return None

    return schema_core.validate_and_derive(schema_fields)


def get_icl_eligible_count(
    project_id: str,
    workspace_root: str,
) -> int | None:
    """Count ICL-eligible Edits: non-pool Verified Edits under the active Guidance.

    Returns the count (0 when no Guidance is active), or None if the
    project does not exist.
    """
    engine = get_project_engine(project_id, workspace_root)
    if engine is None:
        return None

    with Session(engine) as session:
        project = session.execute(
            select(Project).where(Project.project_id == project_id)
        ).scalar_one_or_none()
        if project is None:
            return None

        active_guid = project.active_guidance_id
        if not active_guid:
            return 0

        return count_icl_eligible_edits(session, project_id, active_guid)


# ── Schema Refinement Reminders ──────────────────────────────────────────────


def get_reminder_status(
    project_id: str,
    workspace_root: str,
    settings: Settings,
) -> dict[str, Any] | None:
    """Check whether a schema refinement reminder should fire.

    Returns a status dict or None if project not found.
    """
    engine = get_project_engine(project_id, workspace_root)
    if engine is None:
        return None

    threshold_1 = settings.SCHEMA_REFINEMENT_REMINDER_THRESHOLD_1
    threshold_2 = settings.SCHEMA_REFINEMENT_REMINDER_THRESHOLD_2

    with Session(engine) as session:
        project = session.query(Project).filter_by(project_id=project_id).first()
        if project is None:
            return None

        dismissed = project.schema_refinement_reminders_dismissed

        # Count total Verified labels
        verified_count = (
            session.execute(
                select(func.count())
                .select_from(Label)
                .where(
                    Label.project_id == project_id,
                    Label.label_status == "verified",
                )
            ).scalar()
            or 0
        )

        # Check if Guidance has been edited post-save (version_number > 1)
        guidance_edited = (
            session.execute(
                select(func.count())
                .select_from(Guidance)
                .where(
                    Guidance.project_id == project_id,
                    Guidance.version_number > 1,
                )
            ).scalar()
            or 0
        )

        # Determine active reminder
        active_reminder: int | None = None

        if guidance_edited > 0:
            # Suppressed — SME already edited Guidance
            active_reminder = None
        elif threshold_1 == 0 and threshold_2 == 0:
            # Both disabled
            active_reminder = None
        else:
            # Check which reminders are eligible
            reminder_1_eligible = (
                threshold_1 > 0 and verified_count >= threshold_1 and dismissed < 1
            )
            reminder_2_eligible = (
                threshold_2 > 0 and verified_count >= threshold_2 and dismissed < 2
            )

            if reminder_2_eligible:
                # If both crossed before either fired, only higher fires
                active_reminder = 2
            elif reminder_1_eligible:
                active_reminder = 1

        return {
            "active_reminder": active_reminder,
            "verified_count": verified_count,
            "threshold_1": threshold_1,
            "threshold_2": threshold_2,
            "dismissed_count": dismissed,
        }


def dismiss_reminder(
    project_id: str,
    workspace_root: str,
) -> dict[str, Any] | str:
    """Dismiss the current schema refinement reminder.

    Increments the dismissed counter on the Project record.
    """
    engine = get_project_engine(project_id, workspace_root)
    if engine is None:
        return f"Project not found: {project_id}"

    with Session(engine) as session:
        project = session.query(Project).filter_by(project_id=project_id).first()
        if project is None:
            return f"Project not found: {project_id}"

        project.schema_refinement_reminders_dismissed += 1
        new_count = project.schema_refinement_reminders_dismissed
        session.commit()

        return {"dismissed_count": new_count}
