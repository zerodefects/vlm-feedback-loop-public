# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Schema evolution service: edit preview, in-place propagation, atomic evolution.

Covers the SchemaCore edit policy, the 8-step atomic schema evolution
procedure, Label deletion, and Project resets.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Literal, cast

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from vlm_feedback_loop.db.base import generate_uuid4, utc_now
from vlm_feedback_loop.db.models.example import Example
from vlm_feedback_loop.db.models.guidance import Guidance
from vlm_feedback_loop.db.models.label import Label
from vlm_feedback_loop.db.models.operation import OperationRecord
from vlm_feedback_loop.db.models.project import Project
from vlm_feedback_loop.db.models.run import ACTIVE_RUN_STATUSES, RunRecord
from vlm_feedback_loop.services import (
    batch_label_service,
    edit_classification,
    evaluation_service,
    schema_core,
)
from vlm_feedback_loop.services.project_service import get_project_engine

logger = logging.getLogger("vlm_feedback_loop.services.schema_evolution")


# ── Result types ────────────────────────────────────────────────────────────


@dataclass
class EditPreview:
    """Returned by the dry-run call for the confirmation dialog."""

    classification: edit_classification.EditClassificationResult
    verified_count: int
    auto_labeled_count: int


@dataclass
class EditResult:
    """Returned after executing the edit."""

    guidance: Guidance
    edit_type: Literal["in_place", "semantic", "no_change"]
    classification: edit_classification.EditClassificationResult
    verified_reverted_count: int
    auto_labeled_reverted_count: int


# ── Public API ──────────────────────────────────────────────────────────────


def _load_validate_classify(
    session: Session,
    project_id: str,
    new_schema_fields: list[dict[str, Any]],
) -> (
    tuple[
        Project,
        Guidance,
        schema_core.ValidationResult,
        edit_classification.EditClassificationResult,
    ]
    | str
    | None
):
    """Shared prelude for ``preview_edit`` and ``execute_edit``.

    Loads the project and its active Guidance, validates the edited schema
    (preserving field_ids), and classifies the changes. Preview is the
    confirmation dialog for execute, so the two must classify identically —
    sharing this prelude makes that true by construction.

    Returns:
        (project, old_guidance, validation, classification) on success.
        str: guidance-state or validation error message.
        None: project not found.
    """
    project = session.execute(
        select(Project).where(Project.project_id == project_id)
    ).scalar_one_or_none()
    if project is None:
        return None

    if not project.active_guidance_id:
        return "No active guidance to edit"

    old_guidance = session.execute(
        select(Guidance).where(Guidance.guidance_id == project.active_guidance_id)
    ).scalar_one_or_none()
    if old_guidance is None:
        return "Active guidance not found"

    old_fields = old_guidance.schema.get("fields", [])

    # Validate new schema (preserving field_ids for classification)
    validation = schema_core.validate_and_derive_edit(old_fields, new_schema_fields)
    if not validation.save_allowed:
        return "; ".join(
            f"[{i.code}] {i.message}"
            for i in validation.issues
            if i.severity == "error"
        )

    # Classify edits — validation.save_allowed=True guarantees processed_fields is non-None.
    assert validation.processed_fields is not None
    classification = edit_classification.classify_edits(
        old_fields, validation.processed_fields
    )

    return project, old_guidance, validation, classification


def preview_edit(
    project_id: str,
    description: str,  # noqa: ARG001 — part of the edit request contract; description is prompt content, not a validation input
    new_schema_fields: list[dict[str, Any]],
    rules: str,  # noqa: ARG001 — part of the edit request contract; rules is prompt content, not a validation input
    workspace_root: str,
) -> EditPreview | str | None:
    """Dry-run: classify edits, return affected counts.

    Returns:
        EditPreview: classification + counts for confirmation dialog.
        str: validation error message.
        None: project not found or no active guidance.
    """
    engine = get_project_engine(project_id, workspace_root)
    if engine is None:
        return None

    with Session(engine) as session:
        prelude = _load_validate_classify(session, project_id, new_schema_fields)
        if prelude is None or isinstance(prelude, str):
            return prelude
        _, _, _, classification = prelude

        # Count affected examples
        verified_count = (
            session.execute(
                select(func.count()).where(
                    Example.project_id == project_id,
                    Example.state == "Verified",
                )
            ).scalar()
            or 0
        )
        auto_labeled_count = (
            session.execute(
                select(func.count()).where(
                    Example.project_id == project_id,
                    Example.state == "Auto-Labeled",
                )
            ).scalar()
            or 0
        )

        return EditPreview(
            classification=classification,
            verified_count=verified_count,
            auto_labeled_count=auto_labeled_count,
        )


def execute_edit(
    project_id: str,
    description: str,
    new_schema_fields: list[dict[str, Any]],
    rules: str,
    workspace_root: str,
    schema_change_context_example_key: str | None = None,
) -> EditResult | str | None:
    """Execute the guidance edit: in-place propagation or atomic evolution.

    Returns:
        EditResult: on success.
        str: validation or classification error message.
        None: project not found or no active guidance.
    """
    engine = get_project_engine(project_id, workspace_root)
    if engine is None:
        return None

    with Session(engine) as session:
        prelude = _load_validate_classify(session, project_id, new_schema_fields)
        if prelude is None or isinstance(prelude, str):
            return prelude
        project, old_guidance, validation, classification = prelude
        old_guidance_id = old_guidance.guidance_id

        # Determine edit type
        if not classification.changes:
            edit_type: Literal["in_place", "semantic", "no_change"] = "no_change"
        elif classification.has_semantic_changes:
            edit_type = "semantic"
        else:
            edit_type = "in_place"

        # Create new Guidance version
        max_ver = session.execute(
            select(func.max(Guidance.version_number)).where(
                Guidance.project_id == project_id
            )
        ).scalar()
        version_number = (max_ver or 0) + 1

        envelope: dict[str, Any] = {
            "fields": validation.processed_fields,
            "derived_json_schema": validation.derived_json_schema,
            "generation_order": validation.generation_order,
            "schema_hash": validation.schema_hash,
        }

        new_guidance = Guidance(
            guidance_id=generate_uuid4(),
            project_id=project_id,
            version_number=version_number,
            description=description,
            schema=envelope,
            rules=rules,
            semantic_core_change_from_guidance_id=(
                old_guidance_id if edit_type == "semantic" else None
            ),
            schema_change_summary=(
                classification.change_summary if edit_type == "semantic" else None
            ),
        )
        session.add(new_guidance)
        session.flush()

        verified_reverted = 0
        auto_labeled_reverted = 0

        if edit_type in ("in_place", "no_change"):
            # Both keep the existing labels valid — the SchemaCore is
            # unchanged (no_change edits only Description/Rules; in_place
            # edits rename fields/enum values, add/remove Aux fields, or touch
            # presentation metadata). Aux additions/removals deliberately do
            # not backfill or erase historical Label JSON. But every edit
            # mints a new immutable Guidance version and makes it active, so
            # labels MUST be re-pointed to it or the corpus orphans from the
            # active guidance (ICL, evaluation, and Compare filter by it).
            if edit_type == "in_place":
                if classification.field_renames:
                    _propagate_field_renames(
                        session,
                        project_id,
                        old_guidance_id,
                        classification.field_renames,
                    )
                if classification.enum_value_renames:
                    _propagate_enum_value_renames(
                        session,
                        project_id,
                        old_guidance_id,
                        classification.enum_value_renames,
                    )
            _update_label_guidance_ids(
                session, project_id, old_guidance_id, new_guidance.guidance_id
            )

        elif edit_type == "semantic":
            # Atomic schema evolution (steps a-h below)
            verified_reverted, auto_labeled_reverted = _execute_semantic_evolution(
                session,
                project,
                schema_change_context_example_key,
            )

        # Every edit re-points (or, for a semantic edit, wipes) the labels
        # that exist NOW — but an in-flight batch/eval run keeps stamping
        # its Phase-A guidance snapshot, so every label it writes AFTER
        # this commit would orphan from the active guidance (and a rename
        # would additionally carry retired names). Stop active runs so
        # their output cannot straddle the edit. Non-semantic edits keep
        # the runs' already-written labels (just re-pointed); the semantic
        # wipe discards them — hence the distinct status reasons.
        canceled_eval_run_ids, canceled_batch_run_ids = _cancel_active_runs(
            session,
            project_id,
            utc_now(),
            status_reason=(
                "schema_evolution_canceled"
                if edit_type == "semantic"
                else "guidance_edited_during_run"
            ),
        )

        # Set new guidance as active
        project.active_guidance_id = new_guidance.guidance_id

        session.commit()

        # Now that the failed status is durable, stop the live executor
        # tasks so they do not keep invoking the Teacher for a wiped
        # Guidance. (The evaluation finalizer independently refuses to
        # overwrite terminal rows, so a missed signal cannot resurrect
        # the run.)
        for canceled_run_id in canceled_eval_run_ids:
            evaluation_service.signal_run_cancellation(canceled_run_id)
        for canceled_run_id in canceled_batch_run_ids:
            batch_label_service.signal_run_cancellation(canceled_run_id)

        session.refresh(new_guidance)
        session.expunge(new_guidance)

        logger.info(
            "Guidance edit %s for project %s: %s (v%d, verified_reverted=%d, auto_labeled_reverted=%d)",
            new_guidance.guidance_id,
            project_id,
            edit_type,
            version_number,
            verified_reverted,
            auto_labeled_reverted,
        )

        return EditResult(
            guidance=new_guidance,
            edit_type=edit_type,
            classification=classification,
            verified_reverted_count=verified_reverted,
            auto_labeled_reverted_count=auto_labeled_reverted,
        )


# ── In-place propagation ───────────────────────────────────────────────────


def _propagate_field_renames(
    session: Session,
    project_id: str,
    guidance_id: str,
    field_renames: dict[str, str],
) -> int:
    """Update JSON keys in label_json for all Label records under this guidance.

    Assigns a new dict (not in-place mutation) so SQLAlchemy detects the change.
    """
    labels = (
        session.execute(
            select(Label).where(
                Label.project_id == project_id,
                Label.guidance_id == guidance_id,
            )
        )
        .scalars()
        .all()
    )

    count = 0
    for label in labels:
        label_json = dict(label.label_json)
        # Resolve every rename against the ORIGINAL dict, then merge with
        # renamed values winning. Applying renames one at a time corrupts
        # swapped (A<->B) and chained (A->B, B->C) renames — an earlier
        # rename overwrites a key a later one still needs — and a key the
        # schema retired in the same edit must not shadow a field renamed
        # onto its name.
        renamed = {
            new_name: label_json[old_name]
            for old_name, new_name in field_renames.items()
            if old_name in label_json
        }
        if renamed:
            rest = {k: v for k, v in label_json.items() if k not in field_renames}
            label.label_json = {**rest, **renamed}
            count += 1
    return count


def _propagate_enum_value_renames(
    session: Session,
    project_id: str,
    guidance_id: str,
    enum_value_renames: dict[str, dict[str, str]],
) -> int:
    """Update enum values in label_json for all Label records.

    Handles both scalar enum fields and list enum_set fields.
    """
    labels = (
        session.execute(
            select(Label).where(
                Label.project_id == project_id,
                Label.guidance_id == guidance_id,
            )
        )
        .scalars()
        .all()
    )

    count = 0
    for label in labels:
        label_json = dict(label.label_json)
        changed = False
        for field_name, renames in enum_value_renames.items():
            if field_name not in label_json:
                continue
            val = label_json[field_name]
            if isinstance(val, str) and val in renames:
                # Scalar enum
                label_json[field_name] = renames[val]
                changed = True
            elif isinstance(val, list):
                # Enum set — replace matching values
                val_list = cast("list[Any]", val)
                new_list = [renames.get(v, v) for v in val_list]
                if new_list != val_list:
                    label_json[field_name] = new_list
                    changed = True
        if changed:
            label.label_json = label_json
            count += 1
    return count


def _update_label_guidance_ids(
    session: Session,
    project_id: str,
    old_guidance_id: str,
    new_guidance_id: str,
) -> None:
    """Point all labels from the old guidance to the new one after in-place edit."""
    labels = (
        session.execute(
            select(Label).where(
                Label.project_id == project_id,
                Label.guidance_id == old_guidance_id,
            )
        )
        .scalars()
        .all()
    )
    for label in labels:
        label.guidance_id = new_guidance_id


# ── Atomic schema evolution ────────────────────────────────────────────────


def _execute_semantic_evolution(
    session: Session,
    project: Project,
    schema_change_context_example_key: str | None,
) -> tuple[int, int]:
    """Execute the 8-step atomic schema evolution within an active session.

    All steps happen in the same transaction (caller commits).

    Returns (verified_reverted_count, auto_labeled_reverted_count).
    """
    project_id = project.project_id

    # ── Step (a): New Guidance already created by caller with
    # semantic_core_change_from_guidance_id and schema_change_summary set.
    # project.active_guidance_id will be set by caller after this returns.

    # ── Step (b): Verified → snapshot + delete Label + transition to Unlabeled
    verified_count = 0
    verified_examples = (
        session.execute(
            select(Example).where(
                Example.project_id == project_id,
                Example.state == "Verified",
            )
        )
        .scalars()
        .all()
    )

    for example in verified_examples:
        label = session.execute(
            select(Label).where(
                Label.project_id == project_id,
                Label.example_key == example.example_key,
                Label.label_status == "verified",
            )
        ).scalar_one_or_none()

        if label is not None:
            snapshot = _snapshot_verified_label(session, label)
            example.prior_verified_label_ref = json.dumps(
                snapshot, separators=(",", ":")
            )
            example.prior_verified_outcome = label.verified_outcome
            session.delete(label)

        example.state = "Unlabeled"
        verified_count += 1

    # ── Step (c): Auto-Labeled → delete Label + transition to Unlabeled
    auto_labeled_count = 0
    auto_labeled_examples = (
        session.execute(
            select(Example).where(
                Example.project_id == project_id,
                Example.state == "Auto-Labeled",
            )
        )
        .scalars()
        .all()
    )

    for example in auto_labeled_examples:
        label = session.execute(
            select(Label).where(
                Label.project_id == project_id,
                Label.example_key == example.example_key,
                Label.label_status == "auto_labeled",
            )
        ).scalar_one_or_none()

        if label is not None:
            session.delete(label)

        example.state = "Unlabeled"
        auto_labeled_count += 1

    # ── Step (d): Pool clearing — consequence of Label deletion.
    # pool_assignment lives on Label records which are now deleted.
    # No separate action needed.

    # ── Step (e): Reset evaluation baselines. (In-progress eval/batch
    # runs are canceled by execute_edit — every guidance edit cancels
    # active runs, not just semantic ones.)
    project.icl_recommendation_dismissed_at_count = 0

    # ── Step (f): Reset selector state
    project.review_selector_scheduler_state = None

    # ── Step (g): Record context example key
    project.schema_change_context_example_key = schema_change_context_example_key

    # ── Step (h): Reset schema refinement reminders
    project.schema_refinement_reminders_dismissed = 0

    logger.info(
        "Schema evolution for project %s: %d verified reverted, %d auto-labeled reverted",
        project_id,
        verified_count,
        auto_labeled_count,
    )

    return verified_count, auto_labeled_count


def _snapshot_verified_label(
    session: Session,
    label: Label,
) -> dict[str, Any]:
    """Build the self-contained prior_verified_label_ref JSON snapshot.

    Includes: final label_json, VLM proposal (from OperationRecord),
    edited_core_fields, edited_aux_fields, and verified_outcome. The optional
    rationale_note is copied only when it exists on the label.
    """
    # Look up the VLM proposal from the OperationRecord
    vlm_proposal = None
    if label.inference_invocation_id:
        op = session.execute(
            select(OperationRecord).where(
                OperationRecord.inference_invocation_id == label.inference_invocation_id
            )
        ).scalar_one_or_none()
        if op is not None:
            vlm_proposal = op.normalized_json_ref

    snapshot = {
        "label_json": label.label_json,
        "vlm_proposal_json": vlm_proposal,
        "edited_core_fields": label.edited_core_fields or [],
        "edited_aux_fields": label.edited_aux_fields or [],
        "verified_outcome": label.verified_outcome,
        "guidance_id": label.guidance_id,
    }
    if "rationale_note" in label.label_json:
        snapshot["rationale_note"] = label.label_json["rationale_note"]
    return snapshot


def _cancel_active_runs(
    session: Session,
    project_id: str,
    now: str,
    status_reason: str,
) -> tuple[list[str], list[str]]:
    """Cancel in-progress evaluation AND batch-label runs on a guidance edit.

    Both run families score/write labels under the guidance snapshot
    being replaced: an evaluation would publish metrics against
    re-pointed (or, after a semantic wipe, deleted) ground truth, and a
    batch run would keep writing labels that orphan from the new active
    version. Transitions active runs (paused batch runs included —
    they would otherwise resume under the new schema) to failed and
    returns their ids per family. The DB flip alone does not stop a
    live executor task — the caller MUST signal each returned id via
    the owning service's ``signal_run_cancellation`` AFTER its
    transaction commits (signaling before commit could stop a task for
    an edit that subsequently rolls back).
    """
    runs = (
        session.execute(
            select(RunRecord).where(
                RunRecord.project_id == project_id,
                RunRecord.run_type.in_(["evaluation_run", "batch_label_run"]),
                RunRecord.status.in_(ACTIVE_RUN_STATUSES | {"paused"}),
            )
        )
        .scalars()
        .all()
    )

    for run in runs:
        run.status = "failed"
        run.status_reason = status_reason
        run.completed_at = now

    return (
        [r.run_id for r in runs if r.run_type == "evaluation_run"],
        [r.run_id for r in runs if r.run_type == "batch_label_run"],
    )
