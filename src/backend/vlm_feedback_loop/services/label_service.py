# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Label service — Save, Skip, Restore, and diff computation.

Handles the SME review actions that create Verified labels, omit images,
and restore omitted images.  Pool routing delegates to pool_service.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, cast

from sqlalchemy.orm import Session

from vlm_feedback_loop.db.base import generate_uuid4, utc_now
from vlm_feedback_loop.db.models.example import Example
from vlm_feedback_loop.db.models.guidance import Guidance
from vlm_feedback_loop.db.models.label import Label
from vlm_feedback_loop.db.models.operation import OperationRecord
from vlm_feedback_loop.db.models.project import Project
from vlm_feedback_loop.services import pool_service
from vlm_feedback_loop.services.exact_match_evaluator import normalize_field_value
from vlm_feedback_loop.services.project_service import get_project_engine
from vlm_feedback_loop.services.schema_core import rationale_note_enabled

logger = logging.getLogger("vlm_feedback_loop.label_service")


# ── Diff computation ─────────────────────────────────────────────────────────


def compute_label_diff(
    label_json: dict[str, Any],
    proposal_json: dict[str, Any],
    guidance_fields: list[dict[str, Any]],
) -> tuple[str, list[str], list[str]]:
    """Compute a deterministic diff between the submitted label and the proposal.

    Returns ``(verified_outcome, edited_core_fields, edited_aux_fields)``.
    No diff → ``("Accept", [], [])``.  Any diff → ``("Edit", [...], [...])``.

    Only fields present in the active Guidance participate. When the optional
    ``rationale_note`` field is disabled, an unexpected model-emitted key is
    ignored rather than turning an otherwise unchanged label into an Edit.
    """
    field_roles: dict[str, str] = {}
    for f in guidance_fields:
        field_roles[f["field_name"]] = f.get("role", "core")

    edited_core: list[str] = []
    edited_aux: list[str] = []

    # Compare all fields present in either dict that are known to guidance
    all_field_names = set(field_roles.keys())
    for fname in sorted(all_field_names):
        submitted_val = label_json.get(fname)
        proposal_val = proposal_json.get(fname)

        if submitted_val != proposal_val:
            role = field_roles.get(fname, "aux")
            if role == "core":
                edited_core.append(fname)
            else:
                edited_aux.append(fname)

    if edited_core or edited_aux:
        return "Edit", sorted(edited_core), sorted(edited_aux)
    return "Accept", [], []


def _core_schema_fingerprint(guidance: Guidance) -> str:
    """Fingerprint the Core fields that determine whether a label is valid.

    Two Guidance versions share a fingerprint iff a proposal valid under one
    is still valid — and still correctly named — under the other. It covers
    only Core fields (Aux values are never evaluated) and only the
    validity-bearing attributes (name, type, enum values, numeric/length
    bounds); presentation metadata and field order are excluded, so a
    no_change or presentation-only edit leaves the fingerprint unchanged
    while a rename, an enum-value change, or a bound change alters it.
    """
    fields = (guidance.schema or {}).get("fields", [])
    core: list[dict[str, Any]] = []
    for f in fields:
        if f.get("role", "core") != "core":
            continue
        core.append(
            {
                "name": f.get("field_name"),
                "type": f.get("type"),
                "enum": sorted(f.get("allowed_values") or []),
                "minimum": f.get("minimum"),
                "maximum": f.get("maximum"),
                "min_length": f.get("min_length"),
                "max_length": f.get("max_length"),
            }
        )
    core.sort(key=lambda d: d["name"] or "")
    return json.dumps(core, sort_keys=True, separators=(",", ":"))


# ── Save ─────────────────────────────────────────────────────────────────────


def save_label(
    project_id: str,
    *,
    example_key: str,
    inference_invocation_id: str,
    label_json: dict[str, Any],
    rationale_source: str | None,
    rationale_regeneration_invocation_id: str | None,
    workspace_root: str,
) -> dict[str, Any] | str:
    """Create or promote a Label with verification metadata.

    Returns a response dict on success or an error string for HTTP mapping.
    Error strings contain category hints: "not found" → 404, "conflict" → 409,
    others → 400.
    """
    engine = get_project_engine(project_id, workspace_root)
    if engine is None:
        return f"Project not found: {project_id}"

    with Session(engine) as session:
        # ── 1. Load and validate the OperationRecord ─────────────────────
        op_record = (
            session.query(OperationRecord)
            .filter_by(
                inference_invocation_id=inference_invocation_id,
            )
            .first()
        )

        if op_record is None:
            return f"Invocation not found: {inference_invocation_id}"

        if op_record.project_id != project_id:
            return f"Invocation not found in project: {inference_invocation_id}"

        if op_record.example_key != example_key:
            return (
                f"Invocation {inference_invocation_id} pertains to example "
                f"'{op_record.example_key}', not '{example_key}'"
            )

        allowed_purposes = {"interactive_proposal", "batch_label"}
        if op_record.purpose not in allowed_purposes:
            return (
                f"Invocation purpose '{op_record.purpose}' is not allowed for save; "
                f"expected one of {allowed_purposes}"
            )

        superseding = (
            session.query(OperationRecord)
            .filter_by(
                project_id=project_id,
                example_key=example_key,
                retry_of_inference_invocation_id=inference_invocation_id,
            )
            .first()
        )
        if superseding is not None:
            return (
                f"Stale proposal conflict: invocation {inference_invocation_id} "
                f"has been superseded by {superseding.inference_invocation_id}"
            )

        # ── 2. Load and validate the Example ────────────────────────────
        example = (
            session.query(Example)
            .filter_by(
                project_id=project_id,
                example_key=example_key,
            )
            .first()
        )

        if example is None:
            return f"Example not found: {example_key}"

        if example.state not in ("Unlabeled", "Auto-Labeled"):
            return (
                f"Example '{example_key}' is in state '{example.state}'; "
                f"only Unlabeled or Auto-Labeled examples can be saved"
            )

        # ── 3. Load proposal JSON from artifact file ─────────────────────
        # A null normalized_json_ref is legitimate: failed proposals
        # (schema-invalid / timeout / endpoint error) have no normalized
        # output, and the SME's manual label is then a true Edit against an
        # empty proposal. But when a ref exists and cannot be loaded, the
        # diff must NOT silently run against {} — that flips a true Accept
        # into Edit, records bogus edited_* field lists, and poisons pool
        # routing and the ICL candidate set (both key on
        # verified_outcome="Edit"). Surface it as a conflict so the SME can
        # Retry for a fresh proposal instead.
        # The conflict below only protects records whose proposal was VALID
        # (a genuine Accept could otherwise silently flip to Edit). For a
        # model-invalid proposal (schema_invalid / schema_valid_core false)
        # there is no Accept/Edit ambiguity — the SME's label is a true Edit
        # against an empty proposal — and refusing the save deadlocks the
        # SME whenever a Teacher deterministically emits degenerate output
        # for an image ("Retry" reproduces the same artifact at temp 0;
        # observed with cosmos3-super emitting a non-object normalized
        # artifact for one trashnet image, 2026-07-14).
        # Conservative gate: only records AFFIRMATIVELY marked model-invalid
        # bypass the conflict; success records (including legacy rows whose
        # schema_valid_core is NULL) keep the full Accept protection.
        proposal_was_valid = not (
            op_record.invocation_status == "schema_invalid"
            or op_record.schema_valid_core is False
        )
        proposal_json: dict[str, Any] = {}
        if op_record.normalized_json_ref:
            parsed: Any = None
            try:
                artifact_path = Path(op_record.normalized_json_ref)
                raw = artifact_path.read_text(encoding="utf-8")
                parsed = json.loads(raw)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning(
                    "Could not load normalized_json from %s: %s",
                    op_record.normalized_json_ref,
                    exc,
                )
                if proposal_was_valid:
                    return (
                        f"conflict: proposal artifact for invocation "
                        f"{inference_invocation_id} is missing or unreadable; "
                        f"the Accept/Edit outcome cannot be determined. Retry "
                        f"to get a fresh proposal."
                    )
            if parsed is not None and not isinstance(parsed, dict):
                logger.warning(
                    "normalized_json at %s is not a JSON object",
                    op_record.normalized_json_ref,
                )
                if proposal_was_valid:
                    return (
                        f"conflict: proposal artifact for invocation "
                        f"{inference_invocation_id} is missing or unreadable; "
                        f"the Accept/Edit outcome cannot be determined. Retry "
                        f"to get a fresh proposal."
                    )
                parsed = None
            if isinstance(parsed, dict):
                proposal_json = cast("dict[str, Any]", parsed)

        # ── 4. Load guidance fields for diff computation ─────────────────
        project = (
            session.query(Project)
            .filter_by(
                project_id=project_id,
            )
            .first()
        )
        active_guidance: Guidance | None = None
        guidance_fields: list[dict[str, Any]] = []
        if project and project.active_guidance_id:
            active_guidance = (
                session.query(Guidance)
                .filter_by(
                    project_id=project_id,
                    guidance_id=project.active_guidance_id,
                )
                .first()
            )
            if active_guidance and active_guidance.schema:
                guidance_fields = active_guidance.schema.get("fields", [])

        # ── 4a. Reject a proposal generated under a superseded Core schema ─
        # A proposal carries the Guidance version it was generated under
        # (OperationRecord.guidance_id). If that version's Core schema no
        # longer matches the active one — a rename, an enum-value change,
        # or a semantic Core change happened after the proposal was made —
        # the proposal is stale: saving it would stamp old-shape or
        # retired-name JSON as ground truth under the active Guidance (and
        # an in-place rename would leave a kept Auto-Labeled proposal that
        # verbatim-Accept could not match). Judge staleness by the
        # generating Guidance, not by field presence: a fresh manual label
        # over a failed proposal is generated under the active Guidance and
        # is never stale, even when it omits an unset core field.
        if (
            active_guidance is not None
            and op_record.guidance_id
            and op_record.guidance_id != active_guidance.guidance_id
        ):
            op_guidance = (
                session.query(Guidance)
                .filter_by(project_id=project_id, guidance_id=op_record.guidance_id)
                .first()
            )
            if op_guidance is None or _core_schema_fingerprint(
                op_guidance
            ) != _core_schema_fingerprint(active_guidance):
                return (
                    "conflict: this proposal predates the current schema "
                    "(its Core fields changed). Retry for a fresh proposal."
                )

        # ── 4b. Normalize the submitted label against SchemaCore ─────────
        # Verified label_json is the evaluation ground truth, and
        # match_fields assumes both sides are already canonical: an
        # untrimmed SME string edit or a numeric boolean from an API
        # client would otherwise become ground truth no correct answer
        # can ever match, and a whitespace-only change would classify as
        # Edit. Core values that fail canonical normalization are
        # rejected; invalid aux values (never evaluated) pass through.
        # The stored label is built from the active schema's fields only —
        # a key the schema does not know (a retired name from a pre-edit
        # tab, or client garbage) is dropped, matching the derived
        # schema's additionalProperties: false. A core field the
        # submission omits is filled with its at-rest value where one is
        # unambiguous (an unchecked boolean is false; an unselected
        # enum_set is empty) so a manual label over a failed proposal
        # produces complete ground truth; a required enum/number/string
        # with no value is a client validation error.
        if guidance_fields:
            normalized_json: dict[str, Any] = {}
            invalid_core: list[str] = []
            missing_core: list[str] = []
            for fdef in guidance_fields:
                fname = fdef.get("field_name")
                if not fname:
                    continue
                if fname not in label_json:
                    if fdef.get("role") == "core":
                        ftype = fdef.get("type")
                        if ftype == "boolean":
                            normalized_json[fname] = False
                        elif ftype == "enum_set":
                            normalized_json[fname] = []
                        else:
                            missing_core.append(fname)
                    continue
                res = normalize_field_value(label_json[fname], fdef)
                if res.valid:
                    normalized_json[fname] = res.normalized_value
                elif fdef.get("role") == "core":
                    invalid_core.append(f"'{fname}': {res.error}")
                else:
                    normalized_json[fname] = label_json[fname]
            if missing_core:
                return "validation: required Core field(s) not provided — " + ", ".join(
                    f"'{f}'" for f in missing_core
                )
            if invalid_core:
                return (
                    "validation: label values do not match the schema — "
                    + "; ".join(invalid_core)
                )
            label_json = normalized_json

        # ── 5. Compute diff ──────────────────────────────────────────────
        verified_outcome, edited_core, edited_aux = compute_label_diff(
            label_json,
            proposal_json,
            guidance_fields,
        )

        # ── 6. Validate rationale metadata when the Guidance enables it ───
        rationale_enabled = rationale_note_enabled(guidance_fields)
        if rationale_enabled:
            if rationale_source is None:
                return (
                    "validation: rationale_source is required while rationale_note "
                    "is enabled in the active Guidance"
                )
            if verified_outcome == "Edit" and rationale_source == "teacher_proposal":
                return (
                    "Edited labels must not use rationale_source='teacher_proposal'; "
                    "the rationale must be reviewed when fields are modified"
                )
        else:
            # Compatibility with older clients: ignore rationale metadata they
            # still send after the Guidance toggle is turned off. label_json was
            # already reduced to active-schema fields above.
            rationale_source = None
            rationale_regeneration_invocation_id = None

        # ── 7. Create or promote Label ───────────────────────────────────
        now = utc_now()

        existing_label = (
            session.query(Label)
            .filter_by(
                project_id=project_id,
                example_key=example_key,
                label_status="auto_labeled",
            )
            .first()
        )

        if existing_label is not None:
            # Auto-Labeled promotion: update existing record
            existing_label.label_status = "verified"
            # Re-point to the active Guidance: a historical or imported
            # Auto-Labeled row can carry a retired guidance_id, and promoting
            # it as ground truth under a stale version would orphan it from
            # ICL, exports, and the gate.
            if active_guidance is not None:
                existing_label.guidance_id = active_guidance.guidance_id
            existing_label.inference_invocation_id = inference_invocation_id
            existing_label.label_json = label_json
            existing_label.labeled_at = now
            existing_label.verified_outcome = verified_outcome
            existing_label.verified_at = now
            existing_label.edited_core_fields = edited_core
            existing_label.edited_aux_fields = edited_aux
            existing_label.rationale_source = rationale_source
            existing_label.rationale_regeneration_invocation_id = (
                rationale_regeneration_invocation_id
            )
            # batch_label_run_id is intentionally left unchanged — retained
            # for lineage
            label = existing_label
        else:
            # Create new Label
            label = Label(
                label_id=generate_uuid4(),
                project_id=project_id,
                example_key=example_key,
                label_status="verified",
                guidance_id=project.active_guidance_id if project else "",
                inference_invocation_id=inference_invocation_id,
                label_json=label_json,
                labeled_at=now,
                verified_outcome=verified_outcome,
                verified_at=now,
                edited_core_fields=edited_core,
                edited_aux_fields=edited_aux,
                rationale_source=rationale_source,
                rationale_regeneration_invocation_id=rationale_regeneration_invocation_id,
            )
            session.add(label)

        # ── 8. Transition Example state ──────────────────────────────────
        example.state = "Verified"

        # ── 9. Pool routing ──────────────────────────────────────────────
        # The earlier project lookup populates ``project`` and the example-
        # key check upstream rejects unknown projects with 404, so the row
        # is non-None here.
        assert project is not None
        pool_assignment = pool_service.route_pool(
            project_id,
            session,
            engine,
            project,
            label,
            verified_outcome,
        )

        # ── 10. Commit guard against a concurrent guidance edit ─────────
        # Everything above ran on autocommit reads, so a guidance edit can
        # commit inside this handler's span — its label sweep only sees
        # committed rows, and this save would then stamp a retired version
        # and retired field names as Verified ground truth. The flush takes
        # the write lock, so the re-read below is post-edit truth; on a
        # mismatch the save is abandoned (session closes without commit).
        session.flush()
        current_active_gid = (
            session.query(Project.active_guidance_id)
            .filter_by(project_id=project_id)
            .scalar()
        )
        if current_active_gid != project.active_guidance_id:
            return (
                "conflict: the guidance changed while this label was being "
                "saved. Retry for a fresh proposal."
            )

        session.commit()

        return {
            "example_key": example_key,
            "label_status": "verified",
            "verified_outcome": verified_outcome,
            "verified_at": now,
            "edited_core_fields": edited_core,
            "edited_aux_fields": edited_aux,
            "pool_assignment": pool_assignment,
        }


# ── Skip ─────────────────────────────────────────────────────────────────────


def skip_example(
    project_id: str,
    example_key: str,
    workspace_root: str,
) -> dict[str, Any] | str:
    """Omit an example from the workflow.

    Returns response dict on success, error string on failure.
    """
    engine = get_project_engine(project_id, workspace_root)
    if engine is None:
        return f"Project not found: {project_id}"

    with Session(engine) as session:
        example = (
            session.query(Example)
            .filter_by(
                project_id=project_id,
                example_key=example_key,
            )
            .first()
        )

        if example is None:
            return f"Example not found: {example_key}"

        if example.state not in ("Unlabeled", "Auto-Labeled"):
            return (
                f"Example '{example_key}' is in state '{example.state}'; "
                f"only Unlabeled or Auto-Labeled examples can be skipped"
            )

        # An Auto-Labeled row is the machine proposal represented by the
        # Example's current state, not durable ground truth. Skip rejects that
        # proposal; deleting it in this Session keeps the label lifecycle and
        # Omitted transition atomic while preserving invocation audit records.
        (
            session.query(Label)
            .filter_by(
                project_id=project_id,
                example_key=example_key,
                label_status="auto_labeled",
            )
            .delete(synchronize_session=False)
        )

        now = utc_now()
        example.state = "Omitted"
        example.omitted_source = "sme_skip"
        example.omitted_at = now

        session.commit()

        return {
            "example_key": example_key,
            "state": "Omitted",
            "omitted_at": now,
        }


# ── Restore Omitted ─────────────────────────────────────────────────────────


def restore_omitted(
    project_id: str,
    workspace_root: str,
) -> dict[str, Any] | str:
    """Bulk-restore all Omitted examples to Unlabeled.

    Returns response dict with ``restored_count``.
    """
    engine = get_project_engine(project_id, workspace_root)
    if engine is None:
        return f"Project not found: {project_id}"

    with Session(engine) as session:
        omitted = (
            session.query(Example)
            .filter_by(
                project_id=project_id,
                state="Omitted",
            )
            .all()
        )

        count = len(omitted)
        for ex in omitted:
            ex.state = "Unlabeled"
            ex.omitted_source = None
            ex.omitted_at = None

        session.commit()

        return {"restored_count": count}
