# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the label service — Save, Skip, Restore.

Covers: the label save endpoint, skip, restore omitted,
diff computation, stale-proposal validation, Auto-Labeled promotion,
pool routing, and rationale source validation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sqlalchemy import event
from sqlalchemy.orm import Session

from conftest import (
    FIXTURE_FIELDS,
    FIXTURE_SCHEMA,
    add_endpoint_row,
    add_example_row,
    add_guidance_row,
    add_model_config_row,
    add_project_row,
    open_project_workspace,
)
from vlm_feedback_loop.db.base import generate_uuid4, utc_now
from vlm_feedback_loop.db.models.example import Example
from vlm_feedback_loop.db.models.label import Label
from vlm_feedback_loop.db.models.operation import OperationRecord
from vlm_feedback_loop.db.models.project import Project
from vlm_feedback_loop.services.label_service import (
    compute_label_diff,
    restore_omitted,
    save_label,
    skip_example,
)

# ── Helpers ──────────────────────────────────────────────────────────────────


def _setup_project_db(tmp_path: Path, project_id: str = "test-proj"):
    engine, project_dir, _ = open_project_workspace(tmp_path, project_id)
    return engine, str(project_dir)


def _add_operation_record(
    session,
    project_id,
    invocation_id,
    example_key,
    purpose="interactive_proposal",
    normalized_json=None,
    project_dir=None,
    guidance_id=None,
):
    """Add an OperationRecord and optionally write the normalized_json artifact."""
    normalized_ref = None
    if normalized_json is not None and project_dir:
        artifacts_dir = Path(project_dir) / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = artifacts_dir / f"{invocation_id}_normalized.json"
        artifact_path.write_text(json.dumps(normalized_json), encoding="utf-8")
        normalized_ref = str(artifact_path)

    session.add(
        OperationRecord(
            inference_invocation_id=invocation_id,
            project_id=project_id,
            purpose=purpose,
            example_key=example_key,
            invocation_status="success",
            normalized_json_ref=normalized_ref,
            guidance_id=guidance_id,
        )
    )


def _setup_full(
    tmp_path,
    *,
    num_examples=1,
    example_state="Unlabeled",
    rationale_enabled=True,
):
    """Create project with guidance, endpoint, model config, examples, and an OperationRecord."""
    pid = "test-proj"
    gid = generate_uuid4()
    mcid = generate_uuid4()
    eid = generate_uuid4()
    iid = generate_uuid4()
    engine, project_dir = _setup_project_db(tmp_path, pid)
    workspace = str(tmp_path / "workspace")

    proposal_json = {"severity": "high", "damaged": True}
    guidance_schema = FIXTURE_SCHEMA
    if rationale_enabled:
        proposal_json["rationale_note"] = "visible dent"
    else:
        guidance_schema = {
            **FIXTURE_SCHEMA,
            "fields": [
                field
                for field in FIXTURE_SCHEMA["fields"]
                if field["field_name"] != "rationale_note"
            ],
            "generation_order": ["severity", "damaged"],
        }

    with Session(engine) as session:
        add_project_row(
            session,
            pid,
            project_dir,
            active_guidance_id=gid,
            teacher_model_config_id=mcid,
        )
        add_guidance_row(session, pid, gid, guidance_schema)
        add_endpoint_row(session, pid, eid)
        add_model_config_row(session, pid, mcid, eid)
        keys = []
        for i in range(num_examples):
            k = f"img_{i:03d}"
            add_example_row(session, pid, k, state=example_state)
            keys.append(k)
        _add_operation_record(
            session,
            pid,
            iid,
            keys[0],
            normalized_json=proposal_json,
            project_dir=project_dir,
        )
        session.commit()

    from vlm_feedback_loop.services.project_service import set_project_engine

    set_project_engine(pid, engine)

    return engine, pid, gid, mcid, keys, iid, workspace, proposal_json


# ══════════════════════════════════════════════════════════════════════════════
# Section A: Diff computation
# ══════════════════════════════════════════════════════════════════════════════


class TestComputeLabelDiff:
    """AC: Deterministic diff between submitted label and proposal."""

    def test_identical_labels_return_accept(self):
        proposal = {"severity": "high", "damaged": True, "rationale_note": "ok"}
        label = {"severity": "high", "damaged": True, "rationale_note": "ok"}
        outcome, core, aux = compute_label_diff(label, proposal, FIXTURE_FIELDS)
        assert outcome == "Accept"
        assert core == []
        assert aux == []

    def test_core_field_diff_returns_edit(self):
        proposal = {"severity": "high", "damaged": True}
        label = {"severity": "low", "damaged": True}
        outcome, core, aux = compute_label_diff(label, proposal, FIXTURE_FIELDS)
        assert outcome == "Edit"
        assert "severity" in core

    def test_aux_field_diff_returns_edit(self):
        proposal = {"severity": "high", "damaged": True, "rationale_note": "old"}
        label = {"severity": "high", "damaged": True, "rationale_note": "new"}
        outcome, core, aux = compute_label_diff(label, proposal, FIXTURE_FIELDS)
        assert outcome == "Edit"
        assert "rationale_note" in aux
        assert core == []

    def test_rationale_note_follows_its_reserved_aux_schema_role(self):
        proposal = {"rationale_note": "old"}
        label = {"rationale_note": "new"}
        outcome, core, aux = compute_label_diff(label, proposal, FIXTURE_FIELDS)
        assert outcome == "Edit"
        assert "rationale_note" in aux
        assert core == []

    def test_multiple_diffs(self):
        proposal = {"severity": "high", "damaged": True, "rationale_note": "old"}
        label = {"severity": "low", "damaged": False, "rationale_note": "new"}
        outcome, core, aux = compute_label_diff(label, proposal, FIXTURE_FIELDS)
        assert outcome == "Edit"
        assert "severity" in core
        assert "damaged" in core
        assert "rationale_note" in aux

    def test_missing_field_in_submission_is_diff(self):
        proposal = {"severity": "high", "damaged": True}
        label = {"damaged": True}  # severity missing
        outcome, core, aux = compute_label_diff(label, proposal, FIXTURE_FIELDS)
        assert outcome == "Edit"
        assert "severity" in core


# ══════════════════════════════════════════════════════════════════════════════
# Section B: Save Label
# ══════════════════════════════════════════════════════════════════════════════


class TestSaveLabel:
    """The label save endpoint."""

    def test_save_creates_verified_label(self, tmp_path):
        engine, pid, gid, mcid, keys, iid, ws, proposal = _setup_full(tmp_path)
        result = save_label(
            pid,
            example_key=keys[0],
            inference_invocation_id=iid,
            label_json=proposal,
            rationale_source="teacher_proposal",
            rationale_regeneration_invocation_id=None,
            workspace_root=ws,
        )
        assert not isinstance(result, str), f"Error: {result}"
        assert result["verified_outcome"] == "Accept"
        assert result["label_status"] == "verified"

        with Session(engine) as s:
            lbl = s.query(Label).filter_by(project_id=pid, example_key=keys[0]).first()
            assert lbl is not None
            assert lbl.label_status == "verified"
            assert lbl.verified_outcome == "Accept"

    def test_save_normalizes_core_values_before_diff_and_persistence(self, tmp_path):
        """Verified label_json is the evaluation ground truth: match_fields
        (A.2) assumes canonical values on both sides. A whitespace/case
        variant of the proposal's enum value must persist canonicalized
        and classify as Accept — previously it became un-matchable ground
        truth and flipped a whitespace-only save into Edit."""
        engine, pid, gid, mcid, keys, iid, ws, proposal = _setup_full(tmp_path)
        submitted = dict(proposal)
        submitted["severity"] = "  HIGH "
        result = save_label(
            pid,
            example_key=keys[0],
            inference_invocation_id=iid,
            label_json=submitted,
            rationale_source="teacher_proposal",
            rationale_regeneration_invocation_id=None,
            workspace_root=ws,
        )
        assert not isinstance(result, str), f"Error: {result}"
        assert result["verified_outcome"] == "Accept"
        with Session(engine) as s:
            lbl = s.query(Label).filter_by(project_id=pid, example_key=keys[0]).first()
            assert lbl is not None
            assert lbl.label_json["severity"] == "high"

    def test_save_rejects_boolean_proxy_core_value(self, tmp_path):
        """Strict boolean normalization is a Spec Decision: numeric proxies
        are schema-invalid on every path. A save submitting damaged=1 must
        be rejected (400 validation), not stored as ground truth that the
        canonical evaluator can never match."""
        engine, pid, gid, mcid, keys, iid, ws, proposal = _setup_full(tmp_path)
        submitted = dict(proposal)
        submitted["damaged"] = 1
        result = save_label(
            pid,
            example_key=keys[0],
            inference_invocation_id=iid,
            label_json=submitted,
            rationale_source="teacher_proposal",
            rationale_regeneration_invocation_id=None,
            workspace_root=ws,
        )
        assert isinstance(result, str)
        assert result.startswith("validation:")
        assert "damaged" in result
        with Session(engine) as s:
            assert (
                s.query(Label).filter_by(project_id=pid, example_key=keys[0]).first()
                is None
            )

    def _activate_new_core_schema(self, engine, pid, iid, gid, new_core_fields):
        """Stamp the proposal at the old guidance, then activate a new
        guidance whose Core schema is ``new_core_fields``. Returns the new
        guidance_id. Shared by the stale-proposal-by-guidance tests."""
        from vlm_feedback_loop.db.models.project import Project

        new_gid = "guidance-v2"
        with Session(engine) as s:
            op = s.query(OperationRecord).filter_by(inference_invocation_id=iid).one()
            op.guidance_id = gid
            new_schema = {
                "fields": [
                    {
                        "field_id": "f0",
                        "field_name": "rationale_note",
                        "type": "string",
                        "role": "aux",
                    },
                    *new_core_fields,
                ],
                "generation_order": ["rationale_note"]
                + [f["field_name"] for f in new_core_fields],
                "derived_json_schema": {},
                "schema_hash": "v2",
            }
            add_guidance_row(s, pid, new_gid, new_schema, version_number=2)
            s.query(Project).filter_by(project_id=pid).update(
                {"active_guidance_id": new_gid}
            )
            s.commit()
        return new_gid

    def test_save_rejects_proposal_whose_core_was_renamed(self, tmp_path):
        """A proposal is stale when the Guidance version it was generated
        under (OperationRecord.guidance_id) no longer shares the active
        Core schema — here an in-place rename of 'severity' to
        'severity_level'. Saving it would stamp retired-name JSON as ground
        truth, so it is a conflict (retry for a fresh proposal)."""
        engine, pid, gid, mcid, keys, iid, ws, proposal = _setup_full(tmp_path)
        self._activate_new_core_schema(
            engine,
            pid,
            iid,
            gid,
            [
                {
                    "field_id": "f1",
                    "field_name": "severity_level",
                    "type": "enum",
                    "role": "core",
                    "allowed_values": ["low", "medium", "high"],
                },
                {
                    "field_id": "f2",
                    "field_name": "damaged",
                    "type": "boolean",
                    "role": "core",
                },
            ],
        )

        result = save_label(
            pid,
            example_key=keys[0],
            inference_invocation_id=iid,
            label_json=proposal,
            rationale_source="teacher_proposal",
            rationale_regeneration_invocation_id=None,
            workspace_root=ws,
        )
        assert isinstance(result, str)
        assert result.startswith("conflict:")
        assert "predates the current schema" in result
        with Session(engine) as s:
            assert (
                s.query(Label).filter_by(project_id=pid, example_key=keys[0]).first()
                is None
            )

    def test_save_rejects_proposal_across_value_compatible_semantic_change(
        self, tmp_path
    ):
        """Staleness is judged by the Core schema, not just field presence:
        a semantic change that only WIDENS an enum keeps every old field
        name and value valid, but the example was reverted to Unlabeled to
        be re-proposed. A proposal from before the change is still stale and
        must be rejected — the field-presence check this replaced missed it."""
        engine, pid, gid, mcid, keys, iid, ws, proposal = _setup_full(tmp_path)
        self._activate_new_core_schema(
            engine,
            pid,
            iid,
            gid,
            [
                {
                    "field_id": "f1",
                    "field_name": "severity",
                    "type": "enum",
                    "role": "core",
                    # widened: 'none' added — the stale proposal's 'high' is
                    # still valid, and no field is missing.
                    "allowed_values": ["none", "low", "medium", "high"],
                },
                {
                    "field_id": "f2",
                    "field_name": "damaged",
                    "type": "boolean",
                    "role": "core",
                },
            ],
        )

        result = save_label(
            pid,
            example_key=keys[0],
            inference_invocation_id=iid,
            label_json=proposal,
            rationale_source="teacher_proposal",
            rationale_regeneration_invocation_id=None,
            workspace_root=ws,
        )
        assert isinstance(result, str)
        assert result.startswith("conflict:")
        assert "predates the current schema" in result

    def test_save_manual_label_over_failed_proposal_fills_unset_boolean(self, tmp_path):
        """A fresh manual label after a failed Teacher proposal is generated
        under the ACTIVE Guidance, so it is never stale — even when the SME
        leaves a core boolean at its unchecked (false) rest state and the UI
        omits it from the payload. The save must succeed and the boolean is
        filled to false, not rejected as a stale proposal (the regression the
        field-presence conflict introduced)."""
        engine, pid, gid, mcid, keys, iid, ws, proposal = _setup_full(tmp_path)
        # A failed proposal: point the op at the active guidance, mark it
        # schema_invalid with no artifact (the deadlock-avoidance path).
        with Session(engine) as s:
            op = s.query(OperationRecord).filter_by(inference_invocation_id=iid).one()
            op.guidance_id = gid
            op.invocation_status = "schema_invalid"
            op.normalized_json_ref = None
            s.commit()

        # SME fills severity, agrees with damaged=false (unchecked → omitted).
        result = save_label(
            pid,
            example_key=keys[0],
            inference_invocation_id=iid,
            label_json={"severity": "low", "rationale_note": "manual"},
            rationale_source="sme_authored",
            rationale_regeneration_invocation_id=None,
            workspace_root=ws,
        )
        assert isinstance(result, dict)
        with Session(engine) as s:
            label = s.query(Label).filter_by(project_id=pid, example_key=keys[0]).one()
            assert label.label_json["damaged"] is False
            assert label.label_json["severity"] == "low"

    def test_save_conflicts_when_guidance_changes_mid_save(self, tmp_path):
        """save_label runs its reads in autocommit, so a guidance edit can
        commit inside the handler's span — after the staleness gate read
        the pre-edit pointer, before the label INSERT. The edit's label
        sweep only sees committed rows, so without the commit guard this
        save would stamp a retired guidance_id as Verified ground truth.
        The listener below commits a new active version at the exact
        pre-INSERT moment; the save must abandon with a conflict. A positive
        pool target also proves the current assignment and one promotion stay
        inside that same rollback boundary."""
        import json as _json

        from sqlalchemy import event, text

        engine, pid, gid, mcid, keys, iid, ws, proposal = _setup_full(
            tmp_path,
            num_examples=4,
        )

        with Session(engine) as s:
            project = s.get(Project, pid)
            assert project is not None
            project.test_pool_fraction = 0.75

            historical = (
                (keys[1], "0000000000000000", "test_pool"),
                (keys[2], "ffffffffffffffff", None),
                (keys[3], "00000000ffffffff", None),
            )
            for key, phash, pool_assignment in historical:
                example = s.get(Example, key)
                assert example is not None
                example.state = "Verified"
                example.phash = phash
                s.add(
                    Label(
                        label_id=generate_uuid4(),
                        project_id=pid,
                        example_key=key,
                        label_status="verified",
                        guidance_id=gid,
                        inference_invocation_id=generate_uuid4(),
                        label_json=proposal,
                        labeled_at=utc_now(),
                        verified_outcome="Accept",
                        verified_at=utc_now(),
                        edited_core_fields=[],
                        edited_aux_fields=[],
                        rationale_source="teacher_proposal",
                        pool_assignment=pool_assignment,
                    )
                )

            current = s.get(Example, keys[0])
            assert current is not None
            current.phash = "ffffffffffffffff"
            s.commit()

        flipped = {"done": False}

        def _flip_guidance_before_first_label_write(
            conn, cursor, statement, parameters, context, executemany
        ):
            # Fire at the save transaction's FIRST write — before it takes
            # the SQLite write lock — so the edit below can commit on a
            # second connection without deadlocking against it.
            if flipped["done"] or not statement.startswith(("INSERT INTO", "UPDATE")):
                return
            flipped["done"] = True
            with engine.begin() as c2:
                c2.execute(
                    text(
                        "INSERT INTO guidances (guidance_id, project_id, "
                        "version_number, description, schema, rules, "
                        "created_at) VALUES ('g2-mid-save', :pid, 2, 'v2', "
                        ":schema, '', '2026-07-22T00:00:00Z')"
                    ),
                    {"pid": pid, "schema": _json.dumps(FIXTURE_SCHEMA)},
                )
                c2.execute(
                    text(
                        "UPDATE projects SET active_guidance_id = "
                        "'g2-mid-save' WHERE project_id = :pid"
                    ),
                    {"pid": pid},
                )

        event.listen(
            engine, "before_cursor_execute", _flip_guidance_before_first_label_write
        )
        try:
            result = save_label(
                pid,
                example_key=keys[0],
                inference_invocation_id=iid,
                label_json=proposal,
                rationale_source="teacher_proposal",
                rationale_regeneration_invocation_id=None,
                workspace_root=ws,
            )
        finally:
            event.remove(
                engine, "before_cursor_execute", _flip_guidance_before_first_label_write
            )

        assert flipped["done"], "the mid-save edit never fired"
        assert isinstance(result, str)
        assert result.startswith("conflict:")
        assert "changed while" in result
        with Session(engine) as s:
            assert (
                s.query(Label).filter_by(project_id=pid, example_key=keys[0]).first()
                is None
            )
            current = s.get(Example, keys[0])
            assert current is not None
            assert current.state == "Unlabeled"

            assignments = {
                label.example_key: label.pool_assignment
                for label in s.query(Label).filter_by(project_id=pid).all()
            }
            assert assignments == {
                keys[1]: "test_pool",
                keys[2]: None,
                keys[3]: None,
            }

    def test_save_drops_keys_the_active_schema_does_not_know(self, tmp_path):
        """The stored label is built from the active schema's fields only
        (additionalProperties: false): a retired field name from a pre-edit
        browser tab — or any client-sent garbage key — must not enter
        Verified ground truth, where it would pollute exports and diverge
        from every label the rename propagation rewrote."""
        engine, pid, gid, mcid, keys, iid, ws, proposal = _setup_full(tmp_path)
        submitted = dict(proposal)
        submitted["retired_note"] = "from a stale tab"
        result = save_label(
            pid,
            example_key=keys[0],
            inference_invocation_id=iid,
            label_json=submitted,
            rationale_source="sme_edited",
            rationale_regeneration_invocation_id=None,
            workspace_root=ws,
        )
        assert isinstance(result, dict)
        with Session(engine) as s:
            label = s.query(Label).filter_by(project_id=pid, example_key=keys[0]).one()
            assert "retired_note" not in label.label_json
            assert label.label_json["severity"] == "high"

    def test_save_rejects_missing_required_core_as_validation(self, tmp_path):
        """A required core field with no unambiguous rest value (an enum) that
        the submission omits is a client validation error (400), not a stale-
        proposal conflict — the API client sent incomplete data under the
        current schema."""
        engine, pid, gid, mcid, keys, iid, ws, proposal = _setup_full(tmp_path)
        with Session(engine) as s:
            op = s.query(OperationRecord).filter_by(inference_invocation_id=iid).one()
            op.guidance_id = gid
            op.invocation_status = "schema_invalid"
            op.normalized_json_ref = None
            s.commit()

        result = save_label(
            pid,
            example_key=keys[0],
            inference_invocation_id=iid,
            label_json={"rationale_note": "no severity picked"},
            rationale_source="sme_authored",
            rationale_regeneration_invocation_id=None,
            workspace_root=ws,
        )
        assert isinstance(result, str)
        assert result.startswith("validation:")
        assert "severity" in result

    def test_save_with_unreadable_proposal_artifact_is_conflict(self, tmp_path):
        """A save whose proposal artifact is missing/corrupt must fail as a
        conflict, not silently diff against {} — that would flip a true
        Accept into Edit, corrupting verified_outcome, pool routing, and
        the ICL candidate set."""
        engine, pid, gid, mcid, keys, iid, ws, proposal = _setup_full(tmp_path)
        with Session(engine) as s:
            op = s.query(OperationRecord).filter_by(inference_invocation_id=iid).first()
            Path(op.normalized_json_ref).unlink()
            s.commit()
        result = save_label(
            pid,
            example_key=keys[0],
            inference_invocation_id=iid,
            label_json=proposal,  # identical to the proposal — a true Accept
            rationale_source="teacher_proposal",
            rationale_regeneration_invocation_id=None,
            workspace_root=ws,
        )
        assert isinstance(result, str)
        assert "conflict" in result.lower()
        assert "retry" in result.lower()
        # And no Verified label was written.
        with Session(engine) as s:
            lbl = s.query(Label).filter_by(project_id=pid, example_key=keys[0]).first()
            assert lbl is None

    def test_save_with_corrupt_proposal_artifact_is_conflict(self, tmp_path):
        """Same contract when the artifact exists but is not parseable JSON."""
        engine, pid, gid, mcid, keys, iid, ws, proposal = _setup_full(tmp_path)
        with Session(engine) as s:
            op = s.query(OperationRecord).filter_by(inference_invocation_id=iid).first()
            Path(op.normalized_json_ref).write_text("{not json", encoding="utf-8")
            s.commit()
        result = save_label(
            pid,
            example_key=keys[0],
            inference_invocation_id=iid,
            label_json=proposal,
            rationale_source="teacher_proposal",
            rationale_regeneration_invocation_id=None,
            workspace_root=ws,
        )
        assert isinstance(result, str)
        assert "conflict" in result.lower()

    def test_save_against_schema_invalid_proposal_with_degenerate_artifact(
        self, tmp_path
    ):
        """An SME correction must save even when the proposal was model-invalid
        and its normalized artifact is not a JSON object. A schema_invalid
        proposal carries no Accept/Edit ambiguity — the label is a true Edit
        against an empty proposal — and refusing the save deadlocks the SME
        whenever the Teacher deterministically emits degenerate output for an
        image (Retry reproduces the identical artifact at temperature 0)."""
        engine, pid, gid, mcid, keys, iid, ws, proposal = _setup_full(tmp_path)
        with Session(engine) as s:
            op = s.query(OperationRecord).filter_by(inference_invocation_id=iid).first()
            op.invocation_status = "schema_invalid"
            Path(op.normalized_json_ref).write_text('"bare string"', encoding="utf-8")
            s.commit()
        result = save_label(
            pid,
            example_key=keys[0],
            inference_invocation_id=iid,
            label_json=proposal,
            rationale_source="sme_edited",
            rationale_regeneration_invocation_id=None,
            workspace_root=ws,
        )
        assert not isinstance(result, str), f"Error: {result}"
        assert result["verified_outcome"] == "Edit"
        with Session(engine) as s:
            lbl = s.query(Label).filter_by(project_id=pid, example_key=keys[0]).first()
            assert lbl is not None
            assert lbl.label_status == "verified"

    def test_save_edit_outcome_with_diff(self, tmp_path):
        engine, pid, gid, mcid, keys, iid, ws, proposal = _setup_full(tmp_path)
        edited = {**proposal, "severity": "low"}
        result = save_label(
            pid,
            example_key=keys[0],
            inference_invocation_id=iid,
            label_json=edited,
            rationale_source="sme_edited",
            rationale_regeneration_invocation_id=None,
            workspace_root=ws,
        )
        assert not isinstance(result, str)
        assert result["verified_outcome"] == "Edit"
        assert "severity" in result["edited_core_fields"]

    def test_save_edit_rejects_teacher_proposal_rationale(self, tmp_path):
        """Edit + rationale_source='teacher_proposal' → 400."""
        engine, pid, gid, mcid, keys, iid, ws, proposal = _setup_full(tmp_path)
        edited = {**proposal, "severity": "low"}
        result = save_label(
            pid,
            example_key=keys[0],
            inference_invocation_id=iid,
            label_json=edited,
            rationale_source="teacher_proposal",
            rationale_regeneration_invocation_id=None,
            workspace_root=ws,
        )
        assert isinstance(result, str)
        assert "teacher_proposal" in result.lower() or "rationale" in result.lower()

    def test_save_requires_rationale_source_only_when_feature_is_enabled(
        self, tmp_path
    ):
        engine, pid, gid, mcid, keys, iid, ws, proposal = _setup_full(tmp_path)
        result = save_label(
            pid,
            example_key=keys[0],
            inference_invocation_id=iid,
            label_json=proposal,
            rationale_source=None,
            rationale_regeneration_invocation_id=None,
            workspace_root=ws,
        )
        assert isinstance(result, str)
        assert result.startswith("validation:")
        assert "rationale_source is required" in result
        with Session(engine) as session:
            assert session.query(Label).filter_by(project_id=pid).first() is None

    def test_save_bypasses_rationale_when_feature_is_disabled(self, tmp_path):
        """A disabled Guidance neither validates nor persists rationale data,
        including stale metadata sent by a client that has not refreshed."""
        engine, pid, gid, mcid, keys, iid, ws, proposal = _setup_full(
            tmp_path, rationale_enabled=False
        )
        submitted = {
            **proposal,
            "severity": "low",
            "rationale_note": "stale rationale from an old browser tab",
        }
        result = save_label(
            pid,
            example_key=keys[0],
            inference_invocation_id=iid,
            label_json=submitted,
            rationale_source="teacher_proposal",
            rationale_regeneration_invocation_id="stale-regeneration-id",
            workspace_root=ws,
        )
        assert not isinstance(result, str), f"Error: {result}"
        assert result["verified_outcome"] == "Edit"
        assert result["edited_core_fields"] == ["severity"]
        assert result["edited_aux_fields"] == []

        with Session(engine) as session:
            label = session.query(Label).filter_by(project_id=pid).one()
            assert label.label_json == {"severity": "low", "damaged": True}
            assert label.rationale_source is None
            assert label.rationale_regeneration_invocation_id is None

    def test_save_transitions_example_to_verified(self, tmp_path):
        engine, pid, gid, mcid, keys, iid, ws, proposal = _setup_full(tmp_path)
        save_label(
            pid,
            example_key=keys[0],
            inference_invocation_id=iid,
            label_json=proposal,
            rationale_source="teacher_proposal",
            rationale_regeneration_invocation_id=None,
            workspace_root=ws,
        )
        with Session(engine) as s:
            ex = s.query(Example).filter_by(example_key=keys[0]).first()
            assert ex.state == "Verified"

    def test_save_auto_labeled_promotion(self, tmp_path):
        engine, pid, gid, mcid, keys, iid, ws, proposal = _setup_full(
            tmp_path,
            example_state="Auto-Labeled",
        )
        batch_run_id = generate_uuid4()
        with Session(engine) as s:
            s.add(
                Label(
                    label_id=generate_uuid4(),
                    project_id=pid,
                    example_key=keys[0],
                    label_status="auto_labeled",
                    guidance_id=gid,
                    inference_invocation_id=generate_uuid4(),
                    label_json={"severity": "medium"},
                    labeled_at=utc_now(),
                    batch_label_run_id=batch_run_id,
                )
            )
            s.commit()

        result = save_label(
            pid,
            example_key=keys[0],
            inference_invocation_id=iid,
            label_json=proposal,
            rationale_source="teacher_proposal",
            rationale_regeneration_invocation_id=None,
            workspace_root=ws,
        )
        assert not isinstance(result, str)
        with Session(engine) as s:
            lbl = s.query(Label).filter_by(project_id=pid, example_key=keys[0]).first()
            assert lbl.label_status == "verified"
            assert lbl.batch_label_run_id == batch_run_id  # retained
            assert lbl.inference_invocation_id == iid  # updated

    def test_save_promotion_repoints_stale_guidance_id(self, tmp_path):
        """Promoting an Auto-Labeled row to Verified re-points it to the
        active Guidance. A historical or imported auto_labeled row can carry a
        retired guidance_id; promoting it as ground truth under the stale
        version would orphan it from ICL, exports, and the gate — the Spec
        requires Verified labels to use the active_guidance_id."""
        engine, pid, gid, mcid, keys, iid, ws, proposal = _setup_full(
            tmp_path,
            example_state="Auto-Labeled",
        )
        with Session(engine) as s:
            s.add(
                Label(
                    label_id=generate_uuid4(),
                    project_id=pid,
                    example_key=keys[0],
                    label_status="auto_labeled",
                    guidance_id="retired-v0",
                    inference_invocation_id=generate_uuid4(),
                    label_json={"severity": "medium"},
                    labeled_at=utc_now(),
                )
            )
            s.commit()

        result = save_label(
            pid,
            example_key=keys[0],
            inference_invocation_id=iid,
            label_json=proposal,
            rationale_source="teacher_proposal",
            rationale_regeneration_invocation_id=None,
            workspace_root=ws,
        )
        assert not isinstance(result, str)
        with Session(engine) as s:
            lbl = s.query(Label).filter_by(project_id=pid, example_key=keys[0]).one()
            assert lbl.guidance_id == gid  # re-pointed from retired-v0

    def test_stale_proposal_wrong_project(self, tmp_path):
        engine, pid, gid, mcid, keys, iid, ws, proposal = _setup_full(tmp_path)
        # Create a record for a different project
        with Session(engine) as s:
            s.add(
                OperationRecord(
                    inference_invocation_id="other-iid",
                    project_id="other-proj",
                    purpose="interactive_proposal",
                    example_key=keys[0],
                    invocation_status="success",
                )
            )
            s.commit()
        result = save_label(
            pid,
            example_key=keys[0],
            inference_invocation_id="other-iid",
            label_json=proposal,
            rationale_source="teacher_proposal",
            rationale_regeneration_invocation_id=None,
            workspace_root=ws,
        )
        assert isinstance(result, str)
        assert "not found" in result.lower()

    def test_stale_proposal_wrong_example(self, tmp_path):
        engine, pid, gid, mcid, keys, iid, ws, proposal = _setup_full(
            tmp_path, num_examples=2
        )
        result = save_label(
            pid,
            example_key=keys[1],  # different example
            inference_invocation_id=iid,
            label_json=proposal,
            rationale_source="teacher_proposal",
            rationale_regeneration_invocation_id=None,
            workspace_root=ws,
        )
        assert isinstance(result, str)
        assert "pertains to" in result.lower() or "example" in result.lower()

    def test_stale_proposal_wrong_purpose(self, tmp_path):
        engine, pid, gid, mcid, keys, iid, ws, proposal = _setup_full(tmp_path)
        # Override the OperationRecord purpose
        with Session(engine) as s:
            op = s.query(OperationRecord).filter_by(inference_invocation_id=iid).first()
            op.purpose = "evaluation"
            s.commit()
        result = save_label(
            pid,
            example_key=keys[0],
            inference_invocation_id=iid,
            label_json=proposal,
            rationale_source="teacher_proposal",
            rationale_regeneration_invocation_id=None,
            workspace_root=ws,
        )
        assert isinstance(result, str)
        assert "purpose" in result.lower()

    def test_superseded_proposal_returns_conflict(self, tmp_path):
        engine, pid, gid, mcid, keys, iid, ws, proposal = _setup_full(tmp_path)
        # Create a retry OperationRecord that supersedes this one
        with Session(engine) as s:
            s.add(
                OperationRecord(
                    inference_invocation_id=generate_uuid4(),
                    project_id=pid,
                    purpose="interactive_proposal",
                    example_key=keys[0],
                    invocation_status="success",
                    retry_of_inference_invocation_id=iid,
                )
            )
            s.commit()
        result = save_label(
            pid,
            example_key=keys[0],
            inference_invocation_id=iid,
            label_json=proposal,
            rationale_source="teacher_proposal",
            rationale_regeneration_invocation_id=None,
            workspace_root=ws,
        )
        assert isinstance(result, str)
        assert "superseded" in result.lower()

    def test_nonexistent_invocation(self, tmp_path):
        engine, pid, gid, mcid, keys, iid, ws, proposal = _setup_full(tmp_path)
        result = save_label(
            pid,
            example_key=keys[0],
            inference_invocation_id="nonexistent-id",
            label_json=proposal,
            rationale_source="teacher_proposal",
            rationale_regeneration_invocation_id=None,
            workspace_root=ws,
        )
        assert isinstance(result, str)
        assert "not found" in result.lower()

    def test_already_verified_returns_error(self, tmp_path):
        engine, pid, gid, mcid, keys, iid, ws, proposal = _setup_full(
            tmp_path,
            example_state="Unlabeled",
        )
        # First save
        save_label(
            pid,
            example_key=keys[0],
            inference_invocation_id=iid,
            label_json=proposal,
            rationale_source="teacher_proposal",
            rationale_regeneration_invocation_id=None,
            workspace_root=ws,
        )
        # Second save on same example (now Verified)
        iid2 = generate_uuid4()
        with Session(engine) as s:
            _add_operation_record(s, pid, iid2, keys[0])
            s.commit()
        result = save_label(
            pid,
            example_key=keys[0],
            inference_invocation_id=iid2,
            label_json=proposal,
            rationale_source="teacher_proposal",
            rationale_regeneration_invocation_id=None,
            workspace_root=ws,
        )
        assert isinstance(result, str)
        assert "verified" in result.lower() or "state" in result.lower()

    def test_rationale_regen_invocation_id_stored(self, tmp_path):
        engine, pid, gid, mcid, keys, iid, ws, proposal = _setup_full(tmp_path)
        regen_id = generate_uuid4()
        edited = {**proposal, "severity": "low", "rationale_note": "new rationale"}
        result = save_label(
            pid,
            example_key=keys[0],
            inference_invocation_id=iid,
            label_json=edited,
            rationale_source="teacher_regenerated_approved",
            rationale_regeneration_invocation_id=regen_id,
            workspace_root=ws,
        )
        assert not isinstance(result, str)
        with Session(engine) as s:
            lbl = s.query(Label).filter_by(project_id=pid, example_key=keys[0]).first()
            assert lbl.rationale_regeneration_invocation_id == regen_id
            assert lbl.rationale_source == "teacher_regenerated_approved"

    def test_pool_assignment_null_when_target_zero(self, tmp_path):
        """With 1 Verified, floor(1*0.40)=0, so pool target is 0 -> None."""
        engine, pid, gid, mcid, keys, iid, ws, proposal = _setup_full(tmp_path)
        result = save_label(
            pid,
            example_key=keys[0],
            inference_invocation_id=iid,
            label_json=proposal,
            rationale_source="teacher_proposal",
            rationale_regeneration_invocation_id=None,
            workspace_root=ws,
        )
        assert not isinstance(result, str)
        assert result["pool_assignment"] is None


# ══════════════════════════════════════════════════════════════════════════════
# Section C: Skip
# ══════════════════════════════════════════════════════════════════════════════


class TestSkipExample:
    """Skip transitions the example to Omitted."""

    def test_skip_unlabeled(self, tmp_path):
        engine, pid, *_, ws, _ = _setup_full(tmp_path)
        keys = ["img_000"]
        result = skip_example(pid, keys[0], ws)
        assert not isinstance(result, str)
        assert result["state"] == "Omitted"
        assert result["omitted_at"] is not None
        with Session(engine) as s:
            ex = s.query(Example).filter_by(example_key=keys[0]).first()
            assert ex.state == "Omitted"
            assert ex.omitted_source == "sme_skip"

    def test_skip_auto_labeled_discards_machine_label_and_restore_stays_clean(
        self, tmp_path
    ):
        """Skip rejects the machine proposal; Restore must not resurrect it."""
        engine, pid, gid, _, keys, iid, ws, proposal = _setup_full(
            tmp_path,
            num_examples=2,
            example_state="Auto-Labeled",
        )
        with Session(engine) as s:
            s.add_all(
                [
                    Label(
                        label_id=generate_uuid4(),
                        project_id=pid,
                        example_key=keys[0],
                        label_status="auto_labeled",
                        guidance_id=gid,
                        inference_invocation_id=iid,
                        label_json=proposal,
                        labeled_at=utc_now(),
                        batch_label_run_id=generate_uuid4(),
                    ),
                    Label(
                        label_id=generate_uuid4(),
                        project_id=pid,
                        example_key=keys[1],
                        label_status="auto_labeled",
                        guidance_id=gid,
                        inference_invocation_id=generate_uuid4(),
                        label_json=proposal,
                        labeled_at=utc_now(),
                        batch_label_run_id=generate_uuid4(),
                    ),
                ]
            )
            s.commit()

        result = skip_example(pid, keys[0], ws)
        assert not isinstance(result, str)
        assert result["state"] == "Omitted"
        with Session(engine) as s:
            assert (
                s.query(Label).filter_by(project_id=pid, example_key=keys[0]).count()
                == 0
            )
            sibling = (
                s.query(Label).filter_by(project_id=pid, example_key=keys[1]).one()
            )
            assert sibling.label_status == "auto_labeled"
            assert (
                s.query(Example)
                .filter_by(project_id=pid, example_key=keys[1])
                .one()
                .state
                == "Auto-Labeled"
            )
            assert (
                s.query(OperationRecord)
                .filter_by(
                    project_id=pid,
                    inference_invocation_id=iid,
                )
                .count()
                == 1
            )

        restored = restore_omitted(pid, ws)
        assert not isinstance(restored, str)
        assert restored["restored_count"] == 1
        with Session(engine) as s:
            example = (
                s.query(Example).filter_by(project_id=pid, example_key=keys[0]).one()
            )
            assert example.state == "Unlabeled"
            assert (
                s.query(Label).filter_by(project_id=pid, example_key=keys[0]).count()
                == 0
            )
            assert (
                s.query(Label).filter_by(project_id=pid, example_key=keys[1]).count()
                == 1
            )

    def test_skip_auto_labeled_deletes_only_the_owned_machine_label(self, tmp_path):
        """Malformed neighboring rows cannot turn Skip into ground-truth loss."""
        engine, pid, gid, _, keys, iid, ws, proposal = _setup_full(
            tmp_path,
            example_state="Auto-Labeled",
        )
        target_id = generate_uuid4()
        verified_id = generate_uuid4()
        foreign_id = generate_uuid4()
        with Session(engine) as s:
            s.add_all(
                [
                    Label(
                        label_id=target_id,
                        project_id=pid,
                        example_key=keys[0],
                        label_status="auto_labeled",
                        guidance_id=gid,
                        inference_invocation_id=iid,
                        label_json=proposal,
                        labeled_at=utc_now(),
                    ),
                    Label(
                        label_id=verified_id,
                        project_id=pid,
                        example_key=keys[0],
                        label_status="verified",
                        guidance_id=gid,
                        inference_invocation_id=generate_uuid4(),
                        label_json=proposal,
                        labeled_at=utc_now(),
                    ),
                    Label(
                        label_id=foreign_id,
                        project_id="foreign-project",
                        example_key=keys[0],
                        label_status="auto_labeled",
                        guidance_id=gid,
                        inference_invocation_id=generate_uuid4(),
                        label_json=proposal,
                        labeled_at=utc_now(),
                    ),
                ]
            )
            s.commit()

        result = skip_example(pid, keys[0], ws)
        assert not isinstance(result, str)
        with Session(engine) as s:
            remaining_ids = {row[0] for row in s.query(Label.label_id).all()}
            assert target_id not in remaining_ids
            assert {verified_id, foreign_id}.issubset(remaining_ids)

    @pytest.mark.parametrize(
        "statement_prefix",
        ["DELETE FROM LABELS", "UPDATE EXAMPLES"],
    )
    def test_skip_auto_labeled_rolls_back_on_write_failure(
        self, tmp_path, statement_prefix
    ):
        """The rejected proposal and Omitted transition commit as one unit."""
        engine, pid, gid, _, keys, iid, ws, proposal = _setup_full(
            tmp_path,
            example_state="Auto-Labeled",
        )
        with Session(engine) as s:
            s.add(
                Label(
                    label_id=generate_uuid4(),
                    project_id=pid,
                    example_key=keys[0],
                    label_status="auto_labeled",
                    guidance_id=gid,
                    inference_invocation_id=iid,
                    label_json=proposal,
                    labeled_at=utc_now(),
                )
            )
            s.commit()

        def fail_target_statement(
            _conn,
            _cursor,
            statement,
            _parameters,
            _context,
            _executemany,
        ):
            if statement.lstrip().upper().startswith(statement_prefix):
                raise RuntimeError("injected Skip write failure")

        event.listen(engine, "before_cursor_execute", fail_target_statement)
        try:
            with pytest.raises(RuntimeError, match="injected Skip write failure"):
                skip_example(pid, keys[0], ws)
        finally:
            event.remove(engine, "before_cursor_execute", fail_target_statement)

        with Session(engine) as s:
            example = (
                s.query(Example).filter_by(project_id=pid, example_key=keys[0]).one()
            )
            assert example.state == "Auto-Labeled"
            assert (
                s.query(Label).filter_by(project_id=pid, example_key=keys[0]).count()
                == 1
            )

    def test_skip_already_omitted_returns_error(self, tmp_path):
        engine, pid, *_, ws, _ = _setup_full(tmp_path)
        skip_example(pid, "img_000", ws)  # first skip
        result = skip_example(pid, "img_000", ws)  # second skip
        assert isinstance(result, str)

    def test_skip_verified_returns_error(self, tmp_path):
        engine, pid, gid, mcid, keys, iid, ws, proposal = _setup_full(tmp_path)
        save_label(
            pid,
            example_key=keys[0],
            inference_invocation_id=iid,
            label_json=proposal,
            rationale_source="teacher_proposal",
            rationale_regeneration_invocation_id=None,
            workspace_root=ws,
        )
        result = skip_example(pid, keys[0], ws)
        assert isinstance(result, str)

    def test_skip_nonexistent(self, tmp_path):
        engine, pid, *_, ws, _ = _setup_full(tmp_path)
        result = skip_example(pid, "nonexistent", ws)
        assert isinstance(result, str)
        assert "not found" in result.lower()


# ══════════════════════════════════════════════════════════════════════════════
# Section D: Restore Omitted
# ══════════════════════════════════════════════════════════════════════════════


class TestRestoreOmitted:
    """Restore returns all Omitted examples to Unlabeled."""

    def test_restore_all(self, tmp_path):
        engine, pid, *_, ws, _ = _setup_full(tmp_path, num_examples=3)
        for k in ["img_000", "img_001", "img_002"]:
            skip_example(pid, k, ws)
        result = restore_omitted(pid, ws)
        assert not isinstance(result, str)
        assert result["restored_count"] == 3
        with Session(engine) as s:
            for k in ["img_000", "img_001", "img_002"]:
                ex = s.query(Example).filter_by(example_key=k).first()
                assert ex.state == "Unlabeled"

    def test_restore_none_returns_zero(self, tmp_path):
        engine, pid, *_, ws, _ = _setup_full(tmp_path)
        result = restore_omitted(pid, ws)
        assert not isinstance(result, str)
        assert result["restored_count"] == 0

    def test_restore_clears_omission_fields(self, tmp_path):
        engine, pid, *_, ws, _ = _setup_full(tmp_path)
        skip_example(pid, "img_000", ws)
        restore_omitted(pid, ws)
        with Session(engine) as s:
            ex = s.query(Example).filter_by(example_key="img_000").first()
            assert ex.omitted_source is None
            assert ex.omitted_at is None

    def test_restore_does_not_affect_other_states(self, tmp_path):
        engine, pid, gid, mcid, keys, iid, ws, proposal = _setup_full(
            tmp_path,
            num_examples=2,
        )
        # Save one example (Verified), skip the other (Omitted)
        save_label(
            pid,
            example_key=keys[0],
            inference_invocation_id=iid,
            label_json=proposal,
            rationale_source="teacher_proposal",
            rationale_regeneration_invocation_id=None,
            workspace_root=ws,
        )
        skip_example(pid, keys[1], ws)
        # Restore
        restore_omitted(pid, ws)
        with Session(engine) as s:
            verified_ex = s.query(Example).filter_by(example_key=keys[0]).first()
            assert verified_ex.state == "Verified"  # unchanged
            restored_ex = s.query(Example).filter_by(example_key=keys[1]).first()
            assert restored_ex.state == "Unlabeled"  # restored
