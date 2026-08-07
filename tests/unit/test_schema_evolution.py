# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance tests for schema evolution.

Covers the SchemaCore Core-edit policy and the schema evolution resets.

Uses synthetic test fixtures: programmatically created Example, Label,
OperationRecord, RunRecord, and Pool records inserted directly into the
project database.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import contextmanager
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy.orm import Session
from starlette.testclient import TestClient

from conftest import create_project_via_api
from vlm_feedback_loop.db.base import generate_uuid4, utc_now
from vlm_feedback_loop.db.models.example import Example
from vlm_feedback_loop.db.models.guidance import Guidance
from vlm_feedback_loop.db.models.label import Label
from vlm_feedback_loop.db.models.operation import OperationRecord
from vlm_feedback_loop.db.models.pool import Pool
from vlm_feedback_loop.db.models.project import Project
from vlm_feedback_loop.db.models.run import RunRecord
from vlm_feedback_loop.services import batch_label_service, evaluation_service
from vlm_feedback_loop.services.project_service import get_project_engine

# ── Helpers ─────────────────────────────────────────────────────────────────


def _create_project(client: TestClient) -> str:
    return create_project_via_api(client, name="EvolutionTest")["project_id"]


def _core_enum_schema(
    field_id: str = "fid-cat",
    name: str = "damage_type",
    values: list[str] | None = None,
) -> list[dict]:
    return [
        {
            "field_name": name,
            "type": "enum",
            "role": "core",
            "allowed_values": values or ["crack", "dent"],
            "display_order": 1,
            "field_id": field_id,
        }
    ]


def _create_guidance_via_api(
    client: TestClient,
    project_id: str,
    schema: list[dict] | None = None,
    description: str = "Classify damage.",
) -> dict:
    """Create guidance via the POST endpoint (fresh field_ids assigned)."""
    create_schema = [
        {key: value for key, value in field.items() if key != "field_id"}
        for field in (schema or _core_enum_schema())
    ]
    resp = client.post(
        f"/v1/projects/{project_id}/guidance",
        json={
            "description": description,
            "schema": create_schema,
            "rules": "",
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _set_active_guidance(client: TestClient, project_id: str, guidance_id: str) -> None:
    resp = client.patch(
        f"/v1/projects/{project_id}",
        json={"active_guidance_id": guidance_id},
    )
    assert resp.status_code == 200, resp.text


def _get_engine(client: TestClient, project_id: str):
    """Get the project's SQLAlchemy engine via the cached path."""
    from vlm_feedback_loop.config import Settings
    from vlm_feedback_loop.routers.projects import get_current_settings

    settings: Settings = client.app.dependency_overrides[get_current_settings]()
    return get_project_engine(project_id, settings.WORKSPACE_ROOT)


def _insert_verified_example(
    engine,
    project_id: str,
    example_key: str,
    guidance_id: str,
    label_json: dict,
    verified_outcome: str = "Edit",
    edited_core_fields: list[str] | None = None,
    vlm_proposal_json: str | None = None,
) -> tuple[str, str]:
    """Insert a synthetic Verified Example + Label + OperationRecord.

    Returns (label_id, inference_invocation_id).
    """
    inv_id = generate_uuid4()
    label_id = generate_uuid4()
    now = utc_now()

    with Session(engine) as session:
        # OperationRecord (for VLM proposal snapshot)
        op = OperationRecord(
            inference_invocation_id=inv_id,
            project_id=project_id,
            purpose="interactive_proposal",
            example_key=example_key,
            guidance_id=guidance_id,
            invocation_status="success",
            normalized_json_ref=vlm_proposal_json or json.dumps(label_json),
        )
        session.add(op)

        ex = Example(
            example_key=example_key,
            project_id=project_id,
            storage_ref=f"/fake/{example_key}.jpg",
            ingested_at=now,
            source_metadata={},
            state="Verified",
            clip_embedding_present=False,
        )
        session.add(ex)

        lbl = Label(
            label_id=label_id,
            project_id=project_id,
            example_key=example_key,
            label_status="verified",
            guidance_id=guidance_id,
            inference_invocation_id=inv_id,
            label_json=label_json,
            labeled_at=now,
            verified_outcome=verified_outcome,
            verified_at=now,
            edited_core_fields=edited_core_fields or [],
            edited_aux_fields=[],
            rationale_source="teacher_proposal"
            if verified_outcome == "Accept"
            else "sme_edited",
        )
        session.add(lbl)
        session.commit()

    return label_id, inv_id


def _insert_verified_with_pool(
    engine,
    project_id: str,
    example_key: str,
    guidance_id: str,
    label_json: dict,
) -> str:
    """Insert a Verified Example with pool_assignment=test_pool."""
    label_id, _ = _insert_verified_example(
        engine,
        project_id,
        example_key,
        guidance_id,
        label_json,
        verified_outcome="Accept",
    )
    with Session(engine) as session:
        lbl = session.query(Label).filter_by(label_id=label_id).one()
        lbl.pool_assignment = "test_pool"
        session.commit()
    return label_id


def _insert_auto_labeled_example(
    engine,
    project_id: str,
    example_key: str,
    guidance_id: str,
    label_json: dict,
) -> str:
    """Insert a synthetic Auto-Labeled Example + Label + OperationRecord."""
    inv_id = generate_uuid4()
    label_id = generate_uuid4()
    now = utc_now()

    with Session(engine) as session:
        op = OperationRecord(
            inference_invocation_id=inv_id,
            project_id=project_id,
            purpose="batch_label",
            example_key=example_key,
            guidance_id=guidance_id,
            invocation_status="success",
            normalized_json_ref=json.dumps(label_json),
        )
        session.add(op)

        ex = Example(
            example_key=example_key,
            project_id=project_id,
            storage_ref=f"/fake/{example_key}.jpg",
            ingested_at=now,
            source_metadata={},
            state="Auto-Labeled",
            clip_embedding_present=False,
        )
        session.add(ex)

        lbl = Label(
            label_id=label_id,
            project_id=project_id,
            example_key=example_key,
            label_status="auto_labeled",
            guidance_id=guidance_id,
            inference_invocation_id=inv_id,
            label_json=label_json,
            labeled_at=now,
        )
        session.add(lbl)
        session.commit()

    return label_id


def _insert_omitted_example(engine, project_id: str, example_key: str) -> None:
    """Insert a synthetic Omitted Example (no label)."""
    now = utc_now()
    with Session(engine) as session:
        ex = Example(
            example_key=example_key,
            project_id=project_id,
            storage_ref=f"/fake/{example_key}.jpg",
            ingested_at=now,
            source_metadata={},
            state="Omitted",
            omitted_source="sme_skip",
            omitted_at=now,
            clip_embedding_present=False,
        )
        session.add(ex)
        session.commit()


def _insert_run(
    engine,
    project_id: str,
    guidance_id: str,
    run_type: str = "evaluation_run",
    status: str = "running",
    **kw,
) -> str:
    """Insert a synthetic RunRecord. Returns run_id."""
    run_id = generate_uuid4()
    with Session(engine) as session:
        run = RunRecord(
            run_id=run_id,
            project_id=project_id,
            run_type=run_type,
            status=status,
            guidance_id=guidance_id,
            recovered_from_restart=False,
            **kw,
        )
        session.add(run)
        session.commit()
    return run_id


@contextmanager
def _registered_cancel_event(service, run_id: str):
    """Install an asyncio.Event in the service's cancel registry; pop on exit.

    Guidance edits signal canceled runs via the owning service's
    in-process ``_cancel_events`` — tests install an Event there to
    observe the signal, and must remove it even on assertion failure.
    """
    evt = asyncio.Event()
    service._cancel_events[run_id] = evt
    try:
        yield evt
    finally:
        service._cancel_events.pop(run_id, None)


def _insert_pool_snapshot(
    engine, project_id: str, guidance_id: str, keys: list[str]
) -> str:
    """Insert a Pool snapshot. Returns pool_id."""
    pool_id = generate_uuid4()
    with Session(engine) as session:
        pool = Pool(
            pool_id=pool_id,
            project_id=project_id,
            pool_type="test_pool",
            pool_version=1,
            member_example_keys=keys,
            member_count=len(keys),
            guidance_id=guidance_id,
        )
        session.add(pool)
        session.commit()
    return pool_id


def _edit_guidance(
    client: TestClient,
    project_id: str,
    schema: list[dict],
    description: str = "Classify damage.",
    rules: str = "",
    dry_run: bool = False,
    context_key: str | None = None,
) -> dict:
    """Call the :edit endpoint. Returns response JSON."""
    body: dict = {
        "description": description,
        "schema": schema,
        "rules": rules,
        "dry_run": dry_run,
    }
    if context_key is not None:
        body["schema_change_context_example_key"] = context_key
    resp = client.post(f"/v1/projects/{project_id}/guidance:edit", json=body)
    return resp.json(), resp.status_code


def _get_field_ids(guidance_data: dict) -> dict[str, str]:
    """Extract {field_name: field_id} from a GuidanceResponse."""
    return {f["field_name"]: f["field_id"] for f in guidance_data["schema_fields"]}


# ══════════════════════════════════════════════════════════════════════════════
# In-place propagation — renames rewrite labels without invalidating them
# ══════════════════════════════════════════════════════════════════════════════


class TestInPlacePropagation:
    """In-place edits propagate without label invalidation."""

    def test_core_field_rename_propagates(self, test_app_client: TestClient):
        """Core field rename propagates to all Verified labels."""
        pid = _create_project(test_app_client)
        g = _create_guidance_via_api(test_app_client, pid)
        gid = g["guidance_id"]
        _set_active_guidance(test_app_client, pid, gid)
        field_ids = _get_field_ids(g)

        engine = _get_engine(test_app_client, pid)
        _insert_verified_example(
            engine,
            pid,
            "img1",
            gid,
            {"rationale_note": "visible crack", "damage_type": "crack"},
        )

        # Edit: rename damage_type → defect_type (preserve field_id)
        old_fid = field_ids["damage_type"]
        new_schema = [
            {
                "field_name": "defect_type",
                "type": "enum",
                "role": "core",
                "allowed_values": ["crack", "dent"],
                "display_order": 1,
                "field_id": old_fid,
            },
        ]
        data, status = _edit_guidance(test_app_client, pid, new_schema)
        assert status == 200, data
        assert data["edit_type"] == "in_place"
        assert data["verified_reverted_count"] == 0

        # Label should have the renamed key
        with Session(engine) as s:
            lbl = s.query(Label).filter_by(project_id=pid, example_key="img1").one()
            assert "defect_type" in lbl.label_json
            assert "damage_type" not in lbl.label_json
            # Example still Verified
            ex = s.query(Example).filter_by(example_key="img1").one()
            assert ex.state == "Verified"

    def test_swapped_field_names_propagate_without_data_loss(
        self, test_app_client: TestClient
    ):
        """Renames resolve against the pre-edit label, not one at a time:
        an edit that swaps two field names (A<->B, one legal in_place edit)
        must move each value to its new name — sequential application
        destroyed one value and left the other under the wrong name."""
        pid = _create_project(test_app_client)
        schema = [
            {
                "field_name": "color",
                "type": "enum",
                "role": "core",
                "allowed_values": ["red", "blue"],
                "display_order": 1,
            },
            {
                "field_name": "shade",
                "type": "enum",
                "role": "core",
                "allowed_values": ["red", "blue"],
                "display_order": 2,
            },
        ]
        g = _create_guidance_via_api(test_app_client, pid, schema=schema)
        gid = g["guidance_id"]
        _set_active_guidance(test_app_client, pid, gid)
        field_ids = _get_field_ids(g)

        engine = _get_engine(test_app_client, pid)
        _insert_verified_example(
            engine,
            pid,
            "img1",
            gid,
            {"rationale_note": "ok", "color": "red", "shade": "blue"},
        )

        # Swap the names, keeping each field_id (in_place rename x2).
        new_schema = [
            {
                "field_name": "shade",
                "type": "enum",
                "role": "core",
                "allowed_values": ["red", "blue"],
                "display_order": 1,
                "field_id": field_ids["color"],
            },
            {
                "field_name": "color",
                "type": "enum",
                "role": "core",
                "allowed_values": ["red", "blue"],
                "display_order": 2,
                "field_id": field_ids["shade"],
            },
        ]
        data, status = _edit_guidance(test_app_client, pid, new_schema)
        assert status == 200, data
        assert data["edit_type"] == "in_place"

        with Session(engine) as s:
            lbl = s.query(Label).filter_by(project_id=pid, example_key="img1").one()
            assert lbl.label_json["shade"] == "red"  # followed field_id of color
            assert lbl.label_json["color"] == "blue"  # followed field_id of shade

    def test_rename_onto_removed_aux_name_keeps_the_core_value(
        self, test_app_client: TestClient
    ):
        """One edit may remove an aux field and rename a core field onto
        the freed name. The renamed core value must win over the retired
        key still sitting in the stored label — not lose to whichever key
        iterates last."""
        pid = _create_project(test_app_client)
        schema = [
            {
                "field_name": "quality",
                "type": "enum",
                "role": "core",
                "allowed_values": ["good", "bad"],
                "display_order": 1,
            },
            {
                "field_name": "condition",
                "type": "string",
                "role": "aux",
                "display_order": 2,
            },
        ]
        g = _create_guidance_via_api(test_app_client, pid, schema=schema)
        gid = g["guidance_id"]
        _set_active_guidance(test_app_client, pid, gid)
        field_ids = _get_field_ids(g)

        engine = _get_engine(test_app_client, pid)
        _insert_verified_example(
            engine,
            pid,
            "img1",
            gid,
            {
                "rationale_note": "ok",
                "quality": "good",
                "condition": "worn, scratched lid",
            },
        )

        # Remove aux 'condition'; rename core 'quality' -> 'condition'.
        new_schema = [
            {
                "field_name": "condition",
                "type": "enum",
                "role": "core",
                "allowed_values": ["good", "bad"],
                "display_order": 1,
                "field_id": field_ids["quality"],
            },
        ]
        data, status = _edit_guidance(test_app_client, pid, new_schema)
        assert status == 200, data
        assert data["edit_type"] == "in_place"

        with Session(engine) as s:
            lbl = s.query(Label).filter_by(project_id=pid, example_key="img1").one()
            assert lbl.label_json["condition"] == "good"

    def test_enum_value_rename_propagates(self, test_app_client: TestClient):
        """Exact 1:1 enum value rename propagates."""
        pid = _create_project(test_app_client)
        g = _create_guidance_via_api(test_app_client, pid)
        gid = g["guidance_id"]
        _set_active_guidance(test_app_client, pid, gid)
        field_ids = _get_field_ids(g)

        engine = _get_engine(test_app_client, pid)
        _insert_verified_example(
            engine,
            pid,
            "img1",
            gid,
            {"rationale_note": "visible crack", "damage_type": "crack"},
        )

        old_fid = field_ids["damage_type"]
        new_schema = [
            {
                "field_name": "damage_type",
                "type": "enum",
                "role": "core",
                "allowed_values": ["fracture", "dent"],
                "display_order": 1,
                "field_id": old_fid,
            },
        ]
        data, status = _edit_guidance(test_app_client, pid, new_schema)
        assert status == 200, data
        assert data["edit_type"] == "in_place"

        with Session(engine) as s:
            lbl = s.query(Label).filter_by(project_id=pid, example_key="img1").one()
            assert lbl.label_json["damage_type"] == "fracture"

    def test_display_order_change_no_invalidation(self, test_app_client: TestClient):
        """Presentation metadata change does not invalidate."""
        pid = _create_project(test_app_client)
        g = _create_guidance_via_api(test_app_client, pid)
        gid = g["guidance_id"]
        _set_active_guidance(test_app_client, pid, gid)
        field_ids = _get_field_ids(g)

        engine = _get_engine(test_app_client, pid)
        _insert_verified_example(
            engine,
            pid,
            "img1",
            gid,
            {"rationale_note": "ok", "damage_type": "crack"},
        )

        old_fid = field_ids["damage_type"]
        new_schema = [
            {
                "field_name": "damage_type",
                "type": "enum",
                "role": "core",
                "allowed_values": ["crack", "dent"],
                "display_order": 99,
                "field_id": old_fid,
            },
        ]
        data, status = _edit_guidance(test_app_client, pid, new_schema)
        assert status == 200, data
        assert data["edit_type"] == "in_place"
        assert data["verified_reverted_count"] == 0


class TestEditDescriptionOptional:
    """The Task Description is optional on guidance:edit — empty descriptions save; ≥1 valid Core field is the only structural requirement."""

    def _same_schema(self, guidance_data: dict) -> list[dict]:
        """Rebuild the current schema preserving field_ids (an unchanged schema)."""
        old_fid = _get_field_ids(guidance_data)["damage_type"]
        return [
            {
                "field_name": "damage_type",
                "type": "enum",
                "role": "core",
                "allowed_values": ["crack", "dent"],
                "display_order": 1,
                "field_id": old_fid,
            },
        ]

    def test_successful_edit_checks_backend_auto_evaluate_triggers(
        self, test_app_client: TestClient
    ):
        """A new active Guidance version feeds backend Auto-Evaluate."""
        pid = _create_project(test_app_client)
        guidance = _create_guidance_via_api(test_app_client, pid)
        _set_active_guidance(test_app_client, pid, guidance["guidance_id"])

        with patch(
            "vlm_feedback_loop.services.evaluation_service.maybe_start_auto_evaluation",
            new_callable=AsyncMock,
        ) as maybe_start:
            data, status = _edit_guidance(
                test_app_client,
                pid,
                self._same_schema(guidance),
            )

        assert status == 200, data
        maybe_start.assert_awaited_once()
        assert maybe_start.await_args.args[0] == pid

    def test_in_place_rename_cancels_active_batch_run(
        self, test_app_client: TestClient
    ):
        """An in-place enum/field rename rewrites the labels that exist now,
        but an in-flight batch run keeps writing under its old snapshot —
        those later labels would carry the retired names. The rename must
        fail active runs so their output cannot straddle it."""
        pid = _create_project(test_app_client)
        g = _create_guidance_via_api(test_app_client, pid)
        gid = g["guidance_id"]
        _set_active_guidance(test_app_client, pid, gid)
        field_ids = _get_field_ids(g)
        engine = _get_engine(test_app_client, pid)
        _insert_verified_example(
            engine, pid, "img1", gid, {"rationale_note": "x", "damage_type": "crack"}
        )
        run_id = _insert_run(
            engine, pid, gid, run_type="batch_label_run", status="running"
        )

        with _registered_cancel_event(batch_label_service, run_id) as evt:
            # Rename the enum value crack → fracture (1:1, in-place).
            new_schema = [
                {
                    "field_name": "damage_type",
                    "type": "enum",
                    "role": "core",
                    "allowed_values": ["fracture", "dent"],
                    "display_order": 1,
                    "field_id": field_ids["damage_type"],
                }
            ]
            data, status = _edit_guidance(test_app_client, pid, new_schema)
            assert status == 200, data
            assert data["edit_type"] == "in_place"
            assert evt.is_set()
        with Session(engine) as s:
            run = s.query(RunRecord).filter_by(run_id=run_id).one()
            assert run.status == "failed"
            assert run.status_reason == "guidance_edited_during_run"

    def test_no_change_edit_cancels_active_batch_run(self, test_app_client: TestClient):
        """Every Guidance edit (even a Description-only no_change) re-points
        existing labels and makes a new version active, so an in-flight
        run would keep stamping the old version and orphan its post-edit
        output. Non-semantic edits stop active runs with a distinct reason
        (their already-written labels are kept, unlike the semantic wipe)."""
        pid = _create_project(test_app_client)
        g = _create_guidance_via_api(test_app_client, pid)
        gid = g["guidance_id"]
        _set_active_guidance(test_app_client, pid, gid)
        engine = _get_engine(test_app_client, pid)
        run_id = _insert_run(
            engine, pid, gid, run_type="batch_label_run", status="running"
        )

        with _registered_cancel_event(batch_label_service, run_id) as evt:
            data, status = _edit_guidance(
                test_app_client, pid, self._same_schema(g), description="Reworded."
            )
            assert status == 200, data
            assert data["edit_type"] == "no_change"
            assert evt.is_set()
        with Session(engine) as s:
            run = s.query(RunRecord).filter_by(run_id=run_id).one()
            assert run.status == "failed"
            assert run.status_reason == "guidance_edited_during_run"

    def test_dry_run_accepts_empty_description(self, test_app_client: TestClient):
        """A dry_run edit with an empty description and valid Core fields validates clean (no rejection)."""
        pid = _create_project(test_app_client)
        g = _create_guidance_via_api(test_app_client, pid)
        _set_active_guidance(test_app_client, pid, g["guidance_id"])

        data, status = _edit_guidance(
            test_app_client, pid, self._same_schema(g), description="", dry_run=True
        )
        assert status == 200, data
        # Schema unchanged → classified no_change; the point is validation passed.
        assert data["edit_type"] == "no_change"

    def test_no_change_edit_repoints_labels_to_new_version(
        self, test_app_client: TestClient
    ):
        """A Description/Rules-only edit mints a new active Guidance
        version but the SchemaCore is unchanged, so every existing Label
        must be re-pointed to it — otherwise the whole labeled corpus
        orphans from the active guidance (ICL, evaluation, exports all
        filter Label.guidance_id == active_guidance_id)."""
        pid = _create_project(test_app_client)
        g = _create_guidance_via_api(test_app_client, pid)
        gid = g["guidance_id"]
        _set_active_guidance(test_app_client, pid, gid)
        engine = _get_engine(test_app_client, pid)
        _insert_verified_example(
            engine,
            pid,
            "img1",
            gid,
            {"rationale_note": "ok", "damage_type": "crack"},
        )

        data, status = _edit_guidance(
            test_app_client, pid, self._same_schema(g), description="Reworded."
        )
        assert status == 200, data
        assert data["edit_type"] == "no_change"
        new_gid = data["guidance"]["guidance_id"]

        from vlm_feedback_loop.db.models.label import Label
        from vlm_feedback_loop.db.models.project import Project

        with Session(engine) as s:
            proj = s.query(Project).filter_by(project_id=pid).one()
            assert proj.active_guidance_id == new_gid
            labels = s.query(Label).filter_by(project_id=pid).all()
            assert labels, "sanity: the verified label exists"
            orphaned = [
                lbl.example_key
                for lbl in labels
                if lbl.guidance_id != proj.active_guidance_id
            ]
            assert orphaned == [], f"labels orphaned from active guidance: {orphaned}"

    def test_execute_persists_empty_description(self, test_app_client: TestClient):
        """Executing an edit with an empty description persists it as the new active version."""
        pid = _create_project(test_app_client)
        g = _create_guidance_via_api(test_app_client, pid)
        _set_active_guidance(test_app_client, pid, g["guidance_id"])

        data, status = _edit_guidance(
            test_app_client, pid, self._same_schema(g), description=""
        )
        assert status == 200, data
        assert data["guidance"]["description"] == ""

        # Stored, not just echoed: re-read the new version
        new_gid = data["guidance"]["guidance_id"]
        resp = test_app_client.get(f"/v1/projects/{pid}/guidance/{new_gid}")
        assert resp.status_code == 200
        assert resp.json()["description"] == ""

    def test_edit_rejects_schema_with_no_core_fields(self, test_app_client: TestClient):
        """An edit whose schema has no Core fields is rejected with
        NO_CORE_FIELDS — the structural save requirement applies to
        guidance:edit exactly as it does to create."""
        pid = _create_project(test_app_client)
        g = _create_guidance_via_api(test_app_client, pid)
        _set_active_guidance(test_app_client, pid, g["guidance_id"])

        aux_only = [
            {
                "field_name": "observation",
                "type": "string",
                "role": "aux",
                "display_order": 1,
            },
        ]
        data, status = _edit_guidance(test_app_client, pid, aux_only)
        assert status == 400, data
        assert "NO_CORE_FIELDS" in str(data)


# ══════════════════════════════════════════════════════════════════════════════
# Semantic triggers — which Core edits invalidate Verified labels
# ══════════════════════════════════════════════════════════════════════════════


class TestSemanticTriggers:
    """Semantic edits trigger label invalidation."""

    @pytest.mark.parametrize(
        "new_fields",
        [
            pytest.param(
                [
                    {
                        "field_name": "damage_type",
                        "type": "enum",
                        "role": "core",
                        "allowed_values": ["crack", "dent"],
                        "display_order": 1,
                    },
                    {
                        "field_name": "severity",
                        "type": "integer",
                        "role": "core",
                        "minimum": 0,
                        "maximum": 5,
                        "display_order": 2,
                    },
                ],
                id="add_core_field",
            ),
            pytest.param(
                [
                    {
                        "field_name": "damage_type",
                        "type": "string",
                        "role": "core",
                        "display_order": 1,
                    },
                ],
                id="type_change",
            ),
            pytest.param(
                # Add a new value (not a 1:1 rename)
                [
                    {
                        "field_name": "damage_type",
                        "type": "enum",
                        "role": "core",
                        "allowed_values": ["crack", "dent", "scratch"],
                        "display_order": 1,
                    },
                ],
                id="allowed_value_change",
            ),
            pytest.param(
                # Demote Core to Aux
                [
                    {
                        "field_name": "damage_type",
                        "type": "enum",
                        "role": "aux",
                        "allowed_values": ["crack", "dent"],
                        "display_order": 1,
                    },
                    {
                        "field_name": "flag",
                        "type": "boolean",
                        "role": "core",
                        "display_order": 2,
                    },
                ],
                id="role_change",
            ),
        ],
    )
    def test_semantic_edit_triggers_invalidation(
        self, test_app_client: TestClient, new_fields: list[dict]
    ):
        """Adding a Core field, changing a Core field's type, changing its
        allowed values (beyond a 1:1 rename), or demoting it to Aux are
        all semantic edits: the Verified label is reverted."""
        pid = _create_project(test_app_client)
        g = _create_guidance_via_api(test_app_client, pid)
        gid = g["guidance_id"]
        _set_active_guidance(test_app_client, pid, gid)
        fids = _get_field_ids(g)

        engine = _get_engine(test_app_client, pid)
        _insert_verified_example(
            engine,
            pid,
            "img1",
            gid,
            {"rationale_note": "ok", "damage_type": "crack"},
        )

        # The mutated damage_type field keeps its field_id — without it
        # the edit would read as remove+add instead of a field mutation.
        new_schema = [
            {**new_fields[0], "field_id": fids["damage_type"]},
            *new_fields[1:],
        ]
        data, status = _edit_guidance(test_app_client, pid, new_schema)
        assert status == 200, data
        assert data["edit_type"] == "semantic"
        assert data["verified_reverted_count"] == 1

    def test_remove_core_field_triggers_invalidation(self, test_app_client: TestClient):
        pid = _create_project(test_app_client)
        schema = [
            {
                "field_name": "cat",
                "type": "enum",
                "role": "core",
                "allowed_values": ["a", "b"],
                "display_order": 1,
            },
            {
                "field_name": "sev",
                "type": "integer",
                "role": "core",
                "minimum": 0,
                "maximum": 5,
                "display_order": 2,
            },
        ]
        g = _create_guidance_via_api(test_app_client, pid, schema=schema)
        gid = g["guidance_id"]
        _set_active_guidance(test_app_client, pid, gid)
        fids = _get_field_ids(g)

        engine = _get_engine(test_app_client, pid)
        _insert_verified_example(
            engine,
            pid,
            "img1",
            gid,
            {"rationale_note": "ok", "cat": "a", "sev": 3},
        )

        # Remove sev field
        new_schema = [
            {
                "field_name": "cat",
                "type": "enum",
                "role": "core",
                "allowed_values": ["a", "b"],
                "display_order": 1,
                "field_id": fids["cat"],
            },
        ]
        data, status = _edit_guidance(test_app_client, pid, new_schema)
        assert status == 200
        assert data["edit_type"] == "semantic"

    def test_constraint_change_triggers_invalidation(self, test_app_client: TestClient):
        pid = _create_project(test_app_client)
        schema = [
            {
                "field_name": "score",
                "type": "integer",
                "role": "core",
                "minimum": 0,
                "maximum": 10,
                "display_order": 1,
            }
        ]
        g = _create_guidance_via_api(test_app_client, pid, schema=schema)
        gid = g["guidance_id"]
        _set_active_guidance(test_app_client, pid, gid)
        fids = _get_field_ids(g)

        engine = _get_engine(test_app_client, pid)
        _insert_verified_example(
            engine,
            pid,
            "img1",
            gid,
            {"rationale_note": "ok", "score": 5},
        )

        new_schema = [
            {
                "field_name": "score",
                "type": "integer",
                "role": "core",
                "minimum": 0,
                "maximum": 20,
                "display_order": 1,
                "field_id": fids["score"],
            }
        ]
        data, status = _edit_guidance(test_app_client, pid, new_schema)
        assert status == 200
        assert data["edit_type"] == "semantic"


# ══════════════════════════════════════════════════════════════════════════════
# Atomic evolution — what a semantic edit deletes, snapshots, and preserves
# ══════════════════════════════════════════════════════════════════════════════


class TestAtomicEvolution:
    """A semantic edit atomically invalidates labels."""

    def _trigger_semantic_edit(self, client):
        """Set up and trigger a semantic edit. Returns (pid, old_gid, engine, data)."""
        pid = _create_project(client)
        g = _create_guidance_via_api(client, pid)
        gid = g["guidance_id"]
        _set_active_guidance(client, pid, gid)
        fids = _get_field_ids(g)

        engine = _get_engine(client, pid)
        _insert_verified_example(
            engine,
            pid,
            "img1",
            gid,
            {"damage_type": "crack"},
            verified_outcome="Edit",
            edited_core_fields=["damage_type"],
            vlm_proposal_json=json.dumps({"damage_type": "dent"}),
        )
        _insert_verified_with_pool(
            engine,
            pid,
            "img2",
            gid,
            {"damage_type": "dent"},
        )

        # Add a new Core field → semantic
        new_schema = [
            {
                "field_name": "damage_type",
                "type": "enum",
                "role": "core",
                "allowed_values": ["crack", "dent"],
                "display_order": 1,
                "field_id": fids["damage_type"],
            },
            {
                "field_name": "severity",
                "type": "integer",
                "role": "core",
                "minimum": 0,
                "maximum": 5,
                "display_order": 2,
            },
        ]
        data, status = _edit_guidance(client, pid, new_schema)
        assert status == 200
        assert data["edit_type"] == "semantic"
        return pid, gid, engine, data

    def test_labels_deleted_and_examples_unlabeled(self, test_app_client: TestClient):
        """Labels are deleted and Examples transition to Unlabeled."""
        pid, _, engine, data = self._trigger_semantic_edit(test_app_client)
        assert data["verified_reverted_count"] == 2

        with Session(engine) as s:
            labels = s.query(Label).filter_by(project_id=pid).all()
            assert len(labels) == 0

            examples = s.query(Example).filter_by(project_id=pid).all()
            for ex in examples:
                assert ex.state == "Unlabeled"

    def test_semantic_core_change_from_persisted(self, test_app_client: TestClient):
        """semantic_core_change_from_guidance_id and schema_change_summary are persisted."""
        pid, old_gid, engine, data = self._trigger_semantic_edit(test_app_client)
        new_gid = data["guidance"]["guidance_id"]

        with Session(engine) as s:
            new_g = s.query(Guidance).filter_by(guidance_id=new_gid).one()
            assert new_g.semantic_core_change_from_guidance_id == old_gid
            assert new_g.schema_change_summary is not None
            assert new_g.schema_change_summary["total_changes"] > 0

    def test_prior_verified_label_ref_snapshot(self, test_app_client: TestClient):
        """prior_verified_label_ref is self-contained JSON."""
        pid, _, engine, _ = self._trigger_semantic_edit(test_app_client)

        with Session(engine) as s:
            ex = s.query(Example).filter_by(example_key="img1").one()
            assert ex.prior_verified_label_ref is not None
            snapshot = json.loads(ex.prior_verified_label_ref)

            assert "label_json" in snapshot
            assert "vlm_proposal_json" in snapshot
            assert "edited_core_fields" in snapshot
            assert "edited_aux_fields" in snapshot
            assert "verified_outcome" in snapshot
            assert "rationale_note" not in snapshot
            assert "guidance_id" in snapshot

            # Specific values
            assert snapshot["verified_outcome"] == "Edit"
            assert snapshot["edited_core_fields"] == ["damage_type"]
            assert snapshot["label_json"]["damage_type"] == "crack"
            # VLM proposal was the original
            vlm = json.loads(snapshot["vlm_proposal_json"])
            assert vlm["damage_type"] == "dent"

    def test_prior_verified_outcome_set(self, test_app_client: TestClient):
        """prior_verified_outcome is preserved on the Example."""
        pid, _, engine, _ = self._trigger_semantic_edit(test_app_client)

        with Session(engine) as s:
            ex1 = s.query(Example).filter_by(example_key="img1").one()
            assert ex1.prior_verified_outcome == "Edit"
            ex2 = s.query(Example).filter_by(example_key="img2").one()
            assert ex2.prior_verified_outcome == "Accept"

    def test_pool_clears(self, test_app_client: TestClient):
        """The Test Pool clears (Labels with pool_assignment deleted)."""
        pid, _, engine, _ = self._trigger_semantic_edit(test_app_client)

        with Session(engine) as s:
            pool_labels = (
                s.query(Label)
                .filter_by(project_id=pid)
                .filter(Label.pool_assignment.isnot(None))
                .all()
            )
            assert len(pool_labels) == 0

    def test_no_stale_labels(self, test_app_client: TestClient):
        """Old labels are deleted — no stale labels remain."""
        pid, old_gid, engine, _ = self._trigger_semantic_edit(test_app_client)

        with Session(engine) as s:
            old_labels = s.query(Label).filter_by(guidance_id=old_gid).all()
            assert len(old_labels) == 0


# ══════════════════════════════════════════════════════════════════════════════
# Auto-Labeled evolution — how Auto-Labeled examples are handled
# ══════════════════════════════════════════════════════════════════════════════


class TestAutoLabeledEvolution:
    """Auto-Labeled examples are handled during evolution."""

    def test_auto_labeled_transition_to_unlabeled(self, test_app_client: TestClient):
        """Auto-Labeled → Unlabeled, OperationRecords preserved."""
        pid = _create_project(test_app_client)
        g = _create_guidance_via_api(test_app_client, pid)
        gid = g["guidance_id"]
        _set_active_guidance(test_app_client, pid, gid)
        fids = _get_field_ids(g)

        engine = _get_engine(test_app_client, pid)
        # Need at least one verified for a valid project state
        _insert_verified_example(
            engine,
            pid,
            "img_v",
            gid,
            {"rationale_note": "ok", "damage_type": "crack"},
        )
        _insert_auto_labeled_example(
            engine,
            pid,
            "img_a",
            gid,
            {"rationale_note": "auto", "damage_type": "dent"},
        )

        # Semantic edit: add Core field
        new_schema = [
            {
                "field_name": "damage_type",
                "type": "enum",
                "role": "core",
                "allowed_values": ["crack", "dent"],
                "display_order": 1,
                "field_id": fids["damage_type"],
            },
            {
                "field_name": "severity",
                "type": "integer",
                "role": "core",
                "minimum": 0,
                "maximum": 5,
                "display_order": 2,
            },
        ]
        data, status = _edit_guidance(test_app_client, pid, new_schema)
        assert status == 200
        assert data["auto_labeled_reverted_count"] == 1

        with Session(engine) as s:
            ex = s.query(Example).filter_by(example_key="img_a").one()
            assert ex.state == "Unlabeled"
            assert (
                s.query(Label).filter_by(project_id=pid, example_key="img_a").count()
                == 0
            )

            # OperationRecords preserved for audit
            ops = (
                s.query(OperationRecord)
                .filter_by(project_id=pid, example_key="img_a")
                .all()
            )
            assert len(ops) >= 1

    def test_dry_run_shows_auto_labeled_count(self, test_app_client: TestClient):
        """dry_run returns auto_labeled_count for the confirmation dialog."""
        pid = _create_project(test_app_client)
        g = _create_guidance_via_api(test_app_client, pid)
        gid = g["guidance_id"]
        _set_active_guidance(test_app_client, pid, gid)
        fids = _get_field_ids(g)

        engine = _get_engine(test_app_client, pid)
        _insert_verified_example(
            engine,
            pid,
            "img_v",
            gid,
            {"rationale_note": "ok", "damage_type": "crack"},
        )
        _insert_auto_labeled_example(
            engine,
            pid,
            "img_a1",
            gid,
            {"rationale_note": "auto1", "damage_type": "crack"},
        )
        _insert_auto_labeled_example(
            engine,
            pid,
            "img_a2",
            gid,
            {"rationale_note": "auto2", "damage_type": "dent"},
        )

        new_schema = [
            {
                "field_name": "damage_type",
                "type": "enum",
                "role": "core",
                "allowed_values": ["crack", "dent"],
                "display_order": 1,
                "field_id": fids["damage_type"],
            },
            {
                "field_name": "new_field",
                "type": "boolean",
                "role": "core",
                "display_order": 2,
            },
        ]
        data, status = _edit_guidance(test_app_client, pid, new_schema, dry_run=True)
        assert status == 200
        assert data["edit_type"] == "semantic"
        assert data["verified_count"] == 1
        assert data["auto_labeled_count"] == 2


# ══════════════════════════════════════════════════════════════════════════════
# Evolution resets — per-project state cleared by a semantic edit
# ══════════════════════════════════════════════════════════════════════════════


class TestEvolutionResets:
    """The per-project state resets a semantic edit triggers."""

    def _setup(self, client):
        """Create a project carrying every kind of state a semantic edit
        resets: trigger counters, a verified + omitted example, a running
        evaluation."""
        pid = _create_project(client)
        g = _create_guidance_via_api(client, pid)
        gid = g["guidance_id"]
        _set_active_guidance(client, pid, gid)
        fids = _get_field_ids(g)

        engine = _get_engine(client, pid)

        with Session(engine) as s:
            proj = s.query(Project).filter_by(project_id=pid).one()
            proj.icl_recommendation_dismissed_at_count = 5
            proj.schema_refinement_reminders_dismissed = 2
            proj.review_selector_scheduler_state = {"recent": ["img0"]}
            s.commit()

        _insert_verified_example(
            engine,
            pid,
            "img1",
            gid,
            {"rationale_note": "ok", "damage_type": "crack"},
        )
        _insert_omitted_example(engine, pid, "img_omit")

        eval_run_id = _insert_run(engine, pid, gid)
        return pid, gid, engine, eval_run_id, fids

    @staticmethod
    def _trigger_semantic_edit(client, pid, fids):
        """Change severity's type (string → integer): a semantic edit."""
        new_schema = [
            {
                "field_name": "damage_type",
                "type": "enum",
                "role": "core",
                "allowed_values": ["crack", "dent"],
                "display_order": 1,
                "field_id": fids["damage_type"],
            },
            {
                "field_name": "severity",
                "type": "integer",
                "role": "core",
                "minimum": 0,
                "maximum": 5,
                "display_order": 2,
            },
        ]
        data, status = _edit_guidance(client, pid, new_schema, context_key="img1")
        assert status == 200
        assert data["edit_type"] == "semantic"

    def _setup_and_trigger(self, client):
        """Create project with various state, trigger semantic edit."""
        pid, gid, engine, eval_run_id, fids = self._setup(client)
        self._trigger_semantic_edit(client, pid, fids)
        return pid, gid, engine, eval_run_id

    def test_icl_recommendation_dismissed_reset(self, test_app_client: TestClient):
        """icl_recommendation_dismissed_at_count → 0."""
        pid, _, engine, _ = self._setup_and_trigger(test_app_client)
        with Session(engine) as s:
            proj = s.query(Project).filter_by(project_id=pid).one()
            assert proj.icl_recommendation_dismissed_at_count == 0

    def test_in_progress_evaluation_canceled(self, test_app_client: TestClient):
        """An in-progress evaluation is canceled."""
        pid, _, engine, eval_run_id = self._setup_and_trigger(test_app_client)
        with Session(engine) as s:
            run = s.query(RunRecord).filter_by(run_id=eval_run_id).one()
            assert run.status == "failed"
            assert run.status_reason == "schema_evolution_canceled"

    def test_semantic_edit_cancels_active_batch_run_and_signals(
        self, test_app_client: TestClient
    ):
        """A mid-flight batch run writes Auto-Labeled rows under the schema
        being wiped; the semantic edit must fail it (like evaluations)
        and signal its live executor so it stops invoking the Teacher."""
        client = test_app_client
        pid = _create_project(client)
        g = _create_guidance_via_api(client, pid)
        gid = g["guidance_id"]
        _set_active_guidance(client, pid, gid)
        fids = _get_field_ids(g)
        engine = _get_engine(client, pid)
        _insert_verified_example(
            engine,
            pid,
            "img1",
            gid,
            {"rationale_note": "ok", "damage_type": "crack"},
        )
        batch_run_id = _insert_run(
            engine, pid, gid, run_type="batch_label_run", status="running"
        )

        with _registered_cancel_event(batch_label_service, batch_run_id) as evt:
            self._trigger_semantic_edit(client, pid, fids)
            assert evt.is_set()
        with Session(engine) as s:
            run = s.query(RunRecord).filter_by(run_id=batch_run_id).one()
            assert run.status == "failed"
            assert run.status_reason == "schema_evolution_canceled"

    def test_semantic_edit_fails_paused_batch_runs_too(
        self, test_app_client: TestClient
    ):
        """A paused batch run would RESUME under the new schema (resume only
        requires status=='paused') — the semantic edit must fail it along
        with the active ones."""
        client = test_app_client
        pid = _create_project(client)
        g = _create_guidance_via_api(client, pid)
        gid = g["guidance_id"]
        _set_active_guidance(client, pid, gid)
        fids = _get_field_ids(g)
        engine = _get_engine(client, pid)
        _insert_verified_example(
            engine,
            pid,
            "img1",
            gid,
            {"rationale_note": "ok", "damage_type": "crack"},
        )
        run_id = _insert_run(
            engine,
            pid,
            gid,
            run_type="batch_label_run",
            status="paused",
            paused_reason="circuit_breaker_threshold_reached",
        )
        self._trigger_semantic_edit(client, pid, fids)
        with Session(engine) as s:
            run = s.query(RunRecord).filter_by(run_id=run_id).one()
            assert run.status == "failed"
            assert run.status_reason == "schema_evolution_canceled"

    def test_semantic_edit_signals_live_executor_cancel_event(
        self, test_app_client: TestClient
    ):
        """The DB flip to failed alone does not stop a live executor task
        — it would keep invoking the Teacher for the wiped Guidance and
        its finalizer used to resurrect the failed row to 'completed'.
        execute_edit must signal the run's in-process cancel event after
        its transaction commits."""
        pid, _, _, eval_run_id, fids = self._setup(test_app_client)
        with _registered_cancel_event(evaluation_service, eval_run_id) as evt:
            self._trigger_semantic_edit(test_app_client, pid, fids)
            assert evt.is_set()

    def test_selector_state_reinitialized(self, test_app_client: TestClient):
        """review_selector_scheduler_state is reset."""
        pid, _, engine, _ = self._setup_and_trigger(test_app_client)
        with Session(engine) as s:
            proj = s.query(Project).filter_by(project_id=pid).one()
            assert proj.review_selector_scheduler_state is None

    def test_schema_refinement_reminders_reset(self, test_app_client: TestClient):
        """schema_refinement_reminders_dismissed → 0."""
        pid, _, engine, _ = self._setup_and_trigger(test_app_client)
        with Session(engine) as s:
            proj = s.query(Project).filter_by(project_id=pid).one()
            assert proj.schema_refinement_reminders_dismissed == 0

    def test_schema_change_context_example_key_recorded(
        self, test_app_client: TestClient
    ):
        """schema_change_context_example_key is recorded."""
        pid, _, engine, _ = self._setup_and_trigger(test_app_client)
        with Session(engine) as s:
            proj = s.query(Project).filter_by(project_id=pid).one()
            assert proj.schema_change_context_example_key == "img1"

    def test_prior_run_records_preserved(self, test_app_client: TestClient):
        """Prior Run Records remain in history for audit."""
        pid, _, engine, eval_run_id = self._setup_and_trigger(test_app_client)
        with Session(engine) as s:
            run = s.query(RunRecord).filter_by(run_id=eval_run_id).one()
            assert run is not None  # Still exists, just status=failed

    def test_omitted_examples_unaffected(self, test_app_client: TestClient):
        """Omitted examples remain Omitted."""
        pid, _, engine, _ = self._setup_and_trigger(test_app_client)
        with Session(engine) as s:
            ex = s.query(Example).filter_by(example_key="img_omit").one()
            assert ex.state == "Omitted"
            assert ex.omitted_source == "sme_skip"
            assert ex.omitted_at is not None


# ══════════════════════════════════════════════════════════════════════════════
# Additional: field_id immutability, Aux extensions, no-change edit
# ══════════════════════════════════════════════════════════════════════════════


class TestAdditional:
    def test_field_id_preserved_across_edit(self, test_app_client: TestClient):
        """field_id immutable: preserved across in-place rename."""
        pid = _create_project(test_app_client)
        g = _create_guidance_via_api(test_app_client, pid)
        gid = g["guidance_id"]
        _set_active_guidance(test_app_client, pid, gid)
        fids = _get_field_ids(g)
        old_fid = fids["damage_type"]

        new_schema = [
            {
                "field_name": "defect_type",
                "type": "enum",
                "role": "core",
                "allowed_values": ["crack", "dent"],
                "display_order": 1,
                "field_id": old_fid,
            },
        ]
        data, status = _edit_guidance(test_app_client, pid, new_schema)
        assert status == 200
        new_fids = {
            f["field_name"]: f["field_id"] for f in data["guidance"]["schema_fields"]
        }
        assert new_fids["defect_type"] == old_fid

    def test_add_aux_field_in_place(self, test_app_client: TestClient):
        """Aux extension: adding an Aux field is in-place."""
        pid = _create_project(test_app_client)
        g = _create_guidance_via_api(test_app_client, pid)
        gid = g["guidance_id"]
        _set_active_guidance(test_app_client, pid, gid)
        fids = _get_field_ids(g)

        engine = _get_engine(test_app_client, pid)
        _insert_verified_example(
            engine,
            pid,
            "img1",
            gid,
            {"rationale_note": "ok", "damage_type": "crack"},
        )

        new_schema = [
            {
                "field_name": "damage_type",
                "type": "enum",
                "role": "core",
                "allowed_values": ["crack", "dent"],
                "display_order": 1,
                "field_id": fids["damage_type"],
            },
            {
                "field_name": "observation",
                "type": "string",
                "role": "aux",
                "display_order": 0,
            },
        ]
        data, status = _edit_guidance(test_app_client, pid, new_schema)
        assert status == 200
        assert data["edit_type"] == "in_place"
        assert data["verified_reverted_count"] == 0

    def test_rationale_toggle_is_in_place_and_preserves_verified_labels(
        self, test_app_client: TestClient
    ):
        """Enabling or disabling the reserved Aux field never invalidates labels."""
        pid = _create_project(test_app_client)
        guidance = _create_guidance_via_api(test_app_client, pid)
        gid = guidance["guidance_id"]
        _set_active_guidance(test_app_client, pid, gid)
        field_ids = _get_field_ids(guidance)
        engine = _get_engine(test_app_client, pid)
        _insert_verified_example(
            engine,
            pid,
            "img1",
            gid,
            {"damage_type": "crack"},
        )

        core_field = {
            "field_name": "damage_type",
            "type": "enum",
            "role": "core",
            "allowed_values": ["crack", "dent"],
            "display_order": 1,
            "field_id": field_ids["damage_type"],
        }
        enabled, status = _edit_guidance(
            test_app_client,
            pid,
            [
                core_field,
                {
                    "field_name": "rationale_note",
                    "type": "string",
                    "role": "aux",
                    "display_order": 0,
                },
            ],
        )
        assert status == 200
        assert enabled["edit_type"] == "in_place"
        assert enabled["verified_reverted_count"] == 0

        enabled_field_ids = _get_field_ids(enabled["guidance"])
        disabled_core = {**core_field, "field_id": enabled_field_ids["damage_type"]}
        disabled, status = _edit_guidance(test_app_client, pid, [disabled_core])
        assert status == 200
        assert disabled["edit_type"] == "in_place"
        assert disabled["verified_reverted_count"] == 0

        with Session(engine) as session:
            example = session.query(Example).filter_by(example_key="img1").one()
            label = session.query(Label).filter_by(example_key="img1").one()
            assert example.state == "Verified"
            assert label.label_status == "verified"
            assert label.label_json == {"damage_type": "crack"}

    def test_no_change_edit(self, test_app_client: TestClient):
        """Identical schema produces no_change."""
        pid = _create_project(test_app_client)
        g = _create_guidance_via_api(test_app_client, pid)
        gid = g["guidance_id"]
        _set_active_guidance(test_app_client, pid, gid)
        fids = _get_field_ids(g)

        # Same schema, same field_ids
        new_schema = [
            {
                "field_name": "damage_type",
                "type": "enum",
                "role": "core",
                "allowed_values": ["crack", "dent"],
                "display_order": 1,
                "field_id": fids["damage_type"],
            },
        ]
        data, status = _edit_guidance(test_app_client, pid, new_schema)
        assert status == 200
        assert data["edit_type"] == "no_change"

    def test_edit_requires_active_guidance(self, test_app_client: TestClient):
        """Edit fails when no active guidance is set."""
        pid = _create_project(test_app_client)
        # No guidance created, active_guidance_id is null
        data, status = _edit_guidance(test_app_client, pid, _core_enum_schema())
        assert status == 400
