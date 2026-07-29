# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the session-scoped dataset export helper.

Confirms that :func:`persist_dataset_export_in_session` honors
caller-owned transaction boundaries (no commit inside) and reports
a missing project as an error string.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from conftest import (
    add_endpoint_row,
    add_example_row,
    add_guidance_row,
    add_model_config_row,
    add_project_row,
    open_project_workspace,
)
from vlm_feedback_loop.db.base import generate_uuid4, utc_now
from vlm_feedback_loop.db.models.dataset_export import DatasetExport
from vlm_feedback_loop.db.models.label import Label
from vlm_feedback_loop.db.models.project import Project
from vlm_feedback_loop.services.dataset_export_service import (
    persist_dataset_export_in_session,
)

PID = "proj-internal"
GID = "g-internal"

FIX_FIELDS = [
    {
        "field_id": "f0",
        "field_name": "rationale_note",
        "type": "string",
        "role": "aux",
        "display_order": -1,
    },
    {
        "field_id": "f1",
        "field_name": "severity",
        "type": "enum",
        "role": "core",
        "display_order": 1,
        "allowed_values": ["low", "high"],
    },
]
SCHEMA_ENV = {
    "fields": FIX_FIELDS,
    "generation_order": ["rationale_note", "severity"],
    "derived_json_schema": {},
    "schema_hash": "x",
}


def _setup(tmp_path):
    engine, pdir, workspace = open_project_workspace(
        tmp_path, PID, register_engine=True, subdirs=("exports",)
    )

    with Session(engine) as s:
        add_project_row(s, PID, str(pdir), name="X", active_guidance_id=GID)
        add_guidance_row(s, PID, GID, SCHEMA_ENV, description="Classify severity.")
        add_endpoint_row(s, PID, "e", display_name="t", base_url="https://t/v1")
        add_model_config_row(s, PID, "mc", "e", model_name="m")
        # Seed one non-pool verified label.
        img = tmp_path / "x.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
        add_example_row(s, PID, "k1", storage_ref=str(img), state="Verified")
        s.add(
            Label(
                label_id=generate_uuid4(),
                project_id=PID,
                example_key="k1",
                label_status="verified",
                guidance_id=GID,
                inference_invocation_id=generate_uuid4(),
                label_json={"rationale_note": "r", "severity": "low"},
                labeled_at=utc_now(),
                verified_outcome="Accept",
                verified_at=utc_now(),
                edited_core_fields=[],
                edited_aux_fields=[],
                rationale_source="teacher_proposal",
            )
        )
        s.commit()
    return engine, workspace


class TestInternalHelper:
    def test_helper_without_commit_stages_rollback_safe(self, tmp_path):
        engine, workspace = _setup(tmp_path)

        # The helper stages the row but does NOT commit. A rollback must
        # leave zero DatasetExport rows.
        with Session(engine) as s:
            result = persist_dataset_export_in_session(
                s,
                PID,
                dataset_intent="training",
                label_tier_filter="verified_only",
                export_field_mode="all",
                workspace_root=str(workspace),
            )
            assert not isinstance(result, str), result
            s.rollback()

        with Session(engine) as s:
            assert s.query(DatasetExport).count() == 0

    def test_helper_returns_error_on_missing_project(self, tmp_path):
        engine, workspace = _setup(tmp_path)
        # Delete the project to force the error branch.
        with Session(engine) as s:
            s.query(Project).filter_by(project_id=PID).delete()
            s.commit()

        with Session(engine) as s:
            result = persist_dataset_export_in_session(
                s,
                PID,
                dataset_intent="training",
                label_tier_filter="verified_only",
                export_field_mode="all",
                workspace_root=str(workspace),
            )
        assert isinstance(result, str)
        assert "not found" in result.lower()
