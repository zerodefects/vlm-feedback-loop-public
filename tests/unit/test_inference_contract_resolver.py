# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the canonical Inference Contract derivation helper.

``resolve_training_inference_contract`` is the single derivation path
shared by Student registration, the Student NIM lifecycle, and the
deployment-handoff mismatch gate. These tests pin its lookup priority
(training-intent row > first listed row) and its tolerance of dangling
export references — the lifecycle suite covers only the
single-training-export and empty-list cases through its delegate.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from conftest import open_project_workspace
from vlm_feedback_loop.db.base import generate_uuid4
from vlm_feedback_loop.db.models.dataset_export import DatasetExport
from vlm_feedback_loop.services.inference_contract_resolver import (
    resolve_training_inference_contract,
)

PID = "proj-contract-resolver"


def _add_export(session: Session, *, intent: str, field_mode: str) -> str:
    export_id = generate_uuid4()
    session.add(
        DatasetExport(
            dataset_export_id=export_id,
            project_id=PID,
            dataset_intent=intent,
            export_field_mode=field_mode,
            guidance_id="g-1",
            label_tier_filter="verified",
            selection_definition_snapshot={},
            artifact_refs={},
            manifest_ref="manifest",
            example_count=10,
        )
    )
    return export_id


def test_training_export_wins_over_listed_first_non_training(tmp_path):
    """The training-intent export defines the Student's contract even when a
    non-training export appears earlier in ``dataset_export_ids``."""
    engine, _, _ = open_project_workspace(tmp_path, PID)
    with Session(engine) as session:
        test_id = _add_export(session, intent="testing", field_mode="all")
        train_id = _add_export(session, intent="training", field_mode="core_only")
        session.commit()

        contract = resolve_training_inference_contract(session, [test_id, train_id])

    assert contract["output_field_mode"] == "core_only"
    assert contract["icl_field_mode"] == "core_only"


def test_no_training_intent_falls_back_to_first_export(tmp_path):
    """With no training-intent row, the first listed export's field mode is
    used — not a silent reset to the Teacher's ``all``."""
    engine, _, _ = open_project_workspace(tmp_path, PID)
    with Session(engine) as session:
        first_id = _add_export(session, intent="testing", field_mode="aux_and_core")
        second_id = _add_export(session, intent="evaluation", field_mode="core_only")
        session.commit()

        contract = resolve_training_inference_contract(session, [first_id, second_id])

    assert contract["output_field_mode"] == "aux_and_core"
    assert contract["icl_field_mode"] == "aux_and_core"


def test_dangling_export_id_is_skipped_not_fatal(tmp_path):
    """A stale/deleted export reference ahead of the training export must not
    crash derivation or mask the training export's field mode."""
    engine, _, _ = open_project_workspace(tmp_path, PID)
    with Session(engine) as session:
        train_id = _add_export(session, intent="training", field_mode="core_only")
        session.commit()

        contract = resolve_training_inference_contract(
            session, [generate_uuid4(), train_id]
        )

    assert contract["output_field_mode"] == "core_only"
    assert contract["icl_field_mode"] == "core_only"
