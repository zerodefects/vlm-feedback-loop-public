# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cross-module pipeline integration smoke tests.

These tests pin down behaviors that emerge from interactions between
the Batch Labeling, schema-evolution, and foreground-priority-dispatch
subsystems — not the subsystems in isolation. Each one is the
fastest-to-fail signal for a specific cross-module invariant the
unit-level tests do not exercise:

  * **Auto-Labeled examples revert to Unlabeled on a semantic Core
    schema change.** Verifies the schema-evolution flow handles
    Auto-Labeled state transitions identically to Verified state
    transitions (the spec requires both).

  * **Batch Labeling honors foreground priority dispatch.** When an
    interactive proposal arrives mid-batch, the per-example dispatch
    must wait for the foreground request to clear. Evaluation already
    has equivalent coverage; this is the batch counterpart.

  * **Core field rename propagates through the labeling pipeline.**
    Renaming a Core field must update all Verified ``Label.label_json``
    entries, a dataset export's gpt turn, and the active Guidance
    schema — three downstream consumers of the same field name, all of
    which must observe the rename.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from conftest import (
    FIXTURE_FIELDS,
    add_endpoint_row,
    add_example_row,
    add_guidance_row,
    add_model_config_row,
    add_project_row,
    make_settings,
    open_project_workspace,
)
from vlm_feedback_loop.db.base import generate_uuid4, utc_now
from vlm_feedback_loop.db.models.example import Example
from vlm_feedback_loop.db.models.guidance import Guidance
from vlm_feedback_loop.db.models.label import Label
from vlm_feedback_loop.db.models.operation import OperationRecord
from vlm_feedback_loop.db.models.project import Project
from vlm_feedback_loop.db.models.run import RunRecord
from vlm_feedback_loop.services.batch_label_service import (
    BatchExampleResult,
    _execute_batch_label,
)
from vlm_feedback_loop.services.priority import ForegroundPriorityDispatch

# ── Shared fixtures ─────────────────────────────────────────────────────────

PID = "proj-pipeline-smoke"
GID = "guid-rerun"
MCID = "mc-rerun"
EID = "ep-rerun"

FIXTURE_SCHEMA = {
    "fields": FIXTURE_FIELDS,
    "generation_order": ["rationale_note", "severity", "damaged"],
    "derived_json_schema": {
        "type": "object",
        "properties": {
            "rationale_note": {"type": "string"},
            "severity": {"type": "string", "enum": ["low", "medium", "high"]},
            "damaged": {"type": "boolean"},
        },
        "required": ["severity", "damaged"],
        "additionalProperties": False,
    },
    "schema_hash": "rerun_hash",
}


def _setup_project(tmp_path: Path, n_unlabeled: int = 3):
    """Create a project with ready-to-batch state (gate ready, Verified train)."""
    engine, project_dir, workspace = open_project_workspace(
        tmp_path, PID, register_engine=True
    )

    with Session(engine) as s:
        add_project_row(
            s,
            PID,
            str(project_dir),
            name="Pipeline Integration Smoke",
            active_guidance_id=GID,
            teacher_model_config_id=MCID,
        )
        add_endpoint_row(s, PID, EID)
        add_model_config_row(s, PID, MCID, EID)
        add_guidance_row(s, PID, GID, FIXTURE_SCHEMA, description="Classify damage.")
        for i in range(n_unlabeled):
            add_example_row(s, PID, f"ex_{i:03d}")
        s.commit()

    settings = make_settings(workspace)
    return engine, project_dir, settings


def _insert_verified(
    engine,
    key: str,
    label_json: dict,
    verified_outcome: str = "Edit",
    pool: str | None = None,
) -> str:
    """Create an Example + Label (verified) + Operation artifact.

    Returns the Label ID.  Used to stand up ICL-eligible and Test Pool
    labels for the rename propagation test.
    """
    inv_id = generate_uuid4()
    label_id = generate_uuid4()
    now = utc_now()
    with Session(engine) as s:
        # If the example exists already, just attach the label; otherwise
        # create both.
        ex = (
            s.query(Example)
            .filter_by(
                project_id=PID,
                example_key=key,
            )
            .first()
        )
        if ex is None:
            s.add(
                Example(
                    example_key=key,
                    project_id=PID,
                    storage_ref=f"/fake/{key}.jpg",
                    ingested_at=now,
                    source_metadata={},
                    state="Verified",
                    phash="b" * 16,
                )
            )
        else:
            ex.state = "Verified"
        s.add(
            OperationRecord(
                inference_invocation_id=inv_id,
                project_id=PID,
                purpose="interactive_proposal",
                example_key=key,
                guidance_id=GID,
                model_config_id=MCID,
                endpoint_id=EID,
                model_name="test-model",
                invocation_status="success",
                schema_valid_core=True,
                normalized_json_ref=json.dumps(label_json),
                label_tier="proposal",
            )
        )
        s.add(
            Label(
                label_id=label_id,
                project_id=PID,
                example_key=key,
                label_status="verified",
                guidance_id=GID,
                inference_invocation_id=inv_id,
                label_json=label_json,
                labeled_at=now,
                verified_outcome=verified_outcome,
                verified_at=now,
                edited_core_fields=["severity"] if verified_outcome == "Edit" else [],
                edited_aux_fields=[],
                rationale_source="sme_edited"
                if verified_outcome == "Edit"
                else "teacher_proposal",
                pool_assignment=pool,
            )
        )
        s.commit()
    return label_id


# ═══════════════════════════════════════════════════════════════════════════════
# Schema evolution with pipeline-produced Auto-Labeled data
# ═══════════════════════════════════════════════════════════════════════════════


class TestEvolutionWithPipelineAutoLabeled:
    """Schema evolution after a real Batch Labeling run.

    Synthetic-fixture schema evolution tests exist already; this case
    closes a different gap by producing Auto-Labeled Labels through the
    real batch pipeline and then triggering a semantic Core edit.
    Verifies the Auto-Labeled deletion path handles
    pipeline-produced data the same as hand-inserted fixtures.
    """

    @pytest.mark.asyncio
    async def test_batch_run_auto_labeled_reverted_by_evolution(self, tmp_path):
        from vlm_feedback_loop.services.schema_evolution_service import (
            execute_edit,
        )

        engine, _, settings = _setup_project(tmp_path, n_unlabeled=3)

        # Also seed a Verified example so the evolution has both tiers.
        _insert_verified(
            engine,
            "verified_000",
            {"rationale_note": "dent", "severity": "high", "damaged": True},
            verified_outcome="Edit",
        )

        # ── 1. Run a batch labeling pipeline against the 3 Unlabeled examples,
        #    producing Auto-Labeled Labels via the real service. ──
        run_id = generate_uuid4()
        keys = ["ex_000", "ex_001", "ex_002"]
        with Session(engine) as s:
            s.add(
                RunRecord(
                    run_id=run_id,
                    project_id=PID,
                    run_type="batch_label_run",
                    status="queued",
                    guidance_id=GID,
                    model_config_id=MCID,
                    generation_preset_key="precise",
                    thinking_mode_effective="on",
                    visual_budget_preset_key="balanced",
                    structured_generation_mode_effective="auto",
                    examples_total=3,
                    metrics={"input_keys": keys, "include_auto_labeled": False},
                )
            )
            s.commit()

        async def _mock_invoke(pid, rid, ek, **kw):
            # Simulate the real Profile-B contract: create an
            # OperationRecord with purpose='batch_label' before returning.
            # This matches what the real _invoke_for_batch_label will do
            # when wired against live NIM.
            inv_id = generate_uuid4()
            proposal = {
                "rationale_note": "pipeline",
                "severity": "low",
                "damaged": False,
            }
            with Session(engine) as s:
                s.add(
                    OperationRecord(
                        inference_invocation_id=inv_id,
                        project_id=pid,
                        purpose="batch_label",
                        example_key=ek,
                        guidance_id=GID,
                        model_config_id=MCID,
                        endpoint_id=EID,
                        model_name="test-model",
                        invocation_status="success",
                        schema_valid_core=True,
                        normalized_json_ref=json.dumps(proposal),
                        batch_label_run_id=rid,
                        label_tier="auto_labeled",
                        generation_preset_key="precise",
                        sampling_params_effective={"temperature": 0.0, "top_p": 1.0},
                        thinking_mode_effective="on",
                        max_tokens_effective=256,
                    )
                )
                s.commit()
            return BatchExampleResult(
                example_key=ek,
                invocation_id=inv_id,
                invocation_status="success",
                proposal_json=proposal,
                schema_valid_core=True,
            )

        with (
            patch(
                "vlm_feedback_loop.services.batch_label_service._invoke_for_batch_label",
                side_effect=_mock_invoke,
            ),
            patch(
                "vlm_feedback_loop.services.batch_label_service.sse_manager"
            ) as mock_sse,
        ):
            mock_sse.emit = AsyncMock()
            await _execute_batch_label(PID, run_id, settings)

        # Verify the pipeline produced Auto-Labeled Labels.
        with Session(engine) as s:
            auto_labels = (
                s.query(Label)
                .filter_by(
                    project_id=PID,
                    label_status="auto_labeled",
                    batch_label_run_id=run_id,
                )
                .all()
            )
            assert len(auto_labels) == 3, (
                "Batch pipeline must create Auto-Labeled Labels for all 3"
            )
            assert all(lbl.inference_invocation_id is not None for lbl in auto_labels)

        # ── 2. Trigger a semantic Core edit (add a new Core field). ──
        new_fields = [
            dict(FIXTURE_FIELDS[0]),  # rationale_note
            dict(FIXTURE_FIELDS[1]),  # severity
            dict(FIXTURE_FIELDS[2]),  # damaged
            {
                "field_name": "new_core_field",
                "type": "boolean",
                "role": "core",
                "display_order": 3,
            },
        ]
        result = execute_edit(
            project_id=PID,
            description="Classify damage.",
            new_schema_fields=new_fields,
            rules="",
            workspace_root=settings.WORKSPACE_ROOT,
        )

        # ── 3. Verify Auto-Labeled Labels were deleted; Examples reverted. ──
        assert result.edit_type == "semantic"
        assert result.auto_labeled_reverted_count == 3, (
            "all 3 pipeline-produced Auto-Labeled Labels "
            "MUST be deleted and their Examples MUST revert to Unlabeled"
        )

        with Session(engine) as s:
            # All auto_labeled Labels for these keys are gone.
            remaining_auto = (
                s.query(Label)
                .filter_by(project_id=PID, label_status="auto_labeled")
                .count()
            )
            assert remaining_auto == 0

            # Example states are Unlabeled for the batch-labeled keys.
            for k in keys:
                ex = (
                    s.query(Example)
                    .filter_by(
                        project_id=PID,
                        example_key=k,
                    )
                    .one()
                )
                assert ex.state == "Unlabeled", (
                    f"Example {k} must revert to Unlabeled after semantic "
                    f"Core change; got state={ex.state}"
                )

            # OperationRecords are preserved for audit.
            ops = (
                s.query(OperationRecord)
                .filter_by(
                    project_id=PID,
                    purpose="batch_label",
                    batch_label_run_id=run_id,
                )
                .all()
            )
            assert len(ops) >= 3, (
                "Operation Records from batch labeling MUST survive "
                "schema evolution for audit"
            )

            # Verified Example was also reverted (now Unlabeled with
            # prior_verified_label_ref populated).
            verified_ex = (
                s.query(Example)
                .filter_by(
                    project_id=PID,
                    example_key="verified_000",
                )
                .one()
            )
            assert verified_ex.state == "Unlabeled"
            assert verified_ex.prior_verified_label_ref is not None


# ═══════════════════════════════════════════════════════════════════════════════
# Foreground priority with batch as background
# ═══════════════════════════════════════════════════════════════════════════════


class TestForegroundPriorityWithBatchBackground:
    """Batch Labeling honors the foreground-priority dispatch.

    The ``ForegroundPriorityDispatch`` primitive is unit-tested in
    isolation; this case verifies the *batch service* actually calls
    ``priority_dispatch.wait_for_background()`` per example, i.e. the
    integration point is wired correctly and a live foreground activity
    would block batch progress.
    """

    @pytest.mark.asyncio
    async def test_batch_calls_wait_for_background_per_example(self, tmp_path):
        """Each example in a batch run MUST await the priority dispatch.

        This is the structural check: if the call is missing, the batch
        run would proceed during interactive proposals, degrading SME
        latency.  We replace priority_dispatch with a probe that counts
        invocations and verify one wait per example.
        """
        engine, _, settings = _setup_project(tmp_path, n_unlabeled=3)

        run_id = generate_uuid4()
        keys = ["ex_000", "ex_001", "ex_002"]
        with Session(engine) as s:
            s.add(
                RunRecord(
                    run_id=run_id,
                    project_id=PID,
                    run_type="batch_label_run",
                    status="queued",
                    guidance_id=GID,
                    model_config_id=MCID,
                    generation_preset_key="precise",
                    thinking_mode_effective="on",
                    visual_budget_preset_key="balanced",
                    structured_generation_mode_effective="auto",
                    examples_total=3,
                    metrics={"input_keys": keys, "include_auto_labeled": False},
                )
            )
            s.commit()

        wait_count = 0

        class _CountingDispatch:
            async def wait_for_background(self) -> None:
                nonlocal wait_count
                wait_count += 1

        async def _mock_invoke(pid, rid, ek, **kw):
            return BatchExampleResult(
                example_key=ek,
                invocation_id=generate_uuid4(),
                invocation_status="success",
                proposal_json={
                    "rationale_note": "ok",
                    "severity": "medium",
                    "damaged": True,
                },
                schema_valid_core=True,
            )

        with (
            patch(
                "vlm_feedback_loop.services.batch_label_service._invoke_for_batch_label",
                side_effect=_mock_invoke,
            ),
            patch(
                "vlm_feedback_loop.services.batch_label_service.priority_dispatch",
                new=_CountingDispatch(),
            ),
            patch(
                "vlm_feedback_loop.services.batch_label_service.sse_manager"
            ) as mock_sse,
        ):
            mock_sse.emit = AsyncMock()
            await _execute_batch_label(PID, run_id, settings)

        assert wait_count == 3, (
            f"Batch labeling MUST call priority_dispatch.wait_for_background() "
            f"once per example so foreground interactive proposals preempt. "
            f"Expected 3 waits, got {wait_count}"
        )

    @pytest.mark.asyncio
    async def test_batch_blocks_while_foreground_active(self, tmp_path):
        """End-to-end: active foreground blocks a running batch example.

        Start a real ForegroundPriorityDispatch with foreground in the
        active state.  Kick off _execute_batch_label.  Within a short
        window the batch must NOT complete (because each example awaits
        ``wait_for_background`` which blocks).  Release foreground,
        batch completes.
        """
        engine, _, settings = _setup_project(tmp_path, n_unlabeled=2)
        run_id = generate_uuid4()
        keys = ["ex_000", "ex_001"]
        with Session(engine) as s:
            s.add(
                RunRecord(
                    run_id=run_id,
                    project_id=PID,
                    run_type="batch_label_run",
                    status="queued",
                    guidance_id=GID,
                    model_config_id=MCID,
                    generation_preset_key="precise",
                    thinking_mode_effective="on",
                    visual_budget_preset_key="balanced",
                    structured_generation_mode_effective="auto",
                    examples_total=2,
                    metrics={"input_keys": keys, "include_auto_labeled": False},
                )
            )
            s.commit()

        dispatch = ForegroundPriorityDispatch()
        await dispatch.enter_foreground()

        async def _mock_invoke(pid, rid, ek, **kw):
            return BatchExampleResult(
                example_key=ek,
                invocation_id=generate_uuid4(),
                invocation_status="success",
                proposal_json={
                    "rationale_note": "ok",
                    "severity": "low",
                    "damaged": False,
                },
                schema_valid_core=True,
            )

        with (
            patch(
                "vlm_feedback_loop.services.batch_label_service._invoke_for_batch_label",
                side_effect=_mock_invoke,
            ),
            patch(
                "vlm_feedback_loop.services.batch_label_service.priority_dispatch",
                new=dispatch,
            ),
            patch(
                "vlm_feedback_loop.services.batch_label_service.sse_manager"
            ) as mock_sse,
        ):
            mock_sse.emit = AsyncMock()

            batch_task = asyncio.create_task(
                _execute_batch_label(PID, run_id, settings)
            )
            # Short wait: the batch MUST be held up by foreground.
            await asyncio.sleep(0.1)

            with Session(engine) as s:
                run = s.query(RunRecord).filter_by(run_id=run_id).first()
                assert run.status in ("queued", "running"), (
                    "Batch must NOT have reached a terminal state while "
                    "foreground is active"
                )
                # No Labels yet — they only appear after the first
                # successful example.
                labels = (
                    s.query(Label)
                    .filter_by(project_id=PID, batch_label_run_id=run_id)
                    .count()
                )
                assert labels == 0, (
                    "Batch labeling must not produce any Labels while "
                    "blocked by foreground"
                )

            # Release foreground — batch resumes and completes.
            await dispatch.exit_foreground()
            await batch_task

            with Session(engine) as s:
                run = s.query(RunRecord).filter_by(run_id=run_id).first()
                assert run.status == "completed"
                labels = (
                    s.query(Label)
                    .filter_by(project_id=PID, batch_label_run_id=run_id)
                    .all()
                )
                assert len(labels) == 2, (
                    "After releasing foreground, batch must complete and "
                    "create Labels for all 2 examples"
                )


# ═══════════════════════════════════════════════════════════════════════════════
# In-place rename propagation through ICL + dataset export
# ═══════════════════════════════════════════════════════════════════════════════


class TestRenamePropagationThroughLivePipeline:
    """Rename a Core field and verify key propagates through all consumers.

    Verifies that an in-place field rename propagates to:
      (i)   Verified Label.label_json (persisted rename)
      (ii)  Dataset export gpt turn
      (iii) The active Guidance schema
    """

    @pytest.mark.asyncio
    async def test_rename_propagates_to_labels_and_export(self, tmp_path):
        from vlm_feedback_loop.services.dataset_export_service import (
            create_dataset_export,
        )
        from vlm_feedback_loop.services.schema_evolution_service import (
            execute_edit,
        )

        engine, _, settings = _setup_project(tmp_path, n_unlabeled=0)

        # Insert 3 Verified Edits (ICL-eligible) with the old field name.
        for i in range(3):
            k = f"verified_{i:03d}"
            _insert_verified(
                engine,
                k,
                {
                    "rationale_note": f"dent case {i}",
                    "severity": "medium",
                    "damaged": True,
                },
                verified_outcome="Edit",
                pool=None,  # ICL-eligible
            )

        # ── Apply in-place rename: severity → damage_level ──────────────
        new_fields = [
            dict(FIXTURE_FIELDS[0]),  # rationale_note unchanged
            {**FIXTURE_FIELDS[1], "field_name": "damage_level"},  # renamed
            dict(FIXTURE_FIELDS[2]),  # damaged unchanged
        ]
        result = execute_edit(
            project_id=PID,
            description="Classify damage.",
            new_schema_fields=new_fields,
            rules="",
            workspace_root=settings.WORKSPACE_ROOT,
        )
        assert result.edit_type == "in_place", (
            f"Field rename (field_id preserved) MUST be classified as "
            f"in-place, got {result.edit_type}"
        )
        # In-place rename preserves Labels — no verified/auto_labeled revert.
        assert result.verified_reverted_count == 0
        assert result.auto_labeled_reverted_count == 0

        # (i) ── Verified Label.label_json entries must have renamed key ──
        with Session(engine) as s:
            verified_labels = (
                s.query(Label).filter_by(project_id=PID, label_status="verified").all()
            )
            assert len(verified_labels) == 3
            for lbl in verified_labels:
                assert "damage_level" in lbl.label_json, (
                    f"Rename MUST propagate into Label.label_json; "
                    f"got keys={list(lbl.label_json.keys())}"
                )
                assert "severity" not in lbl.label_json, (
                    "Old key MUST be gone after rename"
                )
                assert lbl.label_json["damage_level"] == "medium"

        # (ii) ── Dataset export gpt turn must use the new key ─────────
        # The export ships only examples whose image exists on disk;
        # point the seeded rows at real files first.
        imgs_dir = tmp_path / "imgs"
        imgs_dir.mkdir(exist_ok=True)
        with Session(engine) as s:
            for ex in s.query(Example).filter_by(project_id=PID).all():
                real = imgs_dir / f"{ex.example_key}.jpg"
                real.write_bytes(b"img")
                ex.storage_ref = str(real)
            s.commit()

        export = create_dataset_export(
            project_id=PID,
            dataset_intent="training",
            label_tier_filter="verified_only",
            export_field_mode="all",
            batch_label_run_id=None,
            selection_filters=None,
            settings=settings,
        )
        assert export["example_count"] == 3

        import tarfile

        archive_path = Path(export["artifact_refs"]["archive_path"])
        assert archive_path.exists()
        with tarfile.open(archive_path, "r:gz") as tf:
            # Tar entries are namespaced under ``{export_id}/`` for the
            # cosmos-rl extraction-layout contract.
            ann_member = next(
                (m for m in tf.getmembers() if m.name.endswith("annotations.json")),
                None,
            )
            assert ann_member is not None, "annotations.json not in archive"
            ann_file = tf.extractfile(ann_member)
            assert ann_file is not None
            annotations = json.loads(ann_file.read().decode("utf-8"))

        assert len(annotations) == 3
        for sample in annotations:
            gpt_value = sample["conversations"][1]["value"]
            # The gpt turn MUST be a JSON string
            assert isinstance(gpt_value, str)
            parsed = json.loads(gpt_value)
            assert "damage_level" in parsed, (
                f"Dataset export gpt turn MUST contain renamed key; "
                f"got {list(parsed.keys())}"
            )
            assert "severity" not in parsed, (
                "Dataset export must NOT carry old pre-rename key"
            )

        # (iii) ── Guidance schema reflects the rename ──────────────────
        with Session(engine) as s:
            active_gid = s.execute(
                select(Project.active_guidance_id).where(Project.project_id == PID)
            ).scalar()
            g = s.query(Guidance).filter_by(guidance_id=active_gid).one()
            field_names = {f["field_name"] for f in g.schema["fields"]}
            assert "damage_level" in field_names
            assert "severity" not in field_names
