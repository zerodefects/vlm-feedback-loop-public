# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for dataset_export_service.

Covers: Cosmos-RL archive creation, annotations.json wire format, export field
modes, label selection by intent/tier, archive structure, DatasetExport record
persistence, get/list, schema-invalid manifest, structural validation.
"""

from __future__ import annotations

import asyncio
import json
import tarfile
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy.orm import Session

from conftest import (
    FIXTURE_FIELDS,
    GID,
    PID,
    add_endpoint_and_model_rows,
    add_example_row,
    add_fixture_guidance_row,
    add_standard_project_row,
    make_settings,
    setup_project_db,
)
from vlm_feedback_loop.db.base import generate_uuid4, utc_now
from vlm_feedback_loop.db.models.dataset_export import DatasetExport
from vlm_feedback_loop.db.models.example import Example
from vlm_feedback_loop.db.models.label import Label
from vlm_feedback_loop.db.models.operation import OperationRecord
from vlm_feedback_loop.db.models.run import RunRecord
from vlm_feedback_loop.services.dataset_export_service import (
    _select_labels,
    _serialize_label_for_export,
    create_dataset_export,
    get_dataset_export,
    get_schema_invalid_manifest,
    list_dataset_exports,
    persist_dataset_export_in_session,
    recover_dataset_exports,
    start_dataset_export,
)
from vlm_feedback_loop.services.sse import sse_manager

# ── Helpers ──────────────────────────────────────────────────────────────────

SAMPLE_LABEL_JSON = {
    "rationale_note": "visible dent on panel",
    "severity": "high",
    "damaged": True,
}


def _add_guidance(session, project_id=PID, guidance_id=GID, version_number=1):
    add_fixture_guidance_row(
        session,
        project_id,
        guidance_id,
        version_number=version_number,
        description="Classify damage and return JSON only.",
    )


def _create_image_file(tmp_path: Path, key: str) -> str:
    """Create a minimal fake image file and return its path."""
    images_dir = tmp_path / "images"
    images_dir.mkdir(exist_ok=True)
    img_path = images_dir / f"{key}.jpg"
    img_path.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)  # fake JPEG header
    return str(img_path)


def _add_example(session, project_id, key, state="Verified", storage_ref="/fake.jpg"):
    add_example_row(session, project_id, key, state=state, storage_ref=storage_ref)


def _add_verified_label(
    session, project_id, key, guidance_id=GID, pool_assignment=None, label_json=None
):
    session.add(
        Label(
            label_id=generate_uuid4(),
            project_id=project_id,
            example_key=key,
            label_status="verified",
            guidance_id=guidance_id,
            inference_invocation_id=generate_uuid4(),
            label_json=label_json or dict(SAMPLE_LABEL_JSON),
            labeled_at=utc_now(),
            verified_outcome="Accept",
            verified_at=utc_now(),
            edited_core_fields=[],
            edited_aux_fields=[],
            rationale_source="teacher_proposal",
            pool_assignment=pool_assignment,
        )
    )


def _add_auto_label(
    session, project_id, key, guidance_id=GID, run_id="batch-run-1", label_json=None
):
    session.add(
        Label(
            label_id=generate_uuid4(),
            project_id=project_id,
            example_key=key,
            label_status="auto_labeled",
            guidance_id=guidance_id,
            inference_invocation_id=generate_uuid4(),
            label_json=label_json or dict(SAMPLE_LABEL_JSON),
            labeled_at=utc_now(),
            batch_label_run_id=run_id,
        )
    )


def _setup_export_project(tmp_path, n_verified=3, n_pool=2, n_auto=2):
    """Create project with Guidance + examples with images + labels."""
    engine, pdir = setup_project_db(tmp_path, subdirs=("artifacts", "exports"))
    settings = make_settings(str(tmp_path / "workspace"))

    with Session(engine) as s:
        add_standard_project_row(s, PID, pdir)
        _add_guidance(s)
        add_endpoint_and_model_rows(s)

        for i in range(n_verified):
            key = f"ver_{i:03d}"
            img_path = _create_image_file(tmp_path, key)
            _add_example(s, PID, key, state="Verified", storage_ref=img_path)
            _add_verified_label(s, PID, key, pool_assignment=None)

        for i in range(n_pool):
            key = f"pool_{i:03d}"
            img_path = _create_image_file(tmp_path, key)
            _add_example(s, PID, key, state="Verified", storage_ref=img_path)
            _add_verified_label(s, PID, key, pool_assignment="test_pool")

        for i in range(n_auto):
            key = f"auto_{i:03d}"
            img_path = _create_image_file(tmp_path, key)
            _add_example(s, PID, key, state="Auto-Labeled", storage_ref=img_path)
            _add_auto_label(s, PID, key)

        s.commit()

    return engine, pdir, settings


def _extract_annotations(tar_path: str | Path) -> list[dict[str, Any]]:
    """Extract and parse annotations.json from a .tar.gz archive.

    Tar contents are flat — ``annotations.json`` at the archive root
    (see ``_build_tar_archive``). Find the member by its
    ``annotations.json`` suffix to stay layout-agnostic.
    """
    with tarfile.open(tar_path, "r:gz") as tf:
        ann_member = next(
            (m for m in tf.getmembers() if m.name.endswith("annotations.json")),
            None,
        )
        assert ann_member is not None, "annotations.json not in archive"
        f = tf.extractfile(ann_member)
        assert f is not None
        return json.loads(f.read())


def _list_archive_files(tar_path: str | Path) -> list[str]:
    """List all file names in a .tar.gz archive."""
    with tarfile.open(tar_path, "r:gz") as tf:
        return [m.name for m in tf.getmembers()]


# ══════════════════════════════════════════════════════════════════════════════
# Section A: Create Export — Basic Intents
# ══════════════════════════════════════════════════════════════════════════════


class TestCreateExport:
    def test_verified_only_training(self, tmp_path):
        engine, pdir, settings = _setup_export_project(
            tmp_path, n_verified=3, n_pool=2, n_auto=2
        )
        result = create_dataset_export(
            PID,
            dataset_intent="training",
            label_tier_filter="verified_only",
            export_field_mode="all",
            settings=settings,
        )
        assert not isinstance(result, str), result
        assert result["dataset_intent"] == "training"
        assert result["example_count"] == 3  # non-pool Verified only

    def test_missing_image_excluded_from_annotations_and_tar(self, tmp_path):
        """An example whose by-reference image has vanished must not ship:
        the tar can't include it, so the annotations (in-tar + sidecar) and
        counts must exclude it too — otherwise TAO trains against
        annotations referencing images absent from the archive.
        The skipped keys are recorded in the manifest."""
        engine, pdir, settings = _setup_export_project(
            tmp_path, n_verified=3, n_pool=0, n_auto=0
        )
        # Delete one image from disk after ingest (moved/removed by user).
        with Session(engine) as s:
            gone = (
                s.query(Example).filter_by(project_id=PID, example_key="ver_001").one()
            )
            Path(gone.storage_ref).unlink()

        result = create_dataset_export(
            PID,
            dataset_intent="training",
            label_tier_filter="verified_only",
            export_field_mode="all",
            settings=settings,
        )
        assert not isinstance(result, str), result
        assert result["example_count"] == 2

        ann = _extract_annotations(result["artifact_refs"]["archive_path"])
        assert sorted(a["id"] for a in ann) == ["ver_000", "ver_002"]

        sidecar = json.loads(
            Path(result["artifact_refs"]["annotations_path"]).read_text()
        )
        assert sorted(a["id"] for a in sidecar) == ["ver_000", "ver_002"]

        manifest = json.loads(Path(result["manifest_ref"]).read_text())
        assert manifest["example_count"] == 2
        assert manifest["skipped_missing_images"] == ["ver_001"]

    def test_export_fails_fast_when_all_images_missing(self, tmp_path):
        """If labels match the selection but EVERY image is gone (typical
        after a workspace migration without the image tree), the export must
        fail fast instead of producing a zero-sample archive that would
        upload cleanly and then fail — or silently no-op — inside TAO."""
        engine, pdir, settings = _setup_export_project(
            tmp_path, n_verified=3, n_pool=0, n_auto=0
        )
        with Session(engine) as s:
            for ex in s.query(Example).filter_by(project_id=PID).all():
                Path(ex.storage_ref).unlink()

        result = create_dataset_export(
            PID,
            dataset_intent="training",
            label_tier_filter="verified_only",
            export_field_mode="all",
            settings=settings,
        )
        assert isinstance(result, str)
        assert "validation:" in result
        assert "image file" in result

    def test_verified_only_evaluation(self, tmp_path):
        engine, pdir, settings = _setup_export_project(
            tmp_path, n_verified=3, n_pool=2, n_auto=2
        )
        result = create_dataset_export(
            PID,
            dataset_intent="evaluation",
            label_tier_filter="verified_only",
            export_field_mode="all",
            settings=settings,
        )
        assert not isinstance(result, str), result
        assert result["example_count"] == 2  # pool members only

    def test_auto_labeled_only(self, tmp_path):
        engine, pdir, settings = _setup_export_project(
            tmp_path, n_verified=3, n_pool=2, n_auto=2
        )
        result = create_dataset_export(
            PID,
            dataset_intent="training",
            label_tier_filter="auto_labeled_only",
            export_field_mode="all",
            settings=settings,
        )
        assert not isinstance(result, str), result
        assert result["example_count"] == 2  # auto-labeled only

    def test_auto_labeled_export_is_scoped_to_batch_run_and_records_filter(
        self, tmp_path
    ):
        """A run export excludes another run and records every selection filter."""
        engine, pdir, settings = _setup_export_project(
            tmp_path, n_verified=0, n_pool=0, n_auto=2
        )
        with Session(engine) as s:
            other_run = (
                s.query(Label).filter_by(project_id=PID, example_key="auto_001").one()
            )
            other_run.batch_label_run_id = "batch-run-2"
            s.commit()

        result = create_dataset_export(
            PID,
            dataset_intent="training",
            label_tier_filter="auto_labeled_only",
            export_field_mode="core_only",
            batch_label_run_id="batch-run-1",
            selection_filters={"guidance_id": GID},
            settings=settings,
        )
        assert not isinstance(result, str), result
        assert result["example_count"] == 1
        expected_snapshot = {
            "dataset_intent": "training",
            "label_tier_filter": "auto_labeled_only",
            "export_field_mode": "core_only",
            "batch_label_run_id": "batch-run-1",
            "guidance_id": GID,
            "selection_filters": {"guidance_id": GID},
        }
        with Session(engine) as s:
            record = s.get(DatasetExport, result["dataset_export_id"])
            assert record is not None
            assert record.selection_definition_snapshot == expected_snapshot

        annotations = _extract_annotations(result["artifact_refs"]["archive_path"])
        sidecar = json.loads(
            Path(result["artifact_refs"]["annotations_path"]).read_text()
        )
        assert [row["id"] for row in annotations] == ["auto_000"]
        assert [row["id"] for row in sidecar] == ["auto_000"]
        assert all(
            "auto_001" not in member
            for member in _list_archive_files(result["artifact_refs"]["archive_path"])
        )

        manifest = json.loads(Path(result["manifest_ref"]).read_text())
        assert manifest == {
            "dataset_export_id": result["dataset_export_id"],
            **expected_snapshot,
            "example_count": 1,
            "example_keys": ["auto_000"],
            "skipped_missing_images": [],
        }

    def test_empty_batch_run_id_cannot_broaden_export(self, tmp_path):
        """An explicit empty run filter selects no run, never every run."""
        _engine, _pdir, settings = _setup_export_project(
            tmp_path, n_verified=0, n_pool=0, n_auto=2
        )

        result = create_dataset_export(
            PID,
            dataset_intent="training",
            label_tier_filter="auto_labeled_only",
            batch_label_run_id="",
            settings=settings,
        )

        assert isinstance(result, str)
        assert (
            result == "validation: batch_label_run_id must be non-empty when provided"
        )

    def test_auto_labeled_only_requires_live_example_state(self, tmp_path):
        """A stale machine Label cannot make an Omitted image exportable."""
        engine, pdir, settings = _setup_export_project(
            tmp_path, n_verified=0, n_pool=0, n_auto=2
        )
        with Session(engine) as s:
            omitted = (
                s.query(Example).filter_by(project_id=PID, example_key="auto_001").one()
            )
            omitted.state = "Omitted"
            omitted.omitted_source = "sme_skip"
            omitted.omitted_at = utc_now()
            s.commit()

        result = create_dataset_export(
            PID,
            dataset_intent="training",
            label_tier_filter="auto_labeled_only",
            export_field_mode="all",
            settings=settings,
        )
        assert not isinstance(result, str), result
        assert result["example_count"] == 1
        annotations = _extract_annotations(result["artifact_refs"]["archive_path"])
        assert [row["id"] for row in annotations] == ["auto_000"]

    def test_auto_labeled_selection_requires_exact_project_owner(self, tmp_path):
        """Another project's Example state cannot authorize this project's Label."""
        engine, pdir = setup_project_db(tmp_path, subdirs=("artifacts", "exports"))
        key = "cross-project"
        with Session(engine) as s:
            add_standard_project_row(s, PID, pdir)
            _add_guidance(s)
            _add_example(
                s,
                "foreign-project",
                key,
                state="Auto-Labeled",
                storage_ref=_create_image_file(tmp_path, key),
            )
            _add_auto_label(s, PID, key)
            s.commit()

            selected = _select_labels(
                s,
                project_id=PID,
                guidance_id=GID,
                dataset_intent="training",
                label_tier_filter="auto_labeled_only",
                batch_label_run_id=None,
            )

        assert selected == []

    @pytest.mark.parametrize("dataset_intent", ["evaluation", "testing"])
    @pytest.mark.parametrize("label_tier_filter", ["auto_labeled_only", "combined"])
    def test_ground_truth_export_rejects_auto_labeled_tiers(
        self, tmp_path, dataset_intent, label_tier_filter
    ):
        """Evaluation/testing answer keys must contain Verified labels only."""
        engine, pdir, settings = _setup_export_project(
            tmp_path, n_verified=3, n_pool=2, n_auto=2
        )

        result = create_dataset_export(
            PID,
            dataset_intent=dataset_intent,
            label_tier_filter=label_tier_filter,
            export_field_mode="all",
            settings=settings,
        )

        assert isinstance(result, str)
        assert "verified_only" in result

    def test_combined(self, tmp_path):
        engine, pdir, settings = _setup_export_project(
            tmp_path, n_verified=3, n_pool=2, n_auto=2
        )
        result = create_dataset_export(
            PID,
            dataset_intent="training",
            label_tier_filter="combined",
            export_field_mode="all",
            settings=settings,
        )
        assert not isinstance(result, str), result
        assert result["example_count"] == 5  # 3 verified non-pool + 2 auto

    def test_testing_intent(self, tmp_path):
        engine, pdir, settings = _setup_export_project(
            tmp_path, n_verified=3, n_pool=2, n_auto=2
        )
        result = create_dataset_export(
            PID,
            dataset_intent="testing",
            label_tier_filter="verified_only",
            export_field_mode="all",
            settings=settings,
        )
        assert not isinstance(result, str), result
        assert result["example_count"] == 2  # same as evaluation (pool members)

    def test_no_guidance_returns_error(self, tmp_path):
        engine, pdir = setup_project_db(tmp_path, subdirs=("artifacts", "exports"))
        settings = make_settings(str(tmp_path / "workspace"))
        with Session(engine) as s:
            add_standard_project_row(s, PID, pdir, active_guidance_id=None)
            s.commit()
        result = create_dataset_export(
            PID,
            dataset_intent="training",
            settings=settings,
        )
        assert isinstance(result, str)
        assert "guidance" in result.lower()


# ══════════════════════════════════════════════════════════════════════════════
# Section B: annotations.json Wire Format
# ══════════════════════════════════════════════════════════════════════════════


class TestAnnotationsFormat:
    def test_top_level_json_array(self, tmp_path):
        engine, pdir, settings = _setup_export_project(
            tmp_path, n_verified=2, n_pool=0, n_auto=0
        )
        result = create_dataset_export(
            PID,
            dataset_intent="training",
            settings=settings,
        )
        annotations = _extract_annotations(result["artifact_refs"]["archive_path"])
        assert isinstance(annotations, list)
        assert len(annotations) == 2

    def test_sample_has_id_images_conversations(self, tmp_path):
        engine, pdir, settings = _setup_export_project(
            tmp_path, n_verified=1, n_pool=0, n_auto=0
        )
        result = create_dataset_export(
            PID,
            dataset_intent="training",
            settings=settings,
        )
        annotations = _extract_annotations(result["artifact_refs"]["archive_path"])
        sample = annotations[0]
        assert "id" in sample
        assert "images" in sample
        assert "conversations" in sample
        assert isinstance(sample["images"], list)
        assert len(sample["images"]) == 1
        assert isinstance(sample["conversations"], list)
        assert len(sample["conversations"]) == 2

    def test_human_turn_renders_serving_prompt(self, tmp_path):
        """The human turn carries the rendered serving prompt, not a bare
        task sentence.

        Students are served and serving-evaluated with the full proposal
        prompt; a student trained on a short prompt is off-distribution at
        deployment (measured 3.4× exact-match loss). The export human turn
        must be the zero-shot, self-contained serving prompt with the
        literal ``<image>`` token at the query-image position.
        """
        engine, pdir, settings = _setup_export_project(
            tmp_path, n_verified=1, n_pool=0, n_auto=0
        )
        result = create_dataset_export(
            PID,
            dataset_intent="training",
            settings=settings,
        )
        annotations = _extract_annotations(result["artifact_refs"]["archive_path"])
        human = annotations[0]["conversations"][0]
        assert human["from"] == "human"
        value = human["value"]
        # System text folded into the human turn (two-turn contract has
        # no system slot).
        assert value.startswith("You are a vision labeling assistant.")
        # Compact production prompt omits verbose Guidance prose but embeds
        # the actual field contract because training has no response_format.
        assert "Task Description:" not in value
        assert "Classify damage and return JSON only." not in value
        assert "Label schema:" in value
        assert '- severity (required): "low" | "medium" | "high"' in value
        assert "- damaged (required): boolean" in value
        assert "schema described above" not in value
        assert "Add rationale_note last." in value
        # Query image: exactly one literal token, no unresolved sentinels,
        # and zero-shot (students serve without ICL).
        assert value.count("<image>") == 1
        assert "<<<__VLM_" not in value
        assert "ICL Example" not in value

    def test_empty_description_human_turn_mirrors_serving_prompt(self, tmp_path):
        """The export human turn is the rendered serving prompt even when the
        Guidance description is empty: an empty description omits its labeled
        block, and the exported turn stays byte-identical to what a deployed
        Student is served.
        """
        engine, pdir = setup_project_db(tmp_path, subdirs=("artifacts", "exports"))
        settings = make_settings(str(tmp_path / "workspace"))
        with Session(engine) as s:
            add_standard_project_row(s, PID, pdir)
            add_fixture_guidance_row(s, PID, GID, description="")
            add_endpoint_and_model_rows(s)
            key = "ver_000"
            img_path = _create_image_file(tmp_path, key)
            _add_example(s, PID, key, state="Verified", storage_ref=img_path)
            _add_verified_label(s, PID, key)
            s.commit()

        result = create_dataset_export(
            PID,
            dataset_intent="training",
            settings=settings,
        )
        assert not isinstance(result, str), result
        annotations = _extract_annotations(result["artifact_refs"]["archive_path"])
        human = annotations[0]["conversations"][0]
        assert human["from"] == "human"
        value = human["value"]
        assert "Task Description:" not in value
        # Still a complete serving prompt: one query image, output contract.
        assert value.count("<image>") == 1
        assert "Label schema:" in value
        assert '- severity (required): "low" | "medium" | "high"' in value
        assert "schema described above" not in value

    def test_evaluation_export_mirrors_training_human_turn(self, tmp_path):
        """Training and evaluation exports render byte-identical human turns.

        The FTMS evaluate leg scores the student against the evaluation
        export; if only training carried the serving prompt the evaluate
        leg would go back out of distribution.
        """
        engine, pdir, settings = _setup_export_project(
            tmp_path, n_verified=1, n_pool=1, n_auto=0
        )
        training = create_dataset_export(
            PID,
            dataset_intent="training",
            settings=settings,
        )
        evaluation = create_dataset_export(
            PID,
            dataset_intent="evaluation",
            settings=settings,
        )
        train_human = _extract_annotations(training["artifact_refs"]["archive_path"])[
            0
        ]["conversations"][0]["value"]
        eval_human = _extract_annotations(evaluation["artifact_refs"]["archive_path"])[
            0
        ]["conversations"][0]["value"]
        assert train_human == eval_human

    def test_gpt_turn_is_json_string_not_object(self, tmp_path):
        engine, pdir, settings = _setup_export_project(
            tmp_path, n_verified=1, n_pool=0, n_auto=0
        )
        result = create_dataset_export(
            PID,
            dataset_intent="training",
            settings=settings,
        )
        annotations = _extract_annotations(result["artifact_refs"]["archive_path"])
        gpt = annotations[0]["conversations"][1]
        assert gpt["from"] == "gpt"
        # value MUST be a string (not nested JSON object)
        assert isinstance(gpt["value"], str)
        # Parse the string as JSON to verify it's valid
        parsed = json.loads(gpt["value"])
        assert isinstance(parsed, dict)

    def test_conversations_use_from_value_not_role_content(self, tmp_path):
        engine, pdir, settings = _setup_export_project(
            tmp_path, n_verified=1, n_pool=0, n_auto=0
        )
        result = create_dataset_export(
            PID,
            dataset_intent="training",
            settings=settings,
        )
        annotations = _extract_annotations(result["artifact_refs"]["archive_path"])
        for turn in annotations[0]["conversations"]:
            assert "from" in turn
            assert "value" in turn
            assert "role" not in turn
            assert "content" not in turn

    def test_image_path_resolves_in_archive(self, tmp_path):
        engine, pdir, settings = _setup_export_project(
            tmp_path, n_verified=1, n_pool=0, n_auto=0
        )
        result = create_dataset_export(
            PID,
            dataset_intent="training",
            settings=settings,
        )
        tar_path = result["artifact_refs"]["archive_path"]
        annotations = _extract_annotations(tar_path)
        archive_files = _list_archive_files(tar_path)
        _ = result["dataset_export_id"]  # captured for trace-context only
        for sample in annotations:
            image_path = sample["images"][0]
            # Tar contents are flat, so each annotation's relative image
            # path (``images/{key}.ext``) is itself a tar entry.
            expected = image_path
            assert expected in archive_files, (
                f"{expected} not found in archive ({archive_files})"
            )


# ══════════════════════════════════════════════════════════════════════════════
# Section C: Export Field Mode
# ══════════════════════════════════════════════════════════════════════════════


class TestExportFieldMode:
    def test_mode_all_includes_rationale_aux_core(self, tmp_path):
        engine, pdir, settings = _setup_export_project(
            tmp_path, n_verified=1, n_pool=0, n_auto=0
        )
        result = create_dataset_export(
            PID,
            dataset_intent="training",
            export_field_mode="all",
            settings=settings,
        )
        annotations = _extract_annotations(result["artifact_refs"]["archive_path"])
        parsed = json.loads(annotations[0]["conversations"][1]["value"])
        assert "rationale_note" in parsed
        assert "severity" in parsed
        assert "damaged" in parsed

    def test_mode_aux_and_core_excludes_rationale(self, tmp_path):
        engine, pdir, settings = _setup_export_project(
            tmp_path, n_verified=1, n_pool=0, n_auto=0
        )
        result = create_dataset_export(
            PID,
            dataset_intent="training",
            export_field_mode="aux_and_core",
            settings=settings,
        )
        annotations = _extract_annotations(result["artifact_refs"]["archive_path"])
        parsed = json.loads(annotations[0]["conversations"][1]["value"])
        assert "rationale_note" not in parsed
        assert "severity" in parsed
        assert "damaged" in parsed

    def test_mode_core_only(self, tmp_path):
        engine, pdir, settings = _setup_export_project(
            tmp_path, n_verified=1, n_pool=0, n_auto=0
        )
        result = create_dataset_export(
            PID,
            dataset_intent="training",
            export_field_mode="core_only",
            settings=settings,
        )
        annotations = _extract_annotations(result["artifact_refs"]["archive_path"])
        parsed = json.loads(annotations[0]["conversations"][1]["value"])
        assert "rationale_note" not in parsed
        assert "severity" in parsed
        assert "damaged" in parsed

    def test_field_ordering_matches_serving_prompt_order(self, tmp_path):
        engine, pdir, settings = _setup_export_project(
            tmp_path, n_verified=1, n_pool=0, n_auto=0
        )
        result = create_dataset_export(
            PID,
            dataset_intent="training",
            export_field_mode="all",
            settings=settings,
        )
        annotations = _extract_annotations(result["artifact_refs"]["archive_path"])
        gpt_str = annotations[0]["conversations"][1]["value"]
        parsed = json.loads(gpt_str)
        keys = list(parsed.keys())
        # The production serving prompt requests rationale last.
        assert keys == ["severity", "damaged", "rationale_note"]

    def test_serialize_label_field_ordering(self):
        """Direct unit test of the serialization helper."""
        label = {"damaged": True, "severity": "high", "rationale_note": "dent"}
        gen_order = ["rationale_note", "severity", "damaged"]
        result = _serialize_label_for_export(label, gen_order, FIXTURE_FIELDS, "all")
        parsed = json.loads(result)
        assert list(parsed.keys()) == ["rationale_note", "severity", "damaged"]


# ══════════════════════════════════════════════════════════════════════════════
# Section D: Label Selection
# ══════════════════════════════════════════════════════════════════════════════


class TestLabelSelection:
    def test_training_excludes_pool_members(self, tmp_path):
        engine, pdir, settings = _setup_export_project(
            tmp_path, n_verified=3, n_pool=2, n_auto=0
        )
        result = create_dataset_export(
            PID,
            dataset_intent="training",
            label_tier_filter="verified_only",
            settings=settings,
        )
        annotations = _extract_annotations(result["artifact_refs"]["archive_path"])
        keys = {s["id"] for s in annotations}
        # Should not include pool members
        assert not any(k.startswith("pool_") for k in keys)
        assert all(k.startswith("ver_") for k in keys)

    def test_evaluation_includes_only_pool_members(self, tmp_path):
        engine, pdir, settings = _setup_export_project(
            tmp_path, n_verified=3, n_pool=2, n_auto=0
        )
        result = create_dataset_export(
            PID,
            dataset_intent="evaluation",
            label_tier_filter="verified_only",
            settings=settings,
        )
        annotations = _extract_annotations(result["artifact_refs"]["archive_path"])
        keys = {s["id"] for s in annotations}
        assert all(k.startswith("pool_") for k in keys)

    def test_empty_export_succeeds(self, tmp_path):
        engine, pdir, settings = _setup_export_project(
            tmp_path, n_verified=0, n_pool=0, n_auto=0
        )
        result = create_dataset_export(
            PID,
            dataset_intent="training",
            settings=settings,
        )
        assert not isinstance(result, str), result
        assert result["example_count"] == 0

    def test_scoped_to_current_guidance(self, tmp_path):
        """Labels under a different guidance_id should not be exported."""
        engine, pdir = setup_project_db(tmp_path, subdirs=("artifacts", "exports"))
        settings = make_settings(str(tmp_path / "workspace"))
        other_gid = "other-guid"
        with Session(engine) as s:
            add_standard_project_row(s, PID, pdir)
            _add_guidance(s, guidance_id=GID, version_number=1)
            _add_guidance(s, guidance_id=other_gid, version_number=2)
            add_endpoint_and_model_rows(s)
            # Label under current guidance
            img = _create_image_file(tmp_path, "current")
            _add_example(s, PID, "current", state="Verified", storage_ref=img)
            _add_verified_label(s, PID, "current", guidance_id=GID)
            # Label under other guidance
            img2 = _create_image_file(tmp_path, "other")
            _add_example(s, PID, "other", state="Verified", storage_ref=img2)
            _add_verified_label(s, PID, "other", guidance_id=other_gid)
            s.commit()

        result = create_dataset_export(
            PID,
            dataset_intent="training",
            settings=settings,
        )
        assert result["example_count"] == 1  # only current guidance


# ══════════════════════════════════════════════════════════════════════════════
# Section E: Archive Structure
# ══════════════════════════════════════════════════════════════════════════════


class TestArchiveStructure:
    def test_archive_is_tar_gz(self, tmp_path):
        engine, pdir, settings = _setup_export_project(
            tmp_path, n_verified=1, n_pool=0, n_auto=0
        )
        result = create_dataset_export(
            PID,
            dataset_intent="training",
            settings=settings,
        )
        archive_path = result["artifact_refs"]["archive_path"]
        assert archive_path.endswith(".tar.gz")
        assert tarfile.is_tarfile(archive_path)

    def test_archive_contains_annotations_and_images(self, tmp_path):
        engine, pdir, settings = _setup_export_project(
            tmp_path, n_verified=2, n_pool=0, n_auto=0
        )
        result = create_dataset_export(
            PID,
            dataset_intent="training",
            settings=settings,
        )
        files = _list_archive_files(result["artifact_refs"]["archive_path"])
        # Tar contents are flat (annotations.json + images/...) at the
        # archive root. The spec-level fix uses a parent-directory URL
        # for media_path so TAO's path substitution aligns with
        # ``_extract_images`` extraction layout.
        assert any(f.endswith("annotations.json") for f in files), files
        image_files = [f for f in files if f.startswith("images/")]
        assert len(image_files) == 2, files

    def test_image_files_present_for_every_sample(self, tmp_path):
        engine, pdir, settings = _setup_export_project(
            tmp_path, n_verified=3, n_pool=0, n_auto=0
        )
        result = create_dataset_export(
            PID,
            dataset_intent="training",
            settings=settings,
        )
        tar_path = result["artifact_refs"]["archive_path"]
        annotations = _extract_annotations(tar_path)
        archive_files = set(_list_archive_files(tar_path))
        # annotations.json image paths are relative (``images/{key}.jpg``)
        # and tar contents are flat, so each annotation's path is itself
        # a tar entry. Verify every sample's image resolves to an
        # archive member.
        _ = result["dataset_export_id"]  # captured for trace-context only
        for sample in annotations:
            expected = sample["images"][0]
            assert expected in archive_files, (expected, archive_files)


# ══════════════════════════════════════════════════════════════════════════════
# Section F: DatasetExport Record
# ══════════════════════════════════════════════════════════════════════════════


class TestRecordPersistence:
    def test_record_persisted_with_all_fields(self, tmp_path):
        engine, pdir, settings = _setup_export_project(
            tmp_path, n_verified=2, n_pool=0, n_auto=0
        )
        result = create_dataset_export(
            PID,
            dataset_intent="training",
            export_field_mode="core_only",
            settings=settings,
        )
        with Session(engine) as s:
            record = (
                s.query(DatasetExport)
                .filter_by(
                    dataset_export_id=result["dataset_export_id"],
                )
                .first()
            )
            assert record is not None
            assert record.project_id == PID
            assert record.dataset_intent == "training"
            assert record.export_field_mode == "core_only"
            assert record.guidance_id == GID
            assert record.example_count == 2

    def test_artifact_refs_has_path_and_checksum(self, tmp_path):
        engine, pdir, settings = _setup_export_project(
            tmp_path, n_verified=1, n_pool=0, n_auto=0
        )
        result = create_dataset_export(
            PID,
            dataset_intent="training",
            settings=settings,
        )
        refs = result["artifact_refs"]
        assert "archive_path" in refs
        assert "checksum_sha256" in refs
        assert len(refs["checksum_sha256"]) == 64  # SHA-256 hex

    def test_manifest_ref_points_to_file(self, tmp_path):
        engine, pdir, settings = _setup_export_project(
            tmp_path, n_verified=1, n_pool=0, n_auto=0
        )
        result = create_dataset_export(
            PID,
            dataset_intent="training",
            settings=settings,
        )
        manifest_path = Path(result["manifest_ref"])
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text())
        assert manifest["dataset_export_id"] == result["dataset_export_id"]
        assert manifest["example_count"] == result["example_count"]


# ══════════════════════════════════════════════════════════════════════════════
# Section G: Get / List
# ══════════════════════════════════════════════════════════════════════════════


class TestGetList:
    def test_get_returns_full_detail(self, tmp_path):
        engine, pdir, settings = _setup_export_project(
            tmp_path, n_verified=1, n_pool=0, n_auto=0
        )
        created = create_dataset_export(
            PID,
            dataset_intent="training",
            settings=settings,
        )
        result = get_dataset_export(
            PID,
            created["dataset_export_id"],
            settings=settings,
        )
        assert not isinstance(result, str), result
        assert result["dataset_export_id"] == created["dataset_export_id"]
        assert result["dataset_intent"] == "training"

    def test_get_nonexistent_returns_error(self, tmp_path):
        _setup_export_project(tmp_path)
        settings = make_settings(str(tmp_path / "workspace"))
        result = get_dataset_export(PID, "no-such-id", settings=settings)
        assert isinstance(result, str)
        assert "not found" in result.lower()

    def test_list_returns_items(self, tmp_path):
        engine, pdir, settings = _setup_export_project(
            tmp_path, n_verified=2, n_pool=1, n_auto=1
        )
        create_dataset_export(PID, dataset_intent="training", settings=settings)
        create_dataset_export(PID, dataset_intent="evaluation", settings=settings)

        items, cursor = list_dataset_exports(PID, settings=settings)
        assert len(items) == 2

    def test_list_with_intent_filter(self, tmp_path):
        engine, pdir, settings = _setup_export_project(
            tmp_path, n_verified=2, n_pool=1, n_auto=1
        )
        create_dataset_export(PID, dataset_intent="training", settings=settings)
        create_dataset_export(PID, dataset_intent="evaluation", settings=settings)

        items, _ = list_dataset_exports(
            PID,
            dataset_intent_filter="training",
            settings=settings,
        )
        assert len(items) == 1
        assert items[0]["dataset_intent"] == "training"


# ══════════════════════════════════════════════════════════════════════════════
# Section H: Schema-Invalid Manifest
# ══════════════════════════════════════════════════════════════════════════════


class TestSchemaInvalidManifest:
    def test_returns_schema_invalid_examples(self, tmp_path):
        engine, pdir, settings = _setup_export_project(tmp_path)
        run_id = generate_uuid4()
        with Session(engine) as s:
            s.add(
                RunRecord(
                    run_id=run_id,
                    project_id=PID,
                    run_type="batch_label_run",
                    status="completed",
                    examples_total=3,
                )
            )
            # 2 schema-invalid operations
            s.add(
                OperationRecord(
                    inference_invocation_id=generate_uuid4(),
                    project_id=PID,
                    purpose="batch_label",
                    example_key="ex_001",
                    batch_label_run_id=run_id,
                    invocation_status="success",
                    schema_valid_core=False,
                    validation_errors_core=["severity: invalid value"],
                    model_name="test-model",
                )
            )
            s.add(
                OperationRecord(
                    inference_invocation_id=generate_uuid4(),
                    project_id=PID,
                    purpose="batch_label",
                    example_key="ex_002",
                    batch_label_run_id=run_id,
                    invocation_status="success",
                    schema_valid_core=False,
                    validation_errors_core=["damaged: not a boolean"],
                    model_name="test-model",
                )
            )
            # 1 valid operation
            s.add(
                OperationRecord(
                    inference_invocation_id=generate_uuid4(),
                    project_id=PID,
                    purpose="batch_label",
                    example_key="ex_003",
                    batch_label_run_id=run_id,
                    invocation_status="success",
                    schema_valid_core=True,
                    model_name="test-model",
                )
            )
            s.commit()

        result = get_schema_invalid_manifest(PID, run_id, settings=settings)
        assert not isinstance(result, str), result
        assert result["total_count"] == 2
        assert len(result["schema_invalid_examples"]) == 2
        keys = {e["example_key"] for e in result["schema_invalid_examples"]}
        assert keys == {"ex_001", "ex_002"}

    def test_empty_when_all_valid(self, tmp_path):
        engine, pdir, settings = _setup_export_project(tmp_path)
        run_id = generate_uuid4()
        with Session(engine) as s:
            s.add(
                RunRecord(
                    run_id=run_id,
                    project_id=PID,
                    run_type="batch_label_run",
                    status="completed",
                    examples_total=1,
                )
            )
            s.add(
                OperationRecord(
                    inference_invocation_id=generate_uuid4(),
                    project_id=PID,
                    purpose="batch_label",
                    example_key="ex_001",
                    batch_label_run_id=run_id,
                    invocation_status="success",
                    schema_valid_core=True,
                    model_name="test-model",
                )
            )
            s.commit()

        result = get_schema_invalid_manifest(PID, run_id, settings=settings)
        assert result["total_count"] == 0
        assert result["schema_invalid_examples"] == []

    def test_nonexistent_run_returns_error(self, tmp_path):
        _setup_export_project(tmp_path)
        settings = make_settings(str(tmp_path / "workspace"))
        result = get_schema_invalid_manifest(PID, "no-such-run", settings=settings)
        assert isinstance(result, str)
        assert "not found" in result.lower()


# ══════════════════════════════════════════════════════════════════════════════
# Section I: Structural Validation
# ══════════════════════════════════════════════════════════════════════════════


class TestStructuralValidation:
    def test_all_three_intents_produce_valid_archives(self, tmp_path):
        """Training, evaluation, and testing exports all pass structural checks."""
        engine, pdir, settings = _setup_export_project(
            tmp_path,
            n_verified=2,
            n_pool=2,
            n_auto=1,
        )
        for intent in ("training", "evaluation", "testing"):
            tier = "verified_only"
            result = create_dataset_export(
                PID,
                dataset_intent=intent,
                label_tier_filter=tier,
                settings=settings,
            )
            assert not isinstance(result, str), f"{intent}: {result}"

            # Structural check: top-level array
            annotations = _extract_annotations(result["artifact_refs"]["archive_path"])
            assert isinstance(annotations, list)

            # Structural check: from/value format
            for sample in annotations:
                for turn in sample["conversations"]:
                    assert "from" in turn and "value" in turn

            # Structural check: all image paths resolve to archive
            # members (flat layout — see _build_tar_archive).
            archive_files = set(
                _list_archive_files(result["artifact_refs"]["archive_path"])
            )
            _ = result["dataset_export_id"]  # captured for trace-context only
            for sample in annotations:
                expected = sample["images"][0]
                assert expected in archive_files, (expected, archive_files)

    def test_gpt_value_parseable_as_json(self, tmp_path):
        """Every gpt turn value is a parseable JSON string."""
        engine, pdir, settings = _setup_export_project(
            tmp_path, n_verified=3, n_pool=0, n_auto=0
        )
        result = create_dataset_export(
            PID,
            dataset_intent="training",
            settings=settings,
        )
        annotations = _extract_annotations(result["artifact_refs"]["archive_path"])
        for sample in annotations:
            gpt_value = sample["conversations"][1]["value"]
            assert isinstance(gpt_value, str)
            parsed = json.loads(gpt_value)
            assert isinstance(parsed, dict)


# ══════════════════════════════════════════════════════════════════════════════
# Section: Background export lifecycle (API path)
# ══════════════════════════════════════════════════════════════════════════════


async def _wait_for_terminal_export(
    export_id: str, settings: Any, timeout_s: float = 10.0
) -> dict[str, Any]:
    """Poll the record until the background build reaches a terminal status."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while True:
        rec = get_dataset_export(PID, export_id, settings)
        assert not isinstance(rec, str), rec
        if rec["status"] != "running":
            return rec
        assert loop.time() < deadline, "export still 'running' after timeout"
        await asyncio.sleep(0.05)


class TestBackgroundExportLifecycle:
    """POST-path exports return immediately and build in the background.

    Multi-GB archives take minutes to build; the create call must not
    block the HTTP request on the tar build (it previously did, timing
    out at the proxy at product scale while the server finished silently).
    """

    async def test_start_returns_running_then_completes(self, tmp_path):
        """The create call returns status=running with null artifact_refs
        (nothing guaranteed on disk yet); the record later transitions to
        completed with a real archive + checksum, and an export_completed
        SSE event fires for the UI."""
        _engine, _pdir, settings = _setup_export_project(
            tmp_path, n_verified=3, n_pool=0, n_auto=0
        )
        queue = sse_manager.subscribe(PID)
        try:
            result = await start_dataset_export(
                PID, dataset_intent="training", settings=settings
            )
            assert not isinstance(result, str), result
            assert result["status"] == "running"
            assert result["artifact_refs"] is None
            assert result["manifest_ref"] is None
            assert result["example_count"] == 3

            record = await _wait_for_terminal_export(
                result["dataset_export_id"], settings
            )
            assert record["status"] == "completed"
            refs = record["artifact_refs"]
            assert refs is not None and Path(refs["archive_path"]).exists()
            assert refs["checksum_sha256"]
            assert record["manifest_ref"] is not None
            assert Path(record["manifest_ref"]).exists()
            assert record["completed_at"] is not None
            assert record["progress"] == {"images_written": 3, "images_total": 3}
            ann = _extract_annotations(refs["archive_path"])
            assert sorted(a["id"] for a in ann) == ["ver_000", "ver_001", "ver_002"]

            events: list[str] = []
            while not queue.empty():
                msg = queue.get_nowait()
                if msg is not None:
                    events.append(msg)
            assert any("event: export_completed" in e for e in events), events
        finally:
            sse_manager.unsubscribe(PID, queue)

    async def test_background_export_preserves_run_and_field_filters(self, tmp_path):
        """The API background seam must pass every frozen selection argument."""
        engine, _pdir, settings = _setup_export_project(
            tmp_path, n_verified=0, n_pool=0, n_auto=2
        )
        with Session(engine) as s:
            other_run = (
                s.query(Label).filter_by(project_id=PID, example_key="auto_001").one()
            )
            other_run.batch_label_run_id = "batch-run-2"
            s.commit()

        result = await start_dataset_export(
            PID,
            dataset_intent="training",
            label_tier_filter="auto_labeled_only",
            export_field_mode="core_only",
            batch_label_run_id="batch-run-1",
            selection_filters={"guidance_id": GID},
            settings=settings,
        )
        assert not isinstance(result, str), result
        assert result["example_count"] == 1

        record = await _wait_for_terminal_export(result["dataset_export_id"], settings)
        assert record["status"] == "completed"
        expected_snapshot = {
            "dataset_intent": "training",
            "label_tier_filter": "auto_labeled_only",
            "export_field_mode": "core_only",
            "batch_label_run_id": "batch-run-1",
            "guidance_id": GID,
            "selection_filters": {"guidance_id": GID},
        }
        assert record["selection_definition_snapshot"] == expected_snapshot

        refs = record["artifact_refs"]
        assert refs is not None
        annotations = _extract_annotations(refs["archive_path"])
        sidecar = json.loads(Path(refs["annotations_path"]).read_text())
        assert [row["id"] for row in annotations] == ["auto_000"]
        assert sidecar == annotations
        exported_label = json.loads(annotations[0]["conversations"][1]["value"])
        assert "rationale_note" not in exported_label

        assert record["manifest_ref"] is not None
        manifest = json.loads(Path(record["manifest_ref"]).read_text())
        assert manifest == {
            "dataset_export_id": result["dataset_export_id"],
            **expected_snapshot,
            "example_count": 1,
            "example_keys": ["auto_000"],
            "skipped_missing_images": [],
        }

    async def test_second_start_while_running_returns_conflict(self, tmp_path):
        """One archive build at a time per project: a re-POST while an
        export is still building (the natural click after navigating
        away and back) must 409 instead of doubling the multi-GB build."""
        engine, _pdir, settings = _setup_export_project(
            tmp_path, n_verified=3, n_pool=0, n_auto=0
        )
        with Session(engine) as s:
            s.add(
                DatasetExport(
                    dataset_export_id="de-running",
                    project_id=PID,
                    dataset_intent="training",
                    label_tier_filter="verified_only",
                    export_field_mode="all",
                    guidance_id=GID,
                    example_count=3,
                    selection_definition_snapshot={},
                    status="running",
                )
            )
            s.commit()
        result = await start_dataset_export(
            PID, dataset_intent="training", settings=settings
        )
        assert isinstance(result, str)
        assert result.startswith("conflict:")
        assert "de-running" in result

    async def test_selection_errors_return_before_any_record(self, tmp_path):
        """Selection-phase validation failures surface as HTTP-mappable
        error strings from the create call itself — they never leave a
        running record behind."""
        engine, pdir = setup_project_db(tmp_path, subdirs=("artifacts", "exports"))
        settings = make_settings(str(tmp_path / "workspace"))
        with Session(engine) as s:
            add_standard_project_row(s, PID, pdir, active_guidance_id=None)
            s.commit()

        result = await start_dataset_export(
            PID, dataset_intent="training", settings=settings
        )
        assert result == "No active Guidance configured"
        items, _ = list_dataset_exports(PID, settings=settings)
        assert items == []

    async def test_failed_build_marks_record_failed(self, tmp_path):
        """A build-phase failure flips the record to failed with a reason
        instead of leaving it running forever, and no artifact refs are
        published."""
        _engine, pdir, settings = _setup_export_project(
            tmp_path, n_verified=2, n_pool=0, n_auto=0
        )
        # Occupy the exports/ path with a regular file: the archive
        # directory cannot be created, so the build phase raises.
        exports_dir = Path(pdir) / "exports"
        exports_dir.rmdir()
        exports_dir.write_text("occupied")

        result = await start_dataset_export(
            PID, dataset_intent="training", settings=settings
        )
        assert not isinstance(result, str), result
        assert result["status"] == "running"

        record = await _wait_for_terminal_export(result["dataset_export_id"], settings)
        assert record["status"] == "failed"
        assert record["status_reason"]
        assert record["artifact_refs"] is None
        assert record["manifest_ref"] is None

    def test_recovery_fails_interrupted_exports_and_deletes_partials(self, tmp_path):
        """Startup recovery: a 'running' export left behind by a restart
        is marked failed (backend_restart_interrupted) and its partial
        artifact files are deleted — without this the record would show
        running forever after a crash mid-build."""
        engine, pdir, settings = _setup_export_project(
            tmp_path, n_verified=1, n_pool=0, n_auto=0
        )
        export_id = generate_uuid4()
        with Session(engine) as s:
            s.add(
                DatasetExport(
                    dataset_export_id=export_id,
                    project_id=PID,
                    dataset_intent="training",
                    export_field_mode="all",
                    guidance_id=GID,
                    label_tier_filter="verified_only",
                    selection_definition_snapshot={},
                    status="running",
                    started_at=utc_now(),
                    artifact_refs=None,
                    manifest_ref=None,
                    example_count=1,
                )
            )
            s.commit()
        partial_tar = Path(pdir) / "exports" / f"{export_id}.tar.gz"
        partial_tar.write_bytes(b"partial")

        recover_dataset_exports(settings)

        rec = get_dataset_export(PID, export_id, settings)
        assert not isinstance(rec, str), rec
        assert rec["status"] == "failed"
        assert rec["status_reason"] == "backend_restart_interrupted"
        assert not partial_tar.exists()

    def test_suite_path_stays_synchronous_and_completed(self, tmp_path):
        """persist_dataset_export_in_session (the training-suite path)
        blocks until the archive is fully built and stages a terminal
        completed row visible in the caller's own transaction — it must
        not regress to the background create-running-then-build path,
        whose record the suite could observe mid-build."""
        engine, _pdir, _settings = _setup_export_project(
            tmp_path, n_verified=2, n_pool=0, n_auto=0
        )
        with Session(engine) as s:
            result = persist_dataset_export_in_session(
                s,
                PID,
                dataset_intent="training",
                workspace_root=str(tmp_path / "workspace"),
            )
            assert not isinstance(result, str), result
            # The artifact exists before the caller ever commits.
            assert Path(result["artifact_refs"]["archive_path"]).exists()
            row = s.get(DatasetExport, result["dataset_export_id"])
            assert row is not None
            assert row.status == "completed"
            assert row.completed_at is not None
            assert row.artifact_refs is not None
