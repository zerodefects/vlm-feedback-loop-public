# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for TAO re-scoring.

Covers:
* Happy path: real predictions + real ground truth → RunRecord with
  both tao_native_metrics and rescored_metrics; paired StudentModel
  flipped to quality_status="validated".
* Empty predictions / no predictions file → None return, caller
  marks failed.
* Schema-invalid predictions counted correctly.
* Ground-truth archive round-trip (tar.gz → dict).
* Coverage gaps computed via the shared helper.
* Returning/New/previous fields are None for TAO-sourced runs.
* Boolean-proxy predictions (string "true", numeric 1/0) treated as
  schema-invalid by the canonical evaluator: the whole sample
  zero-matches every Core field, same as the NIM evaluation path.
"""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from conftest import (
    add_endpoint_row,
    add_guidance_row,
    add_model_config_row,
    add_project_row,
    make_settings,
    open_project_workspace,
)
from vlm_feedback_loop.db.base import generate_uuid4
from vlm_feedback_loop.db.models.dataset_export import DatasetExport
from vlm_feedback_loop.db.models.guidance import Guidance
from vlm_feedback_loop.db.models.pool import Pool
from vlm_feedback_loop.db.models.run import RunRecord
from vlm_feedback_loop.db.models.student_model import StudentModel
from vlm_feedback_loop.db.models.tao_job import TAOJob
from vlm_feedback_loop.db.models.training_suite import TrainingSuite
from vlm_feedback_loop.services import (
    student_model_service,
    tao_rescoring_service,
)
from vlm_feedback_loop.services.hashing import sha256_file

PID = "proj-rescore"
GID = "gid-rescore"


# ── Schema fixture (Core: damage_type enum + severity integer) ─────────────


def _schema_fields() -> list[dict]:
    return [
        {
            "field_id": "f-rationale",
            "field_name": "rationale_note",
            "type": "string",
            "role": "aux",
            "display_order": 0,
        },
        {
            "field_id": "f-damage",
            "field_name": "damage_type",
            "type": "enum",
            "role": "core",
            "allowed_values": ["crush", "dent", "scratch"],
            "display_order": 1,
        },
        {
            "field_id": "f-severity",
            "field_name": "severity",
            "type": "integer",
            "role": "core",
            "minimum": 0,
            "maximum": 4,
            "display_order": 2,
        },
    ]


# ── DB setup ───────────────────────────────────────────────────────────────


def _seed_project(tmp_path: Path):
    engine, pdir, workspace = open_project_workspace(
        tmp_path, PID, register_engine=True, subdirs=()
    )

    schema = {"fields": _schema_fields()}

    with Session(engine) as s:
        add_project_row(s, PID, str(pdir), name="T")
        add_endpoint_row(s, PID, "ep-1", display_name="t", base_url="https://test/v1")
        add_model_config_row(
            s,
            PID,
            "mc-1",
            "ep-1",
            model_name="nvidia/cosmos-reason2-8b",
            eligible_roles=json.dumps(["student_base"]),
            thinking_toggle_mode="qwen_enable_thinking",
            thinking_toggle_support="supported",
            visual_budget_mode="mm_processor_size",
            visual_budget_support="supported",
        )
        add_guidance_row(s, PID, GID, schema, description="Damage classification")
        # Create a Pool for pool_version_id linkage.
        s.add(
            Pool(
                pool_id="pool-v1",
                project_id=PID,
                pool_type="test_pool",
                pool_version=1,
                member_example_keys=["ex-01", "ex-02", "ex-03"],
                member_count=3,
                guidance_id=GID,
            )
        )
        s.commit()
    return engine, workspace, pdir


def _swap_schema_to_boolean_core(engine) -> None:
    """Replace the seeded Guidance schema with boolean + enum Core fields."""
    with Session(engine) as s:
        g = s.query(Guidance).filter_by(guidance_id=GID).one()
        g.schema = {
            "fields": [
                {
                    "field_id": "rn",
                    "field_name": "rationale_note",
                    "type": "string",
                    "role": "aux",
                    "display_order": 0,
                },
                {
                    "field_id": "f-dmg",
                    "field_name": "is_damaged",
                    "type": "boolean",
                    "role": "core",
                    "display_order": 1,
                },
                {
                    "field_id": "f-kind",
                    "field_name": "damage_kind",
                    "type": "enum",
                    "role": "core",
                    "allowed_values": ["dent", "crush"],
                    "display_order": 2,
                },
            ]
        }
        s.commit()


def _gt_sample(example_key: str, label: dict) -> dict:
    """One annotations.json sample whose gpt turn carries *label*."""
    return {
        "id": example_key,
        "images": [f"images/{example_key}.jpg"],
        "conversations": [
            {"from": "human", "value": "<image>\nLabel."},
            {"from": "gpt", "value": json.dumps(label)},
        ],
    }


def _make_eval_archive(path: Path, samples: list[dict]) -> None:
    """Write a Cosmos-RL-shaped .tar.gz at *path* containing annotations.json.

    Each sample must be ``{"id", "images", "conversations": [{human}, {gpt}]}``
    with the gpt turn value being a JSON **string** (the Cosmos-RL wire
    format).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(str(path), "w:gz") as tf:
        ann_bytes = json.dumps(samples).encode("utf-8")
        info = tarfile.TarInfo("annotations.json")
        info.size = len(ann_bytes)
        tf.addfile(info, io.BytesIO(ann_bytes))


def _write_predictions_single_file(cache_dir: Path, preds: dict[str, dict]):
    """Single-file layout: per_sample_predictions is a JSON file."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    arr = [{"id": ek, "prediction": json.dumps(v)} for ek, v in preds.items()]
    (cache_dir / "per_sample_predictions").write_text(json.dumps(arr), encoding="utf-8")


def _write_predictions_dict_file(cache_dir: Path, preds: dict[str, dict]):
    """Single-file layout: dict of {id: prediction} shape."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "per_sample_predictions").write_text(
        json.dumps({k: json.dumps(v) for k, v in preds.items()}),
        encoding="utf-8",
    )


def _write_predictions_directory(cache_dir: Path, preds: dict[str, dict]):
    """Directory layout: per_sample_predictions/ with one .json per example."""
    d = cache_dir / "per_sample_predictions"
    d.mkdir(parents=True, exist_ok=True)
    for ek, v in preds.items():
        (d / f"{ek}.json").write_text(json.dumps(v), encoding="utf-8")


def _seed_chain_with_artifacts(
    engine,
    *,
    chain_id: str,
    pdir: Path,
    eval_archive_path: Path,
    eval_cache_dir: Path,
    eval_example_count: int,
) -> tuple[str, str]:
    """Seed a suite + train/eval chain (no StudentModel — callers register
    one via the real ``register_from_tao_terminal`` service so the paired
    lookup works end-to-end).

    Returns ``(train_id, evaluate_id)``.
    """
    suite_id = generate_uuid4()
    train_id = generate_uuid4()
    eval_id = generate_uuid4()

    with Session(engine) as s:
        s.add(
            TrainingSuite(
                training_suite_id=suite_id,
                project_id=PID,
                idempotency_key=f"idem-{suite_id}",
                guidance_id=GID,
                training_preset="standard",
                export_field_mode="all",
                include_auto_labeled=False,
                training_dataset_export_id="de-train",
                evaluation_dataset_export_id="de-eval",
                selected_student_base_model_config_ids=["mc-1"],
                quantization_schemes=[],
                chain_ids_ordered=[chain_id],
                status="running",
            )
        )
        s.add(
            DatasetExport(
                dataset_export_id="de-train",
                project_id=PID,
                dataset_intent="training",
                export_field_mode="all",
                guidance_id=GID,
                label_tier_filter="verified_only",
                selection_definition_snapshot={},
                artifact_refs={"archive_path": str(pdir / "de-train.tar.gz")},
                manifest_ref="/tmp/m",
                example_count=5,
            )
        )
        s.add(
            DatasetExport(
                dataset_export_id="de-eval",
                project_id=PID,
                dataset_intent="evaluation",
                export_field_mode="all",
                guidance_id=GID,
                label_tier_filter="verified_only",
                selection_definition_snapshot={},
                artifact_refs={
                    "archive_path": str(eval_archive_path),
                    "checksum_sha256": sha256_file(eval_archive_path),
                },
                manifest_ref="/tmp/m",
                example_count=eval_example_count,
            )
        )
        train_cache = pdir / "artifacts" / "tao_jobs" / train_id
        best = train_cache / "best_model"
        best.mkdir(parents=True, exist_ok=True)
        (best / "config.json").write_text("{}")
        (best / "model.safetensors").write_bytes(b"x")
        (best / "tokenizer.json").write_text("{}")

        s.add(
            TAOJob(
                tao_job_id=train_id,
                project_id=PID,
                student_base_model_config_id="mc-1",
                dataset_export_ids=["de-train"],
                action="train",
                status="succeeded",
                training_backend="cosmos_rl_tao_vlm",
                training_policy_type="sft",
                job_config={
                    "training_preset": "standard",
                    "lora_config": {"enable_lora": True},
                    "resolved_training_fields": {
                        "policy": {"model_name_or_path": "nv/cosmos-reason-2-8b"}
                    },
                },
                tao_create_job_request={
                    "kind": "experiment",
                    "action": "train",
                    "specs": {},
                },
                outputs={"artifact_cache_dir": str(train_cache)},
                chain_id=chain_id,
                chain_sequence=1,
            )
        )
        s.add(
            TAOJob(
                tao_job_id=eval_id,
                project_id=PID,
                student_base_model_config_id="mc-1",
                dataset_export_ids=["de-eval"],
                action="evaluate",
                status="succeeded",
                training_backend="cosmos_rl_tao_vlm",
                job_config={"training_preset": "standard"},
                tao_create_job_request={
                    "kind": "experiment",
                    "action": "evaluate",
                    "specs": {},
                },
                outputs={
                    "artifact_cache_dir": str(eval_cache_dir),
                    "tao_aggregate_metrics": {
                        "accuracy": 0.66,
                        "samples": 3,
                    },
                },
                parent_tao_job_id=train_id,
                chain_id=chain_id,
                chain_sequence=2,
            )
        )
        s.commit()

    return train_id, eval_id


async def _prepare_rescore_case(
    tmp_path: Path,
    *,
    samples: list[dict],
    predictions: list[dict],
    eval_example_count: int,
    chain_id: str,
):
    """Build one real frozen-export/prediction case and register its Student."""
    engine, workspace, pdir = _seed_project(tmp_path)
    archive = pdir / f"{chain_id}-eval.tar.gz"
    _make_eval_archive(archive, samples)
    cache = pdir / "artifacts" / "tao_jobs" / f"{chain_id}-eval"
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "per_sample_predictions").write_text(
        json.dumps(predictions),
        encoding="utf-8",
    )
    train_id, eval_id = _seed_chain_with_artifacts(
        engine,
        chain_id=chain_id,
        pdir=pdir,
        eval_archive_path=archive,
        eval_cache_dir=cache,
        eval_example_count=eval_example_count,
    )
    settings = make_settings(workspace)
    student_id = await student_model_service.register_from_tao_terminal(
        PID,
        train_id,
        settings=settings,
    )
    assert student_id is not None
    return engine, settings, student_id, eval_id


# ═══════════════════════════════════════════════════════════════════════════
# Ground-truth archive round-trip
# ═══════════════════════════════════════════════════════════════════════════


class TestGroundTruthLoader:
    def test_parses_top_level_json_array_gpt_turn_string(self, tmp_path):
        samples = [
            {
                "id": "ex-01",
                "images": ["images/ex-01.jpg"],
                "conversations": [
                    {"from": "human", "value": "<image>\nLabel this."},
                    {
                        "from": "gpt",
                        "value": json.dumps({"damage_type": "crush", "severity": 2}),
                    },
                ],
            },
            {
                "id": "ex-02",
                "images": ["images/ex-02.jpg"],
                "conversations": [
                    {"from": "human", "value": "<image>\nLabel this."},
                    {
                        "from": "gpt",
                        "value": json.dumps({"damage_type": "dent", "severity": 1}),
                    },
                ],
            },
        ]
        archive = tmp_path / "eval.tar.gz"
        _make_eval_archive(archive, samples)

        gt, duplicate_keys = tao_rescoring_service._load_ground_truth_from_archive(
            archive
        )
        assert gt == {
            "ex-01": {"damage_type": "crush", "severity": 2},
            "ex-02": {"damage_type": "dent", "severity": 1},
        }
        assert duplicate_keys == frozenset()

    def test_missing_archive_returns_empty_dict(self, tmp_path):
        gt, duplicate_keys = tao_rescoring_service._load_ground_truth_from_archive(
            tmp_path / "nope.tar.gz"
        )
        assert gt == {}
        assert duplicate_keys == frozenset()

    @pytest.mark.parametrize("corruption", ["not-tar", "truncated-gzip"])
    def test_corrupt_archive_returns_empty_evidence(self, tmp_path, corruption):
        archive = tmp_path / "corrupt.tar.gz"
        if corruption == "not-tar":
            archive.write_bytes(b"not a tar archive")
        else:
            _make_eval_archive(
                archive,
                [_gt_sample("ex-01", {"damage_type": "crush", "severity": 2})],
            )
            archive.write_bytes(archive.read_bytes()[:50])

        gt, duplicate_keys = tao_rescoring_service._load_ground_truth_from_archive(
            archive
        )

        assert gt == {}
        assert duplicate_keys == frozenset()

    def test_reports_duplicate_ground_truth_ids_without_last_wins(self, tmp_path):
        archive = tmp_path / "duplicates.tar.gz"
        _make_eval_archive(
            archive,
            [
                _gt_sample("ex-01", {"damage_type": "crush", "severity": 2}),
                _gt_sample("ex-01", {"damage_type": "dent", "severity": 1}),
            ],
        )

        gt, duplicate_keys = tao_rescoring_service._load_ground_truth_from_archive(
            archive
        )

        assert gt == {"ex-01": {"damage_type": "crush", "severity": 2}}
        assert duplicate_keys == frozenset({"ex-01"})

    def test_malformed_annotations_returns_empty_dict(self, tmp_path):
        # Write an archive whose annotations.json is not a list.
        archive = tmp_path / "bad.tar.gz"
        with tarfile.open(str(archive), "w:gz") as tf:
            bad = json.dumps({"not": "an array"}).encode()
            info = tarfile.TarInfo("annotations.json")
            info.size = len(bad)
            tf.addfile(info, io.BytesIO(bad))

        gt, duplicate_keys = tao_rescoring_service._load_ground_truth_from_archive(
            archive
        )
        assert gt == {}
        assert duplicate_keys == frozenset()


# ═══════════════════════════════════════════════════════════════════════════
# Prediction loader shapes
# ═══════════════════════════════════════════════════════════════════════════


class TestPredictionLoader:
    def test_single_file_list_shape(self, tmp_path):
        cache = tmp_path / "cache"
        _write_predictions_single_file(
            cache,
            {
                "ex-01": {"damage_type": "crush", "severity": 2},
                "ex-02": {"damage_type": "dent", "severity": 1},
            },
        )
        preds, duplicate_keys = tao_rescoring_service._fetch_tao_predictions(cache)
        assert set(preds) == {"ex-01", "ex-02"}
        assert json.loads(preds["ex-01"]) == {"damage_type": "crush", "severity": 2}
        assert duplicate_keys == frozenset()

    def test_single_file_dict_shape(self, tmp_path):
        cache = tmp_path / "cache"
        _write_predictions_dict_file(
            cache,
            {
                "ex-01": {"damage_type": "crush", "severity": 2},
            },
        )
        preds, duplicate_keys = tao_rescoring_service._fetch_tao_predictions(cache)
        assert "ex-01" in preds
        assert duplicate_keys == frozenset()

    def test_directory_shape(self, tmp_path):
        cache = tmp_path / "cache"
        _write_predictions_directory(
            cache,
            {
                "ex-01": {"damage_type": "crush", "severity": 2},
            },
        )
        preds, duplicate_keys = tao_rescoring_service._fetch_tao_predictions(cache)
        assert "ex-01" in preds
        assert duplicate_keys == frozenset()

    def test_reports_duplicate_ids_from_list_and_directory_shapes(self, tmp_path):
        list_cache = tmp_path / "list-cache"
        list_cache.mkdir()
        (list_cache / "per_sample_predictions").write_text(
            json.dumps(
                [
                    {"id": "ex-01", "prediction": '{"label":"first"}'},
                    {"id": "ex-01", "prediction": '{"label":"second"}'},
                ]
            ),
            encoding="utf-8",
        )

        list_preds, list_duplicates = tao_rescoring_service._fetch_tao_predictions(
            list_cache
        )

        assert list_preds == {"ex-01": '{"label":"first"}'}
        assert list_duplicates == frozenset({"ex-01"})

        directory_cache = tmp_path / "directory-cache"
        prediction_dir = directory_cache / "per_sample_predictions"
        prediction_dir.mkdir(parents=True)
        for filename, prediction in (
            ("first.json", '{"label":"first"}'),
            ("second.json", '{"label":"second"}'),
        ):
            (prediction_dir / filename).write_text(
                json.dumps({"id": "ex-01", "prediction": prediction}),
                encoding="utf-8",
            )

        directory_preds, directory_duplicates = (
            tao_rescoring_service._fetch_tao_predictions(directory_cache)
        )

        assert directory_preds == {"ex-01": '{"label":"first"}'}
        assert directory_duplicates == frozenset({"ex-01"})

    def test_schema_invalid_value_cannot_hide_a_duplicate_id(self, tmp_path):
        cache = tmp_path / "cache"
        cache.mkdir()
        (cache / "per_sample_predictions").write_text(
            json.dumps(
                [
                    {"id": "ex-01", "prediction": []},
                    {"id": "ex-01", "prediction": {"label": "second"}},
                ]
            ),
            encoding="utf-8",
        )

        preds, duplicate_keys = tao_rescoring_service._fetch_tao_predictions(cache)

        assert preds == {"ex-01": "[]"}
        assert duplicate_keys == frozenset({"ex-01"})

    @pytest.mark.parametrize(
        ("prediction", "serialized"),
        [
            (None, "null"),
            ([], "[]"),
            (7, "7"),
            (False, "false"),
        ],
        ids=("null", "list", "number", "boolean"),
    )
    def test_preserves_present_non_object_json_values(
        self,
        tmp_path,
        prediction,
        serialized,
    ):
        cache = tmp_path / "cache"
        cache.mkdir()
        (cache / "per_sample_predictions").write_text(
            json.dumps([{"id": "ex-01", "prediction": prediction}]),
            encoding="utf-8",
        )

        preds, duplicate_keys = tao_rescoring_service._fetch_tao_predictions(cache)

        assert preds == {"ex-01": serialized}
        assert duplicate_keys == frozenset()

    def test_missing_cache_dir_returns_empty(self, tmp_path):
        preds, duplicate_keys = tao_rescoring_service._fetch_tao_predictions(
            tmp_path / "gone"
        )
        assert preds == {}
        assert duplicate_keys == frozenset()

    def test_empty_cache_dir_returns_empty(self, tmp_path):
        cache = tmp_path / "cache"
        cache.mkdir()
        assert tao_rescoring_service._fetch_tao_predictions(cache) == (
            {},
            frozenset(),
        )


# ═══════════════════════════════════════════════════════════════════════════
# Happy-path re-scoring
# ═══════════════════════════════════════════════════════════════════════════


class TestHappyPath:
    @pytest.mark.asyncio
    async def test_full_rescore_creates_runrecord_and_flips_quality(self, tmp_path):
        engine, workspace, pdir = _seed_project(tmp_path)

        archive = pdir / "de-eval.tar.gz"
        _make_eval_archive(
            archive,
            [
                {
                    "id": "ex-01",
                    "images": ["images/ex-01.jpg"],
                    "conversations": [
                        {"from": "human", "value": "<image>\nLabel."},
                        {
                            "from": "gpt",
                            "value": json.dumps(
                                {"damage_type": "crush", "severity": 2}
                            ),
                        },
                    ],
                },
                {
                    "id": "ex-02",
                    "images": ["images/ex-02.jpg"],
                    "conversations": [
                        {"from": "human", "value": "<image>\nLabel."},
                        {
                            "from": "gpt",
                            "value": json.dumps({"damage_type": "dent", "severity": 1}),
                        },
                    ],
                },
                {
                    "id": "ex-03",
                    "images": ["images/ex-03.jpg"],
                    "conversations": [
                        {"from": "human", "value": "<image>\nLabel."},
                        {
                            "from": "gpt",
                            "value": json.dumps(
                                {"damage_type": "scratch", "severity": 1}
                            ),
                        },
                    ],
                },
            ],
        )

        cache = pdir / "artifacts" / "tao_jobs" / "eval-job"
        _write_predictions_single_file(
            cache,
            {
                # 2 correct, 1 wrong → 2/3 exact match.
                "ex-01": {"damage_type": "crush", "severity": 2},
                "ex-02": {"damage_type": "dent", "severity": 1},
                "ex-03": {"damage_type": "crush", "severity": 1},
                # Outside the frozen export: must not change the denominator.
                "outside-export": {"damage_type": "scratch", "severity": 1},
            },
        )

        train_id, eval_id = _seed_chain_with_artifacts(
            engine,
            chain_id="chain-happy",
            pdir=pdir,
            eval_archive_path=archive,
            eval_cache_dir=cache,
            eval_example_count=3,
        )

        # Update eval TAOJob outputs to point at the real cache.
        with Session(engine) as s:
            ej = s.query(TAOJob).filter_by(tao_job_id=eval_id).one()
            outputs = dict(ej.outputs or {})
            outputs["artifact_cache_dir"] = str(cache)
            ej.outputs = outputs
            s.commit()

        settings = make_settings(workspace)
        # Register the baseline StudentModel first so find_student_for_evaluate_job works.
        student_id = await student_model_service.register_from_tao_terminal(
            PID, train_id, settings=settings
        )
        assert student_id is not None

        run_id = await tao_rescoring_service.rescore_evaluate_job(
            PID, eval_id, settings=settings
        )
        assert run_id is not None

        with Session(engine) as s:
            run = s.query(RunRecord).filter_by(run_id=run_id).one()
            assert run.run_type == "evaluation_run"
            assert run.status == "completed"
            assert run.evaluation_source == "tao"
            assert run.tao_job_id == eval_id
            assert run.icl_mode == "disabled"
            # A TAO quality run IS a Student evaluation — it must carry
            # the Student's id so the Compare page's Teacher-baseline
            # predicate (student_model_config_id == null) never adopts it.
            assert run.student_model_config_id == student_id
            # Both metric maps present
            assert run.tao_native_metrics == {"accuracy": 0.66, "samples": 3}
            assert isinstance(run.rescored_metrics, dict)
            assert run.rescored_metrics["overall"]["example_count"] == 3
            # 2/3 exact match
            assert run.rescored_metrics["overall"]["exact_match_rate"] == pytest.approx(
                2 / 3
            )
            # Returning/New N/A for TAO
            assert run.returning_example_keys is None
            assert run.new_example_keys is None
            assert run.previous_overall_exact_match is None
            assert run.previous_pool_version is None
            # Coverage gap: only 3 of 5 integer values represented
            assert run.coverage_gaps is not None

            student = s.query(StudentModel).filter_by(student_model_id=student_id).one()
            assert student.quality_status == "validated"
            assert student.quality_evaluation_run_id == run_id

    @pytest.mark.parametrize(
        "schema_invalid_prediction",
        ["", [], 7, None],
        ids=("empty-string", "list", "number", "null"),
    )
    @pytest.mark.asyncio
    async def test_schema_invalid_predictions_count_but_do_not_crash(
        self,
        tmp_path,
        schema_invalid_prediction,
    ):
        engine, workspace, pdir = _seed_project(tmp_path)
        archive = pdir / "de-eval.tar.gz"
        _make_eval_archive(
            archive,
            [
                {
                    "id": "ex-01",
                    "images": ["images/ex-01.jpg"],
                    "conversations": [
                        {"from": "human", "value": "<image>\nLabel."},
                        {
                            "from": "gpt",
                            "value": json.dumps(
                                {"damage_type": "crush", "severity": 2}
                            ),
                        },
                    ],
                },
                {
                    "id": "ex-02",
                    "images": ["images/ex-02.jpg"],
                    "conversations": [
                        {"from": "human", "value": "<image>\nLabel."},
                        {
                            "from": "gpt",
                            "value": json.dumps({"damage_type": "dent", "severity": 1}),
                        },
                    ],
                },
            ],
        )
        cache = pdir / "artifacts" / "tao_jobs" / "eval-si"
        cache.mkdir(parents=True, exist_ok=True)
        # One valid and one present-but-schema-invalid response. The latter
        # is evidence, not an absent prediction.
        (cache / "per_sample_predictions").write_text(
            json.dumps(
                [
                    {
                        "id": "ex-01",
                        "prediction": json.dumps(
                            {"damage_type": "crush", "severity": 2}
                        ),
                    },
                    {"id": "ex-02", "prediction": schema_invalid_prediction},
                ]
            )
        )
        train_id, eval_id = _seed_chain_with_artifacts(
            engine,
            chain_id="chain-si",
            pdir=pdir,
            eval_archive_path=archive,
            eval_cache_dir=cache,
            eval_example_count=2,
        )
        settings = make_settings(workspace)
        await student_model_service.register_from_tao_terminal(
            PID, train_id, settings=settings
        )
        run_id = await tao_rescoring_service.rescore_evaluate_job(
            PID, eval_id, settings=settings
        )
        assert run_id is not None
        with Session(engine) as s:
            run = s.query(RunRecord).filter_by(run_id=run_id).one()
            assert run.examples_total == 2
            assert run.examples_succeeded == 1
            assert run.examples_schema_invalid == 1
            # 1/2 exact match (the valid one is a match)
            assert run.rescored_metrics["overall"]["exact_match_rate"] == pytest.approx(
                0.5
            )

    @pytest.mark.asyncio
    async def test_boolean_string_proxy_zero_matches_the_whole_sample(self, tmp_path):
        """Canonical-boolean rule: string "true" is schema-invalid, and a
        schema-invalid sample zero-matches EVERY Core field (no per-field
        partial credit) while counting in ``examples_schema_invalid`` —
        the same treatment the NIM evaluation path gives schema-invalid
        responses.
        """
        engine, workspace, pdir = _seed_project(tmp_path)
        _swap_schema_to_boolean_core(engine)

        archive = pdir / "de-eval-bool.tar.gz"
        _make_eval_archive(
            archive,
            [
                _gt_sample("ex-01", {"is_damaged": True, "damage_kind": "dent"}),
                _gt_sample("ex-02", {"is_damaged": False, "damage_kind": "crush"}),
            ],
        )
        cache = pdir / "artifacts" / "tao_jobs" / "eval-bool"
        # ex-01 uses string "true" — NOT a canonical boolean; its correct
        # damage_kind must earn no credit. ex-02 is fully valid.
        _write_predictions_single_file(
            cache,
            {
                "ex-01": {"is_damaged": "true", "damage_kind": "dent"},
                "ex-02": {"is_damaged": False, "damage_kind": "crush"},
            },
        )
        train_id, eval_id = _seed_chain_with_artifacts(
            engine,
            chain_id="chain-bool",
            pdir=pdir,
            eval_archive_path=archive,
            eval_cache_dir=cache,
            eval_example_count=2,
        )
        settings = make_settings(workspace)
        await student_model_service.register_from_tao_terminal(
            PID, train_id, settings=settings
        )
        run_id = await tao_rescoring_service.rescore_evaluate_job(
            PID, eval_id, settings=settings
        )
        assert run_id is not None
        with Session(engine) as s:
            run = s.query(RunRecord).filter_by(run_id=run_id).one()
            assert run.examples_total == 2
            assert run.examples_succeeded == 1
            assert run.examples_schema_invalid == 1
            overall = run.rescored_metrics["overall"]
            assert overall["exact_match_rate"] == pytest.approx(0.5)
            assert overall["per_field_match_rates"] == {
                "is_damaged": pytest.approx(0.5),
                "damage_kind": pytest.approx(0.5),
            }


# ═══════════════════════════════════════════════════════════════════════════
# Frozen evaluation-population integrity
# ═══════════════════════════════════════════════════════════════════════════


class TestFrozenEvaluationPopulation:
    @pytest.mark.asyncio
    async def test_missing_prediction_cannot_validate_a_subset(self, tmp_path):
        """Every frozen Test Pool key needs prediction evidence."""
        samples = [
            _gt_sample("ex-01", {"damage_type": "crush", "severity": 2}),
            _gt_sample("ex-02", {"damage_type": "dent", "severity": 1}),
            _gt_sample("ex-03", {"damage_type": "scratch", "severity": 0}),
        ]
        engine, settings, student_id, eval_id = await _prepare_rescore_case(
            tmp_path,
            samples=samples,
            predictions=[
                {
                    "id": "ex-01",
                    "prediction": json.dumps({"damage_type": "crush", "severity": 2}),
                }
            ],
            eval_example_count=3,
            chain_id="chain-missing-prediction",
        )

        run_id = await tao_rescoring_service.rescore_evaluate_job(
            PID,
            eval_id,
            settings=settings,
        )

        assert run_id is None
        with Session(engine) as session:
            assert (
                session.query(RunRecord)
                .filter_by(tao_job_id=eval_id, evaluation_source="tao")
                .count()
                == 0
            )
            student = session.get(StudentModel, student_id)
            assert student is not None
            assert student.quality_status == "pending"
            assert student.quality_evaluation_run_id is None

    @pytest.mark.parametrize(
        "duplicate_predictions",
        [
            [
                {"damage_type": "crush", "severity": 2},
                {"damage_type": "dent", "severity": 1},
            ],
            [
                {"damage_type": "dent", "severity": 1},
                {"damage_type": "crush", "severity": 2},
            ],
            [
                {"damage_type": "crush", "severity": 2},
                {"damage_type": "crush", "severity": 2},
            ],
        ],
        ids=("correct-then-wrong", "wrong-then-correct", "identical"),
    )
    @pytest.mark.asyncio
    async def test_duplicate_required_prediction_is_ambiguous(
        self,
        tmp_path,
        duplicate_predictions,
    ):
        """A frozen key must not select a prediction by artifact order."""
        samples = [
            _gt_sample("ex-01", {"damage_type": "crush", "severity": 2}),
            _gt_sample("ex-02", {"damage_type": "dent", "severity": 1}),
        ]
        predictions = [
            {
                "id": "ex-01",
                "prediction": json.dumps(prediction),
            }
            for prediction in duplicate_predictions
        ]
        predictions.append(
            {
                "id": "ex-02",
                "prediction": json.dumps({"damage_type": "dent", "severity": 1}),
            }
        )
        engine, settings, student_id, eval_id = await _prepare_rescore_case(
            tmp_path,
            samples=samples,
            predictions=predictions,
            eval_example_count=2,
            chain_id="chain-duplicate-required",
        )

        run_id = await tao_rescoring_service.rescore_evaluate_job(
            PID,
            eval_id,
            settings=settings,
        )

        assert run_id is None
        with Session(engine) as session:
            assert (
                session.query(RunRecord)
                .filter_by(tao_job_id=eval_id, evaluation_source="tao")
                .count()
                == 0
            )
            student = session.get(StudentModel, student_id)
            assert student is not None
            assert student.quality_status == "pending"
            assert student.quality_evaluation_run_id is None

    @pytest.mark.asyncio
    async def test_schema_invalid_prediction_cannot_hide_required_duplicate(
        self,
        tmp_path,
    ):
        """Duplicate detection precedes prediction-shape validation."""
        samples = [
            _gt_sample("ex-01", {"damage_type": "crush", "severity": 2}),
            _gt_sample("ex-02", {"damage_type": "dent", "severity": 1}),
        ]
        engine, settings, student_id, eval_id = await _prepare_rescore_case(
            tmp_path,
            samples=samples,
            predictions=[
                {"id": "ex-01", "prediction": []},
                {
                    "id": "ex-01",
                    "prediction": {"damage_type": "crush", "severity": 2},
                },
                {
                    "id": "ex-02",
                    "prediction": {"damage_type": "dent", "severity": 1},
                },
            ],
            eval_example_count=2,
            chain_id="chain-invalid-duplicate-required",
        )

        run_id = await tao_rescoring_service.rescore_evaluate_job(
            PID,
            eval_id,
            settings=settings,
        )

        assert run_id is None
        with Session(engine) as session:
            assert (
                session.query(RunRecord)
                .filter_by(tao_job_id=eval_id, evaluation_source="tao")
                .count()
                == 0
            )
            student = session.get(StudentModel, student_id)
            assert student is not None
            assert student.quality_status == "pending"
            assert student.quality_evaluation_run_id is None

    @pytest.mark.parametrize("duplicate_first", [True, False])
    @pytest.mark.asyncio
    async def test_duplicate_ground_truth_rows_are_ambiguous(
        self,
        tmp_path,
        duplicate_first,
    ):
        """Archive row order cannot select the authoritative ground truth."""
        first = _gt_sample("ex-01", {"damage_type": "crush", "severity": 2})
        duplicate = _gt_sample("ex-01", {"damage_type": "dent", "severity": 1})
        samples = [duplicate, first] if duplicate_first else [first, duplicate]
        samples.append(_gt_sample("ex-02", {"damage_type": "dent", "severity": 1}))
        engine, settings, student_id, eval_id = await _prepare_rescore_case(
            tmp_path,
            samples=samples,
            predictions=[
                {
                    "id": "ex-01",
                    "prediction": {"damage_type": "crush", "severity": 2},
                },
                {
                    "id": "ex-02",
                    "prediction": {"damage_type": "dent", "severity": 1},
                },
            ],
            eval_example_count=2,
            chain_id=f"chain-duplicate-ground-truth-{duplicate_first}",
        )

        run_id = await tao_rescoring_service.rescore_evaluate_job(
            PID,
            eval_id,
            settings=settings,
        )

        assert run_id is None
        with Session(engine) as session:
            assert (
                session.query(RunRecord)
                .filter_by(tao_job_id=eval_id, evaluation_source="tao")
                .count()
                == 0
            )
            student = session.get(StudentModel, student_id)
            assert student is not None
            assert student.quality_status == "pending"
            assert student.quality_evaluation_run_id is None

    @pytest.mark.asyncio
    async def test_duplicate_prediction_outside_export_is_ignored(self, tmp_path):
        """Ambiguous evidence for an unknown key cannot change frozen metrics."""
        samples = [
            _gt_sample("ex-01", {"damage_type": "crush", "severity": 2}),
            _gt_sample("ex-02", {"damage_type": "dent", "severity": 1}),
        ]
        predictions = [
            {
                "id": sample["id"],
                "prediction": sample["conversations"][1]["value"],
            }
            for sample in samples
        ]
        predictions.extend(
            [
                {"id": "unknown", "prediction": '{"damage_type":"crush"}'},
                {"id": "unknown", "prediction": '{"damage_type":"dent"}'},
            ]
        )
        engine, settings, student_id, eval_id = await _prepare_rescore_case(
            tmp_path,
            samples=samples,
            predictions=predictions,
            eval_example_count=2,
            chain_id="chain-duplicate-extra",
        )

        run_id = await tao_rescoring_service.rescore_evaluate_job(
            PID,
            eval_id,
            settings=settings,
        )

        assert run_id is not None
        with Session(engine) as session:
            run = session.get(RunRecord, run_id)
            assert run is not None
            assert run.examples_total == 2
            assert run.rescored_metrics["overall"]["exact_match_rate"] == 1.0
            student = session.get(StudentModel, student_id)
            assert student is not None
            assert student.quality_status == "validated"

    @pytest.mark.parametrize("recorded_count", [2, 4])
    @pytest.mark.asyncio
    async def test_archive_count_must_match_export_record(
        self,
        tmp_path,
        recorded_count,
    ):
        """A completed export's row count must agree with its archive."""
        samples = [
            _gt_sample("ex-01", {"damage_type": "crush", "severity": 2}),
            _gt_sample("ex-02", {"damage_type": "dent", "severity": 1}),
            _gt_sample("ex-03", {"damage_type": "scratch", "severity": 0}),
        ]
        predictions = [
            {"id": sample["id"], "prediction": sample["conversations"][1]["value"]}
            for sample in samples
        ]
        engine, settings, student_id, eval_id = await _prepare_rescore_case(
            tmp_path,
            samples=samples,
            predictions=predictions,
            eval_example_count=recorded_count,
            chain_id=f"chain-count-{recorded_count}",
        )

        run_id = await tao_rescoring_service.rescore_evaluate_job(
            PID,
            eval_id,
            settings=settings,
        )

        assert run_id is None
        with Session(engine) as session:
            assert (
                session.query(RunRecord)
                .filter_by(tao_job_id=eval_id, evaluation_source="tao")
                .count()
                == 0
            )
            student = session.get(StudentModel, student_id)
            assert student is not None
            assert student.quality_status == "pending"
            assert student.quality_evaluation_run_id is None

    @pytest.mark.asyncio
    async def test_archive_bytes_must_match_recorded_checksum(self, tmp_path):
        """A mutable local path cannot replace the frozen scoring artifact."""
        samples = [
            _gt_sample("ex-01", {"damage_type": "crush", "severity": 2}),
            _gt_sample("ex-02", {"damage_type": "dent", "severity": 1}),
        ]
        engine, settings, student_id, eval_id = await _prepare_rescore_case(
            tmp_path,
            samples=samples,
            predictions=[
                {
                    "id": "ex-01",
                    "prediction": {"damage_type": "dent", "severity": 1},
                },
                {
                    "id": "ex-02",
                    "prediction": {"damage_type": "dent", "severity": 1},
                },
            ],
            eval_example_count=2,
            chain_id="chain-checksum-mismatch",
        )
        with Session(engine) as session:
            export = session.get(DatasetExport, "de-eval")
            assert export is not None
            archive = Path(export.artifact_refs["archive_path"])
        _make_eval_archive(
            archive,
            [
                _gt_sample("ex-01", {"damage_type": "dent", "severity": 1}),
                _gt_sample("ex-02", {"damage_type": "dent", "severity": 1}),
            ],
        )

        run_id = await tao_rescoring_service.rescore_evaluate_job(
            PID,
            eval_id,
            settings=settings,
        )

        assert run_id is None
        with Session(engine) as session:
            assert session.query(RunRecord).filter_by(tao_job_id=eval_id).count() == 0
            student = session.get(StudentModel, student_id)
            assert student is not None
            assert student.quality_status == "pending"
            assert student.quality_evaluation_run_id is None

    @pytest.mark.parametrize(
        "recorded_checksum",
        [None, "", "not-a-sha256"],
        ids=("missing", "empty", "malformed"),
    )
    @pytest.mark.asyncio
    async def test_export_requires_valid_recorded_archive_checksum(
        self,
        tmp_path,
        recorded_checksum,
    ):
        """Quality evidence without a valid frozen digest is not authoritative."""
        samples = [
            _gt_sample("ex-01", {"damage_type": "crush", "severity": 2}),
            _gt_sample("ex-02", {"damage_type": "dent", "severity": 1}),
        ]
        engine, settings, student_id, eval_id = await _prepare_rescore_case(
            tmp_path,
            samples=samples,
            predictions=[
                {
                    "id": "ex-01",
                    "prediction": {"damage_type": "crush", "severity": 2},
                },
                {
                    "id": "ex-02",
                    "prediction": {"damage_type": "dent", "severity": 1},
                },
            ],
            eval_example_count=2,
            chain_id=f"chain-invalid-checksum-{recorded_checksum}",
        )
        with Session(engine) as session:
            export = session.get(DatasetExport, "de-eval")
            assert export is not None
            refs = dict(export.artifact_refs)
            if recorded_checksum is None:
                refs.pop("checksum_sha256")
            else:
                refs["checksum_sha256"] = recorded_checksum
            export.artifact_refs = refs
            session.commit()

        run_id = await tao_rescoring_service.rescore_evaluate_job(
            PID,
            eval_id,
            settings=settings,
        )

        assert run_id is None
        with Session(engine) as session:
            assert session.query(RunRecord).filter_by(tao_job_id=eval_id).count() == 0
            student = session.get(StudentModel, student_id)
            assert student is not None
            assert student.quality_status == "pending"
            assert student.quality_evaluation_run_id is None

    @pytest.mark.asyncio
    async def test_rerescore_cannot_promote_incomplete_artifacts(self, tmp_path):
        """Operator replay cannot turn an incomplete artifact into quality proof."""
        samples = [
            _gt_sample("ex-01", {"damage_type": "crush", "severity": 2}),
            _gt_sample("ex-02", {"damage_type": "dent", "severity": 1}),
        ]
        engine, settings, student_id, _eval_id = await _prepare_rescore_case(
            tmp_path,
            samples=samples,
            predictions=[
                {
                    "id": "ex-01",
                    "prediction": json.dumps({"damage_type": "crush", "severity": 2}),
                }
            ],
            eval_example_count=2,
            chain_id="chain-rerescore-incomplete",
        )
        with Session(engine) as session:
            student = session.get(StudentModel, student_id)
            assert student is not None
            student.quality_status = "failed"
            student.quality_evaluation_run_id = None
            session.commit()

        result = await tao_rescoring_service.rerescore_student_model_quality(
            PID,
            student_id,
            settings=settings,
        )

        assert result == {
            "run_id": None,
            "quality_status": "failed",
            "error": "rescore_returned_none",
        }
        with Session(engine) as session:
            student = session.get(StudentModel, student_id)
            assert student is not None
            assert student.quality_status == "failed"
            assert student.quality_evaluation_run_id is None
            assert (
                session.query(RunRecord).filter_by(evaluation_source="tao").count() == 0
            )


# ═══════════════════════════════════════════════════════════════════════════
# C2 coverage — empty / missing predictions
# ═══════════════════════════════════════════════════════════════════════════


class TestC2EmptyPredictions:
    @pytest.mark.asyncio
    async def test_no_predictions_file_returns_none(self, tmp_path):
        engine, workspace, pdir = _seed_project(tmp_path)
        archive = pdir / "de-eval.tar.gz"
        _make_eval_archive(
            archive,
            [
                {
                    "id": "ex-01",
                    "images": ["x"],
                    "conversations": [
                        {"from": "human", "value": "<image>"},
                        {
                            "from": "gpt",
                            "value": json.dumps(
                                {"damage_type": "crush", "severity": 1}
                            ),
                        },
                    ],
                }
            ],
        )
        cache = pdir / "artifacts" / "tao_jobs" / "eval-empty"
        cache.mkdir(parents=True, exist_ok=True)  # empty dir, no file
        train_id, eval_id = _seed_chain_with_artifacts(
            engine,
            chain_id="chain-empty",
            pdir=pdir,
            eval_archive_path=archive,
            eval_cache_dir=cache,
            eval_example_count=1,
        )
        settings = make_settings(workspace)
        await student_model_service.register_from_tao_terminal(
            PID, train_id, settings=settings
        )
        run_id = await tao_rescoring_service.rescore_evaluate_job(
            PID, eval_id, settings=settings
        )
        assert run_id is None

    @pytest.mark.asyncio
    async def test_predictions_all_unparseable_returns_none(self, tmp_path):
        engine, workspace, pdir = _seed_project(tmp_path)
        archive = pdir / "de-eval.tar.gz"
        _make_eval_archive(
            archive,
            [
                {
                    "id": "ex-01",
                    "images": ["x"],
                    "conversations": [
                        {"from": "human", "value": "<image>"},
                        {
                            "from": "gpt",
                            "value": json.dumps(
                                {"damage_type": "crush", "severity": 1}
                            ),
                        },
                    ],
                }
            ],
        )
        cache = pdir / "artifacts" / "tao_jobs" / "eval-bad"
        cache.mkdir(parents=True, exist_ok=True)
        (cache / "per_sample_predictions").write_text(
            json.dumps([{"id": "ex-01", "prediction": "{not valid json"}])
        )
        train_id, eval_id = _seed_chain_with_artifacts(
            engine,
            chain_id="chain-unparseable",
            pdir=pdir,
            eval_archive_path=archive,
            eval_cache_dir=cache,
            eval_example_count=1,
        )
        settings = make_settings(workspace)
        await student_model_service.register_from_tao_terminal(
            PID, train_id, settings=settings
        )
        run_id = await tao_rescoring_service.rescore_evaluate_job(
            PID, eval_id, settings=settings
        )
        # 1 attempt, 0 parseable → C2 path returns None
        assert run_id is None

    @pytest.mark.asyncio
    async def test_all_predictions_schema_invalid_returns_none(self, tmp_path):
        """Zero schema-valid predictions is the same C2 outcome as zero
        parseable predictions: no RunRecord is created and the caller
        marks the Student's quality failed, instead of validating a run
        whose metrics score nothing but invalid samples."""
        engine, workspace, pdir = _seed_project(tmp_path)
        _swap_schema_to_boolean_core(engine)
        archive = pdir / "de-eval-allinvalid.tar.gz"
        _make_eval_archive(
            archive,
            [_gt_sample("ex-01", {"is_damaged": True, "damage_kind": "dent"})],
        )
        cache = pdir / "artifacts" / "tao_jobs" / "eval-allinvalid"
        # The only prediction carries a numeric boolean proxy.
        _write_predictions_single_file(
            cache, {"ex-01": {"is_damaged": 1, "damage_kind": "dent"}}
        )
        train_id, eval_id = _seed_chain_with_artifacts(
            engine,
            chain_id="chain-allinvalid",
            pdir=pdir,
            eval_archive_path=archive,
            eval_cache_dir=cache,
            eval_example_count=1,
        )
        settings = make_settings(workspace)
        await student_model_service.register_from_tao_terminal(
            PID, train_id, settings=settings
        )
        run_id = await tao_rescoring_service.rescore_evaluate_job(
            PID, eval_id, settings=settings
        )
        assert run_id is None

    @pytest.mark.asyncio
    async def test_missing_evaluation_dataset_export_returns_none(self, tmp_path):
        engine, workspace, pdir = _seed_project(tmp_path)
        # Create a minimally-valid TAOJob with no dataset_export_ids.
        train_id = generate_uuid4()
        eval_id = generate_uuid4()
        with Session(engine) as s:
            s.add(
                TAOJob(
                    tao_job_id=train_id,
                    project_id=PID,
                    student_base_model_config_id="mc-1",
                    dataset_export_ids=[],
                    action="train",
                    status="succeeded",
                    training_backend="cosmos_rl_tao_vlm",
                    job_config={"training_preset": "standard"},
                    tao_create_job_request={},
                    chain_id="chain-dangling",
                    chain_sequence=1,
                )
            )
            s.add(
                TAOJob(
                    tao_job_id=eval_id,
                    project_id=PID,
                    student_base_model_config_id="mc-1",
                    dataset_export_ids=[],
                    action="evaluate",
                    status="succeeded",
                    training_backend="cosmos_rl_tao_vlm",
                    job_config={},
                    tao_create_job_request={},
                    parent_tao_job_id=train_id,
                    chain_id="chain-dangling",
                    chain_sequence=2,
                )
            )
            s.commit()
        settings = make_settings(workspace)
        run_id = await tao_rescoring_service.rescore_evaluate_job(
            PID, eval_id, settings=settings
        )
        assert run_id is None

    @pytest.mark.asyncio
    async def test_non_evaluate_action_returns_none(self, tmp_path):
        engine, workspace, pdir = _seed_project(tmp_path)
        train_id = generate_uuid4()
        with Session(engine) as s:
            s.add(
                TAOJob(
                    tao_job_id=train_id,
                    project_id=PID,
                    student_base_model_config_id="mc-1",
                    dataset_export_ids=[],
                    action="train",
                    status="succeeded",
                    training_backend="cosmos_rl_tao_vlm",
                    job_config={},
                    tao_create_job_request={},
                    chain_id="c",
                    chain_sequence=1,
                )
            )
            s.commit()
        settings = make_settings(workspace)
        run_id = await tao_rescoring_service.rescore_evaluate_job(
            PID, train_id, settings=settings
        )
        assert run_id is None


# ── Markdown-fenced JSON predictions ───────────────────────────────


class TestStripMarkdownFences:
    """Cosmos-Reason2 served via cosmos-rl's vLLM evaluation path produces
    JSON wrapped in a ```json...``` markdown fence. Without fence
    stripping, strict ``json.loads`` rejects these as schema-invalid and
    the rescoring service marks every sample failed even when the
    underlying payload is perfectly valid JSON.

    The rescoring path routes through the CANONICAL stripper
    (exact_match_evaluator.strip_code_fence) — this file used to carry a
    drifted copy. These tests pin the canonical semantics as seen from
    the rescoring import."""

    def test_unwraps_canonical_fence(self):
        """The exact fence shape cosmos-rl emits in real
        per_sample_predictions files."""
        s = '```json\n{\n  "gesture": "rock"\n}\n```'
        assert tao_rescoring_service.strip_code_fence(s) == '{\n  "gesture": "rock"\n}'

    def test_unwraps_fence_with_no_language_hint(self):
        s = '```\n{"gesture": "paper"}\n```'
        assert tao_rescoring_service.strip_code_fence(s) == '{"gesture": "paper"}'

    def test_idempotent_on_unfenced_json(self):
        """A model that DOES emit raw JSON gets passed through unchanged."""
        s = '{"gesture": "scissors"}'
        assert tao_rescoring_service.strip_code_fence(s) == s

    def test_idempotent_on_blank(self):
        """Fence-less input is returned unchanged (json.loads rejects
        blank input either way — the parse path treats both the same)."""
        assert tao_rescoring_service.strip_code_fence("") == ""
        assert tao_rescoring_service.strip_code_fence("   ") == "   "

    def test_handles_outer_whitespace(self):
        s = '  \n```json\n{"g": "r"}\n```\n  '
        assert tao_rescoring_service.strip_code_fence(s) == '{"g": "r"}'

    def test_missing_closing_fence_left_intact(self):
        """Canonical rule: no closing fence → unchanged, so a truncated
        response surfaces as a parse error instead of being silently
        altered (the old drifted copy stripped the opening fence anyway)."""
        s = '```json\n{"gesture": "rock"}'
        assert tao_rescoring_service.strip_code_fence(s) == s


class TestParseFencedPrediction:
    """Fenced prediction strings as emitted by cosmos-rl parse correctly
    through the full ``_parse_and_normalize_prediction`` path."""

    def test_fenced_prediction_normalizes(self):
        fields = [
            {
                "field_name": "gesture",
                "type": "enum",
                "role": "core",
                "allowed_values": ["rock", "paper", "scissors"],
            }
        ]
        s = '```json\n{\n  "gesture": "rock"\n}\n```'
        result = tao_rescoring_service._parse_and_normalize_prediction(s, fields)
        assert result == {"gesture": "rock"}

    def test_unfenced_prediction_still_works(self):
        fields = [
            {
                "field_name": "gesture",
                "type": "enum",
                "role": "core",
                "allowed_values": ["rock", "paper", "scissors"],
            }
        ]
        result = tao_rescoring_service._parse_and_normalize_prediction(
            '{"gesture": "scissors"}', fields
        )
        assert result == {"gesture": "scissors"}

    def test_truly_malformed_still_returns_none(self):
        """The fix MUST NOT mask genuinely-broken predictions — non-JSON
        garbage still fails parsing and contributes a zero-match example."""
        fields = [
            {
                "field_name": "gesture",
                "type": "enum",
                "role": "core",
                "allowed_values": ["rock", "paper", "scissors"],
            }
        ]
        assert (
            tao_rescoring_service._parse_and_normalize_prediction(
                "not json at all", fields
            )
            is None
        )
        assert (
            tao_rescoring_service._parse_and_normalize_prediction(
                "```json\n{not valid", fields
            )
            is None
        )

    def test_none_input_returns_none(self):
        assert (
            tao_rescoring_service._parse_and_normalize_prediction(
                None, [{"field_name": "x", "type": "string", "role": "core"}]
            )
            is None
        )


class TestInvalidCoreField:
    """``_parse_and_normalize_prediction`` mirrors ``validate_proposal``'s
    ``schema_valid_core`` semantics: ANY invalid Core field invalidates
    the whole prediction (returns None), so the caller zero-matches every
    Core field — no per-field partial credit for invalid samples."""

    def test_one_invalid_core_field_invalidates_whole_prediction(self):
        fields = [f for f in _schema_fields() if f["role"] == "core"]
        assert (
            tao_rescoring_service._parse_and_normalize_prediction(
                '{"damage_type": "bogus", "severity": 2}', fields
            )
            is None
        )

    def test_missing_core_field_invalidates_whole_prediction(self):
        fields = [f for f in _schema_fields() if f["role"] == "core"]
        assert (
            tao_rescoring_service._parse_and_normalize_prediction(
                '{"severity": 2}', fields
            )
            is None
        )

    def test_numeric_boolean_proxy_invalidates_whole_prediction(self):
        """Numeric proxies (1/0) for a boolean Core field are
        schema-invalid — Python's ``1 == True`` must never earn a field
        match."""
        fields = [
            {
                "field_id": "f-b",
                "field_name": "is_damaged",
                "type": "boolean",
                "role": "core",
                "display_order": 0,
            }
        ]
        assert (
            tao_rescoring_service._parse_and_normalize_prediction(
                '{"is_damaged": 1}', fields
            )
            is None
        )


# ═══════════════════════════════════════════════════════════════════════════
# Re-rescore a Student whose quality_status="failed" under since-fixed
# rescore-side behavior.
# ═══════════════════════════════════════════════════════════════════════════


class TestRerescore:
    @pytest.mark.asyncio
    async def test_rerescore_flips_failed_to_validated_when_predictions_now_parse(
        self, tmp_path
    ):
        """A Student whose quality_status='failed' due to a since-fixed
        fix to the rescore service (e.g., markdown-fence
        stripping) flips to validated when re-rescore runs against the
        same on-disk predictions under current code."""
        engine, workspace, pdir = _seed_project(tmp_path)
        archive = pdir / "de-eval-f33.tar.gz"
        _make_eval_archive(
            archive,
            [
                {
                    "id": "ex-01",
                    "images": ["images/ex-01.jpg"],
                    "conversations": [
                        {"from": "human", "value": "<image>\nLabel."},
                        {
                            "from": "gpt",
                            "value": json.dumps(
                                {"damage_type": "crush", "severity": 2}
                            ),
                        },
                    ],
                },
                {
                    "id": "ex-02",
                    "images": ["images/ex-02.jpg"],
                    "conversations": [
                        {"from": "human", "value": "<image>\nLabel."},
                        {
                            "from": "gpt",
                            "value": json.dumps({"damage_type": "dent", "severity": 1}),
                        },
                    ],
                },
            ],
        )
        cache = pdir / "artifacts" / "tao_jobs" / "eval-f33"
        _write_predictions_single_file(
            cache,
            {
                "ex-01": {"damage_type": "crush", "severity": 2},
                "ex-02": {"damage_type": "dent", "severity": 1},
            },
        )

        train_id, eval_id = _seed_chain_with_artifacts(
            engine,
            chain_id="chain-f33",
            pdir=pdir,
            eval_archive_path=archive,
            eval_cache_dir=cache,
            eval_example_count=2,
        )

        with Session(engine) as s:
            ej = s.query(TAOJob).filter_by(tao_job_id=eval_id).one()
            outputs = dict(ej.outputs or {})
            outputs["artifact_cache_dir"] = str(cache)
            ej.outputs = outputs
            s.commit()

        settings = make_settings(workspace)
        student_id = await student_model_service.register_from_tao_terminal(
            PID, train_id, settings=settings
        )
        assert student_id is not None

        # Manually flip the Student to quality_status='failed' to simulate
        # a prior C2 path under old rescore code.
        with Session(engine) as s:
            sm = s.query(StudentModel).filter_by(student_model_id=student_id).one()
            sm.quality_status = "failed"
            sm.quality_evaluation_run_id = None
            s.commit()

        # Run the re-rescore.
        result = await tao_rescoring_service.rerescore_student_model_quality(
            PID, student_id, settings=settings
        )

        assert result.get("error") is None, result
        assert result["run_id"] is not None
        assert result["quality_status"] == "validated"

        # Re-read the Student to confirm the flip.
        with Session(engine) as s:
            sm = s.query(StudentModel).filter_by(student_model_id=student_id).one()
            assert sm.quality_status == "validated"
            assert sm.quality_evaluation_run_id == result["run_id"]
            run = s.query(RunRecord).filter_by(run_id=result["run_id"]).one()
            assert run.evaluation_source == "tao"
            assert run.tao_job_id == eval_id

    @pytest.mark.asyncio
    async def test_rerescore_refuses_validated_student(self, tmp_path):
        """Re-rescore MUST refuse to overwrite a Student whose quality_status
        is already 'validated'. The Student stays unchanged."""
        engine, workspace, pdir = _seed_project(tmp_path)
        archive = pdir / "de-eval-f33-v.tar.gz"
        _make_eval_archive(
            archive,
            [
                {
                    "id": "ex-01",
                    "images": ["images/ex-01.jpg"],
                    "conversations": [
                        {"from": "human", "value": "<image>\nLabel."},
                        {
                            "from": "gpt",
                            "value": json.dumps(
                                {"damage_type": "crush", "severity": 2}
                            ),
                        },
                    ],
                },
            ],
        )
        cache = pdir / "artifacts" / "tao_jobs" / "eval-f33-v"
        _write_predictions_single_file(
            cache, {"ex-01": {"damage_type": "crush", "severity": 2}}
        )
        train_id, _eval_id = _seed_chain_with_artifacts(
            engine,
            chain_id="chain-f33-v",
            pdir=pdir,
            eval_archive_path=archive,
            eval_cache_dir=cache,
            eval_example_count=1,
        )
        settings = make_settings(workspace)
        student_id = await student_model_service.register_from_tao_terminal(
            PID, train_id, settings=settings
        )
        with Session(engine) as s:
            sm = s.query(StudentModel).filter_by(student_model_id=student_id).one()
            sm.quality_status = "validated"
            s.commit()

        result = await tao_rescoring_service.rerescore_student_model_quality(
            PID, student_id, settings=settings
        )
        assert result == {
            "error": "student_not_failed",
            "quality_status": "validated",
        }

    @pytest.mark.asyncio
    async def test_rerescore_refuses_pending_student(self, tmp_path):
        """Re-rescore MUST refuse to operate on a quality_status='pending'
        Student — those are still mid-pipeline."""
        engine, workspace, pdir = _seed_project(tmp_path)
        archive = pdir / "de-eval-f33-p.tar.gz"
        _make_eval_archive(
            archive,
            [
                {
                    "id": "ex-01",
                    "images": ["images/ex-01.jpg"],
                    "conversations": [
                        {"from": "human", "value": "<image>\nLabel."},
                        {
                            "from": "gpt",
                            "value": json.dumps(
                                {"damage_type": "crush", "severity": 2}
                            ),
                        },
                    ],
                },
            ],
        )
        cache = pdir / "artifacts" / "tao_jobs" / "eval-f33-p"
        _write_predictions_single_file(
            cache, {"ex-01": {"damage_type": "crush", "severity": 2}}
        )
        train_id, _eval_id = _seed_chain_with_artifacts(
            engine,
            chain_id="chain-f33-p",
            pdir=pdir,
            eval_archive_path=archive,
            eval_cache_dir=cache,
            eval_example_count=1,
        )
        settings = make_settings(workspace)
        student_id = await student_model_service.register_from_tao_terminal(
            PID, train_id, settings=settings
        )
        # default quality_status from register_from_tao_terminal is
        # 'pending' until the post-train evaluate completes.
        with Session(engine) as s:
            sm = s.query(StudentModel).filter_by(student_model_id=student_id).one()
            sm.quality_status = "pending"
            s.commit()

        result = await tao_rescoring_service.rerescore_student_model_quality(
            PID, student_id, settings=settings
        )
        assert result == {
            "error": "student_not_failed",
            "quality_status": "pending",
        }

    @pytest.mark.asyncio
    async def test_rerescore_refuses_partial_student(self, tmp_path):
        """``quality_status="partial"`` MUST be refused with the same
        ``student_not_failed``
        body the existing path uses for ``"validated"`` and ``"pending"``.
        Partial Students are set by NIM-eval, not by a stale TAO rescore;
        the remediation path is to re-run NIM eval, not :rerescore."""
        engine, workspace, pdir = _seed_project(tmp_path)
        archive = pdir / "de-eval-f33-partial.tar.gz"
        _make_eval_archive(
            archive,
            [
                {
                    "id": "ex-01",
                    "images": ["images/ex-01.jpg"],
                    "conversations": [
                        {"from": "human", "value": "<image>\nLabel."},
                        {
                            "from": "gpt",
                            "value": json.dumps(
                                {"damage_type": "crush", "severity": 2}
                            ),
                        },
                    ],
                },
            ],
        )
        cache = pdir / "artifacts" / "tao_jobs" / "eval-f33-partial"
        _write_predictions_single_file(
            cache, {"ex-01": {"damage_type": "crush", "severity": 2}}
        )
        train_id, _eval_id = _seed_chain_with_artifacts(
            engine,
            chain_id="chain-f33-partial",
            pdir=pdir,
            eval_archive_path=archive,
            eval_cache_dir=cache,
            eval_example_count=1,
        )
        settings = make_settings(workspace)
        student_id = await student_model_service.register_from_tao_terminal(
            PID, train_id, settings=settings
        )
        with Session(engine) as s:
            sm = s.query(StudentModel).filter_by(student_model_id=student_id).one()
            sm.quality_status = "partial"
            s.commit()

        result = await tao_rescoring_service.rerescore_student_model_quality(
            PID, student_id, settings=settings
        )
        assert result == {
            "error": "student_not_failed",
            "quality_status": "partial",
        }

    @pytest.mark.asyncio
    async def test_rerescore_student_not_found(self, tmp_path):
        """Re-rescore returns student_not_found for a missing student id."""
        _engine, workspace, _pdir = _seed_project(tmp_path)
        settings = make_settings(workspace)
        result = await tao_rescoring_service.rerescore_student_model_quality(
            PID, "non-existent-student-id", settings=settings
        )
        assert result == {"error": "student_not_found"}
