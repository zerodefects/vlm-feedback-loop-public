# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TAO evaluate re-scoring.

After a TAO ``evaluate`` TAOJob reaches ``succeeded``, this service:

1. Loads the evaluate TAOJob + the Test Pool DatasetExport it scored
   against (from ``dataset_export_ids``, selecting the record with
   ``dataset_intent="evaluation"``).
2. Reads TAO's per-sample prediction outputs from the local artifact
   cache TAO downloaded in ``_handle_succeeded``
   (``{project_dir}/artifacts/tao_jobs/{tao_job_id}/per_sample_predictions``).
3. Reads ground truth from the DatasetExport archive's
   ``annotations.json`` (frozen snapshot — the same file TAO scored
   against, immune to pool drift).
4. Re-scores every prediction using the canonical Exact Match evaluator
   (:mod:`exact_match_evaluator`) — no separate scoring
   implementation.
5. Creates a new evaluation :class:`RunRecord` with
   ``run_type="evaluation_run"``, ``evaluation_source="tao"``,
   ``tao_job_id`` linked, ``tao_native_metrics`` (informational) and
   ``rescored_metrics`` (authoritative) both persisted, plus
   ``coverage_gaps`` via the shared helper re-exported from
   ``evaluation_service``.
6. Flips the paired :class:`StudentModel`'s ``quality_status`` to
   ``"validated"`` and sets ``quality_evaluation_run_id``.

**C2 widened coverage.** When prediction evidence is empty or incomplete,
the frozen archive count disagrees with its DatasetExport record, zero
predictions are schema-valid, or TAO reported ``succeeded`` with no
per-sample-prediction file at all, the re-scoring returns ``None`` and the
caller marks the paired StudentModel ``quality_status="failed"`` via
:func:`student_model_service.mark_student_quality_failed`. This handles
the spec-implicit "succeeded-with-empty" edge case alongside the obvious
TAO ``failed`` path (handled in ``handle_terminal_failure``).

**Returning vs New is N/A for TAO-sourced evaluations** — TAO produces
the first authoritative evaluation for this Student, and there is no
"previous" Run Record for the same Student to diff against. The
``returning_example_keys``, ``new_example_keys``,
``previous_overall_exact_match``, ``previous_pool_version`` fields on
the Run Record are left as ``None``. Coverage gaps ARE still computed
— those depend on the ground truth, not on prior evaluations.
"""

from __future__ import annotations

import asyncio
import json
import logging
import tarfile
from pathlib import Path
from typing import Any, cast

from sqlalchemy.orm import Session

from vlm_feedback_loop.config import Settings
from vlm_feedback_loop.db.base import generate_uuid4, utc_now
from vlm_feedback_loop.db.models.dataset_export import DatasetExport
from vlm_feedback_loop.db.models.guidance import Guidance
from vlm_feedback_loop.db.models.pool import Pool
from vlm_feedback_loop.db.models.run import RunRecord
from vlm_feedback_loop.db.models.student_model import StudentModel
from vlm_feedback_loop.db.models.tao_job import TAOJob
from vlm_feedback_loop.services import evaluation_service, student_model_service
from vlm_feedback_loop.services.exact_match_evaluator import (
    compute_aggregate_metrics,
    match_fields,
    normalize_field_value,
    normalize_ground_truth,
    strip_code_fence,
)
from vlm_feedback_loop.services.hashing import sha256_file
from vlm_feedback_loop.services.project_service import get_project_engine

logger = logging.getLogger("vlm_feedback_loop.services.tao_rescoring_service")
_MISSING = object()


# ── Ground-truth loader (from frozen .tar.gz archive) ──────────────────────


def _load_ground_truth_from_archive(
    archive_path: Path,
) -> tuple[dict[str, dict[str, Any]], frozenset[str]]:
    """Stream-read ``annotations.json`` out of a DatasetExport archive.

    Returns ``({example_key: ground_truth_label_dict}, duplicate_keys)``.
    The gpt-turn ``value`` in each sample is a JSON **string** per the
    export wire format; we parse it here so the caller gets a plain dict
    ready for :func:`match_fields`. Duplicate IDs are retained as separate
    evidence instead of selecting one row by archive order. Samples whose
    gpt turn is missing or unparseable are skipped; the caller rejects any
    duplicate or count mismatch rather than shrinking the denominator.
    """
    ground_truth: dict[str, dict[str, Any]] = {}
    seen_keys: set[str] = set()
    duplicate_keys: set[str] = set()
    if not archive_path.is_file():
        logger.warning("ground_truth: archive missing at %s", archive_path)
        return ground_truth, frozenset()

    try:
        with tarfile.open(str(archive_path), "r:gz") as tf:
            member = tf.getmember("annotations.json")
            extracted = tf.extractfile(member)
            if extracted is None:
                logger.warning(
                    "ground_truth: cannot read annotations.json from %s",
                    archive_path,
                )
                return ground_truth, frozenset()
            raw = extracted.read()
    except KeyError:
        logger.warning("ground_truth: annotations.json not found in %s", archive_path)
        return ground_truth, frozenset()
    except (tarfile.TarError, OSError, EOFError) as exc:
        logger.warning("ground_truth: cannot read archive %s: %s", archive_path, exc)
        return ground_truth, frozenset()

    try:
        ann = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        logger.warning(
            "ground_truth: annotations.json malformed in %s: %s",
            archive_path,
            exc,
        )
        return ground_truth, frozenset()

    if not isinstance(ann, list):
        logger.warning(
            "ground_truth: expected top-level JSON array in %s (got %s)",
            archive_path,
            type(ann).__name__,
        )
        return ground_truth, frozenset()

    ann_list = cast("list[Any]", ann)
    for raw_sample in ann_list:
        if not isinstance(raw_sample, dict):
            continue
        sample = cast("dict[str, Any]", raw_sample)
        key: Any = sample.get("id")
        convs_raw: Any = sample.get("conversations") or []
        if not isinstance(key, str) or not key or not isinstance(convs_raw, list):
            continue
        if key in seen_keys:
            duplicate_keys.add(key)
        else:
            seen_keys.add(key)
        convs = cast("list[Any]", convs_raw)
        # Assistant turn (gpt) carries the serialized label JSON string.
        for raw_turn in convs:
            if not isinstance(raw_turn, dict):
                continue
            turn = cast("dict[str, Any]", raw_turn)
            if turn.get("from") == "gpt":
                value: Any = turn.get("value")
                if not isinstance(value, str):
                    break
                try:
                    parsed = json.loads(value)
                except json.JSONDecodeError:
                    logger.warning(
                        "ground_truth: sample %s gpt turn not valid JSON",
                        key,
                    )
                    break
                if isinstance(parsed, dict) and key not in ground_truth:
                    ground_truth[key] = cast("dict[str, Any]", parsed)
                break
    return ground_truth, frozenset(duplicate_keys)


# ── Prediction loader (from local artifact cache) ──────────────────────────


def _fetch_tao_predictions(
    artifact_cache_dir: Path,
) -> tuple[dict[str, str], frozenset[str]]:
    """Read per-sample predictions TAO downloaded to the local cache.

    TAO ``evaluate`` emits per-sample results as either (a) a single JSON
    file at ``artifact_cache_dir/per_sample_predictions`` containing a
    list of ``{id, prediction, ...}`` samples or a ``{id: prediction}``
    dict, or (b) a ``per_sample_predictions/`` directory of per-example
    JSON files. This loader handles both shapes.

    Returns a ``({example_key: predicted_json_str}, duplicate_keys)`` pair.
    The string is re-parsed later to keep the canonical evaluator's
    validation boundary intact (a mal-JSON prediction counts as
    schema-invalid). Duplicate keys are reported separately because the
    frozen evaluation population permits exactly one prediction per key;
    selecting a winner by artifact order would make quality nondeterministic.
    If the cache is missing or empty, returns ``({}, frozenset())`` — the
    caller treats that as C2 and fails quality_status.
    """
    if not artifact_cache_dir.is_dir():
        logger.warning("rescore: artifact cache dir missing at %s", artifact_cache_dir)
        return {}, frozenset()

    candidate = artifact_cache_dir / "per_sample_predictions"

    def _to_pred_str(val: Any) -> str:
        if isinstance(val, str):
            return val
        # Every value decoded from JSON is serializable back to JSON.
        # Preserving null, scalar, list, and object values keeps malformed
        # but present evidence in the denominator as schema-invalid.
        return json.dumps(val)

    preds: dict[str, str] = {}
    duplicate_keys: set[str] = set()

    def _record_prediction(example_key: str, prediction: Any) -> None:
        if example_key in preds:
            duplicate_keys.add(example_key)
            return
        preds[example_key] = _to_pred_str(prediction)

    # Case A: single file.
    if candidate.is_file():
        try:
            content = candidate.read_text("utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            logger.warning("rescore: failed reading %s: %s", candidate, exc)
            return {}, frozenset()

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            logger.warning("rescore: %s is not valid JSON: %s", candidate, exc)
            return {}, frozenset()

        if isinstance(parsed, list):
            parsed_list = cast("list[Any]", parsed)
            for raw_item in parsed_list:
                if not isinstance(raw_item, dict):
                    continue
                item = cast("dict[str, Any]", raw_item)
                ek: Any = item.get("id") or item.get("example_key")
                pred: Any = _MISSING
                for prediction_key in ("prediction", "predicted", "response"):
                    if prediction_key in item:
                        pred = item[prediction_key]
                        break
                if isinstance(ek, str) and ek and pred is not _MISSING:
                    _record_prediction(ek, pred)
            return preds, frozenset(duplicate_keys)

        if isinstance(parsed, dict):
            parsed_dict = cast("dict[str, Any]", parsed)
            for ek_str, pred_val in parsed_dict.items():
                _record_prediction(ek_str, pred_val)
            return preds, frozenset(duplicate_keys)

        return {}, frozenset()

    # Case B: directory of files.
    if candidate.is_dir():
        for entry in sorted(candidate.iterdir()):
            if not entry.is_file() or not entry.name.endswith(".json"):
                continue
            ek = entry.stem
            try:
                content = entry.read_text("utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            # If the file wraps the prediction in an envelope, unpack; else
            # use the raw contents.
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict) and any(
                k in parsed for k in ("id", "example_key")
            ):
                parsed_dict = cast("dict[str, Any]", parsed)
                prediction: Any = _MISSING
                for prediction_key in ("prediction", "predicted", "response"):
                    if prediction_key in parsed_dict:
                        prediction = parsed_dict[prediction_key]
                        break
                resolved_ek_raw: Any = (
                    parsed_dict.get("id") or parsed_dict.get("example_key") or ek
                )
                resolved_ek = (
                    resolved_ek_raw
                    if isinstance(resolved_ek_raw, str)
                    else str(resolved_ek_raw)
                )
                if prediction is not _MISSING:
                    _record_prediction(resolved_ek, prediction)
            else:
                _record_prediction(ek, content)
        return preds, frozenset(duplicate_keys)

    logger.warning(
        "rescore: no per_sample_predictions file or dir at %s",
        artifact_cache_dir,
    )
    return {}, frozenset()


# ── Prediction normalization + scoring ─────────────────────────────────────


# Fenced-JSON predictions route through the canonical stripper in
# exact_match_evaluator (strip_code_fence) — do not grow a local copy;
# a drifted local copy previously diverged on language tags and
# missing closing fences.


def _parse_and_normalize_prediction(
    pred_str: str | None,
    fields: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Parse + normalize a single prediction against SchemaCore.

    Returns the normalized label dict on success, or ``None`` on parse
    failure / schema-invalid Core. Mirrors
    :func:`exact_match_evaluator.validate_proposal`'s
    ``schema_valid_core`` semantics for Core fields only — Aux
    normalization is irrelevant for TAO re-scoring. ANY invalid or
    missing Core field invalidates the whole prediction (``None``), so
    the caller zero-matches every Core field — keeping the strict
    normalizers authoritative (a numeric boolean proxy must not match
    via Python's ``1 == True``) and denying per-field partial credit to
    schema-invalid samples, exactly like the NIM evaluation path.

    Markdown ```json fences are stripped before ``json.loads`` so
    cosmos-rl's vLLM evaluation path (which doesn't enable structured
    generation) doesn't get rejected for delivering fully-valid JSON
    inside a fence wrapper.
    """
    if pred_str is None:
        return None
    cleaned = strip_code_fence(pred_str)
    try:
        parsed = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(parsed, dict):
        return None
    parsed_dict = cast("dict[str, Any]", parsed)

    normalized: dict[str, Any] = {}
    for f in fields:
        if f.get("role") != "core":
            continue
        res = normalize_field_value(parsed_dict.get(f["field_name"]), f)
        if not res.valid:
            return None
        normalized[f["field_name"]] = res.normalized_value
    return normalized


# ── Main entry point ──────────────────────────────────────────────────────


async def rescore_evaluate_job(
    project_id: str,
    evaluate_tao_job_id: str,
    *,
    settings: Settings,
) -> str | None:
    """Re-score a TAO evaluate job and persist a RunRecord.

    Returns the new RunRecord.run_id on success, or ``None`` on C2 paths
    (missing/incomplete predictions, inconsistent frozen export, no
    schema-valid predictions, no paired StudentModel, or no Test Pool
    DatasetExport). On ``None`` return, the caller is responsible for
    marking ``quality_status="failed"`` on any paired StudentModel.
    """
    engine = get_project_engine(project_id, settings.WORKSPACE_ROOT)
    if engine is None:
        logger.warning("rescore: project %s has no engine", project_id)
        return None

    # ── Phase A: load TAOJob + paired lineage (short read) ─────────────
    with Session(engine) as session:
        evaluate_job = (
            session.query(TAOJob)
            .filter(
                TAOJob.project_id == project_id,
                TAOJob.tao_job_id == evaluate_tao_job_id,
            )
            .first()
        )
        if evaluate_job is None:
            logger.warning("rescore: evaluate TAOJob %s not found", evaluate_tao_job_id)
            return None
        if evaluate_job.action != "evaluate":
            logger.warning(
                "rescore: TAOJob %s action is %r, not 'evaluate' — skipping",
                evaluate_tao_job_id,
                evaluate_job.action,
            )
            return None

        student = student_model_service.find_student_for_evaluate_job(
            session, project_id=project_id, evaluate_job=evaluate_job
        )
        if student is None:
            logger.warning(
                "rescore: no paired StudentModel for evaluate job %s",
                evaluate_tao_job_id,
            )
            return None

        # Find the Test Pool evaluation DatasetExport.
        eval_export: DatasetExport | None = None
        for deid in evaluate_job.dataset_export_ids or []:
            de = (
                session.query(DatasetExport)
                .filter(
                    DatasetExport.project_id == project_id,
                    DatasetExport.dataset_export_id == deid,
                )
                .first()
            )
            if de is None:
                continue
            if de.dataset_intent == "evaluation":
                eval_export = de
                break
        if eval_export is None:
            logger.warning(
                "rescore: no evaluation DatasetExport attached to %s",
                evaluate_tao_job_id,
            )
            return None

        guidance = (
            session.query(Guidance)
            .filter(
                Guidance.project_id == project_id,
                Guidance.guidance_id == eval_export.guidance_id,
            )
            .first()
        )
        guidance_fields: list[dict[str, Any]] = []
        if guidance is not None and guidance.schema:
            guidance_fields = guidance.schema.get("fields", [])

        core_fields = [f for f in guidance_fields if f.get("role") == "core"]

        # Snapshot data needed outside the session.
        archive_path_str = (
            (eval_export.artifact_refs or {}).get("archive_path")
            if eval_export.artifact_refs
            else None
        )
        archive_checksum = (
            (eval_export.artifact_refs or {}).get("checksum_sha256")
            if eval_export.artifact_refs
            else None
        )
        outputs_dict: dict[str, Any] = evaluate_job.outputs or {}
        artifact_cache_dir = outputs_dict.get("artifact_cache_dir")
        _tao_native_raw: Any = outputs_dict.get("tao_aggregate_metrics")
        tao_native_from_outputs: dict[str, Any] = (
            cast("dict[str, Any]", _tao_native_raw)
            if isinstance(_tao_native_raw, dict)
            else {}
        )
        guidance_id = eval_export.guidance_id
        export_field_mode = eval_export.export_field_mode
        expected_example_count = eval_export.example_count
        # Resolve the student-base ModelConfig for the run snapshot — fall
        # back to the chain's train job if lineage chain walking is
        # required (baseline path: evaluate.parent == train; quantize
        # path: evaluate.parent == quantize whose chain has a train).
        model_config_id = evaluate_job.student_base_model_config_id
        student_model_id = student.student_model_id

        # Pool version snapshot: use the latest Pool record for this
        # project. The evaluation snapshot mechanism creates a Pool
        # whenever an evaluation starts; TAO evaluate runs against the
        # frozen export (which preserves the membership already), but we
        # still link pool_version_id for reproducibility.
        pool = (
            session.query(Pool)
            .filter(Pool.project_id == project_id)
            .order_by(Pool.pool_version.desc())
            .first()
        )
        pool_version_id = pool.pool_id if pool is not None else None

    # ── Phase B: load predictions + ground truth (no DB activity) ──────
    if not artifact_cache_dir:
        logger.warning(
            "rescore: evaluate TAOJob %s has no artifact_cache_dir — "
            "marking quality failed (C2)",
            evaluate_tao_job_id,
        )
        return None

    preds, duplicate_prediction_keys = _fetch_tao_predictions(Path(artifact_cache_dir))
    if not preds:
        logger.warning(
            "rescore: no per-sample predictions loaded for %s (C2)",
            evaluate_tao_job_id,
        )
        return None

    if not isinstance(archive_path_str, str) or not archive_path_str:
        logger.warning("rescore: evaluation DatasetExport has no archive_path")
        return None
    if not isinstance(archive_checksum, str) or not archive_checksum:
        logger.warning(
            "rescore: evaluation DatasetExport has no valid archive checksum for %s",
            evaluate_tao_job_id,
        )
        return None
    archive_path = Path(archive_path_str)
    try:
        actual_archive_checksum = await asyncio.to_thread(sha256_file, archive_path)
    except OSError as exc:
        logger.warning("rescore: cannot hash frozen archive %s: %s", archive_path, exc)
        return None
    if actual_archive_checksum.lower() != archive_checksum.lower():
        logger.warning(
            "rescore: frozen archive checksum does not match DatasetExport for %s",
            evaluate_tao_job_id,
        )
        return None

    ground_truth, duplicate_ground_truth_keys = await asyncio.to_thread(
        _load_ground_truth_from_archive, archive_path
    )
    if not ground_truth:
        logger.warning("rescore: no ground truth extracted from %s", archive_path_str)
        return None
    if duplicate_ground_truth_keys:
        logger.warning(
            "rescore: frozen archive contains %d duplicate ground-truth keys "
            "for %s (C2)",
            len(duplicate_ground_truth_keys),
            evaluate_tao_job_id,
        )
        return None
    if len(ground_truth) != expected_example_count:
        logger.warning(
            "rescore: frozen ground-truth count %d does not match "
            "DatasetExport.example_count %d for %s (C2)",
            len(ground_truth),
            expected_example_count,
            evaluate_tao_job_id,
        )
        return None

    missing_prediction_keys = ground_truth.keys() - preds.keys()
    if missing_prediction_keys:
        logger.warning(
            "rescore: %d of %d frozen examples have no materialized "
            "prediction for %s (C2)",
            len(missing_prediction_keys),
            expected_example_count,
            evaluate_tao_job_id,
        )
        return None
    duplicate_frozen_keys = ground_truth.keys() & duplicate_prediction_keys
    if duplicate_frozen_keys:
        logger.warning(
            "rescore: %d of %d frozen examples have multiple materialized "
            "predictions for %s (C2)",
            len(duplicate_frozen_keys),
            expected_example_count,
            evaluate_tao_job_id,
        )
        return None
    extra_prediction_keys = preds.keys() - ground_truth.keys()
    if extra_prediction_keys:
        logger.warning(
            "rescore: ignoring %d predictions outside frozen export for %s",
            len(extra_prediction_keys),
            evaluate_tao_job_id,
        )
    duplicate_extra_keys = duplicate_prediction_keys - ground_truth.keys()
    if duplicate_extra_keys:
        logger.warning(
            "rescore: ignoring duplicate records for %d predictions outside "
            "frozen export for %s",
            len(duplicate_extra_keys),
            evaluate_tao_job_id,
        )

    # Legacy exports may carry pre-normalization ground truth;
    # match_fields assumes canonical values on both sides.
    ground_truth = {
        ek: normalize_ground_truth(gt_json, core_fields)
        for ek, gt_json in ground_truth.items()
    }

    # ── Phase C: score ─────────────────────────────────────────────────
    matched_example_keys: list[str] = []
    all_field_results: list[Any] = []
    schema_valid_count = 0
    for ek, gt in ground_truth.items():
        pred_str = preds[ek]
        normalized_pred = _parse_and_normalize_prediction(pred_str, core_fields)
        if normalized_pred is None:
            # Schema-invalid → produce a zero-match entry per field so
            # the aggregate correctly reflects the failure.
            normalized_pred = {}
        else:
            schema_valid_count += 1
        results = match_fields(normalized_pred, gt, core_fields)
        all_field_results.append(results)
        matched_example_keys.append(ek)

    if schema_valid_count == 0:
        logger.warning(
            "rescore: zero schema-valid predictions for %s — C2",
            evaluate_tao_job_id,
        )
        return None

    # Aggregate via canonical evaluator.
    aggregate = compute_aggregate_metrics(all_field_results, core_fields)
    rescored = evaluation_service.serialize_metrics_overall(aggregate)

    gaps = evaluation_service.compute_coverage_gaps(
        core_fields, ground_truth, matched_example_keys
    )

    # ── Phase D: persist RunRecord + flip StudentModel (short write) ──
    now = utc_now()
    run_id = generate_uuid4()
    with Session(engine) as session:
        run = RunRecord(
            run_id=run_id,
            project_id=project_id,
            run_type="evaluation_run",
            status="completed",
            created_at=now,
            started_at=now,
            completed_at=now,
            pool_version_id=pool_version_id,
            guidance_id=guidance_id,
            model_config_id=model_config_id,
            icl_mode="disabled",
            evaluation_source="tao",
            generation_preset_key=None,
            thinking_mode_effective=None,
            visual_budget_preset_key=None,
            structured_generation_mode_effective=None,
            inference_contract={
                "output_field_mode": export_field_mode,
                "icl_field_mode": export_field_mode,
                "icl_max_examples": None,
            },
            icl_eligible_count_at_start=None,
            icl_eligible_count_at_completion=None,
            tao_job_id=evaluate_tao_job_id,
            tao_native_metrics=tao_native_from_outputs,
            rescored_metrics=rescored,
            previous_pool_version=None,
            returning_example_keys=None,
            new_example_keys=None,
            previous_overall_exact_match=None,
            coverage_gaps=gaps,
            metrics=rescored,
            # Student identity — the Compare page's Teacher-baseline
            # predicate treats a null student_model_config_id as a
            # Teacher run.
            student_model_config_id=student_model_id,
            examples_total=len(all_field_results),
            examples_succeeded=schema_valid_count,
            examples_schema_invalid=len(all_field_results) - schema_valid_count,
            examples_timeout=0,
            examples_endpoint_error=0,
        )
        session.add(run)

        # Flip the paired StudentModel's quality_status.
        student = (
            session.query(StudentModel)
            .filter_by(
                project_id=project_id,
                student_model_id=student_model_id,
            )
            .first()
        )
        if student is not None:
            student.quality_status = "validated"
            student.quality_evaluation_run_id = run_id

        session.commit()

    logger.info(
        "rescore: evaluate job %s → run %s (overall=%.3f, examples=%d, schema_invalid=%d)",
        evaluate_tao_job_id,
        run_id,
        aggregate.overall_exact_match_rate,
        len(all_field_results),
        len(all_field_results) - schema_valid_count,
    )
    return run_id


async def rerescore_student_model_quality(
    project_id: str,
    student_model_id: str,
    *,
    settings: Settings,
) -> dict[str, str | None]:
    """Re-rescore an evaluate TAOJob whose paired StudentModel
    has ``quality_status="failed"`` due to a since-fixed defect in the
    rescoring service.

    The canonical rescore path (:func:`rescore_evaluate_job`) is invoked
    once per evaluate TAOJob during normal post-success polling. If the
    rescoring service is later amended (e.g., markdown-fence
    stripping for vLLM-served evaluate output, or the
    quantize-parent prefix glob in ``_materialize_evaluate_predictions``),
    StudentModels whose original rescore returned a C2 path under the
    old code remain at ``quality_status="failed"`` even though the
    underlying TAO predictions on disk would now parse cleanly. This
    is the operator-callable path to replay those rescores under
    current code.

    **Safety guard.** ``quality_status`` MUST be ``"failed"``. This path
    will not overwrite a ``"validated"`` Student. ``"pending"`` is also
    rejected — those Students are still mid-pipeline; the regular
    polling-driven rescore will reach them.

    **No artifact re-fetch.** The function reuses the per_sample_
    predictions file already materialized under the current
    ``_materialize_evaluate_predictions`` glob. If the file
    is empty or absent, this returns the existing C2 outcome — it does
    NOT re-download the evaluate tarball (that would require an
    upstream change to the ``outputs_fetch_status`` lifecycle).

    Returns a dict with one of these shapes::

        {"run_id": "<new RunRecord>", "quality_status": "validated"}
        {"run_id": null, "quality_status": "failed", "error": "<C2_path>"}
        {"error": "student_not_found"}
        {"error": "student_not_failed", "quality_status": "<current>"}
        {"error": "no_paired_evaluate_job"}
    """
    engine = get_project_engine(project_id, settings.WORKSPACE_ROOT)
    if engine is None:
        return {"error": "project_not_found"}

    with Session(engine) as session:
        student = (
            session.query(StudentModel)
            .filter_by(
                project_id=project_id,
                student_model_id=student_model_id,
            )
            .first()
        )
        if student is None:
            return {"error": "student_not_found"}
        if student.quality_status != "failed":
            return {
                "error": "student_not_failed",
                "quality_status": student.quality_status,
            }

        # Find the paired evaluate TAOJob. The Student's ``tao_job_id``
        # points at the train job (baseline) or the quantize job
        # (quantized variant). The evaluate sibling's
        # ``parent_tao_job_id`` equals the same train/quantize job id.
        parent_id = student.quantize_tao_job_id or student.tao_job_id
        if not parent_id:
            return {"error": "no_paired_evaluate_job"}
        evaluate_job = (
            session.query(TAOJob)
            .filter(
                TAOJob.project_id == project_id,
                TAOJob.parent_tao_job_id == parent_id,
                TAOJob.action == "evaluate",
                TAOJob.status == "succeeded",
            )
            .order_by(TAOJob.created_at.desc())
            .first()
        )
        if evaluate_job is None:
            return {"error": "no_paired_evaluate_job"}
        evaluate_tao_job_id = evaluate_job.tao_job_id

    # Run the canonical rescore path against the persisted predictions.
    run_id = await rescore_evaluate_job(
        project_id,
        evaluate_tao_job_id,
        settings=settings,
    )

    # Re-read final state to report the post-rescore quality_status.
    with Session(engine) as session:
        student = (
            session.query(StudentModel)
            .filter_by(
                project_id=project_id,
                student_model_id=student_model_id,
            )
            .first()
        )
        final_status = student.quality_status if student else None

    if run_id:
        logger.info(
            "rerescore: student %s (eval_job %s) → run %s; quality_status=%s",
            student_model_id,
            evaluate_tao_job_id,
            run_id,
            final_status,
        )
        return {"run_id": run_id, "quality_status": final_status}

    logger.warning(
        "rerescore: student %s (eval_job %s) → C2 path; quality_status=%s",
        student_model_id,
        evaluate_tao_job_id,
        final_status,
    )
    return {
        "run_id": None,
        "quality_status": final_status,
        "error": "rescore_returned_none",
    }


__all__ = [
    "rescore_evaluate_job",
    "rerescore_student_model_quality",
    "_fetch_tao_predictions",
    "_load_ground_truth_from_archive",
]
