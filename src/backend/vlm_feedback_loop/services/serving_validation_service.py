# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Authoritative checks for production-current Student serving validation.

``serving_status="validated"`` predates the production AIPerf workload
contract.  Persisted workspaces can therefore contain successful legacy
``httpx`` latency sweeps that are useful historical evidence but do not prove
the current real-image, cache-disabled, all-concurrency reliability gate.

This module keeps that compatibility decision in the backend so the Compare
screen, deployment handoff, and portable bundle cannot disagree.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

from sqlalchemy.orm import Session

from vlm_feedback_loop.db.models.run import RunRecord
from vlm_feedback_loop.db.models.student_model import StudentModel

AIPERF_DRIVER_NAME = "aiperf"
AIPERF_DRIVER_VERSION = "0.10.0"
PRODUCTION_WORKLOAD_VERSION = "production_vlm_v1"


@dataclass(frozen=True)
class ServingValidationAssessment:
    """Whether a persisted serving result satisfies today's release gate."""

    current: bool
    blocker: str | None = None


def _positive_finite(value: object) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0
    )


def assess_aiperf_serving_run(
    run: RunRecord | None,
    *,
    expected_concurrencies: Sequence[int],
) -> ServingValidationAssessment:
    """Validate the durable AIPerf workload and every configured load cell."""

    if run is None:
        return ServingValidationAssessment(False, "serving_evaluation_run_missing")
    if run.status != "completed":
        return ServingValidationAssessment(False, "serving_evaluation_not_completed")
    if not isinstance(run.metrics, dict):
        return ServingValidationAssessment(False, "aiperf_workload_missing")

    metrics = run.metrics
    workload_raw = metrics.get("benchmark_workload")
    if not isinstance(workload_raw, dict):
        return ServingValidationAssessment(False, "aiperf_workload_missing")
    workload = cast("dict[str, Any]", workload_raw)
    driver_raw = workload.get("driver")
    if not isinstance(driver_raw, dict):
        return ServingValidationAssessment(False, "aiperf_driver_required")
    driver = cast("dict[str, Any]", driver_raw)
    if driver.get("name") != AIPERF_DRIVER_NAME:
        return ServingValidationAssessment(False, "aiperf_driver_required")
    if driver.get("version") != AIPERF_DRIVER_VERSION:
        return ServingValidationAssessment(False, "aiperf_version_mismatch")
    if workload.get("version") != PRODUCTION_WORKLOAD_VERSION:
        return ServingValidationAssessment(False, "aiperf_workload_version_mismatch")
    if workload.get("output_limit_mode") != "uncapped":
        return ServingValidationAssessment(False, "aiperf_output_policy_mismatch")
    if workload.get("kv_cache_reuse") != "disabled":
        return ServingValidationAssessment(False, "aiperf_cache_policy_mismatch")

    selected_count = workload.get("selected_count")
    if not isinstance(selected_count, int) or isinstance(selected_count, bool):
        return ServingValidationAssessment(False, "aiperf_request_count_invalid")
    if selected_count <= 0:
        return ServingValidationAssessment(False, "aiperf_request_count_invalid")
    workload_hash = workload.get("workload_hash")
    prompt_hash = workload.get("prompt_hash")
    evaluated_prompt_hash = workload.get("evaluated_prompt_hash")
    if not isinstance(workload_hash, str) or not workload_hash:
        return ServingValidationAssessment(False, "aiperf_workload_hash_missing")
    if (
        not isinstance(prompt_hash, str)
        or not prompt_hash
        or prompt_hash != evaluated_prompt_hash
    ):
        return ServingValidationAssessment(False, "aiperf_prompt_provenance_mismatch")

    expected = tuple(int(value) for value in expected_concurrencies)
    benchmarks_raw = metrics.get("benchmarks")
    if not isinstance(benchmarks_raw, list):
        return ServingValidationAssessment(False, "aiperf_concurrency_cells_incomplete")
    benchmarks = cast("list[Any]", benchmarks_raw)
    if not expected or len(benchmarks) != len(expected):
        return ServingValidationAssessment(False, "aiperf_concurrency_cells_incomplete")

    by_concurrency: dict[int, dict[str, Any]] = {}
    for raw_cell in benchmarks:
        if not isinstance(raw_cell, dict):
            return ServingValidationAssessment(False, "aiperf_concurrency_cell_invalid")
        cell = cast("dict[str, Any]", raw_cell)
        concurrency = cell.get("concurrency")
        if not isinstance(concurrency, int) or isinstance(concurrency, bool):
            return ServingValidationAssessment(False, "aiperf_concurrency_cell_invalid")
        if concurrency in by_concurrency:
            return ServingValidationAssessment(
                False, "aiperf_concurrency_cells_incomplete"
            )
        by_concurrency[concurrency] = cell

    if set(by_concurrency) != set(expected):
        return ServingValidationAssessment(False, "aiperf_concurrency_cells_incomplete")

    for concurrency in expected:
        cell = by_concurrency[concurrency]
        if (
            cell.get("status") != "passed"
            or cell.get("failed") is not False
            or cell.get("driver") != AIPERF_DRIVER_NAME
            or cell.get("driver_version") != AIPERF_DRIVER_VERSION
        ):
            return ServingValidationAssessment(False, "aiperf_concurrency_cell_failed")
        if (
            cell.get("attempted_request_count") != selected_count
            or cell.get("successful_request_count") != selected_count
            or cell.get("failed_request_count") != 0
            or cell.get("failure_rate") != 0
        ):
            return ServingValidationAssessment(False, "aiperf_reliability_gate_failed")
        if not all(
            _positive_finite(cell.get(field))
            for field in (
                "latency_p50_ms",
                "latency_p90_ms",
                "latency_p99_ms",
                "request_throughput_rps",
            )
        ):
            return ServingValidationAssessment(False, "aiperf_metrics_incomplete")

    return ServingValidationAssessment(True)


def annotate_student_serving_validation(
    session: Session,
    students: Sequence[StudentModel],
    *,
    expected_concurrencies: Sequence[int],
) -> None:
    """Attach response-only current-contract fields without mutating history."""

    run_ids = {
        student.serving_evaluation_run_id
        for student in students
        if student.serving_evaluation_run_id
    }
    runs = (
        session.query(RunRecord).filter(RunRecord.run_id.in_(run_ids)).all()
        if run_ids
        else []
    )
    run_by_id = {run.run_id: run for run in runs}
    for student in students:
        if student.serving_status != "validated":
            assessment = ServingValidationAssessment(
                False, f"serving_status_{student.serving_status}"
            )
        else:
            assessment = assess_aiperf_serving_run(
                run_by_id.get(student.serving_evaluation_run_id or ""),
                expected_concurrencies=expected_concurrencies,
            )
        response_student = cast("Any", student)
        response_student.serving_benchmark_current = assessment.current
        response_student.serving_benchmark_blocker = assessment.blocker


__all__ = [
    "AIPERF_DRIVER_NAME",
    "AIPERF_DRIVER_VERSION",
    "PRODUCTION_WORKLOAD_VERSION",
    "ServingValidationAssessment",
    "annotate_student_serving_validation",
    "assess_aiperf_serving_run",
]
