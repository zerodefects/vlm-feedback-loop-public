# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Production handoff accepts only complete real-workload AIPerf evidence."""

from __future__ import annotations

from copy import deepcopy

from vlm_feedback_loop.db.models.run import RunRecord
from vlm_feedback_loop.services.serving_validation_service import (
    assess_aiperf_serving_run,
)


def _current_metrics() -> dict:
    workload = {
        "version": "production_vlm_v1",
        "driver": {"name": "aiperf", "version": "0.10.0"},
        "output_limit_mode": "uncapped",
        "kv_cache_reuse": "disabled",
        "selected_count": 84,
        "workload_hash": "workload-sha",
        "prompt_hash": "prompt-sha",
        "evaluated_prompt_hash": "prompt-sha",
    }
    cells = []
    for concurrency in (1, 8, 24):
        cells.append(
            {
                "concurrency": concurrency,
                "status": "passed",
                "failed": False,
                "driver": "aiperf",
                "driver_version": "0.10.0",
                "attempted_request_count": 84,
                "successful_request_count": 84,
                "failed_request_count": 0,
                "failure_rate": 0,
                "latency_p50_ms": 10.0,
                "latency_p90_ms": 20.0,
                "latency_p99_ms": 30.0,
                "request_throughput_rps": 40.0,
            }
        )
    return {"benchmark_workload": workload, "benchmarks": cells}


def _run(metrics: dict) -> RunRecord:
    return RunRecord(
        run_id="run-1",
        project_id="project-1",
        run_type="evaluation_run",
        status="completed",
        created_at="2026-08-05T00:00:00Z",
        metrics=metrics,
    )


def test_complete_aiperf_workload_is_current() -> None:
    """Every configured load cell must replay the complete workload cleanly."""

    result = assess_aiperf_serving_run(
        _run(_current_metrics()), expected_concurrencies=(1, 8, 24)
    )
    assert result.current is True
    assert result.blocker is None


def test_legacy_httpx_latency_sweep_requires_revalidation() -> None:
    """A historical synthetic sweep cannot unlock a production handoff."""

    legacy = {
        "benchmarks": [
            {
                "concurrency": concurrency,
                "driver": "httpx",
                "latency_p50_ms": 10.0,
                "latency_p90_ms": 20.0,
                "latency_p99_ms": 30.0,
                "request_count": 100,
                "error_count": 0,
                "failed": False,
            }
            for concurrency in (1, 8, 24)
        ]
    }
    result = assess_aiperf_serving_run(_run(legacy), expected_concurrencies=(1, 8, 24))
    assert result.current is False
    assert result.blocker == "aiperf_workload_missing"


def test_failed_or_incomplete_aiperf_cell_never_passes() -> None:
    """AIPerf branding alone is insufficient when reliability evidence fails."""

    metrics = deepcopy(_current_metrics())
    metrics["benchmarks"][1]["failed_request_count"] = 1
    metrics["benchmarks"][1]["successful_request_count"] = 83
    metrics["benchmarks"][1]["failure_rate"] = 1 / 84
    result = assess_aiperf_serving_run(_run(metrics), expected_concurrencies=(1, 8, 24))
    assert result.current is False
    assert result.blocker == "aiperf_reliability_gate_failed"


def test_prompt_or_concurrency_drift_fails_closed() -> None:
    """Production handoff cannot reuse a different prompt or load-cell matrix."""

    prompt_drift = deepcopy(_current_metrics())
    prompt_drift["benchmark_workload"]["evaluated_prompt_hash"] = "other"
    result = assess_aiperf_serving_run(
        _run(prompt_drift), expected_concurrencies=(1, 8, 24)
    )
    assert result.blocker == "aiperf_prompt_provenance_mismatch"

    missing_cell = deepcopy(_current_metrics())
    missing_cell["benchmarks"].pop()
    result = assess_aiperf_serving_run(
        _run(missing_cell), expected_concurrencies=(1, 8, 24)
    )
    assert result.blocker == "aiperf_concurrency_cells_incomplete"
