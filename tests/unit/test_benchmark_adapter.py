# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavioral tests for the mandatory AIPerf serving adapter."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from vlm_feedback_loop.services.benchmark_adapter import (
    AIPerfAdapter,
    BenchmarkResult,
    parse_aiperf_output,
    select_adapter,
)


def _summary(*, successful: int = 20, failed: int = 0) -> dict:
    result = {
        "schema_version": "1.3",
        "aiperf_version": "0.10.0",
        "start_time": "2026-08-04T10:00:00Z",
        "end_time": "2026-08-04T10:00:10Z",
        "was_cancelled": False,
        "request_latency": {
            "unit": "ms",
            "p50": 164.25,
            "p90": 201.75,
            "p99": 227.5,
            "count": successful,
        },
        "request_throughput": {"unit": "requests/sec", "avg": 2.0},
        "request_count": {"unit": "requests", "avg": float(successful)},
        "input_sequence_length": {"unit": "tokens", "avg": 423.5},
        "output_sequence_length": {"unit": "tokens", "avg": 37.25},
    }
    if failed:
        result["error_request_count"] = {
            "unit": "requests",
            "avg": float(failed),
        }
        result["error_summary"] = [{"count": failed, "message": "HTTP 500"}]
    return result


def test_select_adapter_has_no_synthetic_fallback():
    assert isinstance(select_adapter(), AIPerfAdapter)


def test_parser_preserves_precise_latency_throughput_and_counts(tmp_path):
    output = tmp_path / "profile_export_aiperf.json"
    output.write_text(json.dumps(_summary()))

    result = parse_aiperf_output(
        artifact_dir=tmp_path,
        concurrency=8,
        expected_request_count=20,
        return_code=0,
    )

    assert result.status == "passed"
    assert result.failed is False
    assert result.latency_p50_ms == 164.25
    assert result.latency_p99_ms == 227.5
    assert result.request_throughput_rps == 2.0
    assert result.attempted_request_count == 20
    assert result.successful_request_count == 20
    assert result.failed_request_count == 0
    assert result.failure_rate == 0.0
    assert result.benchmark_duration_s == 10.0
    assert result.input_tokens_mean == 423.5
    assert result.output_tokens_mean == 37.25
    assert result.driver_version == "0.10.0"


def test_any_request_failure_fails_the_concurrency_but_keeps_evidence(tmp_path):
    (tmp_path / "profile_export_aiperf.json").write_text(
        json.dumps(_summary(successful=19, failed=1))
    )

    result = parse_aiperf_output(
        artifact_dir=tmp_path,
        concurrency=24,
        expected_request_count=20,
        return_code=0,
    )

    assert result.failed is True
    assert result.status == "failed"
    assert result.attempted_request_count == 20
    assert result.failed_request_count == 1
    assert result.failure_rate == 0.05
    assert result.latency_p50_ms == 164.25
    assert result.error_summary == [{"count": 1, "message": "HTTP 500"}]


def test_missing_or_non_finite_metrics_never_become_fake_zeroes(tmp_path):
    summary = _summary()
    summary["request_latency"].pop("p99")
    summary["request_throughput"]["avg"] = float("nan")
    (tmp_path / "profile_export_aiperf.json").write_text(json.dumps(summary))

    result = parse_aiperf_output(
        artifact_dir=tmp_path,
        concurrency=1,
        expected_request_count=20,
        return_code=0,
    )

    assert result.failed is True
    assert result.latency_p99_ms is None
    assert result.request_throughput_rps is None
    assert "missing_or_non_finite_required_metric" in (result.failure_reason or "")


def test_process_failure_is_explicit_non_measurement(tmp_path):
    result = parse_aiperf_output(
        artifact_dir=tmp_path,
        concurrency=1,
        expected_request_count=20,
        return_code=2,
    )

    assert result == BenchmarkResult(
        concurrency=1,
        status="failed",
        artifact_dir=str(tmp_path),
        failure_reason="aiperf_exit_2",
        failed=True,
    )
    assert result.latency_p50_ms is None


def test_unpinned_driver_version_fails_closed(tmp_path):
    summary = _summary()
    summary["aiperf_version"] = "0.11.0"
    (tmp_path / "profile_export_aiperf.json").write_text(json.dumps(summary))

    result = parse_aiperf_output(
        artifact_dir=tmp_path,
        concurrency=1,
        expected_request_count=20,
        return_code=0,
    )

    assert result.failed is True
    assert "driver_version_0.11.0_expected_0.10.0" in (result.failure_reason or "")


@pytest.mark.asyncio
async def test_aiperf_cancellation_uses_shared_cleanup_and_propagates(
    tmp_path, monkeypatch
):
    from vlm_feedback_loop.services import benchmark_adapter

    class FakeProcess:
        returncode = None

    async def fake_exec(*_args, **_kwargs):
        return FakeProcess()

    communication = AsyncMock(side_effect=asyncio.CancelledError)
    monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)
    monkeypatch.setattr(
        benchmark_adapter,
        "communicate_with_timeout",
        communication,
    )
    input_file = tmp_path / "workload.jsonl"
    input_file.write_text("{}\n")

    with pytest.raises(asyncio.CancelledError):
        await AIPerfAdapter().run(
            base_url="http://127.0.0.1:8001/v1",
            model="student",
            concurrency=1,
            input_file=input_file,
            artifact_dir=tmp_path / "artifacts",
            request_count=1,
        )

    communication.assert_awaited_once()
