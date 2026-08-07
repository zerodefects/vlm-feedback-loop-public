# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serving benchmark sweep and persistence behavior."""

from __future__ import annotations

import asyncio

from conftest import make_settings
from vlm_feedback_loop.services import student_nim_lifecycle as lifecycle
from vlm_feedback_loop.services.benchmark_adapter import BenchmarkResult
from vlm_feedback_loop.services.serving_benchmark_workload import (
    ServingBenchmarkWorkload,
)


class _Adapter:
    def __init__(self, *, fail_at: int | None = None, timeout_at: int | None = None):
        self.fail_at = fail_at
        self.timeout_at = timeout_at
        self.calls: list[int] = []

    async def run(self, **kwargs):
        concurrency = kwargs["concurrency"]
        self.calls.append(concurrency)
        if concurrency == self.timeout_at:
            await asyncio.sleep(3600)
        failed = concurrency == self.fail_at
        count = kwargs["request_count"]
        return BenchmarkResult(
            concurrency=concurrency,
            status="failed" if failed else "passed",
            latency_p50_ms=None if failed else 10.0 * concurrency,
            latency_p90_ms=None if failed else 20.0 * concurrency,
            latency_p99_ms=None if failed else 30.0 * concurrency,
            request_throughput_rps=None if failed else 5.0,
            request_count=0 if failed else count,
            error_count=count if failed else 0,
            artifact_dir=str(kwargs["artifact_dir"]),
            failure_reason="driver_failed" if failed else None,
            failed=failed,
        )


def _snapshot() -> lifecycle.StudentSnapshot:
    return lifecycle.StudentSnapshot(
        student_model_id="0123456789abcdef",
        project_id="aabbccdd-eeee-ffff-aaaa-000000000000",
        student_base_model_config_id="mc-1",
        nim_checkpoint_ref="/tmp/ckpt",
        quantization_method="FP8_DYNAMIC",
        base_model_name="nvidia/cosmos-reason2-2b",
        base_context_window_tokens=256000,
        base_supports_image_input=True,
        base_local_deploy_metadata=None,
        base_structured_generation_support="supported",
        base_visual_budget_mode="mm_processor_size",
        base_visual_budget_support="supported",
        base_max_images_per_request=5,
        dataset_export_ids=[],
        guidance_id="g-1",
        tao_job_id="train-1",
        quantize_tao_job_id=None,
    )


def _workload(tmp_path) -> ServingBenchmarkWorkload:
    temporary_dir = tmp_path / "payload"
    temporary_dir.mkdir(exist_ok=True)
    input_file = temporary_dir / "requests.jsonl"
    input_file.write_text("{}\n")
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir(exist_ok=True)
    return ServingBenchmarkWorkload(
        input_file=input_file,
        artifact_root=artifact_root,
        temporary_dir=temporary_dir,
        request_count=20,
        manifest={"version": "production_vlm_v1", "selected_count": 20},
    )


def _run(tmp_path, monkeypatch, adapter, concurrencies):
    async def _scrape(_base_url, **_kwargs):
        return {
            "request_failure_total": 2.0,
            "request_success_total": 100.0,
            "gpu_cache_usage_perc": None,
        }

    class _SSE:
        async def emit(self, _project_id, _event_type, _data):
            return None

    monkeypatch.setattr(lifecycle, "scrape_prometheus", _scrape)
    monkeypatch.setattr(lifecycle, "sse_manager", _SSE())
    settings = make_settings(tmp_path)
    settings.STUDENT_LATENCY_TEST_CONCURRENCIES = concurrencies
    settings.NIM_BENCHMARK_TIMEOUT_S = 1
    snapshot = _snapshot()
    return asyncio.run(
        lifecycle._run_benchmark_sweep(
            project_id=snapshot.project_id,
            snapshot=snapshot,
            workspace_root=str(tmp_path),
            settings=settings,
            base_url="http://localhost:8002/v1",
            served_model="student-foo",
            serving_run_id="run-1",
            started_at=0.0,
            deployment_id="dep-id",
            adapter=adapter,
            workload=_workload(tmp_path),
        )
    )


def test_all_concurrencies_must_pass(tmp_path, monkeypatch):
    results, failed, manifest = _run(tmp_path, monkeypatch, _Adapter(), [1, 8, 24])
    assert [row["concurrency"] for row in results] == [1, 8, 24]
    assert failed == []
    assert manifest["selected_count"] == 20
    assert all(row["request_count"] == 20 for row in results)
    assert all(row["prometheus"]["request_success_delta"] == 0.0 for row in results)


def test_failed_cell_is_retained_and_fails_gate(tmp_path, monkeypatch):
    adapter = _Adapter(fail_at=8)
    results, failed, _manifest = _run(tmp_path, monkeypatch, adapter, [1, 8, 24])
    assert [row["concurrency"] for row in results] == [1, 8, 24]
    assert results[1]["status"] == "failed"
    assert failed == [8]
    assert adapter.calls == [1, 8, 24]


def test_passing_adapter_flag_cannot_hide_incomplete_request_count(
    tmp_path, monkeypatch
):
    class _IncompleteAdapter(_Adapter):
        async def run(self, **kwargs):
            result = await super().run(**kwargs)
            result.attempted_request_count -= 1
            result.successful_request_count -= 1
            result.request_count -= 1
            return result

    results, failed, _manifest = _run(tmp_path, monkeypatch, _IncompleteAdapter(), [1])

    assert failed == [1]
    assert results[0]["status"] == "failed"
    assert "attempted_19_expected_20" in results[0]["failure_reason"]


def test_timeout_is_explicit_failed_evidence(tmp_path, monkeypatch):
    results, failed, _manifest = _run(
        tmp_path, monkeypatch, _Adapter(timeout_at=24), [1, 8, 24]
    )
    assert [row["concurrency"] for row in results] == [1, 8, 24]
    assert "benchmark_timeout" in results[-1]["failure_reason"]
    assert failed == [24]


def test_persistence_keeps_quality_metrics_and_workload(tmp_path):
    from sqlalchemy.orm import Session

    from vlm_feedback_loop.db.base import generate_uuid4
    from vlm_feedback_loop.db.models.run import RunRecord
    from vlm_feedback_loop.services import project_service

    settings = make_settings(tmp_path)
    project = project_service.create_project("bench-persist", None, settings)
    engine = project_service.get_project_engine(
        project.project_id, settings.WORKSPACE_ROOT
    )
    assert engine is not None
    run_id = generate_uuid4()
    with Session(engine) as session:
        session.add(
            RunRecord(
                run_id=run_id,
                project_id=project.project_id,
                run_type="evaluation_run",
                status="completed",
                created_at="2026-08-04T00:00:00Z",
                metrics={"overall": {"exact_match_rate": 0.84}},
            )
        )
        session.commit()

    rows = [{"concurrency": 1, "latency_p50_ms": 164.25, "status": "passed"}]
    workload = {"version": "production_vlm_v1", "selected_count": 20}
    lifecycle._persist_benchmarks(
        project.project_id,
        settings.WORKSPACE_ROOT,
        run_id=run_id,
        benchmarks=rows,
        workload=workload,
    )

    with Session(engine) as session:
        run = session.get(RunRecord, run_id)
        assert run is not None
        assert run.metrics == {
            "overall": {"exact_match_rate": 0.84},
            "benchmarks": rows,
            "benchmark_workload": workload,
        }
