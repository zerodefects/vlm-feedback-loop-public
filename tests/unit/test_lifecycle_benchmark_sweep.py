# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark sweep tests for the Student lifecycle.

When a benchmark concurrency level times out mid-sweep,
the level is recorded in ``skipped_concurrencies``, the previously
completed levels remain persisted, the container is stopped, and the
outer queue continues.
"""

from __future__ import annotations

import asyncio

import pytest

from conftest import make_settings
from vlm_feedback_loop.services import student_nim_lifecycle as lifecycle
from vlm_feedback_loop.services.benchmark_adapter import BenchmarkResult

# ── _run_benchmark_sweep direct unit test ────────────────────────────────────


class _SuccessAdapter:
    """Returns a predictable BenchmarkResult immediately."""

    def __init__(self):
        self.calls: list[int] = []

    async def run(self, **kwargs):
        c = kwargs["concurrency"]
        self.calls.append(c)
        return BenchmarkResult(
            concurrency=c,
            latency_p50_ms=10.0 * c,
            latency_p90_ms=20.0 * c,
            latency_p99_ms=30.0 * c,
            ttft_p50_ms=None,
            ttft_p90_ms=None,
            itl_p50_ms=None,
            itl_p90_ms=None,
            request_count=100,
            error_count=0,
            prometheus={},
            artifact_dir="/tmp",
            driver="stub",
        )


class _TimeoutAt24Adapter:
    """Times out only at concurrency=24 (after c=1 + c=8 succeed)."""

    def __init__(self):
        self.calls: list[int] = []

    async def run(self, **kwargs):
        c = kwargs["concurrency"]
        self.calls.append(c)
        if c == 24:
            await asyncio.sleep(3600)  # exceeds the wait_for deadline
        return BenchmarkResult(
            concurrency=c,
            latency_p50_ms=10.0 * c,
            latency_p90_ms=20.0 * c,
            latency_p99_ms=30.0 * c,
            ttft_p50_ms=None,
            ttft_p90_ms=None,
            itl_p50_ms=None,
            itl_p90_ms=None,
            request_count=100,
            error_count=0,
            prometheus={},
            artifact_dir="/tmp",
            driver="stub",
        )


class _FailedAtC8Adapter:
    """Returns a failed (no-measurement) BenchmarkResult at concurrency=8."""

    def __init__(self):
        self.calls: list[int] = []

    async def run(self, **kwargs):
        c = kwargs["concurrency"]
        self.calls.append(c)
        return BenchmarkResult(
            concurrency=c,
            latency_p50_ms=0.0 if c == 8 else 10.0 * c,
            latency_p90_ms=0.0 if c == 8 else 20.0 * c,
            latency_p99_ms=0.0 if c == 8 else 30.0 * c,
            ttft_p50_ms=None,
            ttft_p90_ms=None,
            itl_p50_ms=None,
            itl_p90_ms=None,
            request_count=0 if c == 8 else 100,
            error_count=0,
            prometheus={},
            artifact_dir="/tmp",
            driver="stub",
            failed=(c == 8),
        )


def _make_snapshot(workspace) -> lifecycle.StudentSnapshot:
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
    )


@pytest.mark.timeout(60)
class TestBenchmarkSweep:
    def test_all_concurrencies_succeed(self, tmp_path, monkeypatch):
        """Happy path: all 3 concurrencies produce results, no skips."""
        snapshot = _make_snapshot(tmp_path)
        adapter = _SuccessAdapter()

        async def _fake_scrape(base_url, **kwargs):
            return {
                "request_failure_total": 0.0,
                "request_success_total": 100.0,
                "gpu_cache_usage_perc": 0.5,
            }

        monkeypatch.setattr(lifecycle, "scrape_prometheus", _fake_scrape)

        captured_events = []

        class _SSE:
            async def emit(self, project_id, event_type, data):
                captured_events.append((event_type, data))

        monkeypatch.setattr(lifecycle, "sse_manager", _SSE())

        settings = make_settings(tmp_path)
        settings.STUDENT_LATENCY_TEST_CONCURRENCIES = [1, 8, 24]

        async def _go():
            return await lifecycle._run_benchmark_sweep(
                project_id=snapshot.project_id,
                snapshot=snapshot,
                workspace_root=str(tmp_path),
                settings=settings,
                base_url="http://localhost:8002/v1",
                served_model="student-foo",
                started_at=0.0,
                deployment_id="dep-id",
                adapter=adapter,
            )

        benchmarks, skipped = asyncio.run(_go())
        assert len(benchmarks) == 3
        assert [b["concurrency"] for b in benchmarks] == [1, 8, 24]
        assert skipped == []
        assert all(
            b["prometheus"]["request_success_total"] == 100.0 for b in benchmarks
        )

        # SSE: one progress event per concurrency level.
        progress_events = [
            e for e in captured_events if e[0] == "nim_benchmark_progress"
        ]
        assert len(progress_events) == 3
        assert [e[1]["concurrency"] for e in progress_events] == [1, 8, 24]

    def test_mid_sweep_timeout_at_c24(self, tmp_path, monkeypatch):
        """concurrency=24 times out → skipped, c=1+c=8
        results retained, sweep breaks immediately, queue continues.
        """
        snapshot = _make_snapshot(tmp_path)
        adapter = _TimeoutAt24Adapter()

        async def _fake_scrape(base_url, **kwargs):
            return {
                "request_failure_total": 0.0,
                "request_success_total": 100.0,
                "gpu_cache_usage_perc": 0.5,
            }

        monkeypatch.setattr(lifecycle, "scrape_prometheus", _fake_scrape)

        events = []

        class _SSE:
            async def emit(self, project_id, event_type, data):
                events.append((event_type, data))

        monkeypatch.setattr(lifecycle, "sse_manager", _SSE())

        settings = make_settings(tmp_path)
        settings.STUDENT_LATENCY_TEST_CONCURRENCIES = [1, 8, 24]
        # Tight benchmark timeout so the c=24 sleep(3600) is cut short.
        settings.NIM_BENCHMARK_TIMEOUT_S = 1

        async def _go():
            return await lifecycle._run_benchmark_sweep(
                project_id=snapshot.project_id,
                snapshot=snapshot,
                workspace_root=str(tmp_path),
                settings=settings,
                base_url="http://localhost:8002/v1",
                served_model="student-foo",
                started_at=0.0,
                deployment_id="dep-id",
                adapter=adapter,
            )

        benchmarks, skipped = asyncio.run(_go())

        # c=1 + c=8 returned successfully; c=24 was skipped.
        assert [b["concurrency"] for b in benchmarks] == [1, 8]
        assert skipped == [24]
        # The adapter was called at every concurrency level even though
        # c=24 timed out (the sweep advanced to it before bailing out).
        assert adapter.calls == [1, 8, 24]

    def test_failed_benchmark_recorded_as_skipped_not_persisted(
        self, tmp_path, monkeypatch
    ):
        """A driver failure (no real measurement) must be recorded as skipped,
        not persisted as a fake zero-latency row the ServingMatrix would
        render as valid. Unlike a timeout, the sweep continues to later
        concurrencies."""
        snapshot = _make_snapshot(tmp_path)
        adapter = _FailedAtC8Adapter()

        async def _fake_scrape(base_url, **kwargs):
            return {}

        monkeypatch.setattr(lifecycle, "scrape_prometheus", _fake_scrape)

        class _SSE:
            async def emit(self, project_id, event_type, data):
                pass

        monkeypatch.setattr(lifecycle, "sse_manager", _SSE())

        settings = make_settings(tmp_path)
        settings.STUDENT_LATENCY_TEST_CONCURRENCIES = [1, 8, 24]

        async def _go():
            return await lifecycle._run_benchmark_sweep(
                project_id=snapshot.project_id,
                snapshot=snapshot,
                workspace_root=str(tmp_path),
                settings=settings,
                base_url="http://localhost:8002/v1",
                served_model="student-foo",
                started_at=0.0,
                deployment_id="dep-id",
                adapter=adapter,
            )

        benchmarks, skipped = asyncio.run(_go())

        # c=8 failed → skipped, not in benchmarks; c=1 and c=24 both retained
        # (a failure does not break the sweep, unlike a timeout).
        assert [b["concurrency"] for b in benchmarks] == [1, 24]
        assert skipped == [8]
        assert adapter.calls == [1, 8, 24]

    def test_empty_concurrency_list_returns_no_benchmarks(self, tmp_path, monkeypatch):
        """Defensive: an empty STUDENT_LATENCY_TEST_CONCURRENCIES setting
        produces no benchmarks and no skips — orchestrator continues.
        """
        snapshot = _make_snapshot(tmp_path)
        adapter = _SuccessAdapter()

        class _SSE:
            async def emit(self, project_id, event_type, data):
                pass

        monkeypatch.setattr(lifecycle, "sse_manager", _SSE())

        settings = make_settings(tmp_path)
        settings.STUDENT_LATENCY_TEST_CONCURRENCIES = []

        async def _go():
            return await lifecycle._run_benchmark_sweep(
                project_id=snapshot.project_id,
                snapshot=snapshot,
                workspace_root=str(tmp_path),
                settings=settings,
                base_url="http://localhost:8002/v1",
                served_model="student-foo",
                started_at=0.0,
                deployment_id="dep-id",
                adapter=adapter,
            )

        benchmarks, skipped = asyncio.run(_go())
        assert benchmarks == []
        assert skipped == []
        assert adapter.calls == []


# ── Persistence — sweep results must survive the SSE payload ────────────────


class TestPersistBenchmarks:
    """The sweep's results are durable state, not just an SSE payload:
    the Compare page reacts to ``nim_benchmark_completed`` by refetching
    run records and reads ``metrics.benchmarks`` — a process restart or
    page reload must still find them."""

    def test_persists_onto_serving_run_metrics(self, tmp_path):
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
                    created_at="2026-07-07T00:00:00Z",
                    evaluation_source="nim",
                    metrics={"overall": {"exact_match_rate": 0.84}},
                )
            )
            session.commit()

        benchmarks = [
            {
                "concurrency": 1,
                "latency_p50_ms": 900,
                "latency_p90_ms": 1200,
                "latency_p99_ms": 1500,
            },
            {
                "concurrency": 8,
                "latency_p50_ms": 1600,
                "latency_p90_ms": 2100,
                "latency_p99_ms": 2600,
            },
        ]
        lifecycle._persist_benchmarks(
            project.project_id,
            settings.WORKSPACE_ROOT,
            run_id=run_id,
            benchmarks=benchmarks,
        )

        with Session(engine) as session:
            run = session.get(RunRecord, run_id)
            assert run is not None
            # The sweep landed AND the existing metrics survived.
            assert run.metrics["benchmarks"] == benchmarks
            assert run.metrics["overall"] == {"exact_match_rate": 0.84}

    def test_missing_run_is_a_noop(self, tmp_path):
        from vlm_feedback_loop.services import project_service

        settings = make_settings(tmp_path)
        project = project_service.create_project(
            "bench-persist-missing", None, settings
        )
        # Must not raise — the lifecycle continues to Stage 8 regardless.
        lifecycle._persist_benchmarks(
            project.project_id,
            settings.WORKSPACE_ROOT,
            run_id="does-not-exist",
            benchmarks=[{"concurrency": 1, "latency_p50_ms": 1.0}],
        )
