# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Student NIM lifecycle orchestrator.

Covers the base sequence, with adjacent behaviors tested elsewhere:

  - Base sequence: preflight pass → docker_run → health → smoke → register
    → eval → benchmark → stop sequence; preflight fail short-circuits
    + emits an Action Request; SSE stage ordering; container is stopped
    on every failure category; external mode skips local stages;
    serving_status flip only on success.
  - Per-check Action Request: covered in
    test_local_nim_deploy_generator.py.
  - Variant timeout mid-sweep: covered in
    test_lifecycle_benchmark_sweep.py.
  - Pinned AIPerf parsing and failure behavior: covered in
    test_benchmark_adapter.py.

The lifecycle has many side-effecting collaborators
(``deploy_local_nim``, ``stop_local_nim``, ``start_evaluation_run``,
``select_adapter``, ``scrape_prometheus``, etc.). We monkey-patch each
dependency at the lifecycle module's import path so the orchestrator
can drive its state machine without docker, GPUs, or NIM endpoints.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest
from sqlalchemy.orm import Session

from conftest import make_stub_settings
from vlm_feedback_loop.config import Settings
from vlm_feedback_loop.db.base import generate_uuid4, utc_now
from vlm_feedback_loop.db.models.local_nim_deployment import LocalNimDeployment
from vlm_feedback_loop.db.models.run import RunRecord
from vlm_feedback_loop.db.models.student_model import StudentModel
from vlm_feedback_loop.services import student_nim_lifecycle as lifecycle
from vlm_feedback_loop.services.benchmark_adapter import BenchmarkResult
from vlm_feedback_loop.services.serving_benchmark_workload import (
    ServingBenchmarkWorkload,
)

# ── Fixtures + helpers ────────────────────────────────────────────────────────


def _make_project_and_student(
    client, *, with_dataset_export: bool = True
) -> tuple[str, str, Settings]:
    """Bootstrap a project + Student + base ModelConfig + DatasetExport."""
    from vlm_feedback_loop.db.models.dataset_export import DatasetExport
    from vlm_feedback_loop.db.models.model_config import ModelConfig
    from vlm_feedback_loop.db.models.nim_endpoint import NimEndpoint
    from vlm_feedback_loop.routers.projects import get_current_settings
    from vlm_feedback_loop.services.project_service import get_project_engine

    resp = client.post(
        "/v1/projects",
        json={"name": "step-12-1-lifecycle", "description": "x"},
    )
    assert resp.status_code == 201
    pid = resp.json()["project_id"]

    settings = client.app.dependency_overrides[get_current_settings]()
    engine = get_project_engine(pid, settings.WORKSPACE_ROOT)

    base_mc_id = generate_uuid4()
    sid = generate_uuid4()
    de_id = generate_uuid4()
    endpoint_id = generate_uuid4()

    with Session(engine) as session:
        session.add(
            NimEndpoint(
                endpoint_id=endpoint_id,
                project_id=pid,
                display_name="seed-hosted",
                endpoint_mode="hosted",
                base_url="https://integrate.api.nvidia.com/v1",
                auth_mode="bearer",
                source_kind="seeded_hosted",
            )
        )
        session.add(
            ModelConfig(
                model_config_id=base_mc_id,
                project_id=pid,
                endpoint_id=endpoint_id,
                model_name="nvidia/cosmos-reason2-2b",
                context_window_tokens=256000,
                eligible_roles=["teacher", "student_base"],
                supports_image_input=True,
                local_deploy_metadata={
                    "nim_container_image": "nvcr.io/nim/nvidia/cosmos-reason2-2b:1.6.0",
                    "nim_gpu_memory_minimum_gb": 36,
                    "preferred_host_port": 8000,
                },
            )
        )
        if with_dataset_export:
            session.add(
                DatasetExport(
                    dataset_export_id=de_id,
                    project_id=pid,
                    dataset_intent="training",
                    export_field_mode="core_only",
                    guidance_id="g-1",
                    label_tier_filter="verified",
                    selection_definition_snapshot={},
                    artifact_refs={},
                    manifest_ref="manifest",
                    example_count=10,
                )
            )
        session.add(
            StudentModel(
                student_model_id=sid,
                project_id=pid,
                student_base_model_config_id=base_mc_id,
                tao_job_id="tao-1",
                guidance_id="g-1",
                dataset_export_ids=[de_id] if with_dataset_export else [],
                training_preset="standard",
                lora_config={"enable_lora": True},
                created_at="2026-04-29T00:00:00Z",
                checkpoint_packaging_status="validated",
                nim_checkpoint_ref="/tmp/student-ckpt",
                quality_status="validated",
                serving_status="not_attempted",
                quantization_method="FP8_DYNAMIC",
            )
        )
        session.commit()
    return pid, sid, settings


# ── Stub objects ─────────────────────────────────────────────────────────────


@dataclass
class _PreflightCheck:
    check_name: str
    passed: bool
    diagnostic: str


@dataclass
class _Preflight:
    all_passed: bool
    checks: list[_PreflightCheck]
    docker_run_command: str = "docker run -d ..."


def _make_preflight(passed: bool = True) -> _Preflight:
    checks = [
        _PreflightCheck("docker", True, "ok"),
        _PreflightCheck("nvidia_toolkit", True, "ok"),
        _PreflightCheck(
            "gpu_memory",
            passed,
            "ok" if passed else "24 GB < 36 GB required",
        ),
        _PreflightCheck("ngc_api_key", True, "ok"),
        _PreflightCheck("image_pullable", True, "ok"),
        _PreflightCheck("model_profile", True, "ok"),
    ]
    return _Preflight(all_passed=passed, checks=checks)


class _StubAdapter:
    """In-memory benchmark adapter that returns canned results immediately."""

    driver_name = "stub"

    def __init__(self, *, error_count: int = 0):
        self.calls: list[dict[str, Any]] = []
        self._error_count = error_count

    async def run(self, **kwargs):
        self.calls.append(kwargs)
        return BenchmarkResult(
            concurrency=kwargs["concurrency"],
            latency_p50_ms=10.0,
            latency_p90_ms=20.0,
            latency_p99_ms=30.0,
            request_throughput_rps=25.0,
            ttft_p50_ms=None,
            ttft_p90_ms=None,
            itl_p50_ms=None,
            itl_p90_ms=None,
            request_count=kwargs["request_count"],
            error_count=self._error_count,
            prometheus={},
            artifact_dir="/tmp/bench",
            driver=self.driver_name,
            status="failed" if self._error_count else "passed",
            failed=bool(self._error_count),
            failure_reason="request_failure" if self._error_count else None,
        )


class _CapturingSSE:
    def __init__(self):
        self.events: list[tuple[str, str, dict[str, Any]]] = []

    async def emit(self, project_id, event_type, data):
        self.events.append((project_id, event_type, dict(data)))


# ── Common monkeypatch fixture ───────────────────────────────────────────────


@pytest.fixture()
def patched_lifecycle(monkeypatch, tmp_path):
    """Patch all external collaborators of the lifecycle module."""
    sse = _CapturingSSE()
    state = {
        "preflight": _make_preflight(passed=True),
        "deploy_status": "starting",
        "smoke_ok": True,
        "eval_run_status": "completed",
        "eval_examples_total": None,
        "eval_examples_succeeded": None,
        "deployment_id": "dep-id",
        "endpoint_url": "http://localhost:8002/v1",
        "stop_called": [],
        "ar_calls": [],
        "scrape_calls": [],
        "preflight_calls": [],
        "deploy_calls": [],  # captures kwargs passed to deploy_local_nim
        "displaced_returned": [],  # seeds rows the fake "displaces"
        "restore_calls": [],  # captures _restore_displaced_deployment invocations
    }

    async def _fake_resolve_gpu(role, explicit_gpu, workspace_root):
        return explicit_gpu or "device=0"

    async def _fake_preflight(**kwargs):
        state["preflight_calls"].append(kwargs)
        return state["preflight"]

    async def _fake_deploy(**kwargs):
        state["deploy_calls"].append(kwargs)
        # Build a stub LocalNimDeployment row with whatever the test asked for.
        dep = LocalNimDeployment(
            local_nim_deployment_id=state["deployment_id"],
            project_id=kwargs["project_id"],
            model_config_id=kwargs["model_config_id"],
            role="student",
            nim_container_image=kwargs["nim_container_image"],
            container_name="vlm-student-stub",
            container_id="containerid",
            host_port=8002,
            endpoint_url=state["endpoint_url"],
            gpu_assignment=kwargs["gpu_assignment"],
            status=state["deploy_status"],
            student_model_id=kwargs.get("student_model_id"),
            checkpoint_mount_path=kwargs.get("checkpoint_mount"),
            nim_served_model_name=kwargs.get("nim_served_model_name"),
            nim_model_name_path=kwargs.get("nim_model_name_path"),
            precision_method=kwargs.get("precision_method"),
        )
        if state["deploy_status"] == "running":
            dep.status = "running"
            dep.deployed_at = utc_now()
        # Include the ``displaced`` list so the Student lifecycle
        # can iterate it for auto-restore in stage 8 step 9.
        return {
            "deployment": dep,
            "preflight": state["preflight"],
            "displaced": list(state["displaced_returned"]),
        }

    async def _fake_restore(**kwargs):
        state["restore_calls"].append(kwargs)

    async def _fake_stop(**kwargs):
        state["stop_called"].append(kwargs)
        return None

    async def _fake_wait_running(project_id, deployment_id, workspace_root, deadline_s):
        return state["deploy_status"] == "running"

    async def _fake_smoke(base_url, served_model):
        return state["smoke_ok"]

    async def _fake_register(snapshot, workspace_root, settings_, **kwargs):
        return ("ep-id", "temp-mc-id")

    async def _fake_run_evaluation(snapshot, workspace_root, settings_, target_id):
        if state["eval_run_status"] == "error":
            return None, None, "boom"
        run = RunRecord(
            run_id="run-id",
            project_id=snapshot.project_id,
            run_type="evaluation_run",
            status=state["eval_run_status"],
            evaluation_source="nim",
            model_config_id=target_id,
            inference_contract={
                "output_field_mode": "core_only",
                "icl_field_mode": "core_only",
            },
            rescored_metrics={
                # Canonical EvaluationMetrics bucket shape
                # (evaluation_service._agg_to_dict): per-field rates nest
                # under overall as plain floats.
                "overall": {
                    "exact_match_rate": 0.85,
                    "example_count": 4,
                    "per_field_match_rates": {"answer": 0.85},
                    "per_value_metrics": {},
                },
                "returning": None,
                "new": None,
            },
        )
        # Test seam: when the test sets these knobs the fake fills
        # the parseable counters used by ``_compute_parseable_rate``.
        if state["eval_examples_total"] is not None:
            run.examples_total = state["eval_examples_total"]
        if state["eval_examples_succeeded"] is not None:
            run.examples_succeeded = state["eval_examples_succeeded"]
        return "run-id", run, None

    async def _fake_scrape(base_url, **kwargs):
        state["scrape_calls"].append(base_url)
        return {
            "request_failure_total": 0.0,
            "request_success_total": 100.0,
            "gpu_cache_usage_perc": 0.5,
        }

    async def _fake_workload(**kwargs):
        temp_dir = tmp_path / "benchmark-payload"
        temp_dir.mkdir(exist_ok=True)
        input_file = temp_dir / "requests.jsonl"
        input_file.write_text("{}\n")
        artifact_root = tmp_path / "benchmark-artifacts"
        artifact_root.mkdir(exist_ok=True)
        return ServingBenchmarkWorkload(
            input_file=input_file,
            artifact_root=artifact_root,
            temporary_dir=temp_dir,
            request_count=100,
            manifest={"version": "test", "selected_count": 100},
        )

    def _fake_action_request(*args, **kwargs):
        state["ar_calls"].append((args, kwargs))
        return {"rendered_text": "ar"}

    # Apply patches.
    monkeypatch.setattr(lifecycle, "sse_manager", sse)
    monkeypatch.setattr(
        lifecycle.local_nim_service,
        "resolve_gpu_placement",
        _fake_resolve_gpu,
    )
    monkeypatch.setattr(
        lifecycle.local_nim_service,
        "run_preflight_checks",
        _fake_preflight,
    )
    monkeypatch.setattr(
        lifecycle.local_nim_service,
        "deploy_local_nim",
        _fake_deploy,
    )
    monkeypatch.setattr(
        lifecycle.local_nim_service,
        "stop_local_nim",
        _fake_stop,
    )
    monkeypatch.setattr(lifecycle, "_wait_for_deployment_running", _fake_wait_running)
    monkeypatch.setattr(lifecycle, "_restore_displaced_deployment", _fake_restore)
    monkeypatch.setattr(lifecycle, "_smoke_inference", _fake_smoke)
    monkeypatch.setattr(lifecycle, "_register_temp_endpoint", _fake_register)
    monkeypatch.setattr(lifecycle, "_run_evaluation_phase", _fake_run_evaluation)
    monkeypatch.setattr(lifecycle, "scrape_prometheus", _fake_scrape)
    monkeypatch.setattr(lifecycle, "build_serving_benchmark_workload", _fake_workload)
    monkeypatch.setattr(lifecycle, "generate_action_request", _fake_action_request)

    return state, sse


# ── Happy-path local lifecycle ───────────────────────────────────────────────


class TestLocalHappyPath:
    def test_shared_image_student_preflight_omits_base_profile_for_checkpoint(
        self, test_app_client, patched_lifecycle
    ):
        """A Nano Student probes its mounted weights, not the bundled base profile."""

        from vlm_feedback_loop.db.models.model_config import ModelConfig
        from vlm_feedback_loop.services.project_service import get_project_engine

        state, _sse = patched_lifecycle
        state["deploy_status"] = "running"
        pid, sid, settings = _make_project_and_student(test_app_client)
        engine = get_project_engine(pid, settings.WORKSPACE_ROOT)
        with Session(engine) as session:
            student = session.get(StudentModel, sid)
            base = session.get(ModelConfig, student.student_base_model_config_id)
            base.model_name = "nvidia/cosmos3-nano-reasoner"
            base.local_deploy_metadata = {
                **(base.local_deploy_metadata or {}),
                "nim_container_image": "nvcr.io/nim/nvidia/cosmos3-reasoner:1.7.0",
                "nim_model_size": "nano",
                "nim_model_profile": "nano-base-profile",
            }
            session.commit()

        asyncio.run(
            lifecycle.run_student_deployment_lifecycle(
                project_id=pid,
                student_model_id=sid,
                mode="local",
                nim_endpoint_url=None,
                nim_container_image=None,
                nim_release_version=None,
                gpu_assignment=None,
                auth_mode="none",
                settings=settings,
                workspace_root=settings.WORKSPACE_ROOT,
                benchmark_adapter=_StubAdapter(),
            )
        )

        assert state["preflight_calls"][0]["nim_model_size"] == "nano"
        assert state["preflight_calls"][0]["nim_model_profile"] is None
        assert (
            state["preflight_calls"][0]["nim_served_model_name"]
            == "nvidia/cosmos3-nano-reasoner"
        )

    def test_shared_image_student_preflight_uses_super_selector(
        self, test_app_client, patched_lifecycle
    ):
        """A Super Student must probe the Super profile, not image-default Nano."""
        from vlm_feedback_loop.db.models.model_config import ModelConfig
        from vlm_feedback_loop.services.project_service import get_project_engine

        state, _sse = patched_lifecycle
        state["deploy_status"] = "running"
        pid, sid, settings = _make_project_and_student(test_app_client)
        engine = get_project_engine(pid, settings.WORKSPACE_ROOT)
        with Session(engine) as session:
            student = session.get(StudentModel, sid)
            base = session.get(ModelConfig, student.student_base_model_config_id)
            base.model_name = "nvidia/cosmos3-super-reasoner"
            base.local_deploy_metadata = {
                **(base.local_deploy_metadata or {}),
                "nim_model_size": "super",
            }
            session.commit()

        asyncio.run(
            lifecycle.run_student_deployment_lifecycle(
                project_id=pid,
                student_model_id=sid,
                mode="local",
                nim_endpoint_url=None,
                nim_container_image=None,
                nim_release_version=None,
                gpu_assignment=None,
                auth_mode="none",
                settings=settings,
                workspace_root=settings.WORKSPACE_ROOT,
                benchmark_adapter=_StubAdapter(),
            )
        )

        assert state["preflight_calls"][0]["nim_model_size"] == "super"
        assert state["preflight_calls"][0]["nim_model_profile"] is None
        assert (
            state["preflight_calls"][0]["nim_served_model_name"]
            == "nvidia/cosmos3-super-reasoner"
        )
        assert state["deploy_calls"][0]["nim_served_model_name"].startswith("student-")

    def test_full_local_lifecycle_flips_serving_validated(
        self, test_app_client, patched_lifecycle
    ):
        state, sse = patched_lifecycle
        state["deploy_status"] = "running"
        pid, sid, settings = _make_project_and_student(test_app_client)
        adapter = _StubAdapter()

        asyncio.run(
            lifecycle.run_student_deployment_lifecycle(
                project_id=pid,
                student_model_id=sid,
                mode="local",
                nim_endpoint_url=None,
                nim_container_image=None,
                nim_release_version=None,
                gpu_assignment=None,
                auth_mode="none",
                settings=settings,
                workspace_root=settings.WORKSPACE_ROOT,
                benchmark_adapter=adapter,
            )
        )

        # serving_status flipped to validated; eval_run_id set.
        from vlm_feedback_loop.services.project_service import get_project_engine

        engine = get_project_engine(pid, settings.WORKSPACE_ROOT)
        with Session(engine) as session:
            student = session.get(StudentModel, sid)
            assert student.serving_status == "validated"
            assert student.serving_evaluation_run_id == "run-id"
            assert student.nim_endpoint_url == state["endpoint_url"]
            assert student.nim_container_id == "containerid"
            assert student.nim_preflight_status == "passed"
            assert student.nim_vlm_release_version == "1.6.0"

        # SSE stage progression — every documented stage emitted in order.
        progress_stages = [
            data["stage"]
            for (_, etype, data) in sse.events
            if etype == "nim_benchmark_progress"
        ]
        # Allow benchmark stage repetitions (one per concurrency).
        first_seen = []
        for stage in progress_stages:
            if not first_seen or first_seen[-1] != stage:
                first_seen.append(stage)
        # Required canonical order:
        expected_order = [
            "preflight",
            "docker_run",
            "health_poll",
            "smoke_inference",
            "registering_endpoint",
            "evaluation",
            "benchmark",
            "stopping",
        ]
        assert first_seen == expected_order

        # Completed event present + container stopped.
        assert any(e[1] == "nim_benchmark_completed" for e in sse.events)
        assert state["stop_called"], "Container must be stopped at end"

        # Adapter ran 3 times, one per concurrency level [1, 8, 24].
        assert len(adapter.calls) == 3
        assert [c["concurrency"] for c in adapter.calls] == [1, 8, 24]

    def test_completed_event_includes_summary_metrics(
        self, test_app_client, patched_lifecycle
    ):
        state, sse = patched_lifecycle
        state["deploy_status"] = "running"
        pid, sid, settings = _make_project_and_student(test_app_client)
        adapter = _StubAdapter()

        asyncio.run(
            lifecycle.run_student_deployment_lifecycle(
                project_id=pid,
                student_model_id=sid,
                mode="local",
                nim_endpoint_url=None,
                nim_container_image=None,
                nim_release_version="1.6.0",
                gpu_assignment=None,
                auth_mode="none",
                settings=settings,
                workspace_root=settings.WORKSPACE_ROOT,
                benchmark_adapter=adapter,
            )
        )

        completed = [e for e in sse.events if e[1] == "nim_benchmark_completed"]
        assert len(completed) == 1
        payload = completed[0][2]
        assert payload["serving_status"] == "validated"
        assert payload["evaluation_run_id"] == "run-id"
        assert payload["exact_match"] == 0.85
        assert payload["per_field_match_rates"] == {"answer": 0.85}
        assert len(payload["benchmarks"]) == 3
        assert payload["skipped_concurrencies"] == []


# ── Quality-status promotion via NIM eval ───────────────────────


class TestNimEvalQualityPromotion:
    """Verify the NIM-eval-as-quality-fallback path.

    Background: cosmos-rl 6.26.3's bundled vLLM lacks a Qwen3-VL-dense loader,
    which makes TAO eval fail on every Cosmos-Reason2 (2B/8B) fine-tune. NIM
    1.6.0 has the loader and runs eval cleanly. The Blueprint promotes
    quality_status from a NIM-source eval when TAO did not validate first
    AND the prior TAO failure signature matches a known upstream loader gap
    (the conservative gate in ``services.tao_failure_classifier``).
    See ``docs/blueprint_finding_cosmos_rl_qwen3_vl_dense_gap.md``.
    """

    def test_promotes_quality_when_pending_no_prior_tao_eval(
        self, test_app_client, patched_lifecycle
    ):
        """Cold start / operator-driven NIM-only path: ``quality_status=pending``
        promotes on a clean NIM eval without needing a prior TAO failure to
        classify."""
        from vlm_feedback_loop.services.project_service import get_project_engine

        state, _sse = patched_lifecycle
        state["deploy_status"] = "running"
        pid, sid, settings = _make_project_and_student(test_app_client)

        engine = get_project_engine(pid, settings.WORKSPACE_ROOT)
        with Session(engine) as session:
            student = session.get(StudentModel, sid)
            student.quality_status = "pending"
            student.quality_evaluation_run_id = None
            session.commit()

        asyncio.run(
            lifecycle.run_student_deployment_lifecycle(
                project_id=pid,
                student_model_id=sid,
                mode="local",
                nim_endpoint_url=None,
                nim_container_image=None,
                nim_release_version=None,
                gpu_assignment=None,
                auth_mode="none",
                settings=settings,
                workspace_root=settings.WORKSPACE_ROOT,
                benchmark_adapter=_StubAdapter(),
            )
        )

        with Session(engine) as session:
            student = session.get(StudentModel, sid)
            assert student.quality_status == "validated"
            assert student.quality_evaluation_run_id == "run-id"
            assert student.serving_status == "validated"

    def test_promotes_quality_when_failed_with_matching_loader_signature(
        self, test_app_client, patched_lifecycle
    ):
        """``quality_status=failed`` AND the prior failed TAO eval has a
        log-text matching a known upstream-loader pattern → promote."""
        from vlm_feedback_loop.db.models.tao_job import TAOJob
        from vlm_feedback_loop.services.project_service import get_project_engine

        state, _sse = patched_lifecycle
        state["deploy_status"] = "running"
        pid, sid, settings = _make_project_and_student(test_app_client)

        engine = get_project_engine(pid, settings.WORKSPACE_ROOT)
        with Session(engine) as session:
            student = session.get(StudentModel, sid)
            student.quality_status = "failed"
            student.quality_evaluation_run_id = None
            student.tao_job_id = "train-tao-job-id"
            # Seed: a failed evaluate TAOJob whose log contains the
            # canonical Qwen3-VL-dense loader-gap signature.
            session.add(
                TAOJob(
                    tao_job_id="eval-tao-job-id",
                    project_id=pid,
                    student_base_model_config_id=student.student_base_model_config_id,
                    parent_tao_job_id="train-tao-job-id",
                    action="evaluate",
                    chain_id="chain-1",
                    chain_sequence=2,
                    training_backend="cosmos_rl_tao_vlm",
                    dataset_export_ids=[],
                    job_config={},
                    tao_create_job_request={},
                    status="failed",
                    tao_status_raw="Error",
                    tao_external_job_id="external-eval-id",
                    completed_at=utc_now(),
                    outputs={
                        "tao_logs_text": (
                            "stack trace ... "
                            "vllm/model_executor/layers/vocab_parallel_embedding.py "
                            "AssertionError"
                        ),
                    },
                )
            )
            session.commit()

        asyncio.run(
            lifecycle.run_student_deployment_lifecycle(
                project_id=pid,
                student_model_id=sid,
                mode="local",
                nim_endpoint_url=None,
                nim_container_image=None,
                nim_release_version=None,
                gpu_assignment=None,
                auth_mode="none",
                settings=settings,
                workspace_root=settings.WORKSPACE_ROOT,
                benchmark_adapter=_StubAdapter(),
            )
        )

        with Session(engine) as session:
            student = session.get(StudentModel, sid)
            assert student.quality_status == "validated"
            assert student.quality_evaluation_run_id == "run-id"
            assert student.serving_status == "validated"

    def test_quantized_student_classifies_its_quantize_evaluate_failure(
        self, test_app_client, patched_lifecycle
    ):
        """A quantized Student's failed evaluate is parented by ``quantize``.

        The NIM fallback must classify that exact artifact lineage rather than
        the baseline evaluate parented by ``train``; otherwise a clean FP8 NIM
        evaluation remains incorrectly quality-failed.
        """
        from vlm_feedback_loop.db.models.tao_job import TAOJob
        from vlm_feedback_loop.services.project_service import get_project_engine

        state, _sse = patched_lifecycle
        state["deploy_status"] = "running"
        pid, sid, settings = _make_project_and_student(test_app_client)

        engine = get_project_engine(pid, settings.WORKSPACE_ROOT)
        with Session(engine) as session:
            student = session.get(StudentModel, sid)
            student.quality_status = "failed"
            student.quality_evaluation_run_id = None
            student.tao_job_id = "train-tao-job-id"
            student.quantize_tao_job_id = "quantize-tao-job-id"
            session.add(
                TAOJob(
                    tao_job_id="quantized-eval-tao-job-id",
                    project_id=pid,
                    student_base_model_config_id=student.student_base_model_config_id,
                    parent_tao_job_id="quantize-tao-job-id",
                    action="evaluate",
                    chain_id="chain-1",
                    chain_sequence=4,
                    training_backend="cosmos_rl_tao_vlm",
                    dataset_export_ids=[],
                    job_config={},
                    tao_create_job_request={},
                    status="failed",
                    tao_status_raw="Error",
                    tao_external_job_id="external-quantized-eval-id",
                    completed_at=utc_now(),
                    outputs={
                        "tao_logs_text": (
                            "RuntimeError while loading Qwen3VLForConditionalGeneration"
                        ),
                    },
                )
            )
            session.commit()

        asyncio.run(
            lifecycle.run_student_deployment_lifecycle(
                project_id=pid,
                student_model_id=sid,
                mode="local",
                nim_endpoint_url=None,
                nim_container_image=None,
                nim_release_version=None,
                gpu_assignment=None,
                auth_mode="none",
                settings=settings,
                workspace_root=settings.WORKSPACE_ROOT,
                benchmark_adapter=_StubAdapter(),
            )
        )

        with Session(engine) as session:
            student = session.get(StudentModel, sid)
            assert student.quality_status == "validated"
            assert student.quality_evaluation_run_id == "run-id"
            assert student.serving_status == "validated"

    def test_does_not_promote_when_failed_without_matching_signature(
        self, test_app_client, patched_lifecycle
    ):
        """``quality_status=failed`` with a prior TAO failure that does NOT
        match a known upstream-loader signature (e.g. dataset error, OOM)
        → NIM eval succeeds but quality_status stays ``failed``. NIM eval is
        not a generic rescue for arbitrary TAO failures."""
        from vlm_feedback_loop.db.models.tao_job import TAOJob
        from vlm_feedback_loop.services.project_service import get_project_engine

        state, _sse = patched_lifecycle
        state["deploy_status"] = "running"
        pid, sid, settings = _make_project_and_student(test_app_client)

        engine = get_project_engine(pid, settings.WORKSPACE_ROOT)
        with Session(engine) as session:
            student = session.get(StudentModel, sid)
            student.quality_status = "failed"
            student.quality_evaluation_run_id = None
            student.tao_job_id = "train-tao-job-id"
            # Seed: a failed evaluate TAOJob whose log is generic
            # (dataset / OOM / config error). Plain ``AssertionError``
            # alone is intentionally NOT a trigger.
            session.add(
                TAOJob(
                    tao_job_id="eval-tao-job-id",
                    project_id=pid,
                    student_base_model_config_id=student.student_base_model_config_id,
                    parent_tao_job_id="train-tao-job-id",
                    action="evaluate",
                    chain_id="chain-1",
                    chain_sequence=2,
                    training_backend="cosmos_rl_tao_vlm",
                    dataset_export_ids=[],
                    job_config={},
                    tao_create_job_request={},
                    status="failed",
                    tao_status_raw="Error",
                    tao_external_job_id="external-eval-id",
                    completed_at=utc_now(),
                    outputs={
                        "tao_logs_text": (
                            "RuntimeError: CUDA out of memory. "
                            "Tried to allocate 24.00 GiB. "
                            "AssertionError"
                        ),
                    },
                )
            )
            session.commit()

        asyncio.run(
            lifecycle.run_student_deployment_lifecycle(
                project_id=pid,
                student_model_id=sid,
                mode="local",
                nim_endpoint_url=None,
                nim_container_image=None,
                nim_release_version=None,
                gpu_assignment=None,
                auth_mode="none",
                settings=settings,
                workspace_root=settings.WORKSPACE_ROOT,
                benchmark_adapter=_StubAdapter(),
            )
        )

        with Session(engine) as session:
            student = session.get(StudentModel, sid)
            assert student.quality_status == "failed", (
                "quality_status MUST stay failed for non-loader TAO failures"
            )
            assert student.quality_evaluation_run_id is None
            # serving_status is independent and DOES flip on a clean NIM eval.
            assert student.serving_status == "validated"
            assert student.serving_evaluation_run_id == "run-id"

    def test_preserves_existing_tao_validation_audit_pointer(
        self, test_app_client, patched_lifecycle
    ):
        """If TAO eval already validated this Student, NIM eval MUST NOT overwrite
        ``quality_evaluation_run_id`` — preserves the audit trail that ties the
        Student's quality validation back to the TAO RunRecord (not the NIM one).
        """
        from vlm_feedback_loop.services.project_service import get_project_engine

        state, _sse = patched_lifecycle
        state["deploy_status"] = "running"
        pid, sid, settings = _make_project_and_student(test_app_client)

        engine = get_project_engine(pid, settings.WORKSPACE_ROOT)
        with Session(engine) as session:
            student = session.get(StudentModel, sid)
            student.quality_status = "validated"
            student.quality_evaluation_run_id = "tao-rescored-run-id"
            session.commit()

        asyncio.run(
            lifecycle.run_student_deployment_lifecycle(
                project_id=pid,
                student_model_id=sid,
                mode="local",
                nim_endpoint_url=None,
                nim_container_image=None,
                nim_release_version=None,
                gpu_assignment=None,
                auth_mode="none",
                settings=settings,
                workspace_root=settings.WORKSPACE_ROOT,
                benchmark_adapter=_StubAdapter(),
            )
        )

        with Session(engine) as session:
            student = session.get(StudentModel, sid)
            # TAO's audit pointer survives; serving still flipped via NIM run.
            assert student.quality_status == "validated"
            assert student.quality_evaluation_run_id == "tao-rescored-run-id"
            assert student.serving_evaluation_run_id == "run-id"

    def test_external_mode_also_promotes(self, test_app_client, patched_lifecycle):
        """External mode (operator-supplied NIM endpoint URL) promotes through
        the same conservative gate as local mode."""
        from vlm_feedback_loop.db.models.tao_job import TAOJob
        from vlm_feedback_loop.services.project_service import get_project_engine

        state, _sse = patched_lifecycle
        pid, sid, settings = _make_project_and_student(test_app_client)

        engine = get_project_engine(pid, settings.WORKSPACE_ROOT)
        with Session(engine) as session:
            student = session.get(StudentModel, sid)
            student.quality_status = "failed"
            student.quality_evaluation_run_id = None
            student.tao_job_id = "train-tao-job-id"
            session.add(
                TAOJob(
                    tao_job_id="eval-tao-job-id",
                    project_id=pid,
                    student_base_model_config_id=student.student_base_model_config_id,
                    parent_tao_job_id="train-tao-job-id",
                    action="evaluate",
                    chain_id="chain-1",
                    chain_sequence=2,
                    training_backend="cosmos_rl_tao_vlm",
                    dataset_export_ids=[],
                    job_config={},
                    tao_create_job_request={},
                    status="failed",
                    tao_status_raw="Error",
                    tao_external_job_id="external-eval-id",
                    completed_at=utc_now(),
                    outputs={
                        "tao_logs_text": (
                            "vllm/model_executor/models/qwen2_5_vl.py "
                            "AssertionError on Qwen3VLForConditionalGeneration"
                        ),
                    },
                )
            )
            session.commit()

        asyncio.run(
            lifecycle.run_student_deployment_lifecycle(
                project_id=pid,
                student_model_id=sid,
                mode="external",
                nim_endpoint_url="http://example.com:8000",
                nim_container_image=None,
                nim_release_version=None,
                gpu_assignment=None,
                auth_mode="none",
                settings=settings,
                workspace_root=settings.WORKSPACE_ROOT,
                benchmark_adapter=_StubAdapter(),
            )
        )

        with Session(engine) as session:
            student = session.get(StudentModel, sid)
            assert student.quality_status == "validated"
            assert student.quality_evaluation_run_id == "run-id"


# ── Failure paths ────────────────────────────────────────────────────────────


class TestPreflightFailure:
    def test_preflight_failure_emits_run_failed_and_action_request(
        self, test_app_client, patched_lifecycle
    ):
        state, sse = patched_lifecycle
        state["preflight"] = _make_preflight(passed=False)
        pid, sid, settings = _make_project_and_student(test_app_client)
        adapter = _StubAdapter()

        asyncio.run(
            lifecycle.run_student_deployment_lifecycle(
                project_id=pid,
                student_model_id=sid,
                mode="local",
                nim_endpoint_url=None,
                nim_container_image=None,
                nim_release_version=None,
                gpu_assignment=None,
                auth_mode="none",
                settings=settings,
                workspace_root=settings.WORKSPACE_ROOT,
                benchmark_adapter=adapter,
            )
        )

        # serving_status flipped to failed with the right failure_stage.
        from vlm_feedback_loop.services.project_service import get_project_engine

        engine = get_project_engine(pid, settings.WORKSPACE_ROOT)
        with Session(engine) as session:
            student = session.get(StudentModel, sid)
            assert student.serving_status == "failed"
            assert student.nim_preflight_status == "failed"
            details = student.nim_preflight_details
            assert details["failure_stage"] == "preflight_failed"

        # Action Request was generated (preflight-only — per spec).
        assert state["ar_calls"], "AR must be generated on preflight failure"
        ar_args = state["ar_calls"][0][1]  # kwargs
        assert ar_args["request_type"] == "student_nim_deploy"
        ctx = ar_args["context"]
        assert ctx["student_model_id"] == sid
        assert ctx["nim_checkpoint_ref"] == "/tmp/student-ckpt"
        assert ctx["role"] == "student"

        # SSE: progress[preflight] + run_failed; no completed event.
        assert any(e[1] == "run_failed" for e in sse.events)
        assert not any(e[1] == "nim_benchmark_completed" for e in sse.events)
        # No benchmark calls — short-circuited.
        assert not adapter.calls


class TestDockerRunFailure:
    def test_docker_run_failed_emits_run_failed(
        self, test_app_client, patched_lifecycle
    ):
        state, sse = patched_lifecycle
        state["preflight"] = _make_preflight(passed=True)
        state["deploy_status"] = "failed"
        pid, sid, settings = _make_project_and_student(test_app_client)

        asyncio.run(
            lifecycle.run_student_deployment_lifecycle(
                project_id=pid,
                student_model_id=sid,
                mode="local",
                nim_endpoint_url=None,
                nim_container_image=None,
                nim_release_version=None,
                gpu_assignment=None,
                auth_mode="none",
                settings=settings,
                workspace_root=settings.WORKSPACE_ROOT,
                benchmark_adapter=_StubAdapter(),
            )
        )

        from vlm_feedback_loop.services.project_service import get_project_engine

        engine = get_project_engine(pid, settings.WORKSPACE_ROOT)
        with Session(engine) as session:
            student = session.get(StudentModel, sid)
            assert student.serving_status == "failed"
            assert student.nim_preflight_details["failure_stage"] == "docker_run_failed"

        # Docker-run failed does NOT generate an Action Request (preflight-only).
        assert not state["ar_calls"]


class TestSmokeFailure:
    def test_smoke_failed_stops_container_and_emits_run_failed(
        self, test_app_client, patched_lifecycle
    ):
        state, sse = patched_lifecycle
        state["preflight"] = _make_preflight(passed=True)
        state["deploy_status"] = "running"
        state["smoke_ok"] = False
        pid, sid, settings = _make_project_and_student(test_app_client)

        asyncio.run(
            lifecycle.run_student_deployment_lifecycle(
                project_id=pid,
                student_model_id=sid,
                mode="local",
                nim_endpoint_url=None,
                nim_container_image=None,
                nim_release_version=None,
                gpu_assignment=None,
                auth_mode="none",
                settings=settings,
                workspace_root=settings.WORKSPACE_ROOT,
                benchmark_adapter=_StubAdapter(),
            )
        )

        # Failure detail set; no completed event; the container is NOT
        # stopped here in the smoke path because the lifecycle's smoke
        # branch returns early; the unhandled-failure safety net would.
        # Spec is silent on whether docker is stopped here — but
        # serving_status MUST be failed.
        from vlm_feedback_loop.services.project_service import get_project_engine

        engine = get_project_engine(pid, settings.WORKSPACE_ROOT)
        with Session(engine) as session:
            student = session.get(StudentModel, sid)
            assert student.serving_status == "failed"
            assert student.nim_preflight_details["failure_stage"] == "smoke_failed"


# ── External mode ────────────────────────────────────────────────────────────


class TestExternalMode:
    def test_external_mode_skips_local_stages(self, test_app_client, patched_lifecycle):
        state, sse = patched_lifecycle
        pid, sid, settings = _make_project_and_student(test_app_client)
        adapter = _StubAdapter()

        asyncio.run(
            lifecycle.run_student_deployment_lifecycle(
                project_id=pid,
                student_model_id=sid,
                mode="external",
                nim_endpoint_url="https://student.example/v1",
                nim_container_image=None,
                nim_release_version=None,
                gpu_assignment=None,
                auth_mode="bearer",
                settings=settings,
                workspace_root=settings.WORKSPACE_ROOT,
                benchmark_adapter=adapter,
            )
        )

        # Eval ran (NIM source) and serving_status validated.
        from vlm_feedback_loop.services.project_service import get_project_engine

        engine = get_project_engine(pid, settings.WORKSPACE_ROOT)
        with Session(engine) as session:
            student = session.get(StudentModel, sid)
            assert student.serving_status == "validated"
            assert student.serving_evaluation_run_id == "run-id"
            assert student.nim_endpoint_url == "https://student.example/v1"

        # No local orchestration stages emitted. External endpoints still run
        # the same production serving workload before validation.
        progress_stages = [
            data["stage"]
            for (_, etype, data) in sse.events
            if etype == "nim_benchmark_progress"
        ]
        assert "preflight" not in progress_stages
        assert "docker_run" not in progress_stages
        assert "benchmark" in progress_stages
        assert "registering_endpoint" in progress_stages
        assert "evaluation" in progress_stages

        assert len(adapter.calls) == len(settings.STUDENT_LATENCY_TEST_CONCURRENCIES)
        # No container stop call (we never started one).
        assert not state["stop_called"]

    def test_workload_build_failure_is_a_benchmark_failure(
        self, test_app_client, patched_lifecycle, monkeypatch
    ):
        state, _sse = patched_lifecycle
        pid, sid, settings = _make_project_and_student(test_app_client)

        async def _fail_workload(**_kwargs):
            raise RuntimeError("test pool image unavailable")

        monkeypatch.setattr(
            lifecycle, "build_serving_benchmark_workload", _fail_workload
        )
        adapter = _StubAdapter()
        asyncio.run(
            lifecycle.run_student_deployment_lifecycle(
                project_id=pid,
                student_model_id=sid,
                mode="external",
                nim_endpoint_url="https://student.example/v1",
                nim_container_image=None,
                nim_release_version=None,
                gpu_assignment=None,
                auth_mode="none",
                settings=settings,
                workspace_root=settings.WORKSPACE_ROOT,
                benchmark_adapter=adapter,
            )
        )

        from vlm_feedback_loop.services.project_service import get_project_engine

        engine = get_project_engine(pid, settings.WORKSPACE_ROOT)
        with Session(engine) as session:
            student = session.get(StudentModel, sid)
            assert student.serving_status == "failed"
            assert student.nim_preflight_details["failure_stage"] == "benchmark_failed"
            assert "RuntimeError" in student.nim_preflight_details["error_detail"]
        assert adapter.calls == []
        assert not state["stop_called"]


# ── Inference Contract derivation ────────────────────────────────────────────


class TestInferenceContractDerivation:
    """The Student's training DatasetExport ``export_field_mode``
    becomes the lifecycle's evaluation contract.
    Tested at the pure-function level (no docker, no eval pipeline) since
    the integration is the helper itself.
    """

    def test_contract_inherits_training_export_field_mode(self, test_app_client):
        pid, sid, settings = _make_project_and_student(
            test_app_client, with_dataset_export=True
        )
        snapshot = lifecycle._load_student_snapshot(pid, sid, settings.WORKSPACE_ROOT)
        contract = lifecycle._student_inference_contract(
            snapshot, settings.WORKSPACE_ROOT
        )
        assert contract["output_field_mode"] == "core_only"
        assert contract["icl_field_mode"] == "core_only"

    def test_contract_falls_back_to_all_when_no_dataset_export(self, test_app_client):
        """Defensive: a Student with empty dataset_export_ids should fall
        back to ``"all"`` (Teacher-style contract) so we never crash.
        """
        pid, sid, settings = _make_project_and_student(
            test_app_client, with_dataset_export=False
        )
        snapshot = lifecycle._load_student_snapshot(pid, sid, settings.WORKSPACE_ROOT)
        contract = lifecycle._student_inference_contract(
            snapshot, settings.WORKSPACE_ROOT
        )
        assert contract["output_field_mode"] == "all"
        assert contract["icl_field_mode"] == "all"


# ── Partial quality_status ─────────────────────────────────────────────


class TestPartialPromotion:
    """Threshold-based partial quality promotion.

    A NIM-source eval that finishes ``incomplete`` with parseable rate
    >= ``STUDENT_QUALITY_PARTIAL_PARSEABLE_THRESHOLD`` (default 0.90)
    promotes the paired Student to ``quality_status="partial"`` instead
    of staying at the prior value. ``partial`` is informational, not
    gate-passing — the deployment_handoff bar still requires
    ``validated``. Audit invariant: ``validated`` is never demoted to
    ``partial``.
    """

    def _drive_lifecycle(self, pid, sid, settings):
        asyncio.run(
            lifecycle.run_student_deployment_lifecycle(
                project_id=pid,
                student_model_id=sid,
                mode="local",
                nim_endpoint_url=None,
                nim_container_image=None,
                nim_release_version=None,
                gpu_assignment=None,
                auth_mode="none",
                settings=settings,
                workspace_root=settings.WORKSPACE_ROOT,
                benchmark_adapter=_StubAdapter(),
            )
        )

    def _seed(self, pid, sid, settings, *, prior_quality):
        from vlm_feedback_loop.services.project_service import (
            get_project_engine,
        )

        engine = get_project_engine(pid, settings.WORKSPACE_ROOT)
        with Session(engine) as session:
            student = session.get(StudentModel, sid)
            student.quality_status = prior_quality
            student.quality_evaluation_run_id = None
            student.tao_job_id = "train-tao-job-id"
            session.commit()
        return engine

    def test_incomplete_above_threshold_with_pending_promotes_to_partial(
        self, test_app_client, patched_lifecycle
    ):
        """``status=incomplete`` + parseable rate 0.91 + prior pending → partial."""
        state, _sse = patched_lifecycle
        state["deploy_status"] = "running"
        state["eval_run_status"] = "incomplete"
        state["eval_examples_total"] = 148
        state["eval_examples_succeeded"] = 135  # 0.912 — just above the 0.90 threshold

        pid, sid, settings = _make_project_and_student(test_app_client)
        engine = self._seed(pid, sid, settings, prior_quality="pending")

        self._drive_lifecycle(pid, sid, settings)

        with Session(engine) as session:
            student = session.get(StudentModel, sid)
            assert student.quality_status == "partial"
            assert student.quality_evaluation_run_id == "run-id"
            # Serving lands validated even on incomplete because
            # the parseable rate cleared the bar.
            assert student.serving_status == "validated"

    def test_incomplete_above_threshold_with_failed_promotes_to_partial(
        self, test_app_client, patched_lifecycle
    ):
        """``status=incomplete`` + parseable rate 0.91 + prior failed → partial.

        The ``failed → partial`` transition does NOT consult the
        loader-gap classifier — the threshold-based promotion is the
        authority for the partial path. The classifier-gated path is
        ``failed → validated`` in ``_promote_quality_from_nim_eval``.
        """
        state, _sse = patched_lifecycle
        state["deploy_status"] = "running"
        state["eval_run_status"] = "incomplete"
        state["eval_examples_total"] = 100
        state["eval_examples_succeeded"] = 95

        pid, sid, settings = _make_project_and_student(test_app_client)
        engine = self._seed(pid, sid, settings, prior_quality="failed")

        self._drive_lifecycle(pid, sid, settings)

        with Session(engine) as session:
            student = session.get(StudentModel, sid)
            assert student.quality_status == "partial"
            assert student.quality_evaluation_run_id == "run-id"

    def test_incomplete_above_threshold_with_validated_is_noop(
        self, test_app_client, patched_lifecycle
    ):
        """Audit invariant: ``validated`` is never demoted to ``partial``."""
        from vlm_feedback_loop.services.project_service import (
            get_project_engine,
        )

        state, _sse = patched_lifecycle
        state["deploy_status"] = "running"
        state["eval_run_status"] = "incomplete"
        state["eval_examples_total"] = 100
        state["eval_examples_succeeded"] = 99

        pid, sid, settings = _make_project_and_student(test_app_client)
        engine = get_project_engine(pid, settings.WORKSPACE_ROOT)
        prior_run_id = "tao-rescore-prior-run-id"
        with Session(engine) as session:
            student = session.get(StudentModel, sid)
            student.quality_status = "validated"
            student.quality_evaluation_run_id = prior_run_id
            session.commit()

        self._drive_lifecycle(pid, sid, settings)

        with Session(engine) as session:
            student = session.get(StudentModel, sid)
            # Quality unchanged; pointer preserved.
            assert student.quality_status == "validated"
            assert student.quality_evaluation_run_id == prior_run_id

    def test_incomplete_below_threshold_leaves_quality_unchanged(
        self, test_app_client, patched_lifecycle
    ):
        """``status=incomplete`` + parseable rate 0.80 (default threshold 0.90)
        does NOT promote to partial — quality stays at prior value, and the
        lifecycle fires ``run_failed`` per the existing failure path."""
        state, _sse = patched_lifecycle
        state["deploy_status"] = "running"
        state["eval_run_status"] = "incomplete"
        state["eval_examples_total"] = 100
        state["eval_examples_succeeded"] = 80  # below threshold

        pid, sid, settings = _make_project_and_student(test_app_client)
        engine = self._seed(pid, sid, settings, prior_quality="pending")

        self._drive_lifecycle(pid, sid, settings)

        with Session(engine) as session:
            student = session.get(StudentModel, sid)
            # Quality untouched.
            assert student.quality_status == "pending"
            assert student.quality_evaluation_run_id is None

    def test_external_mode_incomplete_below_threshold_still_validates_serving(
        self, test_app_client, patched_lifecycle
    ):
        """Serving accepts ``incomplete`` in EXTERNAL mode too, even when
        the parseable rate misses the partial threshold. Guards the drift
        class where an external copy of the gate records ``eval_failed``
        here — the reason the gates are consolidated into
        ``_apply_serving_quality_gate``. Quality stays at its prior value
        (no promotion below the threshold)."""
        state, _sse = patched_lifecycle
        state["eval_run_status"] = "incomplete"
        state["eval_examples_total"] = 100
        state["eval_examples_succeeded"] = 80  # below the 0.90 threshold

        pid, sid, settings = _make_project_and_student(test_app_client)
        engine = self._seed(pid, sid, settings, prior_quality="pending")

        asyncio.run(
            lifecycle.run_student_deployment_lifecycle(
                project_id=pid,
                student_model_id=sid,
                mode="external",
                nim_endpoint_url="http://example.com:8000",
                nim_container_image=None,
                nim_release_version=None,
                gpu_assignment=None,
                auth_mode="none",
                settings=settings,
                workspace_root=settings.WORKSPACE_ROOT,
                benchmark_adapter=_StubAdapter(),
            )
        )

        with Session(engine) as session:
            student = session.get(StudentModel, sid)
            assert student.serving_status == "validated"
            assert student.serving_evaluation_run_id == "run-id"
            assert student.quality_status == "pending"

    def test_zero_parseable_rate_is_failure(self, test_app_client, patched_lifecycle):
        """``examples_total=0`` (degenerate run) gives parseable_rate=0.0
        which deterministically misses the threshold — quality stays
        at prior value."""
        state, _sse = patched_lifecycle
        state["deploy_status"] = "running"
        state["eval_run_status"] = "incomplete"
        state["eval_examples_total"] = 0
        state["eval_examples_succeeded"] = 0

        pid, sid, settings = _make_project_and_student(test_app_client)
        engine = self._seed(pid, sid, settings, prior_quality="failed")

        self._drive_lifecycle(pid, sid, settings)

        with Session(engine) as session:
            student = session.get(StudentModel, sid)
            assert student.quality_status == "failed"

    def test_threshold_boundary_exact_match_promotes(
        self, test_app_client, patched_lifecycle
    ):
        """Boundary: parseable rate exactly == threshold promotes (>= comparison)."""
        state, _sse = patched_lifecycle
        state["deploy_status"] = "running"
        state["eval_run_status"] = "incomplete"
        state["eval_examples_total"] = 10
        state["eval_examples_succeeded"] = 9  # exactly 0.90

        pid, sid, settings = _make_project_and_student(test_app_client)
        engine = self._seed(pid, sid, settings, prior_quality="pending")

        self._drive_lifecycle(pid, sid, settings)

        with Session(engine) as session:
            student = session.get(StudentModel, sid)
            assert student.quality_status == "partial"


# ── One-NIM-per-GPU replace semantics ───────────────────────────────────


class TestSingleGpuReplacement:
    """The Student NIM lifecycle invokes the replace semantics by default:

    * Stage 2 (docker_run) calls ``deploy_local_nim`` with
      ``replace_resident=True`` so the service stops any active
      resident on the target GPU before constructing the Student
      ``docker run``.
    * Stage 8 (stopping → step 9) iterates the displaced rows returned
      by step 0 and best-effort re-deploys each via
      ``_restore_displaced_deployment``. Failure is logged as a
      warning, not a hard error.
    """

    def _drive_lifecycle(self, pid, sid, settings):
        asyncio.run(
            lifecycle.run_student_deployment_lifecycle(
                project_id=pid,
                student_model_id=sid,
                mode="local",
                nim_endpoint_url=None,
                nim_container_image=None,
                nim_release_version=None,
                gpu_assignment=None,
                auth_mode="none",
                settings=settings,
                workspace_root=settings.WORKSPACE_ROOT,
                benchmark_adapter=_StubAdapter(),
            )
        )

    def test_lifecycle_calls_deploy_local_nim_with_replace_resident_true(
        self, test_app_client, patched_lifecycle
    ):
        """The Student lifecycle ALWAYS passes ``replace_resident=True``
        to ``deploy_local_nim``. On multi-GPU hosts with a free device
        this is a no-op (no residents on that GPU); on single-GPU
        hosts this stops the resident Teacher / embedding before the
        Student container starts."""
        state, _sse = patched_lifecycle
        state["deploy_status"] = "running"
        pid, sid, settings = _make_project_and_student(test_app_client)

        self._drive_lifecycle(pid, sid, settings)

        assert len(state["deploy_calls"]) == 1, (
            "lifecycle must invoke deploy_local_nim exactly once"
        )
        call = state["deploy_calls"][0]
        assert call.get("replace_resident") is True, (
            "lifecycle must opt into replace semantics via "
            "deploy_local_nim(replace_resident=True)"
        )

    def test_lifecycle_auto_restores_displaced_residents_after_student_stops(
        self, test_app_client, patched_lifecycle
    ):
        """The lifecycle iterates each displaced row returned by the
        deploy stage and invokes ``_restore_displaced_deployment`` per
        row (the auto-restore contract)."""
        state, _sse = patched_lifecycle
        state["deploy_status"] = "running"
        # Seed two displaced residents (a Teacher + an embedding) that
        # the fake _fake_deploy returns as "displaced". The lifecycle
        # must call _restore_displaced_deployment for each.
        teacher_row = LocalNimDeployment(
            local_nim_deployment_id="displaced-teacher",
            project_id="p",
            model_config_id="t-mc",
            role="teacher",
            nim_container_image="cosmos:1.6.0",
            container_name="vlm-teacher-victim",
            host_port=8000,
            endpoint_url="http://localhost:8000/v1",
            gpu_assignment="device=0",
            status="stopped",
        )
        embedding_row = LocalNimDeployment(
            local_nim_deployment_id="displaced-embedding",
            project_id="p",
            model_config_id="e-mc",
            role="embedding",
            nim_container_image="retriever:1.12.0",
            container_name="vlm-embedding-victim",
            host_port=8001,
            endpoint_url="http://localhost:8001/v1",
            gpu_assignment="device=0",
            status="stopped",
        )
        state["displaced_returned"] = [teacher_row, embedding_row]
        pid, sid, settings = _make_project_and_student(test_app_client)

        self._drive_lifecycle(pid, sid, settings)

        assert len(state["restore_calls"]) == 2, (
            "lifecycle must invoke _restore_displaced_deployment "
            "once per row returned in deploy_result['displaced']"
        )
        restored_roles = {call["displaced"].role for call in state["restore_calls"]}
        assert restored_roles == {"teacher", "embedding"}

    def test_lifecycle_no_op_restore_when_nothing_displaced(
        self, test_app_client, patched_lifecycle
    ):
        """Multi-GPU happy path: deploy_local_nim returns
        ``displaced=[]`` because the auto-placer found a free GPU.
        The lifecycle's auto-restore loop is a no-op (no calls)."""
        state, _sse = patched_lifecycle
        state["deploy_status"] = "running"
        state["displaced_returned"] = []  # nothing displaced
        pid, sid, settings = _make_project_and_student(test_app_client)

        self._drive_lifecycle(pid, sid, settings)

        assert state["restore_calls"] == [], (
            "auto-restore must be a no-op when nothing was displaced"
        )

    def test_displaced_teacher_restored_on_deploy_failure(
        self, test_app_client, patched_lifecycle
    ):
        """A FAILED lifecycle must still restore the displaced Teacher.

        On a single-GPU host the deploy stops the resident Teacher to take the
        GPU. If the Student's own container then fails to start, the earlier
        code left the Teacher down. The finally now restores it on every exit
        path — otherwise the SME is stranded with no Teacher after a failed
        benchmark.
        """
        state, _sse = patched_lifecycle
        # deploy returns a displaced Teacher, but the Student container fails.
        state["deploy_status"] = "failed"
        teacher_row = LocalNimDeployment(
            local_nim_deployment_id="displaced-teacher",
            project_id="p",
            model_config_id="t-mc",
            role="teacher",
            nim_container_image="cosmos:1.6.0",
            container_name="vlm-teacher-victim",
            host_port=8000,
            endpoint_url="http://localhost:8000/v1",
            gpu_assignment="device=0",
            status="stopped",
        )
        state["displaced_returned"] = [teacher_row]
        pid, sid, settings = _make_project_and_student(test_app_client)

        self._drive_lifecycle(pid, sid, settings)

        assert len(state["restore_calls"]) == 1, (
            "a failed deploy must still restore the displaced Teacher"
        )
        assert state["restore_calls"][0]["displaced"].role == "teacher"


class TestEvaluationPhaseDeploymentParity:
    @pytest.mark.asyncio
    async def test_student_eval_runs_native_visual_budget_and_no_icl(
        self, monkeypatch, tmp_path
    ):
        """The Student serving eval must match the training distribution:
        zero ICL AND native image resolution.

        §9.3 training consumes images at native size; inheriting the
        project's Teacher-oriented visual budget (default high_detail,
        shortest_edge 1568) upscales the inputs far off the training
        distribution — measured live on the same CR2-2B checkpoint and
        the same 120 held-out keys: EM 0.95 native vs 0.367
        upscaled. A regression here silently mis-scores every Student and
        would mis-serve deployed ones.
        """
        captured: dict[str, Any] = {}

        async def fake_start_evaluation_run(project_id, **kwargs):
            captured.update(kwargs, project_id=project_id)
            return "error: stop here"  # early-exit the phase after capture

        monkeypatch.setattr(
            lifecycle, "start_evaluation_run", fake_start_evaluation_run
        )
        monkeypatch.setattr(
            lifecycle, "_student_inference_contract", lambda *a, **k: {}
        )
        monkeypatch.setattr(lifecycle, "get_project_engine", lambda *a, **k: None)

        snapshot = lifecycle.StudentSnapshot(
            student_model_id="sm-1",
            project_id="proj-1",
            student_base_model_config_id="mc-base",
            nim_checkpoint_ref=str(tmp_path),
            quantization_method=None,
            base_model_name="nvidia/cosmos-reason2-2b",
            base_context_window_tokens=262144,
            base_supports_image_input=True,
            base_local_deploy_metadata=None,
            base_structured_generation_support="supported",
            base_visual_budget_mode="mm_processor_size",
            base_visual_budget_support="supported",
            base_max_images_per_request=5,
            dataset_export_ids=[],
            guidance_id="guid-1",
            tao_job_id="train-1",
            quantize_tao_job_id=None,
        )
        run_id, run, err = await lifecycle._run_evaluation_phase(
            snapshot, str(tmp_path), make_stub_settings(), "mc-target"
        )
        assert err == "error: stop here"
        assert captured["icl_mode"] == "disabled"
        assert captured["visual_budget_preset_key"] == "native"

    @pytest.mark.asyncio
    async def test_student_eval_uses_paired_held_out_export_checksum(
        self, test_app_client, monkeypatch
    ):
        """Serving evaluation provenance follows the evaluate job because a
        Student's own export IDs correctly contain training data only.
        """
        from vlm_feedback_loop.db.models.dataset_export import DatasetExport
        from vlm_feedback_loop.db.models.tao_job import TAOJob
        from vlm_feedback_loop.services.project_service import get_project_engine

        pid, sid, settings = _make_project_and_student(test_app_client)
        engine = get_project_engine(pid, settings.WORKSPACE_ROOT)
        with Session(engine) as session:
            student = session.get(StudentModel, sid)
            assert student is not None
            evaluation_export_id = generate_uuid4()
            session.add(
                DatasetExport(
                    dataset_export_id=evaluation_export_id,
                    project_id=pid,
                    dataset_intent="evaluation",
                    export_field_mode="core_only",
                    guidance_id="g-1",
                    label_tier_filter="verified",
                    selection_definition_snapshot={},
                    artifact_refs={"checksum_sha256": "held-out-sha"},
                    manifest_ref="evaluation-manifest",
                    example_count=20,
                )
            )
            session.add(
                TAOJob(
                    tao_job_id=generate_uuid4(),
                    project_id=pid,
                    student_base_model_config_id=(student.student_base_model_config_id),
                    dataset_export_ids=[evaluation_export_id],
                    action="evaluate",
                    status="succeeded",
                    training_backend="cosmos_rl_tao_vlm",
                    job_config={},
                    tao_create_job_request={},
                    parent_tao_job_id=student.tao_job_id,
                    chain_sequence=2,
                )
            )
            session.commit()

        captured: dict[str, Any] = {}

        async def fake_start_evaluation_run(project_id, **kwargs):
            captured.update(kwargs, project_id=project_id)
            return "error: stop here"

        monkeypatch.setattr(
            lifecycle, "start_evaluation_run", fake_start_evaluation_run
        )
        snapshot = lifecycle._load_student_snapshot(pid, sid, settings.WORKSPACE_ROOT)
        assert snapshot is not None
        _, _, err = await lifecycle._run_evaluation_phase(
            snapshot,
            settings.WORKSPACE_ROOT,
            settings,
            "mc-target",
        )
        assert err == "error: stop here"
        assert captured["dataset_manifest_sha256"] == "held-out-sha"


class TestWriteStudentStateLockRetry:
    def test_retries_lock_then_succeeds(self, monkeypatch, tmp_path):
        """Every lifecycle state write goes through _write_student_state; a
        transient 'database is locked' (WAL snapshot-upgrade conflict under
        FTMS-poller write load) must retry, not crash the whole lifecycle
        and strand serving_status='pending' (observed live
        2026-07-15)."""
        from sqlalchemy.exc import OperationalError

        calls = {"n": 0}

        class _FakeSession:
            def __init__(self, *a, **k):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def get(self, *a):
                class _S:
                    pass

                return _S()

            def commit(self):
                calls["n"] += 1
                if calls["n"] == 1:
                    raise OperationalError("stmt", {}, Exception("database is locked"))

        monkeypatch.setattr(lifecycle, "get_project_engine", lambda *a: object())
        monkeypatch.setattr(lifecycle, "Session", _FakeSession)
        monkeypatch.setattr(lifecycle.time, "sleep", lambda s: None)
        lifecycle._write_student_state(
            "p", "s", "/tmp", fields={"serving_status": "failed"}
        )
        assert calls["n"] == 2
