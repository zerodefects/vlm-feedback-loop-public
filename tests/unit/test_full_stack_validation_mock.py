# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mock-mode tests for ``scripts/full_stack_validation.py`` — fully in-process.

The handoff re-execution proof (re-running a handoff's `docker run` for
both 2B and 8B produces a NIM that responds with a schema-valid label
round-trip) has two evidence states: mock-validated (no GPU hardware
required) and live-validated (against real hardware). These tests cover
the mock-validated state by exercising:

  * The `docker_run_args` shape validator (`_validate_docker_run_args`).
  * The chat/completions round-trip helper (`_send_round_trip`) against
    the in-process MockNIMServer.
  * `phase_d_handoff_reexecution` end-to-end in mock mode for both 2B
    and 8B paired handoffs — proves the closing-smoke's re-execution
    path is correct when run against real hardware (the diff is just whether
    the docker subprocess actually exec's vs. sends to a mock).

The full orchestration (Phase A discover via real HTTP + Phase B deploy
via real :deploy_nim) is exercised live against real hardware. The
helpers + orchestrator-error-path coverage here is what lands in CI.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import full_stack_validation as fsv  # noqa: E402
from full_stack_validation import (  # noqa: E402
    FullStackValidationResult,
    _extract_host_port,
    _send_round_trip,
    _validate_docker_run_args,
    phase_d_handoff_reexecution,
    run_full_stack_validation,
)

from mock_nim_server import MockNIMServer  # noqa: E402


class TestExtractHostPort:
    """Phase D MUST poll the host port that the docker_run_args
    actually mapped to, not the LOCAL_NIM_TEACHER_PORT default. The
    Blueprint's port resolver increments past the dev backend (which
    holds 8000), so the resolved host port can be 8002, 8003, etc."""

    def test_canonical_p_flag(self):
        args = ["docker", "run", "-p", "8002:8000", "image"]
        assert _extract_host_port(args) == 8002

    def test_default_8000_when_p_absent(self):
        """Defensive — every well-formed handoff has -p, but if missing
        fall back to LOCAL_NIM_TEACHER_PORT default."""
        assert _extract_host_port(["docker", "run", "image"]) == 8000

    def test_three_part_iface_host_container(self):
        """Some configurations bind explicitly to an interface."""
        args = ["docker", "run", "-p", "0.0.0.0:8003:8000", "image"]
        assert _extract_host_port(args) == 8003

    def test_real_blueprint_handoff_shape(self):
        """The exact docker_run_args shape a real handoff produces;
        the guarded failure mode is Phase D polling 8000 instead of
        the resolved host port 8002."""
        args = [
            "docker",
            "run",
            "-d",
            "--name",
            "vlm-student-6239d046-4226844e",
            "--runtime=nvidia",
            "--gpus",
            '"device=0"',
            "--shm-size=32GB",
            "-p",
            "8002:8000",
            "-e",
            "NGC_API_KEY",
            "-v",
            "/host/cache:/opt/nim/.cache",
            "-u",
            "1002",
            "-v",
            "/host/checkpoints:/opt/checkpoints/student:ro",
            "-e",
            "NIM_MODEL_NAME=/opt/checkpoints/student",
            "-e",
            "NIM_SERVED_MODEL_NAME=student-4226844e",
            "nvcr.io/nim/nvidia/cosmos-reason2-2b:1.6.0",
        ]
        assert _extract_host_port(args) == 8002


# ── Helper: synthesize a handoff payload for testing ────────────────────────


def _synth_handoff(
    *,
    served_model: str,
    nim_model_name_path: str = "/opt/checkpoints/student",
    container_image: str = "nvcr.io/nim/nvidia/cosmos-reason2-2b:1.6.0",
    container_name: str = "vlm-student-test-2b",
    profile_recommended: str = "vllm-cosmos-2b-bf16",
    gpu_requirements: str = "1× A100 (≥36 GB)",
    nim_env_overrides: dict | None = None,
) -> dict:
    """Produce a handoff dict with realistic docker_run_args (mirrors
    ``local_nim_service._build_docker_run_command`` for role=student)."""
    docker_run_args = [
        "docker",
        "run",
        "-d",
        "--name",
        container_name,
        "--runtime=nvidia",
        "--gpus",
        '"device=0"',
        "--shm-size=32GB",
        "-p",
        "8000:8000",
        "-e",
        "NGC_API_KEY",
        "-v",
        "/home/user/.cache/nim:/opt/nim/.cache",
        "-u",
        "1000",
        "-v",
        f"/tmp/ckpt:{nim_model_name_path}:ro",
        "-e",
        f"NIM_MODEL_NAME={nim_model_name_path}",
        "-e",
        f"NIM_SERVED_MODEL_NAME={served_model}",
        container_image,
    ]
    docker_run_command = (
        f"docker run -d --name {container_name} --runtime=nvidia "
        f'--gpus "device=0" --shm-size=32GB -p 8000:8000 '
        f"-e NGC_API_KEY -v /home/user/.cache/nim:/opt/nim/.cache "
        f"-u $(id -u) -v /tmp/ckpt:{nim_model_name_path}:ro "
        f"-e NIM_MODEL_NAME={nim_model_name_path} "
        f"-e NIM_SERVED_MODEL_NAME={served_model} {container_image}"
    )
    nim_env = {
        "NGC_API_KEY": "$NGC_API_KEY",
        "NIM_MODEL_NAME": nim_model_name_path,
        "NIM_SERVED_MODEL_NAME": served_model,
        "NIM_MODEL_PROFILE": profile_recommended,
    }
    if nim_env_overrides:
        nim_env.update(nim_env_overrides)
    return {
        "request_type": "deployment_handoff",
        "technical_requirements": {
            "nim_container_image": container_image,
            "nim_served_model_name": served_model,
            "nim_model_name_path": nim_model_name_path,
            "nim_model_profile_recommended": profile_recommended,
            "gpu_requirements": gpu_requirements,
            "tensor_parallelism": 1,
            "nim_env_vars_recommended": nim_env,
            "docker_run_args": docker_run_args,
            "docker_run_command": docker_run_command,
        },
        "current_environment": {
            "quality_status": "validated",
            "serving_status": "validated",
        },
        "rendered_text": f"Deployment Handoff Request — {served_model}",
    }


# ── Shape validator ─────────────────────────────────────────────────────────


class TestValidateDockerRunArgs:
    def test_canonical_args_pass(self):
        handoff = _synth_handoff(served_model="stub-2b")
        ok, detail = _validate_docker_run_args(
            handoff["technical_requirements"]["docker_run_args"]
        )
        assert ok, detail
        assert "shape ok" in detail

    def test_empty_args_rejected(self):
        ok, detail = _validate_docker_run_args([])
        assert ok is False
        assert "empty" in detail

    def test_argv_zero_must_be_docker(self):
        ok, detail = _validate_docker_run_args(["podman", "run", "-u", "1000"])
        assert ok is False
        assert "argv[0]" in detail

    def test_run_subcommand_required(self):
        ok, detail = _validate_docker_run_args(["docker", "stop", "x"])
        assert ok is False
        assert "'run'" in detail

    def test_uid_pin_required(self):
        # Lifecycle's _build_docker_run_command always emits -u for student.
        # Strip it and confirm rejection.
        handoff = _synth_handoff(served_model="stub")
        args = [
            a
            for i, a in enumerate(handoff["technical_requirements"]["docker_run_args"])
            if not (a == "-u" or args_at_index(handoff, i - 1, "-u"))
        ]
        ok, detail = _validate_docker_run_args(args)
        assert ok is False
        assert "-u" in detail

    def test_readonly_checkpoint_mount_required(self):
        args = ["docker", "run", "-u", "1000", "--gpus", "device=0", "image"]
        ok, detail = _validate_docker_run_args(args)
        assert ok is False
        assert ":ro" in detail

    def test_gpus_flag_required(self):
        args = ["docker", "run", "-u", "1000", "-v", "x:y:ro", "image"]
        ok, detail = _validate_docker_run_args(args)
        assert ok is False
        assert "--gpus" in detail

    @pytest.mark.parametrize(
        "secret_token",
        [
            "NGC_API_KEY=SENTINEL_LITERAL_NGC",
            "NGC_API_KEY=$NGC_API_KEY",
        ],
    )
    def test_secret_env_requires_name_only_forwarding(self, secret_token):
        handoff = _synth_handoff(served_model="stub")
        args = list(handoff["technical_requirements"]["docker_run_args"])
        ngc_index = args.index("NGC_API_KEY")
        args[ngc_index] = secret_token
        ok, detail = _validate_docker_run_args(args)
        assert ok is False
        assert "name-only" in detail
        assert "SENTINEL_LITERAL_NGC" not in detail


def args_at_index(handoff, idx, value):
    args = handoff["technical_requirements"]["docker_run_args"]
    return 0 <= idx < len(args) and args[idx] == value


# ── Round-trip against MockNIMServer ────────────────────────────────────────


@pytest.mark.asyncio
async def test_send_round_trip_returns_schema_valid_label():
    with MockNIMServer() as mock:
        ok, detail = await _send_round_trip(mock.base_url, "stub-cosmos-2b")
        assert ok, detail
        assert "ok" in detail


@pytest.mark.asyncio
async def test_send_round_trip_rejects_unreachable_endpoint():
    ok, detail = await _send_round_trip(
        "http://127.0.0.1:1/v1", "anything"
    )  # port 1 is unreachable
    assert ok is False


@pytest.mark.asyncio
async def test_send_round_trip_accepts_text_response():
    """The round-trip's intent is to prove the canonical
    ``docker_run_args`` spawn a NIM that responds end-to-end
    — strict JSON-only acceptance is too narrow because Cosmos Reason2 2B
    without JSON-mode prompting responds in natural text (e.g., 'OK',
    'rock paper or scissors'). HTTP 200 + non-empty content is the right
    success signal; the round-trip MUST accept text responses.
    """
    with MockNIMServer() as mock:
        # Override the label payload to a string. The mock wraps in a
        # chat.completion envelope; ``json.dumps("not json")`` becomes
        # ``"not json"`` which IS valid JSON-as-string but NOT a dict.
        # It must be accepted as a valid text response, not rejected
        # for failing dict parsing (the regression shape guarded here).
        mock.set_label_payload("not json")
        ok, detail = await _send_round_trip(mock.base_url, "x")
        assert ok is True, f"text response should pass, got: {detail}"
        assert "text response" in detail or "parsed keys" in detail


@pytest.mark.asyncio
async def test_send_round_trip_accepts_dict_response_with_keys():
    """Sanity: when the response IS a JSON dict, the helper reports
    ``parsed keys`` in the detail (the original happy path)."""
    with MockNIMServer() as mock:
        mock.set_label_payload({"category": "rock", "rationale_note": "test"})
        ok, detail = await _send_round_trip(mock.base_url, "x")
        assert ok is True
        assert "parsed keys" in detail


# ── Phase D end-to-end (mock) ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_phase_d_round_trip_succeeds_for_both_2b_and_8b(tmp_path, monkeypatch):
    """Mock evidence: both handoffs round-trip via the MockNIMServer."""
    with MockNIMServer() as mock:
        monkeypatch.setenv("LOCAL_NIM_MOCK_ENDPOINT_URL", mock.base_url)
        handoffs = {
            "2B": _synth_handoff(
                served_model="student-2b-abc",
                container_image="nvcr.io/nim/nvidia/cosmos-reason2-2b:1.6.0",
                profile_recommended="vllm-cosmos-2b-bf16",
                gpu_requirements="1× A100 (≥36 GB)",
            ),
            "8B": _synth_handoff(
                served_model="student-8b-def",
                container_image="nvcr.io/nim/nvidia/cosmos-reason2-8b:1.6.0",
                profile_recommended="vllm-cosmos-8b-fp8",
                gpu_requirements="1× A100 (≥56 GB)",
            ),
        }
        result = FullStackValidationResult(execution_mode="mock", started_at="t0")
        await phase_d_handoff_reexecution(
            handoffs=handoffs,
            execution_mode="mock",
            result=result,
            evidence_dir=tmp_path,
        )
        assert result.c20_handoff_rerun.two_b is True, (
            result.c20_handoff_rerun.detail_2b
        )
        assert result.c20_handoff_rerun.eight_b is True, (
            result.c20_handoff_rerun.detail_8b
        )
        # Per-handoff evidence files written.
        assert (tmp_path / "handoff_rerun_2b.log").exists()
        assert (tmp_path / "handoff_rerun_8b.log").exists()


@pytest.mark.asyncio
async def test_phase_d_rejects_malformed_docker_run_args(tmp_path, monkeypatch):
    """A handoff with broken docker_run_args fails re-execution even in mock mode."""
    with MockNIMServer() as mock:
        monkeypatch.setenv("LOCAL_NIM_MOCK_ENDPOINT_URL", mock.base_url)
        handoffs = {
            "2B": _synth_handoff(served_model="x"),
            "8B": _synth_handoff(served_model="y"),
        }
        # Sabotage 8B: drop the :ro flag.
        handoffs["8B"]["technical_requirements"]["docker_run_args"] = [
            "docker",
            "run",
            "-u",
            "1000",
            "--gpus",
            "device=0",
            "image",
        ]
        result = FullStackValidationResult(execution_mode="mock", started_at="t0")
        await phase_d_handoff_reexecution(
            handoffs=handoffs,
            execution_mode="mock",
            result=result,
            evidence_dir=tmp_path,
        )
        assert result.c20_handoff_rerun.two_b is True  # 2B still passes
        assert result.c20_handoff_rerun.eight_b is False
        assert ":ro" in result.c20_handoff_rerun.detail_8b


@pytest.mark.asyncio
async def test_phase_d_live_keeps_secret_out_of_argv_and_evidence(
    tmp_path, monkeypatch
):
    """Live re-execution passes NGC only through the Docker child's env."""
    sentinel = "SENTINEL_PHASE_D_NGC"
    monkeypatch.setenv("NGC_API_KEY", sentinel)
    captured: list[tuple[tuple[str, ...], dict]] = []

    def fake_run(args, **kwargs):
        captured.append((tuple(args), dict(kwargs)))
        if args[:2] == ["docker", "run"]:
            return subprocess.CompletedProcess(
                args,
                0,
                stdout=f"a1b2c3d4e5f6\n{sentinel}\n",
                stderr=f"harmless diagnostic around {sentinel}",
            )
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    async def healthy(*_args, **_kwargs):
        return True

    async def round_trip(*_args, **_kwargs):
        return True, "ok"

    monkeypatch.setattr(fsv.subprocess, "run", fake_run)
    monkeypatch.setattr(fsv, "_poll_health", healthy)
    monkeypatch.setattr(fsv, "_send_round_trip", round_trip)
    handoffs = {"2B": _synth_handoff(served_model="student-2b")}
    result = FullStackValidationResult(execution_mode="live", started_at="t0")

    await phase_d_handoff_reexecution(
        handoffs=handoffs,
        execution_mode="live",
        result=result,
        evidence_dir=tmp_path,
    )

    assert result.c20_handoff_rerun.two_b is True
    docker_args, docker_kwargs = captured[0]
    assert docker_args[:2] == ("docker", "run")
    assert sentinel not in " ".join(docker_args)
    ngc_index = docker_args.index("NGC_API_KEY")
    assert docker_args[ngc_index - 1] == "-e"
    assert docker_kwargs["env"]["NGC_API_KEY"] == sentinel
    cleanup_args = [
        args for args, _kwargs in captured if args[:2] == ("docker", "stop")
    ]
    assert cleanup_args == [("docker", "stop", "vlm-student-test-2b")]
    assert all(sentinel not in " ".join(args) for args, _kwargs in captured)
    evidence = (tmp_path / "handoff_rerun_2b.log").read_text()
    assert sentinel not in evidence
    assert "[REDACTED]" in evidence


@pytest.mark.asyncio
async def test_phase_d_live_redacts_before_truncating_failure_detail(
    tmp_path, monkeypatch
):
    """A secret straddling the detail limit cannot survive as a partial value."""
    sentinel = "SENTINEL_PHASE_D_BOUNDARY_" + ("X" * 40)
    padding = "p" * (200 - (len(sentinel) - 1))
    monkeypatch.setenv("NGC_API_KEY", sentinel)

    def fake_run(args, **kwargs):
        return subprocess.CompletedProcess(
            args,
            9,
            stdout="",
            stderr=padding + sentinel,
        )

    monkeypatch.setattr(fsv.subprocess, "run", fake_run)
    result = FullStackValidationResult(execution_mode="live", started_at="t0")

    await phase_d_handoff_reexecution(
        handoffs={"2B": _synth_handoff(served_model="student-2b")},
        execution_mode="live",
        result=result,
        evidence_dir=tmp_path,
    )

    assert result.c20_handoff_rerun.two_b is False
    assert "[REDACTED]" in result.c20_handoff_rerun.detail_2b
    assert sentinel[:-1] not in result.c20_handoff_rerun.detail_2b


# ── Orchestrator end-to-end ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_orchestrator_no_students_writes_evidence(tmp_path, monkeypatch):
    """Happy-ish path: discover returns [] cleanly. Acceptance file written
    with overall_ok=False (no Students)."""

    async def _empty_list(client, project_id):
        return []

    monkeypatch.setattr("full_stack_validation._list_students", _empty_list)

    evidence_dir = tmp_path / "evidence"
    await run_full_stack_validation(
        execution_mode="mock",
        backend_url="http://127.0.0.1:9999",
        student_project_2b="p1",
        student_project_8b="p2",
        evidence_dir=evidence_dir,
    )
    out = evidence_dir / "closing_acceptance.json"
    assert out.exists()
    payload = json.loads(out.read_text())
    assert payload["overall_ok"] is False  # no students discovered
    assert "no quality-validated Students" in payload["error"]


@pytest.mark.asyncio
async def test_acceptance_dict_phase_b_counts_match_targets(tmp_path, monkeypatch):
    """End-to-end mock check: discover returns 4 Students (2B + 2B-FP8 + 8B
    + 8B-FP8); deploy succeeds for all 4 (mocked); the 2B vs 8B handoffs
    are differentiated; the handoff re-execution round-trips against the
    mock; closing_acceptance.json has phase_b_validated_count=4 and
    overall_ok=True.
    """
    # Pre-fabricated discover results.
    students = [
        {
            "student_model_id": "s-2b-a",
            "base_model_name": "nvidia/cosmos-reason2-2b",
            "quantization_method": "none",
            "quality_status": "validated",
        },
        {
            "student_model_id": "s-2b-b",
            "base_model_name": "nvidia/cosmos-reason2-2b",
            "quantization_method": "fp8",
            "quality_status": "validated",
        },
        {
            "student_model_id": "s-8b-a",
            "base_model_name": "nvidia/cosmos-reason2-8b",
            "quantization_method": "none",
            "quality_status": "validated",
        },
        {
            "student_model_id": "s-8b-b",
            "base_model_name": "nvidia/cosmos-reason2-8b",
            "quantization_method": "fp8",
            "quality_status": "validated",
        },
    ]

    async def _list(client, project_id):
        if project_id == "p-2b":
            return [s for s in students if "2b" in s["base_model_name"]]
        return [s for s in students if "8b" in s["base_model_name"]]

    monkeypatch.setattr("full_stack_validation._list_students", _list)

    # Mock the deploy / serving phases so phase_b records validated outcomes
    # without actually hitting the backend.
    async def _stub_deploy(client, *, discovered, result, deploy_timeout_s):
        for _project_id, student in discovered:
            for v in result.variants:
                if v.student_model_id == student["student_model_id"]:
                    v.deploy_ok = True
                    v.serving_status = "validated"
                    result.phase_b_validated_count += 1

    monkeypatch.setattr("full_stack_validation.phase_b_deploy_serving", _stub_deploy)

    # Mock phase_c to inject differentiated handoffs so the
    # differentiation check evaluates True.
    with MockNIMServer() as mock:

        async def _stub_handoffs(client, *, discovered, result, evidence_dir):
            handoffs = {
                "2B": _synth_handoff(
                    served_model="student-2b-abc",
                    container_image="nvcr.io/nim/nvidia/cosmos-reason2-2b:1.6.0",
                    profile_recommended="vllm-cosmos-2b-bf16",
                    gpu_requirements="1× A100 (≥36 GB)",
                ),
                "8B": _synth_handoff(
                    served_model="student-8b-def",
                    container_image="nvcr.io/nim/nvidia/cosmos-reason2-8b:1.6.0",
                    profile_recommended="vllm-cosmos-8b-fp8",
                    gpu_requirements="1× A100 (≥56 GB)",
                ),
            }
            tech_2b = handoffs["2B"]["technical_requirements"]
            tech_8b = handoffs["8B"]["technical_requirements"]
            c21 = result.c21_differentiation
            c21.nim_model_profile_recommended = {
                "two_b": tech_2b["nim_model_profile_recommended"],
                "eight_b": tech_8b["nim_model_profile_recommended"],
                "differs": True,
            }
            c21.gpu_requirements = {
                "two_b": tech_2b["gpu_requirements"],
                "eight_b": tech_8b["gpu_requirements"],
                "differs": True,
            }
            c21.tensor_parallelism = {
                "two_b": tech_2b["tensor_parallelism"],
                "eight_b": tech_8b["tensor_parallelism"],
                "both_int_present": True,
            }
            c21.nim_env_vars_recommended = {
                "two_b": tech_2b["nim_env_vars_recommended"],
                "eight_b": tech_8b["nim_env_vars_recommended"],
                "differs": True,
            }
            c21.differentiated = True
            return handoffs

        monkeypatch.setattr(
            "full_stack_validation.phase_c_handoff_differentiation", _stub_handoffs
        )

        # Phase D uses the real implementation against MockNIMServer.
        monkeypatch.setenv("LOCAL_NIM_MOCK_ENDPOINT_URL", mock.base_url)

        # Phase E is gated on NVIDIA_API_KEY — should auto-skip in CI.
        monkeypatch.delenv("NVIDIA_API_KEY", raising=False)

        evidence_dir = tmp_path / "evidence"
        await run_full_stack_validation(
            execution_mode="mock",
            backend_url="http://127.0.0.1:9999",
            student_project_2b="p-2b",
            student_project_8b="p-8b",
            evidence_dir=evidence_dir,
        )

    out = evidence_dir / "closing_acceptance.json"
    payload = json.loads(out.read_text())
    assert payload["execution_mode"] == "mock"
    assert payload["phase_a_complete"] is True
    assert payload["phase_b_target_count"] == 4
    assert payload["phase_b_validated_count"] == 4
    assert payload["c21_differentiation"]["differentiated"] is True
    assert payload["c20_handoff_rerun"]["two_b"] is True
    assert payload["c20_handoff_rerun"]["eight_b"] is True
    assert payload["final_integration_skipped"] is True  # no API key
    # All required acceptance items pass → overall_ok is True.
    assert payload["overall_ok"] is True


@pytest.mark.asyncio
async def test_phase_a_excludes_listed_student_ids(monkeypatch):
    """Phase A skips Students whose IDs appear in ``exclude_student_ids``.

    A sacrificial sibling Student can live in the 2B project alongside
    the production 2B Students; Phase A must discover the production
    Students but skip the sibling so it never enters Phase B's deploy
    queue. The exclusion takes effect even when the sibling has
    ``quality_status="validated"``.
    """
    students_2b = [
        {
            "student_model_id": "s-2b-prod-a",
            "base_model_name": "nvidia/cosmos-reason2-2b",
            "quantization_method": "none",
            "quality_status": "validated",
        },
        {
            "student_model_id": "s-2b-prod-b",
            "base_model_name": "nvidia/cosmos-reason2-2b",
            "quantization_method": "fp8_dynamic",
            "quality_status": "validated",
        },
        {
            "student_model_id": "s-2b-sibling",  # sacrificial
            "base_model_name": "nvidia/cosmos-reason2-2b",
            "quantization_method": "none",
            "quality_status": "validated",
        },
    ]
    students_8b = [
        {
            "student_model_id": "s-8b-prod-a",
            "base_model_name": "nvidia/cosmos-reason2-8b",
            "quantization_method": "w4a16",
            "quality_status": "validated",
        },
    ]

    async def _list(client, project_id):
        return students_2b if project_id == "p-2b" else students_8b

    monkeypatch.setattr("full_stack_validation._list_students", _list)

    from full_stack_validation import (
        FullStackValidationResult,
        phase_a_discover,
    )

    result = FullStackValidationResult(
        execution_mode="mock", started_at="2026-05-04T00:00:00Z"
    )
    discovered = await phase_a_discover(
        client=None,  # _list_students is monkeypatched; client unused
        project_2b="p-2b",
        project_8b="p-8b",
        result=result,
        exclude_student_ids={"s-2b-sibling"},
    )
    assert {sid for _, s in discovered for sid in [s["student_model_id"]]} == {
        "s-2b-prod-a",
        "s-2b-prod-b",
        "s-8b-prod-a",
    }
    assert result.phase_b_target_count == 3
