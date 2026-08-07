# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the student_nim_deploy Action Request generator.

One generator renders the operator-facing deploy instructions for every
local-NIM role; Student contexts carry extra keys (checkpoint mount,
served-model name, quantization) on top of the base deploy context.
Verifies:

  - Each of the 6 preflight checks produces a distinguishable rendered_text
    when it's the failing one (per-check Action Request copy).
  - rendered_text contains: docker run command, GPU requirements,
    checkpoint path, health check command, smoke test, preflight failure
    details.
  - technical_requirements omits the Student section when student_model_id
    is absent.

The cross-generator no-secrets and technical_requirements-structure
invariants live in test_generator_invariants.py.
"""

from __future__ import annotations

import pytest

import vlm_feedback_loop.services.local_nim_deploy_generator  # noqa: F401 — register
from vlm_feedback_loop.services.action_requests import generate_action_request

# All 6 preflight checks emitted by run_preflight_checks.
PREFLIGHT_CHECKS = [
    ("docker", "Docker not available: command not found"),
    ("nvidia_toolkit", "NVIDIA Container Toolkit missing — gpu passthrough disabled"),
    ("gpu_memory", "24 GB available < 56 GB required for Cosmos Reason2 8B BF16"),
    ("ngc_api_key", "NGC_API_KEY is empty"),
    ("image_pullable", "docker pull failed: 401 Unauthorized"),
    ("model_profile", "list-model-profiles: no compatible profile for SM 8.0"),
]


def _student_context(failing_check: str, diagnostic: str) -> dict:
    checks = []
    for name, _default_diag in PREFLIGHT_CHECKS:
        if name == failing_check:
            checks.append(
                {"check_name": name, "passed": False, "diagnostic": diagnostic}
            )
        else:
            checks.append({"check_name": name, "passed": True, "diagnostic": "OK"})
    return {
        "docker_run_command": (
            "docker run -d \\\n"
            "  --name vlm-student-abc12345-def67890 \\\n"
            "  --runtime=nvidia \\\n"
            '  --gpus "device=0" \\\n'
            "  --shm-size=32GB \\\n"
            "  -p 8002:8000 \\\n"
            "  -e NGC_API_KEY \\\n"
            "  -v /home/u/.cache/nim:/opt/nim/.cache \\\n"
            "  -u $(id -u) \\\n"
            "  -v /tmp/ckpt:/opt/checkpoints/student:ro \\\n"
            "  -e NIM_MODEL_NAME=/opt/checkpoints/student \\\n"
            "  -e NIM_SERVED_MODEL_NAME=student-abc12345 \\\n"
            "  nvcr.io/nim/nvidia/cosmos-reason2-2b:1.6.0"
        ),
        "preflight_checks": checks,
        "role": "student",
        "nim_container_image": "nvcr.io/nim/nvidia/cosmos-reason2-2b:1.6.0",
        "gpu_assignment": "device=0",
        "gpu_memory_minimum_gb": 24,
        "host_port": 8002,
        "student_model_id": "0123456789abcdef0123456789abcdef",
        "nim_checkpoint_ref": "/tmp/ckpt",
        "quantization_method": "FP8_DYNAMIC",
        "nim_served_model_name": "student-abc12345",
        "nim_model_name_path": "/opt/checkpoints/student",
        "checkpoint_directory_structure": [
            "config.json",
            "tokenizer.json|tokenizer.model",
            "*.safetensors|pytorch_model*.bin",
        ],
        "nim_release_version": "1.6.0",
    }


# ═══════════════════════════════════════════════════════════════════════════
# Per-check distinguishable Action Request copy
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("check_name,diag", PREFLIGHT_CHECKS)
class TestPerCheckActionRequest:
    """For each of the 6 preflight checks, when *that* check is the failing
    one, the rendered_text must (a) name it as failed (✗) and (b) include
    its diagnostic substring.
    """

    def test_failing_check_name_appears_with_x_marker(self, check_name, diag):
        result = generate_action_request(
            request_type="student_nim_deploy",
            project_name="Test Project",
            project_id="abc-123",
            context=_student_context(check_name, diag),
        )
        text = result["rendered_text"]
        # The failing check's diagnostic appears verbatim.
        # gpu_memory diagnostic literal contains "GB" and a number.
        assert diag in text
        # The failing check is marked ✗ next to its name.
        assert f"✗ {check_name}" in text

    def test_other_checks_marked_passed(self, check_name, diag):
        """Non-failing checks should appear as ✓."""
        result = generate_action_request(
            request_type="student_nim_deploy",
            project_name="Test Project",
            project_id="abc-123",
            context=_student_context(check_name, diag),
        )
        text = result["rendered_text"]
        # Every non-failing check renders as ✓ {name}.
        for other_name, _ in PREFLIGHT_CHECKS:
            if other_name == check_name:
                continue
            assert f"✓ {other_name}" in text


# ═══════════════════════════════════════════════════════════════════════════
# Rendered-text content requirements
# ═══════════════════════════════════════════════════════════════════════════


class TestRequiredSections:
    def test_rendered_text_contains_required_sections(self):
        result = generate_action_request(
            request_type="student_nim_deploy",
            project_name="Acme",
            project_id="abc-123",
            context=_student_context("gpu_memory", "24 GB < 56 GB"),
        )
        assert result["request_type"] == "student_nim_deploy"
        text = result["rendered_text"]
        # docker run command
        assert "docker run -d" in text
        # GPU requirements (memory minimum)
        assert "GPU: at least 24 GB memory" in text
        assert "GPU memory minimum (this precision): at least 24 GB" in text
        # checkpoint path (host)
        assert "/tmp/ckpt" in text
        # health check command
        assert "/v1/health/ready" in text
        # smoke test
        assert "/v1/models" in text
        # preflight failure details (the failing diagnostic)
        assert "24 GB < 56 GB" in text
        # Student-specific fields
        assert "Student model id" in text
        assert "0123456789abcdef" in text
        assert "/opt/checkpoints/student" in text
        assert "student-abc12345" in text
        assert "FP8_DYNAMIC" in text
        assert "1.6.0" in text
        assert "NVIDIA driver 580.65.06+" in text
        assert (
            result["technical_requirements"]["host_prerequisites"]["nvidia_driver"]
            == "580.65.06+"
        )

    def test_no_student_section_when_student_id_absent(self):
        """Calling the generator with a base Teacher (non-Student) context
        must not produce a Student section in rendered_text."""
        result = generate_action_request(
            request_type="student_nim_deploy",
            project_name="Acme",
            project_id="abc-123",
            context={
                "docker_run_command": "docker run ...",
                "preflight_checks": [
                    {
                        "check_name": "docker",
                        "passed": False,
                        "diagnostic": "missing",
                    }
                ],
                "role": "teacher",
                "nim_container_image": "img",
                "gpu_assignment": "device=0",
                "gpu_memory_minimum_gb": 56,
                "host_port": 8000,
            },
        )
        text = result["rendered_text"]
        assert "Student model id" not in text
        assert "Student deployment context" not in text
        assert "student_deployment" not in result["technical_requirements"]


class TestEnvelope:
    def test_generated_at_is_utc(self):
        """generated_at timestamp ends with Z (UTC)."""
        result = generate_action_request(
            request_type="student_nim_deploy",
            project_name="Test",
            project_id="test-123",
            context={"role": "teacher"},
        )
        assert result["generated_at"].endswith("Z")
