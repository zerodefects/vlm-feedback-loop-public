# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""student_nim_deploy Action Request content generator.

Registered at import time.  Re-used for all NIM deploy preflight failures —
both Student NIM and local Teacher / embedding NIM.

Consumed by the NIM Connection screen's local deploy override and the
Compare & Deploy screen.
"""

from __future__ import annotations

from typing import Any

from vlm_feedback_loop.services.action_requests import register_generator

NIM_MIN_DRIVER_VERSION = "580.65.06"


def _generate_student_nim_deploy(
    project_name: str,
    project_id: str,  # noqa: ARG001 — uniform Action Request generator signature
    context: dict[str, Any],
) -> dict[str, Any]:
    """Generate a student_nim_deploy Action Request.

    Pre-fills: exact docker run command, preflight diagnostics, GPU
    requirements, NIM release, health check and smoke test commands,
    startup and temporary-infrastructure notes.

    Expected *context* keys (all optional — graceful fallback):
      - ``docker_run_command``: the exact command string
      - ``preflight_checks``: list of dicts with check_name/passed/diagnostic
      - ``role``: "teacher" | "embedding" | "student"
      - ``nim_container_image``: target image
      - ``gpu_assignment``: target GPU (e.g., "device=0")
      - ``gpu_memory_minimum_gb``: minimum GPU memory
      - ``host_port``: resolved host port

    Student-specific keys (appended when present):
      - ``student_model_id``
      - ``nim_checkpoint_ref``
      - ``quantization_method``
      - ``nim_served_model_name``
      - ``nim_model_name_path`` (in-container checkpoint path)
      - ``checkpoint_directory_structure`` (list of expected files)
      - ``nim_release_version``
    """
    docker_run_command = context.get("docker_run_command", "(not available)")
    preflight_checks = context.get("preflight_checks", [])
    role = context.get("role", "unknown")
    nim_container_image = context.get("nim_container_image", "(not specified)")
    gpu_assignment = context.get("gpu_assignment", "(not specified)")
    gpu_memory_minimum_gb = context.get("gpu_memory_minimum_gb", "unknown")
    host_port = context.get("host_port", 8000)

    # Build preflight diagnostics section
    diagnostics_lines: list[str] = []
    for check in preflight_checks:
        status = "✓" if check.get("passed") else "✗"
        diagnostics_lines.append(
            f"  {status} {check.get('check_name', '?')}: {check.get('diagnostic', '?')}"
        )
    diagnostics_text = (
        "\n".join(diagnostics_lines) if diagnostics_lines else "  (no preflight data)"
    )

    # Optional Student-specific section
    student_model_id = context.get("student_model_id")
    student_section_text = ""
    student_section_dict: dict[str, Any] = {}
    if student_model_id:
        nim_checkpoint_ref = context.get("nim_checkpoint_ref", "(not specified)")
        quantization_method = context.get("quantization_method", "bf16")
        nim_served_model_name = context.get("nim_served_model_name", "(not specified)")
        nim_model_name_path = context.get(
            "nim_model_name_path", "/opt/checkpoints/student"
        )
        checkpoint_dir_structure = context.get(
            "checkpoint_directory_structure",
            [
                "config.json",
                "tokenizer.json|tokenizer.model",
                "*.safetensors|pytorch_model*.bin",
            ],
        )
        nim_release_version = context.get("nim_release_version", "(not specified)")

        student_section_text = (
            f"\nStudent deployment context:\n"
            f"  Student model id:        {student_model_id}\n"
            f"  Checkpoint path (host):  {nim_checkpoint_ref}\n"
            f"  Expected files:          {', '.join(checkpoint_dir_structure)}\n"
            f"  NIM_MODEL_NAME:          {nim_model_name_path}\n"
            f"  NIM_SERVED_MODEL_NAME:   {nim_served_model_name}\n"
            f"  Quantization:            {quantization_method}\n"
            f"  NIM release:             {nim_release_version}\n"
            f"  GPU memory minimum (this precision): at least {gpu_memory_minimum_gb} GB\n"
        )
        student_section_dict = {
            "student_model_id": student_model_id,
            "nim_checkpoint_ref": nim_checkpoint_ref,
            "quantization_method": quantization_method,
            "nim_served_model_name": nim_served_model_name,
            "nim_model_name_path": nim_model_name_path,
            "checkpoint_directory_structure": checkpoint_dir_structure,
            "nim_release_version": nim_release_version,
        }

    request_title = (
        "Student Serving Validation Deployment Request"
        if student_model_id
        else "NIM Deployment Request"
    )
    rendered_text = (
        f"{request_title}\n"
        f"\n"
        f"Project: {project_name}\n"
        f"Role: {role}\n"
        f"\n"
        f"A local NIM deployment could not be started automatically.\n"
        f"Please deploy the NIM container manually or on suitable infrastructure.\n"
        f"\n"
        f"Docker command:\n"
        f"\n"
        f"{docker_run_command}\n"
        f"\n"
        f"Host prerequisites:\n"
        f"  - Linux OS (Ubuntu 22.04 LTS recommended)\n"
        f"  - NVIDIA driver {NIM_MIN_DRIVER_VERSION}+\n"
        f"  - Docker 29.4.0+\n"
        f"  - NVIDIA Container Toolkit 1.19.0+\n"
        f"  - GPU: at least {gpu_memory_minimum_gb} GB memory\n"
        f"  - NGC_API_KEY configured\n"
        f"\n"
        f"Preflight results:\n"
        f"{diagnostics_text}\n"
        f"{student_section_text}"
        f"\n"
        f"Health check:\n"
        f"  GET http://localhost:{host_port}/v1/health/ready\n"
        f"\n"
        f"Smoke test:\n"
        f"  GET http://localhost:{host_port}/v1/models\n"
        f"\n"
        f"Note: Startup may take several minutes while NIM builds runtime\n"
        f"artifacts. Use a persistent cache mount (~/.cache/nim:/opt/nim/.cache)\n"
        f"to avoid rebuilding on restart.\n"
        f"\n"
        f"This endpoint is for evaluation only. It can be stopped after\n"
        f"results are collected. It is not the permanent production service.\n"
        f"\n"
        f"Setup script: ./scripts/setup-local.sh\n"
    )

    technical_requirements: dict[str, Any] = {
        "docker_run_command": docker_run_command,
        "nim_container_image": nim_container_image,
        "role": role,
        "gpu_assignment": gpu_assignment,
        "gpu_memory_minimum_gb": gpu_memory_minimum_gb,
        "host_port": host_port,
        "host_prerequisites": {
            "os": "Linux (Ubuntu 22.04 LTS)",
            "nvidia_driver": f"{NIM_MIN_DRIVER_VERSION}+",
            "docker": "29.4.0+",
            "nvidia_container_toolkit": "1.19.0+",
            "no_vgpu": True,
        },
        "health_check": f"GET http://localhost:{host_port}/v1/health/ready",
        "smoke_test": f"GET http://localhost:{host_port}/v1/models",
        "preflight_checks": preflight_checks,
    }
    if student_section_dict:
        technical_requirements["student_deployment"] = student_section_dict

    current_environment: dict[str, Any] = {}

    return {
        "technical_requirements": technical_requirements,
        "current_environment": current_environment,
        "rendered_text": rendered_text,
    }


# Register at import time (side-effect import in main.py)
register_generator("student_nim_deploy", _generate_student_nim_deploy)
