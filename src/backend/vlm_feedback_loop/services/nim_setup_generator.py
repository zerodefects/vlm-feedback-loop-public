# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""nim_setup Action Request content generator.

Registered at import time.  Consumed by the NIM Connection screen's
[Request NIM Setup] button.
"""

from __future__ import annotations

from typing import Any

from vlm_feedback_loop.services.action_requests import register_generator
from vlm_feedback_loop.services.project_service import SEEDED_MODEL_CATALOG


def _generate_nim_setup(
    project_name: str,
    project_id: str,  # noqa: ARG001 — uniform Action Request generator signature
    context: dict[str, Any],  # noqa: ARG001 — uniform Action Request generator signature
) -> dict[str, Any]:
    """Generate a nim_setup Action Request.

    Pre-fills: NIM endpoint configuration fields, target model names,
    hosting-team NGC key guidance, GPU requirements per model, and
    verification details.
    """
    model_lines: list[str] = []
    gpu_lines: list[str] = []
    for entry in SEEDED_MODEL_CATALOG:
        model_lines.append(f"  - {entry['model_name']}")
        meta = entry.get("local_deploy_metadata")
        if meta:
            gpu_lines.append(
                f"  - {entry['model_name']}: at least {meta['nim_gpu_memory_minimum_gb']} GB GPU memory"
            )

    models_text = "\n".join(model_lines) if model_lines else "  (none configured)"
    gpu_text = "\n".join(gpu_lines) if gpu_lines else "  (no local deployment metadata)"

    rendered_text = (
        f"Request: Self-Hosted NVIDIA NIM Endpoint\n"
        f"\n"
        f"Project: {project_name}\n"
        f"\n"
        f"I am using VLM Feedback Loop and need help connecting this project "
        f"to a self-hosted NVIDIA NIM vision-language model.\n"
        f"\n"
        f"Please provide an existing endpoint or deploy one with:\n"
        f"  - An OpenAI-compatible base URL ending in /v1 "
        f"(e.g., https://nim.example.internal/v1)\n"
        f"  - Network access from the VLM Feedback Loop backend\n"
        f"  - Access protected by your organization's trusted network controls; "
        f"the application does not send per-request credentials to self-hosted NIMs\n"
        f"\n"
        f"Model options (only one is needed):\n"
        f"{models_text}\n"
        f"\n"
        f"Known minimum GPU memory:\n"
        f"{gpu_text}\n"
        f"  Models without a minimum listed above require sizing from their "
        f"NIM support matrix.\n"
        f"\n"
        f"Hosting team note:\n"
        f"  The team operating the NIM may need an NVIDIA NGC Personal API Key "
        f"to pull the container and model artifacts.\n"
        f"  Please keep that key in the hosting environment and do not send it "
        f"back by email.\n"
        f"\n"
        f"Verification:\n"
        f"  - GET {{base_url}}/models returns the selected model ID\n"
        f"  - POST {{base_url}}/chat/completions accepts a vision request\n"
        f"  (The base URL already includes /v1.)\n"
        f"\n"
        f"Please send back:\n"
        f"  - The NIM base URL\n"
        f"  - The model ID returned by the endpoint\n"
        f"  - Any VPN, firewall allowlisting, or private DNS requirements\n"
        f"  - A technical contact for endpoint issues\n"
        f"\n"
        f"If your security policy requires per-request authentication, please "
        f"let us know. Do not send passwords, API keys, or tokens by email.\n"
        f"\n"
        f"Documentation:\n"
        f"  NVIDIA NIM: https://docs.nvidia.com/nim/\n"
        f"  NGC API keys: https://org.ngc.nvidia.com/account/api-key\n"
    )

    technical_requirements = {
        "endpoint_config": {
            "base_url": "http://<host>:<port>/v1",
        },
        "target_models": [e["model_name"] for e in SEEDED_MODEL_CATALOG],
        "gpu_requirements": {
            entry["model_name"]: {
                "gpu_memory_minimum_gb": entry["local_deploy_metadata"][
                    "nim_gpu_memory_minimum_gb"
                ],
                "nim_container_image": entry["local_deploy_metadata"][
                    "nim_container_image"
                ],
            }
            for entry in SEEDED_MODEL_CATALOG
            if entry.get("local_deploy_metadata")
        },
        "ngc_api_key_required": True,
        "verification_endpoint": "GET {base_url}/models",
    }

    current_environment: dict[str, Any] = {}

    return {
        "technical_requirements": technical_requirements,
        "current_environment": current_environment,
        "rendered_text": rendered_text,
    }


# Register at import time (side-effect import in main.py)
register_generator("nim_setup", _generate_nim_setup)
