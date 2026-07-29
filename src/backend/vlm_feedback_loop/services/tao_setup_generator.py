# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""tao_setup Action Request content generator.

Registered at import time.  Consumed by the Scale-Up hub and Student
Training preflight when TAO is not configured or unreachable.
"""

from __future__ import annotations

from typing import Any

from vlm_feedback_loop.services.action_requests import register_generator
from vlm_feedback_loop.services.project_service import SEEDED_MODEL_CATALOG

# Models eligible for student_base role
_STUDENT_BASE_MODELS = [
    e for e in SEEDED_MODEL_CATALOG if "student_base" in e["eligible_roles"]
]


def _generate_tao_setup(
    project_name: str,
    project_id: str,  # noqa: ARG001 — uniform Action Request generator signature
    context: dict[str, Any],  # noqa: ARG001 — uniform Action Request generator signature
) -> dict[str, Any]:
    """Generate a tao_setup Action Request.

    Pre-fills: required config fields, workspace + base-experiment
    provisioning steps, bootstrap CLI pointer,
    student base models, connection test endpoint, and GPU requirements.
    """
    model_lines = "\n".join(f"  - {m['model_name']}" for m in _STUDENT_BASE_MODELS)
    base_experiment_lines = "\n".join(
        f"  - {m['model_name']} → requires `tao_base_experiment_id` pulled from NGC"
        for m in _STUDENT_BASE_MODELS
    )

    rendered_text = (
        f"TAO Setup Request\n"
        f"\n"
        f"Project: {project_name}\n"
        f"\n"
        f"Student Training requires:\n"
        f"  1. A TAO FTMS endpoint.\n"
        f"  2. A TAO workspace (S3-backed storage owner for every dataset\n"
        f"     and job output — one per deployment).\n"
        f"  3. A Cosmos Reason2 base experiment pulled from NGC into that\n"
        f"     workspace, one per student base model.\n"
        f"\n"
        f"On a fresh FTMS deployment none of these exist. The Blueprint's\n"
        f"`vlm-feedback-loop tao-bootstrap` CLI provisions them one time\n"
        f"per deployment.\n"
        f"\n"
        f"Required configuration (~/.vlm_feedback_loop/.env):\n"
        f"  TAO_API_BASE_URL=https://<tao-host>/api/v2\n"
        f"  TAO_API_KEY=<JWT bearer token from POST /api/v2/login>\n"
        f"  TAO_ORG_NAME=<organization name>\n"
        f"  TAO_WORKSPACE_S3_ACCESS_KEY=<S3 access key>\n"
        f"  TAO_WORKSPACE_S3_SECRET_KEY=<S3 secret key>\n"
        f"\n"
        f"Add all five secret/config values before running the bootstrap.\n"
        f"Managed deployments may inject the two S3 credentials as process\n"
        f"environment variables; `tao-bootstrap` does not persist them.\n"
        f"Workspace identity (workspace id,\n"
        f"bucket, endpoint URLs) is NOT env configuration — `tao-bootstrap`\n"
        f"persists it to the deployment database (deployment.db).\n"
        f"\n"
        f"Authentication:\n"
        f"  1. Obtain an NGC Personal API Key from https://org.ngc.nvidia.com/setup/api-key\n"
        f"  2. Exchange it for a JWT: POST {{TAO_API_BASE_URL}}/login\n"
        f"  3. Use the returned JWT as TAO_API_KEY\n"
        f"\n"
        f"Bootstrap step (one-time per deployment):\n"
        f"  vlm-feedback-loop tao-bootstrap \\\n"
        f"    --workspace-name <name> \\\n"
        f"    --cloud-type seaweedfs \\\n"
        f"    --bucket <bucket>\n"
        f"\n"
        f"The CLI creates the workspace, pulls each student base checkpoint\n"
        f"from Hugging Face, registers it with TAO, persists workspace identity to\n"
        f"deployment.db, leaves credentials in their configured environment\n"
        f"source, and writes the resulting `tao_base_experiment_id` onto every\n"
        f"project's seeded ModelConfig rows. Running it a second time\n"
        f"against an already-bootstrapped deployment is a no-op.\n"
        f"\n"
        f"Student base models requiring base-experiment pulls:\n"
        f"{base_experiment_lines}\n"
        f"\n"
        f"Student base models:\n"
        f"{model_lines}\n"
        f"\n"
        f"GPU requirements (per NVIDIA Cosmos-Reason docs):\n"
        f"  8x A100 80 GB minimum for Cosmos Reason2 training\n"
        f"\n"
        f"Verification:\n"
        f"  GET {{TAO_API_BASE_URL}}/orgs/{{TAO_ORG_NAME}}/jobs?limit=1\n"
        f"    → JSON response with jobs list\n"
        f"  deployment.db `tao_deployment_configs` singleton must show\n"
        f"  `bootstrap_status = 'bootstrapped'` with a non-null\n"
        f"  `tao_workspace_id`.\n"
        f"  Each student base ModelConfig must have `tao_base_experiment_id`\n"
        f"  non-null and `tao_base_experiment_pull_status = 'pull_complete'`.\n"
        f"\n"
        f"Documentation:\n"
        f"  TAO FTMS Setup: https://docs.nvidia.com/tao/tao-toolkit/latest/text/tao_toolkit_api/api_setup.html\n"
        f"  Blueprint TAO Guide: docs/tao-ftms-install.md\n"
    )

    technical_requirements = {
        "required_config_fields": [
            "TAO_API_BASE_URL",
            "TAO_API_KEY",
            "TAO_ORG_NAME",
            "TAO_WORKSPACE_S3_ACCESS_KEY",
            "TAO_WORKSPACE_S3_SECRET_KEY",
        ],
        "tao_workspace": {
            "description": (
                "A TAO workspace is the S3-backed storage owner for every "
                "dataset and job output. One workspace per Blueprint "
                "deployment; its identity (workspace id, bucket, endpoint "
                "URLs) is created by `vlm-feedback-loop tao-bootstrap` and "
                "persisted to deployment.db, not .env."
            ),
            "required_base_experiments": [
                m["model_name"] for m in _STUDENT_BASE_MODELS
            ],
            "bootstrap_cli": "vlm-feedback-loop tao-bootstrap",
            "verification": (
                "deployment.db tao_deployment_configs singleton: "
                "bootstrap_status = 'bootstrapped' with a non-null "
                "tao_workspace_id"
            ),
        },
        "student_base_models": [m["model_name"] for m in _STUDENT_BASE_MODELS],
        "gpu_requirements": "8x A100 80 GB minimum for Cosmos Reason2 training",
        "connection_test": "GET {TAO_API_BASE_URL}/orgs/{TAO_ORG_NAME}/jobs?limit=1",
    }

    current_environment: dict[str, Any] = {}

    return {
        "technical_requirements": technical_requirements,
        "current_environment": current_environment,
        "rendered_text": rendered_text,
    }


# Register at import time (side-effect import in main.py)
register_generator("tao_setup", _generate_tao_setup)
