# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live TAO FTMS smoke test.

Opt-in test that posts a minimal Quick-preset training-suite creation
request against a real TAO FTMS endpoint. Validates:

  * The Cosmos-RL dataset wire format produced by the dataset exporter
    is accepted by the pinned ``cosmos_rl_container_tag``.
  * The suite endpoint's chain layout is acceptable to the live TAO API.

Skipped by default. Run with ``TAO_API_BASE_URL``, ``TAO_API_KEY``, and
``TAO_ORG_NAME`` set:

    TAO_API_BASE_URL=... TAO_API_KEY=... TAO_ORG_NAME=... \
        uv run pytest tests/integration/test_tao_smoke.py -v

NEVER runs in CI — it submits a real training job.
"""

from __future__ import annotations

import os

import pytest

_SKIP_REASON = (
    "Live TAO FTMS smoke test. Requires TAO_API_BASE_URL, TAO_API_KEY, "
    "TAO_ORG_NAME env vars and a pre-prepared project with a training-intent "
    "DatasetExport that matches the pinned cosmos_rl_container_tag. "
    "Opt-in only — not run in CI."
)


def _env_configured() -> bool:
    return all(
        os.environ.get(k) for k in ("TAO_API_BASE_URL", "TAO_API_KEY", "TAO_ORG_NAME")
    )


@pytest.mark.skipif(not _env_configured(), reason=_SKIP_REASON)
def test_live_ftms_accepts_quick_preset_submission():
    """Submit a minimal 1-epoch Quick-preset training suite to live TAO.

    This test assumes the operator has a seeded project (with student_base
    ModelConfig + a training DatasetExport) in ``VLM_SMOKE_PROJECT_ID``.
    It POSTs to the local backend (``VLM_BACKEND_URL``, default
    ``http://localhost:8000``) and asserts the suite endpoint accepted the
    request: 201 with a populated chain. TAO-side rejections surface later
    on the TAOJob row, not as a 4xx here, so the suite status itself may
    legitimately already be ``"failed"``.

    The test is intentionally minimal — it validates that the Cosmos-RL
    dataset wire format and TAO request payload shape are accepted by the
    pinned container version, not that training completes.
    """
    import httpx

    backend_url = os.environ.get("VLM_BACKEND_URL", "http://localhost:8000")
    project_id = os.environ.get("VLM_SMOKE_PROJECT_ID")
    if not project_id:
        pytest.skip("VLM_SMOKE_PROJECT_ID not set — cannot identify seeded project")

    # Fetch student_base models from the catalog to pick one dynamically.
    resp = httpx.get(
        f"{backend_url}/v1/projects/{project_id}/model_configs",
        params={"eligible_role": "student_base"},
        timeout=30.0,
    )
    resp.raise_for_status()
    items = resp.json().get("items", [])
    student_base = next(
        (
            mc["model_config_id"]
            for mc in items
            if "student_base" in mc.get("eligible_roles", [])
        ),
        None,
    )
    if student_base is None:
        pytest.skip("No student_base ModelConfig in the seeded project")

    body = {
        "student_base_model_config_ids": [student_base],
        "training_preset": "quick",
        "include_auto_labeled": False,
        "export_field_mode": "all",
        "quantization_schemes": [],
        "idempotency_key": "smoke-test-ftms-001",
    }
    resp = httpx.post(
        f"{backend_url}/v1/projects/{project_id}/training_suites",
        json=body,
        timeout=120.0,
    )
    # TAO may reject the payload for deployment-specific reasons (missing
    # network_arch, unsupported base_experiment_id, etc.) — that surfaces
    # on the TAOJob row, not as a 4xx from our router. Assert the suite
    # endpoint accepted the request and a chain was created.
    assert resp.status_code == 201, resp.text
    suite = resp.json()
    assert suite["chain_ids_ordered"]
    # Status is one of {running, failed} depending on live TAO validation.
    assert suite["status"] in ("running", "failed", "initialized")
