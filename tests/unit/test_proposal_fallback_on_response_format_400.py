# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end test — structured-generation fallback path with the REAL ``invoke_teacher`` signature.

Pins a real regression class: hosted Qwen rejects
``response_format=json_schema`` mid-conversation, and the spec-compliant
fallback retry in ``proposal_service.py`` can ship with a broken
``invoke_teacher`` call site (e.g. a missing or renamed
required kwarg) — producing HTTP 500 to the SME on every fallback. The unit test
``tests/unit/test_proposal_service.py::TestStructuredGenFallback::test_runtime_rejection_retries_prompt_only``
mocks ``invoke_teacher`` directly, which bypasses kwarg validation and
lets that class of bug ship.

This test exercises the REAL ``invoke_teacher`` end-to-end by stubbing
only the deepest network seam (``nim_client.chat_completions``):

  1. First call → return endpoint_error referencing ``response_format``,
     simulating hosted Qwen mid-run rejection.
  2. Second call → return a valid JSON proposal.

The proposal endpoint MUST return HTTP 200 with
``invocation_status="success"`` and the persisted OperationRecord MUST
have ``structured_generation_fallback_used=True``. If a future commit
breaks the fallback signature again (another missing kwarg, a renamed
param), this test fires before the bug ships.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from sqlalchemy.orm import Session
from starlette.testclient import TestClient

from support import (
    build_test_settings,
    fake_nim_success,
    fake_prepare_result,
    seed_hosted_teacher_project,
)
from vlm_feedback_loop.db.models.operation import OperationRecord
from vlm_feedback_loop.main import app
from vlm_feedback_loop.routers.projects import get_current_settings
from vlm_feedback_loop.services import project_service

# In-process e2e over the real proposal/save/eval pipeline: fast standalone,
# but the bare 30s pytest-timeout ceiling is too tight under full-suite
# xdist with coverage instrumentation — and --timeout-method=thread
# hard-exits the whole worker on overrun, hanging the xdist controller.
# Restore the generous ceiling this file had from the integration conftest.
pytestmark = pytest.mark.timeout(120)

PID = "fallback-e2e"
GID = "g-fallback"
MCID = "mc-qwen-fallback"
EID = "ep-hosted-fallback"


def _seed_project(workspace: Path) -> tuple[Any, str]:
    engine = seed_hosted_teacher_project(
        workspace,
        project_id=PID,
        project_name="Fallback path e2e",
        guidance_id=GID,
        endpoint_id=EID,
        model_config_id=MCID,
        # The seeded ModelConfig probe says structured generation is
        # supported — mid-run hosted-NIM 4xx rejection is the case
        # under test.
        guidance_description="Classify alpha/beta/gamma.",
        max_images_per_request=8,
        example_keys=["ex_query"],
    )
    return engine, PID


def _fake_nim_failure_response_format() -> Any:
    """First call: hosted Qwen mid-run response_format rejection."""
    from vlm_feedback_loop.services.nim_client import NimChatCompletionsResult

    return NimChatCompletionsResult(
        success=False,
        content=None,
        finish_reason=None,
        usage=None,
        status_code=400,
        error="HTTP 400: response_format not supported by this model.",
    )


def _fallback_success() -> Any:
    """Second call: clean fallback response (no response_format)."""
    return fake_nim_success(
        json.dumps({"rationale_note": "fallback rationale", "category": "alpha"})
    )


@pytest.mark.asyncio
async def test_response_format_400_falls_back_through_real_invoke_teacher(
    tmp_path: Path,
) -> None:
    """A response_format-rejection 400 retries via real invoke_teacher signature.

    Catches the regression class "fallback retry call site has wrong
    kwargs" — a production bug that shipped once already. Mocks only
    ``nim_client.chat_completions`` so the rest of the pipeline
    (token-budget, ICL pruning, image prep, prompt rendering, fallback
    dispatch) runs end-to-end.
    """
    project_service.close_project_resources()
    workspace = tmp_path / "workspace"
    engine, pid = _seed_project(workspace)

    settings = build_test_settings(workspace, NVIDIA_API_KEY="nvapi-test-key")
    app.dependency_overrides[get_current_settings] = lambda: settings

    call_count = 0

    async def fake_chat_completions(*args: Any, **kwargs: Any) -> Any:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # Hosted-NIM rejection — body / error contains
            # "response_format" so ``_is_structured_gen_rejection``
            # classifies it as the fallback-trigger case.
            return _fake_nim_failure_response_format()
        # Second call (the fallback retry without response_format) —
        # MUST be reachable. A fallback call site with wrong kwargs
        # (missing or renamed — ``invoke_teacher`` is keyword-only)
        # raises TypeError before this second dispatch, so this
        # call_count never reaches 2.
        return _fallback_success()

    async def fake_prepare(refs: list[str], *args: Any, **kwargs: Any) -> Any:
        return fake_prepare_result(len(refs))

    try:
        with (
            patch(
                "vlm_feedback_loop.services.prompt_service.nim_client.chat_completions",
                new=fake_chat_completions,
            ),
            patch(
                "vlm_feedback_loop.services.prompt_service.prepare_images",
                new=fake_prepare,
            ),
        ):
            client = TestClient(app, raise_server_exceptions=False)
            r = client.post(
                f"/v1/projects/{pid}/proposals",
                json={"example_key": "ex_query"},
            )

            # ── Assertion 1: HTTP 200, no 500 ────────────────────────
            # A broken fallback path raises TypeError → FastAPI returns
            # 500 to the SME; a working fallback returns 200.
            assert r.status_code == 200, (
                f"Expected 200 (fallback succeeded), got {r.status_code}: "
                f"{r.text[:300]}"
            )

            # ── Assertion 2: invocation_status=success ───────────────
            body = r.json()
            assert body["invocation_status"] == "success", body

            # ── Assertion 3: nim_client.chat_completions was called twice ──
            # If call_count is 1, the fallback never ran (the bug
            # would have caused the exception before this point).
            # If call_count is >2, something is retrying more than
            # once which would also be a regression.
            assert call_count == 2, (
                f"Fallback path MUST call chat_completions exactly twice "
                f"(initial + 1 retry); got {call_count}"
            )

            # ── Assertion 4: structured_generation_fallback_used=True ──
            with Session(engine) as s:
                rec = (
                    s.query(OperationRecord)
                    .filter_by(
                        inference_invocation_id=body["inference_invocation_id"],
                    )
                    .first()
                )
                assert rec is not None
                assert rec.structured_generation_fallback_used is True, (
                    "OperationRecord MUST flag structured_generation_fallback_used "
                    "after a successful fallback retry."
                )
    finally:
        app.dependency_overrides.pop(get_current_settings, None)
        project_service.close_project_resources()
