# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end ICL loop integration test.

Drives the full interactive ICL loop in-process via TestClient:

    POST /v1/projects/{id}/proposals  → ProposalResponse (1st)
    POST /v1/projects/{id}/labels     → LabelSaveResponse (Edit)
    POST /v1/projects/{id}/proposals  → ProposalResponse (2nd)
    POST /v1/projects/{id}/labels     → LabelSaveResponse (Edit)
    POST /v1/projects/{id}/proposals  → ProposalResponse (3rd)
    POST /v1/projects/{id}/labels     → LabelSaveResponse (Edit)
    POST /v1/projects/{id}/proposals  → ProposalResponse (4th, asserted on)

Asserts that the 4th proposal:

    1. Returns ``icl_example_keys_used`` containing the 3 prior Edits
       in newest-first order (deterministic by ``labeled_at DESC``).
    2. Returns ``icl_images_attached_count == 3`` — the invariant
       ``icl_images_attached_count == len(icl_example_keys_used)``
       holds at the API contract boundary.
    3. Persists a 64-char hex ``prompt_hash`` on the OperationRecord
       and ``seed_effective is None`` (interactive
       purpose does not set a deterministic seed).
    4. Dispatches a wire payload with exactly 4 image content parts
       (3 ICL + 1 query), interleaved per the Teacher prompt contract.

Catches the regression class "edits land in DB but ICL query stops
seeing them" — a hidden failure mode because every proposal still
succeeds while the ICL context silently degrades.

NIM transport is faked at the module level: ``nim_client.chat_completions``
returns a deterministic JSON proposal and ``prepare_images`` returns an
N-image batch matching the ICL+query refs. Pool routing is forced to
non-pool for every Verified by setting ``test_pool_fraction = 0.0`` so
all 3 Edits remain ICL-eligible.
``utc_now`` is patched in ``label_service`` so each save lands at a
distinct second (default precision is 1s, otherwise the 3 saves can
collide and ``Label.labeled_at DESC`` ordering becomes ambiguous).
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

PID = "icl-loop-e2e"
GID = "g-icl-loop"
MCID = "mc-qwen"
EID = "ep-hosted"

GUIDANCE_DESCRIPTION = (
    "Classify the visible shape as one of: alpha, beta, or gamma. "
    "Provide a short rationale describing what you saw."
)
GUIDANCE_RULES = "If the shape is ambiguous, prefer the most-prominent edge count."


def _seed_project(workspace: Path, n_examples: int = 5) -> tuple[Any, str]:
    engine = seed_hosted_teacher_project(
        workspace,
        project_id=PID,
        project_name="ICL Loop e2e",
        guidance_id=GID,
        endpoint_id=EID,
        model_config_id=MCID,
        guidance_description=GUIDANCE_DESCRIPTION,
        guidance_rules=GUIDANCE_RULES,
        # Qwen carries up to 10 images. Cap = 9 ICL after we subtract 1
        # for the query, so 3 retained Edits never trigger image-budget
        # pruning in this test — the invariant holds because the cap was
        # never the limit. (test_pool_fraction is forced to 0.0 by the
        # seed helper so all 3 Edits stay ICL-eligible.)
        max_images_per_request=10,
        example_keys=[f"ex_{i:03d}" for i in range(n_examples)],
    )
    return engine, PID


@pytest.mark.asyncio
async def test_three_edits_feed_ico_loop_into_fourth_proposal(
    tmp_path: Path,
) -> None:
    """3 Edits → 4th proposal carries all 3 in ICL with images attached."""

    project_service.close_project_resources()
    workspace = tmp_path / "workspace"

    engine, pid = _seed_project(workspace, n_examples=5)

    settings = build_test_settings(workspace, NVIDIA_API_KEY="nvapi-test-key")
    app.dependency_overrides[get_current_settings] = lambda: settings

    captured_messages: list[Any] = []

    async def fake_chat_completions(*args: Any, **kwargs: Any) -> Any:
        # Signature: (base_url, auth_headers, model_name, messages,
        # deadline_s, ...) — capture the messages so the wire shape
        # can be asserted later.
        captured_messages.append(args[3])
        return fake_nim_success(
            json.dumps({"rationale_note": "model proposal", "category": "alpha"})
        )

    async def fake_prepare(refs: list[str], *args: Any, **kwargs: Any) -> Any:
        # Return as many PreparedImage entries as refs requested.
        return fake_prepare_result(len(refs))

    # 1-second timestamp granularity (utc_now uses %S resolution): give
    # each save a distinct timestamp so labeled_at DESC ordering is
    # stable. ex_000 → 12:00, ex_001 → 12:05, ex_002 → 12:10.
    ts_iter = iter(
        [
            "2026-04-27T12:00:00Z",
            "2026-04-27T12:00:05Z",
            "2026-04-27T12:00:10Z",
        ]
    )

    def fake_utc_now() -> str:
        return next(ts_iter)

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
            patch(
                "vlm_feedback_loop.services.label_service.utc_now",
                new=fake_utc_now,
            ),
        ):
            client = TestClient(app, raise_server_exceptions=False)

            # ── Drive 3 proposal+save cycles, each with an Edit ─────
            saved_keys: list[str] = []
            for i in range(3):
                ek = f"ex_{i:03d}"

                pr = client.post(
                    f"/v1/projects/{pid}/proposals",
                    json={"example_key": ek},
                )
                assert pr.status_code == 200, pr.text
                p = pr.json()
                assert p["invocation_status"] == "success"

                # Edit: change category from the proposal's "alpha" to "beta"
                # — verified_outcome MUST be classified Edit because the
                # Core field differs from the proposal.
                edited_label = {
                    "rationale_note": f"SME rationale for {ek}",
                    "category": "beta",
                }
                lr = client.post(
                    f"/v1/projects/{pid}/labels",
                    json={
                        "example_key": ek,
                        "inference_invocation_id": p["inference_invocation_id"],
                        "label_json": edited_label,
                        "rationale_source": "sme_edited",
                    },
                )
                assert lr.status_code == 200, lr.text
                ls = lr.json()
                assert ls["verified_outcome"] == "Edit", (
                    f"Edit on Core field MUST classify as Edit; got "
                    f"{ls['verified_outcome']} for {ek}"
                )
                # Pool routing forced to None by test_pool_fraction=0.0 —
                # all 3 saves remain ICL-eligible.
                assert ls["pool_assignment"] is None
                saved_keys.append(ek)

            # ── 4th proposal: assert the loop ───────────────────────
            ek4 = "ex_003"
            pr4 = client.post(
                f"/v1/projects/{pid}/proposals",
                json={"example_key": ek4},
            )
            assert pr4.status_code == 200, pr4.text
            p4 = pr4.json()
            assert p4["invocation_status"] == "success"

            # ── Assertion 1: all 3 Edits render bookended ───────────
            # Relevance fallback ranks newest-first:
            # ex_002 → ex_001 → ex_000. After pruning, bookend
            # presentation keeps rank-1 first and moves rank-2 last.
            assert p4["icl_example_keys_used"] == [
                "ex_002",
                "ex_000",
                "ex_001",
            ], (
                f"ICL MUST contain all prior Edits in bookend order; got "
                f"{p4['icl_example_keys_used']}"
            )

            # ── Assertion 2: image-budget invariant ─────────────────
            assert p4["icl_images_attached_count"] == 3
            assert p4["icl_images_attached_count"] == len(
                p4["icl_example_keys_used"]
            ), (
                "Every retained ICL example has its image "
                "attached. icl_images_attached_count MUST equal "
                "len(icl_example_keys_used)."
            )

            # ── Assertion 3: persisted prompt_hash + null seed ──────
            with Session(engine) as s:
                rec = (
                    s.query(OperationRecord)
                    .filter_by(
                        inference_invocation_id=p4["inference_invocation_id"],
                    )
                    .first()
                )
            assert rec is not None
            assert isinstance(rec.prompt_hash, str)
            assert len(rec.prompt_hash) == 64, (
                "prompt_hash MUST be a SHA-256 hex string (64 chars). "
                f"Got len={len(rec.prompt_hash) if rec.prompt_hash else 0}."
            )
            assert rec.seed_effective is None, (
                "interactive_proposal MUST NOT set a deterministic seed."
            )
            assert rec.icl_images_attached_count == 3
            assert rec.icl_example_keys_used == [
                "ex_002",
                "ex_000",
                "ex_001",
            ]

            # ── Assertion 4: wire shape — 4 image_url content parts ─
            # The 4th captured dispatch (4 proposals = 4 captures). 0
            # ICL on the first three (no Edits exist yet at proposal
            # time), 3 ICL on the fourth.
            assert len(captured_messages) == 4
            user_content = captured_messages[3][-1]["content"]
            assert isinstance(user_content, list)
            image_parts = [p for p in user_content if p.get("type") == "image_url"]
            assert len(image_parts) == 4, (
                f"4th dispatch MUST carry 4 image content parts (3 ICL + "
                f"1 query); got {len(image_parts)}."
            )
            # First 3 ICL refs come first, query last (per
            # invoke_teacher's `refs_to_prep = icl_refs + [query_ref]`).
            urls = [p["image_url"]["url"] for p in image_parts]
            assert urls == [
                "data:image/jpeg;base64,IMG0",
                "data:image/jpeg;base64,IMG1",
                "data:image/jpeg;base64,IMG2",
                "data:image/jpeg;base64,IMG3",
            ]
    finally:
        app.dependency_overrides.clear()
        project_service.close_project_resources()
