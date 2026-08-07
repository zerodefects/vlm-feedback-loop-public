# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavior pins for the operator-run Profile B acceptance harness."""

from __future__ import annotations

import importlib.util
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType
from unittest.mock import AsyncMock

import httpx
import pytest

from vlm_feedback_loop.model_catalog_constants import STEP_3_7_FLASH

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _import_profile_b() -> ModuleType:
    path = _REPO_ROOT / "scripts" / "profile_b_live_validation.py"
    spec = importlib.util.spec_from_file_location("profile_b_script_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _response(status_code: int, payload: dict[str, object]) -> httpx.Response:
    return httpx.Response(
        status_code,
        json=payload,
        request=httpx.Request("GET", "http://test.invalid"),
    )


def test_curated_cohort_is_exactly_33_33_34(tmp_path: Path):
    """Profile B needs 100 distinct images while retaining near-even RPS
    coverage; 33-per-class accidentally produced only 99."""
    module = _import_profile_b()
    module.RPS_ROOT = tmp_path
    for class_name in ("rock", "paper", "scissors"):
        class_dir = tmp_path / class_name
        class_dir.mkdir()
        for index in range(124):
            (class_dir / f"{index:03d}.png").touch()

    cohort = module._curated_images()

    assert len(cohort) == 100
    assert len({path for _, path in cohort}) == 100
    assert Counter(class_name for class_name, _ in cohort) == {
        "rock": 33,
        "paper": 33,
        "scissors": 34,
    }


def test_truthful_review_preserves_correct_proposal_as_accept():
    """A correct Core prediction stays untouched so the save is an Accept."""
    module = _import_profile_b()
    proposal = {
        "category": "paper",
        "rationale_note": "Five fingers are visible.",
    }

    label, rationale_source = module._truthful_review_label(proposal, "paper")

    assert label == proposal
    assert label is not proposal
    assert rationale_source == "teacher_proposal"


def test_truthful_review_corrects_wrong_core_as_edit():
    """A wrong Core prediction is replaced with known RPS ground truth."""
    module = _import_profile_b()

    label, rationale_source = module._truthful_review_label(
        {"category": "rock", "rationale_note": "Closed hand."}, "scissors"
    )

    assert label == {
        "rationale_note": (
            "Hand is scissors. Verified against the curated RPS test pool."
        ),
        "number_fingers_extended": 2,
        "category": "scissors",
    }
    assert rationale_source == "sme_edited"


@pytest.mark.asyncio
@pytest.mark.parametrize(("ingest_status", "expected_ok"), [(200, False), (202, True)])
async def test_phase_a_requires_accepted_ingest_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ingest_status: int,
    expected_ok: bool,
):
    """A legacy synchronous 200 must fail the live gate; only the current
    202 skeleton-ingest contract may advance Profile B."""
    module = _import_profile_b()
    project_dir = tmp_path / "api-returned-project"
    project_dir.mkdir()
    (project_dir / "project.db").touch()
    cohort = [
        (class_name, tmp_path / class_name / f"{index:03d}.png")
        for class_name, count in module.RPS_CLASS_COUNTS.items()
        for index in range(count)
    ]
    monkeypatch.setattr(module, "_curated_images", lambda: cohort)

    client = AsyncMock()
    client.get.return_value = _response(
        200,
        {"items": [{"model_name": STEP_3_7_FLASH, "model_config_id": "mc-step"}]},
    )
    client.patch.side_effect = [_response(200, {}), _response(200, {})]
    client.post.side_effect = [
        _response(
            201,
            {"project_id": "project-1", "project_dir": str(project_dir)},
        ),
        _response(201, {"guidance_id": "guidance-1"}),
        _response(
            ingest_status,
            {"results": [{"status": "created"} for _ in range(100)]},
        ),
    ]
    report = module.ValidationReport()

    ok, returned_cohort = await module._phase_a_setup(client, report)

    assert ok is expected_ok
    assert report.project_dir == project_dir
    assert len(returned_cohort) == (100 if expected_ok else 0)
    ingest_request = client.post.await_args_list[-1]
    assert len(ingest_request.kwargs["json"]["examples"]) == 100
    project_configuration = client.patch.await_args_list[0].kwargs["json"]
    assert project_configuration["scaleup_min_test_pool_size"] == 20
    assert project_configuration["teacher_model_config_id"] == "mc-step"


@pytest.mark.asyncio
async def test_phase_a_rejects_nonlocal_project_db_before_teacher_calls(tmp_path: Path):
    """A Compose/remote path fails before the first paid proposal or setup call."""
    module = _import_profile_b()
    missing_project_dir = tmp_path / "container-only-project"
    client = AsyncMock()
    client.post.return_value = _response(
        201,
        {"project_id": "project-1", "project_dir": str(missing_project_dir)},
    )
    report = module.ValidationReport()

    ok, cohort = await module._phase_a_setup(client, report)

    assert ok is False
    assert cohort == []
    assert [stage.name for stage in report.stages] == [
        "create_project",
        "project_db_access",
    ]
    assert report.stages[-1].ok is False
    client.get.assert_not_awaited()
    client.patch.assert_not_awaited()
