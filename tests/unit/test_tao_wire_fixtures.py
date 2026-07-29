# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Committed TAO wire-format fixture invariants.

The TAO integration is coded against the committed
``tests/fixtures/tao/*.json`` captures of live TAO FTMS responses.
These tests pin the shapes that integration consumes, so a fixture
refresh (``scripts/capture_tao_fixtures.py``) that drops a consumed
endpoint or field fails CI instead of silently invalidating the
contract.

The live-gated checks (idempotent re-capture, live-vs-committed drift)
stay in ``tests/integration/test_tao_fixture_idempotency.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "tao"


def test_committed_fixtures_exist():
    """Every ambient fixture the TAO code is built against is committed and non-empty."""
    expected = [
        "openapi_v2.json",
        "ftms_version.json",
        "jobs_empty_response.json",
        "workspaces_empty_response.json",
        "datasets_empty_response.json",
    ]
    for name in expected:
        p = _FIXTURE_DIR / name
        assert p.is_file(), f"missing fixture: {p}"
        assert p.stat().st_size > 0


def test_openapi_fixture_pins_consumed_tao_paths():
    """The OpenAPI fixture MUST expose the TAO endpoints the TAO integration consumes.

    This is the wire-format invariant for the Blueprint's TAO integration
    — if these paths disappear from a future OpenAPI capture, the Blueprint
    code must be updated.
    """
    spec = json.loads((_FIXTURE_DIR / "openapi_v2.json").read_text("utf-8"))
    paths = spec.get("paths", {})
    required = [
        "/api/v2/login",
        "/api/v2/orgs/{org_name}/jobs",
        "/api/v2/orgs/{org_name}/jobs/{job_id}",
        "/api/v2/orgs/{org_name}/jobs/{job_id}:download_selective_files",
        "/api/v2/orgs/{org_name}/jobs/{job_id}:list_files",
        "/api/v2/orgs/{org_name}/jobs/{job_id}:logs",
    ]
    missing = [p for p in required if p not in paths]
    assert not missing, f"OpenAPI fixture missing required paths: {missing}"


def test_job_status_fixtures_map_to_canonical_statuses():
    """The live-captured job-status bodies keep the raw statuses and
    ``job_details.{id}`` envelope the poll parser consumes. Captured from
    FTMS 6.26.3 on 2026-07-14 (first live poll bodies)."""
    from vlm_feedback_loop.services.tao_job_service import map_tao_raw_status

    expected = {
        "job_status_submitted.json": ("Started", "running"),
        "job_status_running_early.json": ("Running", "running"),
        "job_status_running_training_complete.json": ("Running", "running"),
        "job_status_running_action_success.json": ("Running", "running"),
        "job_status_done_train.json": ("Done", "succeeded"),
        "job_status_canceled.json": ("Canceled", "canceled"),
    }
    for name, (raw, canonical) in expected.items():
        body = json.loads((_FIXTURE_DIR / name).read_text("utf-8"))
        assert body["status"] == raw, name
        assert map_tao_raw_status(body["status"], current="running") == canonical, name
        job_details = body["job_details"]
        (entry,) = job_details.values()
        assert "detailed_status" in entry, name


def test_running_job_with_success_detail_is_not_terminal():
    """During the post-training artifact-upload window, FTMS reports
    ``detailed_status.status: SUCCESS`` (and a "completed successfully"
    message) while the job's top-level ``status`` is still ``Running``.
    A consumer that keys terminality on ``detailed_status`` would advance
    the chain before checkpoints finish uploading, breaking outputs
    fetch — terminality must come from the top-level status alone.

    Bodies captured live from FTMS 6.26.3 during an 8B LoRA train
    (2026-07-15): polled at 15s cadence for the full 2.6 h run, the
    ``job_details`` progress fields (``epoch``/``cur_iter``/``eta``/
    ``time_per_*``) stayed null throughout — cosmos-rl trains emit no
    iteration progress on this stack, so these upload-window bodies are
    the only observable pre-Done transitions.
    """
    from vlm_feedback_loop.services.tao_job_service import map_tao_raw_status

    for name in (
        "job_status_running_training_complete.json",
        "job_status_running_action_success.json",
    ):
        body = json.loads((_FIXTURE_DIR / name).read_text("utf-8"))
        assert body["status"] == "Running", name
        assert map_tao_raw_status(body["status"], current="running") == "running", name
        (entry,) = body["job_details"].values()
        assert "completed successfully" in entry["detailed_status"]["message"], name
        for field in ("epoch", "cur_iter", "eta", "time_per_epoch", "time_per_iter"):
            assert entry[field] is None, f"{name}: {field}"


def test_jobs_empty_fixture_has_jobs_array():
    """The jobs-list fixture keeps the ``jobs`` array envelope the poller reads."""
    data = json.loads((_FIXTURE_DIR / "jobs_empty_response.json").read_text("utf-8"))
    assert "jobs" in data
    assert isinstance(data["jobs"], list)


def test_version_fixture_reports_semver():
    """The FTMS version fixture carries a semver-shaped ``version`` string."""
    data = json.loads((_FIXTURE_DIR / "ftms_version.json").read_text("utf-8"))
    assert "version" in data
    parts = data["version"].split(".")
    assert len(parts) >= 2
    assert all(p.isdigit() for p in parts[:2])
