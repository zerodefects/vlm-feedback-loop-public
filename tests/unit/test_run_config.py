# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``services/run_config.py`` — the Phase-A snapshot shared by the
evaluation and batch-label executors.

The full-pipeline suites (test_evaluation_service, test_batch_label_service)
exercise the snapshot's happy path end-to-end and pin the structured-
generation mode overrides at the run-outcome level; these tests pin the
snapshot-level contracts those suites do not reach: the per-endpoint
image-cap wiring and the fail-loud guard on schemaless guidance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from conftest import (
    EID,
    GID,
    MCID,
    PID,
    add_endpoint_row,
    add_example_row,
    add_fixture_guidance_row,
    add_guidance_row,
    add_model_config_row,
    add_standard_project_row,
    open_project_workspace,
)
from vlm_feedback_loop.db.models.run import RunRecord
from vlm_feedback_loop.services.run_config import snapshot_run_config


def _seed_project(
    tmp_path: Path,
    *,
    model_overrides: dict[str, Any] | None = None,
    endpoint_overrides: dict[str, Any] | None = None,
    guidance_schema: dict[str, Any] | None = None,
) -> Engine:
    """Project + guidance + endpoint + model config + one example row."""
    engine, project_dir, _ = open_project_workspace(tmp_path, PID)
    with Session(engine) as session:
        add_standard_project_row(session, PID, str(project_dir))
        if guidance_schema is None:
            add_fixture_guidance_row(session)
        else:
            add_guidance_row(session, PID, GID, guidance_schema)
        add_endpoint_row(session, PID, EID, **(endpoint_overrides or {}))
        add_model_config_row(session, PID, MCID, EID, **(model_overrides or {}))
        add_example_row(session, PID, "ex-1")
        session.commit()
    return engine


def _make_run() -> RunRecord:
    """Transient RunRecord carrying the snapshotted config pointers.

    ``snapshot_run_config`` only reads attributes off the run (the
    executors re-read the row themselves), so persistence adds nothing.
    """
    return RunRecord(
        run_id="run-001",
        project_id=PID,
        run_type="evaluation_run",
        status="running",
        guidance_id=GID,
        model_config_id=MCID,
        structured_generation_mode_effective="auto",
    )


class TestImageCapWiring:
    def test_endpoint_image_cap_override_reaches_mc_input(self, tmp_path):
        """The snapshot must load the run's NimEndpoint row and hand it to the
        image-cap resolver, so a per-endpoint ``max_images_per_request``
        override beats the per-model value. If this wiring drops (endpoint
        not passed), probed endpoint caps silently stop applying to eval and
        batch runs and oversized ICL payloads draw HTTP 400s."""
        engine = _seed_project(
            tmp_path,
            model_overrides={"max_images_per_request": 8},
            endpoint_overrides={"max_images_per_request": 3},
        )
        run = _make_run()

        with Session(engine) as session:
            cfg = snapshot_run_config(session, PID, run, example_keys=["ex-1"])

        assert cfg["mc_input"].max_images_per_request == 3


class TestFailLoudGuards:
    def test_schemaless_guidance_aborts_the_snapshot(self, tmp_path):
        """A guidance whose schema envelope is empty must abort the snapshot
        (the executors catch and fail the run) instead of proceeding with an
        empty field list — that would render garbage prompts and score the
        whole run silently wrong."""
        engine = _seed_project(tmp_path, guidance_schema={})
        run = _make_run()

        with Session(engine) as session:
            with pytest.raises(RuntimeError, match=GID):
                snapshot_run_config(session, PID, run, example_keys=["ex-1"])
