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
    make_stub_settings,
    open_project_workspace,
)
from vlm_feedback_loop.config import Settings
from vlm_feedback_loop.db.models.model_config import ModelConfig
from vlm_feedback_loop.db.models.nim_endpoint import NimEndpoint
from vlm_feedback_loop.db.models.run import RunRecord
from vlm_feedback_loop.services.run_config import (
    create_runtime_config_snapshot,
    snapshot_run_config,
)


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
        generation_preset_key="precise",
        visual_budget_preset_key="balanced",
        structured_generation_mode_effective="auto",
    )


def _settings(tmp_path: Path, **overrides: Any) -> Settings:
    return make_stub_settings(
        WORKSPACE_ROOT=str(tmp_path / "workspace"),
        **overrides,
    )


def _create_snapshot(session: Session, settings: Settings) -> dict[str, Any]:
    return create_runtime_config_snapshot(
        session,
        PID,
        MCID,
        settings=settings,
        generation_preset_key="precise",
        visual_budget_preset_key="balanced",
        icl_max_examples=settings.ICL_MAX_EXAMPLES,
        icl_candidate_limit=None,
        icl_sim_gap=settings.ICL_SIM_GAP,
        icl_abs_threshold=settings.ICL_ABS_THRESHOLD,
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
        settings = _settings(tmp_path)

        with Session(engine) as session:
            cfg = snapshot_run_config(
                session,
                PID,
                run,
                example_keys=["ex-1"],
                settings=settings,
            )

        assert cfg["mc_input"].max_images_per_request == 3


class TestImmutableRuntimeSnapshot:
    def test_snapshot_survives_later_model_and_endpoint_removal(self, tmp_path):
        """Delayed execution uses only the model/endpoint values frozen at start."""
        engine = _seed_project(
            tmp_path,
            model_overrides={
                "context_window_tokens": 8192,
                "thinking_toggle_mode": "qwen_enable_thinking",
                "thinking_toggle_support": "supported",
                "visual_budget_mode": "mm_processor_pixels",
                "visual_budget_support": "supported",
                "structured_generation_support": "supported",
                "max_images_per_request": 8,
                "default_icl_max_examples": 4,
            },
            endpoint_overrides={
                "base_url": "https://frozen.example/v1",
                "endpoint_mode": "self_hosted",
                "auth_mode": "none",
                "max_images_per_request": 3,
            },
        )
        run = _make_run()
        settings = _settings(tmp_path)
        with Session(engine) as session:
            run.runtime_config_snapshot = _create_snapshot(session, settings)
            model_config = session.get(ModelConfig, MCID)
            endpoint = session.get(NimEndpoint, EID)
            assert model_config is not None
            assert endpoint is not None
            session.delete(model_config)
            session.delete(endpoint)
            session.commit()

        with Session(engine) as session:
            cfg = snapshot_run_config(
                session,
                PID,
                run,
                example_keys=["ex-1"],
                settings=_settings(tmp_path, ICL_SIM_GAP=0.9),
            )

        assert cfg["model_config_id"] == MCID
        assert cfg["model_name"] == "test-model"
        assert cfg["endpoint_id"] == EID
        assert cfg["endpoint_base_url"] == "https://frozen.example/v1"
        assert cfg["endpoint_mode"] == "self_hosted"
        assert cfg["endpoint_auth_mode"] == "none"
        assert cfg["icl_sim_gap"] == 0.05
        assert cfg["mc_input"] == cfg["mc_input"].__class__(
            context_window_tokens=8192,
            thinking_toggle_mode="qwen_enable_thinking",
            thinking_toggle_support="supported",
            visual_budget_mode="mm_processor_pixels",
            visual_budget_support="supported",
            structured_generation_support="supported",
            max_images_per_request=3,
            default_icl_max_examples=4,
        )

    def test_snapshot_lineage_must_match_the_run(self, tmp_path):
        engine = _seed_project(tmp_path)
        run = _make_run()
        settings = _settings(tmp_path)
        with Session(engine) as session:
            snapshot = _create_snapshot(session, settings)
            snapshot["model_config_id"] = "other-model"
            run.runtime_config_snapshot = snapshot
            with pytest.raises(RuntimeError, match="lineage"):
                snapshot_run_config(
                    session,
                    PID,
                    run,
                    example_keys=["ex-1"],
                    settings=settings,
                )

    def test_snapshot_version_must_be_supported(self, tmp_path):
        """Unknown persisted shapes fail closed instead of reading live rows."""
        engine = _seed_project(tmp_path)
        run = _make_run()
        settings = _settings(tmp_path)
        with Session(engine) as session:
            snapshot = _create_snapshot(session, settings)
            snapshot["version"] = 99
            run.runtime_config_snapshot = snapshot
            with pytest.raises(RuntimeError, match="invalid runtime configuration"):
                snapshot_run_config(
                    session,
                    PID,
                    run,
                    example_keys=["ex-1"],
                    settings=settings,
                )

    @pytest.mark.parametrize(
        ("path", "invalid_value"),
        [
            (("endpoint_mode",), "hostedd"),
            (("endpoint_auth_mode",), "basic"),
            (("thinking_toggle_mode",), "magic"),
            (("thinking_toggle_support",), "maybe"),
            (("visual_budget_mode",), "pixels"),
            (("visual_budget_support",), "maybe"),
            (("structured_generation_support",), "maybe"),
            (("inference_settings", "base_output_tokens_floor"), 0),
            (("inference_settings", "max_output_fraction"), 1.1),
            (("inference_settings", "token_safety_margin"), 0.0),
            (("inference_settings", "sampling_params", "top_p"), 0.0),
        ],
    )
    def test_malformed_snapshot_values_fail_closed(self, tmp_path, path, invalid_value):
        """Recovery rejects corrupted enum and numeric values before dispatch."""
        engine = _seed_project(tmp_path)
        run = _make_run()
        settings = _settings(tmp_path)
        with Session(engine) as session:
            snapshot = _create_snapshot(session, settings)
            target: dict[str, Any] = snapshot
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = invalid_value
            run.runtime_config_snapshot = snapshot

            with pytest.raises(RuntimeError, match="invalid runtime configuration"):
                snapshot_run_config(
                    session,
                    PID,
                    run,
                    example_keys=["ex-1"],
                    settings=settings,
                )

    def test_legacy_snapshot_is_materialized_before_resume(self, tmp_path):
        """A residual legacy NULL snapshot freezes live rows on first execution."""
        engine = _seed_project(
            tmp_path,
            model_overrides={"context_window_tokens": 8192},
            endpoint_overrides={"base_url": "https://first.example/v1"},
        )
        settings = _settings(tmp_path, ICL_SIM_GAP=0.07)
        with Session(engine) as session:
            session.add(_make_run())
            session.commit()

        with Session(engine) as session:
            run = session.get(RunRecord, "run-001")
            assert run is not None
            first = snapshot_run_config(
                session,
                PID,
                run,
                example_keys=["ex-1"],
                settings=settings,
            )
            assert run.runtime_config_snapshot is not None
            session.commit()

        with Session(engine) as session:
            model_config = session.get(ModelConfig, MCID)
            endpoint = session.get(NimEndpoint, EID)
            assert model_config is not None
            assert endpoint is not None
            model_config.context_window_tokens = 16384
            endpoint.base_url = "https://later.example/v1"
            session.commit()

        with Session(engine) as session:
            run = session.get(RunRecord, "run-001")
            assert run is not None
            resumed = snapshot_run_config(
                session,
                PID,
                run,
                example_keys=["ex-1"],
                settings=_settings(tmp_path, ICL_SIM_GAP=0.9),
            )

        assert first["mc_input"].context_window_tokens == 8192
        assert first["endpoint_base_url"] == "https://first.example/v1"
        assert resumed["mc_input"].context_window_tokens == 8192
        assert resumed["endpoint_base_url"] == "https://first.example/v1"
        assert first["icl_sim_gap"] == 0.07
        assert resumed["icl_sim_gap"] == 0.07

    def test_v1_snapshot_upgrades_semantic_settings_once(self, tmp_path):
        """A model-only v1 snapshot gains one immutable Settings boundary."""
        engine = _seed_project(tmp_path)
        initial_settings = _settings(
            tmp_path,
            LABELING_PRESETS={"precise": {"temperature": 0.2, "top_p": 0.8}},
            ICL_SIM_GAP=0.17,
            IMAGE_TRANSPORT_MAX_LONGEST_EDGE=1200,
        )
        with Session(engine) as session:
            snapshot = _create_snapshot(session, initial_settings)
            snapshot.pop("inference_settings")
            snapshot["version"] = 1
            run = _make_run()
            run.runtime_config_snapshot = snapshot
            session.add(run)
            session.commit()

        with Session(engine) as session:
            run = session.get(RunRecord, "run-001")
            assert run is not None
            upgraded = snapshot_run_config(
                session,
                PID,
                run,
                example_keys=["ex-1"],
                settings=initial_settings,
            )
            assert run.runtime_config_snapshot["version"] == 2
            session.commit()

        changed_settings = _settings(
            tmp_path,
            LABELING_PRESETS={"precise": {"temperature": 0.9, "top_p": 0.1}},
            ICL_SIM_GAP=0.9,
            IMAGE_TRANSPORT_MAX_LONGEST_EDGE=4096,
        )
        with Session(engine) as session:
            run = session.get(RunRecord, "run-001")
            assert run is not None
            resumed = snapshot_run_config(
                session,
                PID,
                run,
                example_keys=["ex-1"],
                settings=changed_settings,
            )

        assert upgraded["invoke_settings"]["labeling_presets"] == {
            "precise": {"temperature": 0.2, "top_p": 0.8}
        }
        assert resumed["invoke_settings"] == upgraded["invoke_settings"]
        assert resumed["icl_sim_gap"] == 0.17
        assert resumed["image_transport_max_longest_edge"] == 1200


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
                snapshot_run_config(
                    session,
                    PID,
                    run,
                    example_keys=["ex-1"],
                    settings=_settings(tmp_path),
                )
