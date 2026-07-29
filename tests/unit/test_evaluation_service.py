# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for evaluation_service.

Covers: state machine, config snapshot, pool snapshot, concurrent execution,
sequential retry, metrics aggregation, Returning vs New, SSE events,
cancellation, supersession, trigger status, trigger dismissal, list/get.

All inference calls are mocked via monkeypatch on ``_invoke_for_evaluation``.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import patch

import pytest
from sqlalchemy.orm import Session

from conftest import (
    EID,
    GID,
    MCID,
    PID,
    add_endpoint_and_model_rows,
    add_example_row,
    add_fixture_guidance_row,
    add_model_config_row,
    add_standard_project_row,
    make_stub_settings,
    patched_register,
    setup_project_db,
)
from support import fake_nim_failure, fake_nim_success, fake_prepare_result
from vlm_feedback_loop.db.base import generate_uuid4, utc_now
from vlm_feedback_loop.db.models.label import Label
from vlm_feedback_loop.db.models.pool import Pool
from vlm_feedback_loop.db.models.project import Project
from vlm_feedback_loop.db.models.run import RunRecord
from vlm_feedback_loop.services import evaluation_service as evaluation_service_module
from vlm_feedback_loop.services.evaluation_service import (
    EvalExampleResult,
    _compute_coverage_gaps,
    _limit_icl_candidate_prefix,
    _resolve_eval_concurrency,
    cancel_evaluation_run,
    compute_trigger_status,
    dismiss_trigger,
    get_evaluation_run,
    list_evaluation_runs,
    serialize_metrics_overall,
    start_evaluation_run,
)
from vlm_feedback_loop.services.exact_match_evaluator import (
    AggregateMetrics,
    FieldMatchResult,
    PerValueMetrics,
)

# ── Helpers ──────────────────────────────────────────────────────────────────


def test_candidate_prefix_uses_stable_timestamp_and_key_order() -> None:
    from vlm_feedback_loop.services.icl_service import ICLExample

    candidates = [
        ICLExample("b", {}, "2026-06-02T00:00:00Z"),
        ICLExample("d", {}, "2026-06-02T00:00:00Z"),
        ICLExample("a", {}, "2026-06-01T00:00:00Z"),
        ICLExample("c", {}, "2026-06-01T00:00:00Z"),
    ]

    limited = _limit_icl_candidate_prefix(candidates, 3)

    assert [candidate.example_key for candidate in limited] == ["b", "a", "c"]
    assert _limit_icl_candidate_prefix(candidates, None) is candidates


def _add_example(session, project_id, key, state="Verified", phash=None):
    add_example_row(session, project_id, key, state=state, phash=phash or "a" * 16)


def _add_label(
    session,
    project_id,
    key,
    guidance_id=GID,
    outcome="Accept",
    pool_assignment=None,
    label_json=None,
):
    session.add(
        Label(
            label_id=generate_uuid4(),
            project_id=project_id,
            example_key=key,
            label_status="verified",
            guidance_id=guidance_id,
            inference_invocation_id=generate_uuid4(),
            label_json=label_json
            or {"rationale_note": "test", "severity": "high", "damaged": True},
            labeled_at=utc_now(),
            verified_outcome=outcome,
            verified_at=utc_now(),
            edited_core_fields=[],
            edited_aux_fields=[],
            rationale_source="teacher_proposal"
            if outcome == "Accept"
            else "sme_edited",
            pool_assignment=pool_assignment,
        )
    )


def _setup_project_with_pool(tmp_path, n_pool=5, n_nonpool=3):
    """Create project with Guidance, ModelConfig, NimEndpoint, pool examples."""
    engine, pdir = setup_project_db(tmp_path)
    settings = make_stub_settings(WORKSPACE_ROOT=str(tmp_path / "workspace"))

    with Session(engine) as s:
        add_standard_project_row(s, PID, pdir)
        add_fixture_guidance_row(s)
        add_endpoint_and_model_rows(s)
        for i in range(n_pool):
            key = f"pool_{i:03d}"
            lj = {"rationale_note": "test", "severity": "high", "damaged": True}
            _add_example(s, PID, key)
            _add_label(s, PID, key, pool_assignment="test_pool", label_json=lj)
        for i in range(n_nonpool):
            key = f"nonpool_{i:03d}"
            _add_example(s, PID, key)
            _add_label(s, PID, key, pool_assignment=None, outcome="Edit")
        s.commit()

    return engine, pdir, settings


def _make_success_result(example_key, severity="high", damaged=True):
    """Build a successful EvalExampleResult with controllable label values."""
    return EvalExampleResult(
        example_key=example_key,
        invocation_id=generate_uuid4(),
        invocation_status="success",
        proposal_json={
            "rationale_note": "test",
            "severity": severity,
            "damaged": damaged,
        },
        schema_valid_core=True,
        field_matches=[
            FieldMatchResult("severity", severity == "high", severity, "high"),
            FieldMatchResult("damaged", damaged is True, damaged, True),
        ],
        exact_match_pass=severity == "high" and damaged is True,
    )


def _make_failure_result(example_key, status="timeout"):
    return EvalExampleResult(
        example_key=example_key,
        invocation_id=generate_uuid4(),
        invocation_status=status,
        proposal_json=None,
        schema_valid_core=False,
        field_matches=None,
        exact_match_pass=None,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Section A: State Machine
# ══════════════════════════════════════════════════════════════════════════════


class TestStateMachine:
    @pytest.mark.asyncio
    async def test_start_creates_queued_run(self, tmp_path):
        engine, pdir, settings = _setup_project_with_pool(tmp_path)
        # Don't actually run the background task — mock register
        with patched_register(evaluation_service_module) as mock_reg:
            result = await start_evaluation_run(
                PID,
                icl_mode="enabled",
                settings=settings,
            )
        assert not isinstance(result, str), result
        assert result["status"] == "queued"
        assert result["run_type"] == "evaluation_run"
        assert result["pool_version"] >= 1
        mock_reg.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_pool_returns_error(self, tmp_path):
        """No pool members → error string."""
        engine, pdir = setup_project_db(tmp_path)
        settings = make_stub_settings(WORKSPACE_ROOT=str(tmp_path / "workspace"))
        with Session(engine) as s:
            add_standard_project_row(s, PID, pdir)
            add_fixture_guidance_row(s)
            add_endpoint_and_model_rows(s)
            _add_example(s, PID, "img_001")
            _add_label(s, PID, "img_001", pool_assignment=None)
            s.commit()

        result = await start_evaluation_run(PID, settings=settings)
        assert isinstance(result, str)
        assert "empty" in result.lower() or "pool" in result.lower()

    @pytest.mark.asyncio
    async def test_cancel_terminal_returns_conflict(self, tmp_path):
        engine, pdir, settings = _setup_project_with_pool(tmp_path)
        # Create a completed run manually
        with Session(engine) as s:
            s.add(
                RunRecord(
                    run_id="done-run",
                    project_id=PID,
                    run_type="evaluation_run",
                    status="completed",
                    examples_total=5,
                )
            )
            s.commit()
        result = await cancel_evaluation_run(PID, "done-run", settings=settings)
        assert isinstance(result, str)
        assert "conflict" in result.lower()

    @pytest.mark.asyncio
    async def test_cancel_running_sets_canceling(self, tmp_path):
        engine, pdir, settings = _setup_project_with_pool(tmp_path)
        with Session(engine) as s:
            s.add(
                RunRecord(
                    run_id="run-1",
                    project_id=PID,
                    run_type="evaluation_run",
                    status="running",
                    examples_total=5,
                )
            )
            s.commit()
        from vlm_feedback_loop.services.evaluation_service import _cancel_events

        _cancel_events["run-1"] = asyncio.Event()

        result = await cancel_evaluation_run(PID, "run-1", settings=settings)
        assert not isinstance(result, str)
        assert result["status"] == "canceling"
        assert _cancel_events["run-1"].is_set()
        _cancel_events.pop("run-1", None)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "override,expected_in_run_config",
        [
            (3, 3),  # explicit override flows through
            (None, None),  # default falls back to settings (None in run_config)
        ],
    )
    async def test_icl_max_examples_override(
        self, tmp_path, override, expected_in_run_config
    ):
        """Per-run icl_max_examples override flows through start → execute → invoke.

        Diagnostic-only knob. Verifies the value lands in run_config so
        _invoke_for_evaluation can read it.
        """
        engine, pdir, settings = _setup_project_with_pool(
            tmp_path, n_pool=2, n_nonpool=0
        )

        captured: dict[str, dict] = {}

        async def _mock_invoke(project_id, run_id, example_key, **kwargs):
            captured["run_config"] = kwargs.get("run_config", {})
            return _make_success_result(example_key)

        with patch(
            "vlm_feedback_loop.services.evaluation_service._invoke_for_evaluation",
            side_effect=_mock_invoke,
        ):
            with patched_register(evaluation_service_module):
                kwargs = {"icl_mode": "enabled", "settings": settings}
                if override is not None:
                    kwargs["icl_max_examples"] = override
                result = await start_evaluation_run(PID, **kwargs)
            assert not isinstance(result, str), result
            run_id = result["run_id"]

            from vlm_feedback_loop.services.evaluation_service import (
                _execute_evaluation,
            )

            await _execute_evaluation(PID, run_id, settings, icl_max_examples=override)

        # run_config must carry the override value (None when not set)
        assert "icl_max_examples" in captured["run_config"]
        assert captured["run_config"]["icl_max_examples"] == expected_in_run_config

    @pytest.mark.asyncio
    async def test_run_snapshotted_under_retired_guidance_fails_at_claim(
        self, tmp_path
    ):
        """A run created concurrently with a guidance edit can commit after
        the edit's cancel sweep ran — the one writer the sweep cannot see.
        Phase A re-checks the active Guidance under the claim's write lock
        and fails the run exactly as the sweep would have, before any
        Teacher call."""
        from conftest import add_fixture_guidance_row
        from vlm_feedback_loop.db.models.project import Project
        from vlm_feedback_loop.services.evaluation_service import _execute_evaluation

        engine, pdir, settings = _setup_project_with_pool(
            tmp_path, n_pool=2, n_nonpool=0
        )

        async def _must_not_invoke(*a, **kw):
            raise AssertionError("must not invoke the Teacher")

        with patch(
            "vlm_feedback_loop.services.evaluation_service._invoke_for_evaluation",
            side_effect=_must_not_invoke,
        ):
            with patched_register(evaluation_service_module):
                result = await start_evaluation_run(
                    PID, icl_mode="enabled", settings=settings
                )
            assert not isinstance(result, str), result
            run_id = result["run_id"]

            # An edit activates a new version after the run committed.
            with Session(engine) as s:
                add_fixture_guidance_row(s, PID, "g2-post-edit", version_number=2)
                s.query(Project).filter_by(project_id=PID).update(
                    {"active_guidance_id": "g2-post-edit"}
                )
                s.commit()

            await _execute_evaluation(PID, run_id, settings)

        with Session(engine) as s:
            run = s.query(RunRecord).filter_by(run_id=run_id).one()
            assert run.status == "failed"
            assert run.status_reason == "guidance_edited_during_run"

    @pytest.mark.asyncio
    async def test_eval_concurrency_explicit_override(self, tmp_path):
        """eval_concurrency body field flows through to _execute_evaluation.

        Provider-aware default already picks the right concurrency by endpoint
        mode; this test just locks the explicit-override path.
        """
        engine, pdir, settings = _setup_project_with_pool(
            tmp_path, n_pool=2, n_nonpool=0
        )

        captured: dict[str, dict] = {}

        async def _mock_invoke(project_id, run_id, example_key, **kwargs):
            captured["run_config"] = kwargs.get("run_config", {})
            return _make_success_result(example_key)

        with patch(
            "vlm_feedback_loop.services.evaluation_service._invoke_for_evaluation",
            side_effect=_mock_invoke,
        ):
            with patched_register(evaluation_service_module):
                result = await start_evaluation_run(
                    PID,
                    icl_mode="enabled",
                    eval_concurrency=1,
                    settings=settings,
                )
            assert not isinstance(result, str), result
            run_id = result["run_id"]

            from vlm_feedback_loop.services.evaluation_service import (
                _execute_evaluation,
            )

            await _execute_evaluation(PID, run_id, settings, eval_concurrency=1)

        # The mock was invoked, meaning the run got past Phase B's
        # semaphore allocation with the override in effect.
        assert "run_config" in captured

    @pytest.mark.asyncio
    async def test_default_eval_uses_deployment_adaptive_icl_gates(self, tmp_path):
        """With no per-run override, the eval invocation forwards
        ``settings.ICL_SIM_GAP`` / ``settings.ICL_ABS_THRESHOLD`` to
        ``invoke_teacher`` — the same values interactive proposals and batch
        labeling pass. Forwarding None instead turns similarity gating off
        (fixed-K attach-all), so the gate-certifying eval scores an ICL depth
        nothing in production runs (measured live: attach-all-46 scored 0.033
        EM while the production loop ran ~0.8 on the same Edit pool)."""
        from unittest.mock import AsyncMock

        engine, pdir, settings = _setup_project_with_pool(
            tmp_path, n_pool=2, n_nonpool=0
        )
        # Give the abs threshold a non-None deployment value so its
        # fallback is observable (the shipped default is None).
        settings = make_stub_settings(
            WORKSPACE_ROOT=settings.WORKSPACE_ROOT,
            ICL_SIM_GAP=0.05,
            ICL_ABS_THRESHOLD=0.4,
        )

        captured: dict[str, dict] = {}

        async def _mock_invoke(project_id, run_id, example_key, **kwargs):
            captured["run_config"] = kwargs.get("run_config", {})
            return _make_success_result(example_key)

        with patch(
            "vlm_feedback_loop.services.evaluation_service._invoke_for_evaluation",
            side_effect=_mock_invoke,
        ):
            with patched_register(evaluation_service_module):
                result = await start_evaluation_run(
                    PID, icl_mode="enabled", settings=settings
                )
            assert not isinstance(result, str), result
            run_id = result["run_id"]

            from vlm_feedback_loop.services.evaluation_service import (
                _execute_evaluation,
            )

            await _execute_evaluation(PID, run_id, settings)

        assert captured["run_config"].get("icl_sim_gap") is None

        from vlm_feedback_loop.services import evaluation_service as es
        from vlm_feedback_loop.services.prompt_service import (
            TeacherInvocationResult,
        )

        teacher_result = TeacherInvocationResult(
            inference_invocation_id="",
            content=None,
            finish_reason=None,
            invocation_status="endpoint_error",
            latency_ms=1,
            usage=None,
            icl_example_keys_used=[],
            prompt_hash="wire-mocked",
            structured_generation_attempted=False,
            structured_generation_fallback_used=False,
        )
        member_key = next(iter(captured["run_config"]["ground_truth"]))
        invoke_mock = AsyncMock(return_value=teacher_result)
        with patch.object(es, "invoke_teacher", new=invoke_mock):
            await es._invoke_for_evaluation(
                PID,
                run_id,
                member_key,
                run_config=captured["run_config"],
                engine=engine,
                settings=settings,
            )
        assert invoke_mock.call_args.kwargs["icl_sim_gap"] == 0.05
        assert invoke_mock.call_args.kwargs["icl_abs_threshold"] == 0.4

        # An explicit override — including falsy-but-valid 0.0 — must win
        # over the deployment default (an ``or`` fallback would clobber it).
        captured["run_config"]["icl_sim_gap"] = 0.0
        captured["run_config"]["icl_abs_threshold"] = -1.0
        invoke_mock.reset_mock()
        with patch.object(es, "invoke_teacher", new=invoke_mock):
            await es._invoke_for_evaluation(
                PID,
                run_id,
                member_key,
                run_config=captured["run_config"],
                engine=engine,
                settings=settings,
            )
        assert invoke_mock.call_args.kwargs["icl_sim_gap"] == 0.0
        assert invoke_mock.call_args.kwargs["icl_abs_threshold"] == -1.0


# ══════════════════════════════════════════════════════════════════════════════
# Section B: Config Snapshot
# ══════════════════════════════════════════════════════════════════════════════


class TestConfigSnapshot:
    @pytest.mark.asyncio
    async def test_config_snapshot_persisted(self, tmp_path):
        engine, pdir, settings = _setup_project_with_pool(tmp_path)
        with patched_register(evaluation_service_module):
            result = await start_evaluation_run(
                PID,
                icl_mode="disabled",
                structured_generation_mode="prompt_only",
                settings=settings,
            )
        assert not isinstance(result, str)
        assert result["guidance_id"] == GID
        assert result["model_config_id"] == MCID
        assert result["generation_preset_key"] == "precise"
        assert result["thinking_mode_effective"] == "on"
        assert result["visual_budget_preset_key"] == "balanced"
        assert result["structured_generation_mode_effective"] == "prompt_only"
        assert result["evaluation_source"] == "nim"
        assert result["icl_mode"] == "disabled"

        # Verify persisted on RunRecord
        with Session(engine) as s:
            run = s.query(RunRecord).filter_by(run_id=result["run_id"]).first()
            assert run.guidance_id == GID
            assert run.icl_eligible_count_at_start is not None
            assert run.inference_contract is not None

    @pytest.mark.asyncio
    async def test_generation_preset_override_snapshotted(self, tmp_path):
        """A per-run generation_preset_key override is snapshotted onto the
        RunRecord (audit honesty), overriding the project default."""
        engine, pdir, settings = _setup_project_with_pool(tmp_path)
        with patched_register(evaluation_service_module):
            result = await start_evaluation_run(
                PID,
                generation_preset_key="explore",
                settings=settings,
            )
        assert not isinstance(result, str), result
        # Project default is "precise"; the override wins and is recorded.
        assert result["generation_preset_key"] == "explore"
        with Session(engine) as s:
            run = s.query(RunRecord).filter_by(run_id=result["run_id"]).first()
            assert run.generation_preset_key == "explore"

    @pytest.mark.asyncio
    async def test_invalid_generation_preset_returns_error(self, tmp_path):
        """An unknown generation_preset_key is rejected before a run is created
        (the router maps the error string to a 4xx)."""
        engine, pdir, settings = _setup_project_with_pool(tmp_path)
        result = await start_evaluation_run(
            PID,
            generation_preset_key="bogus",
            settings=settings,
        )
        assert isinstance(result, str)
        assert "invalid generation_preset_key" in result

    @pytest.mark.asyncio
    async def test_thinking_override_snapshotted(self, tmp_path):
        """A per-run thinking_on override wins over project.thinking_default_on
        and is snapshotted onto the RunRecord (audit + config-change trigger)."""
        engine, pdir, settings = _setup_project_with_pool(tmp_path)
        with patched_register(evaluation_service_module):
            result = await start_evaluation_run(
                PID,
                thinking_on=False,
                settings=settings,
            )
        assert not isinstance(result, str), result
        # Project default thinking_default_on=True -> "on"; the override forces off.
        assert result["thinking_mode_effective"] == "off"
        with Session(engine) as s:
            run = s.query(RunRecord).filter_by(run_id=result["run_id"]).first()
            assert run.thinking_mode_effective == "off"

    @pytest.mark.asyncio
    async def test_visual_budget_override_snapshotted(self, tmp_path):
        """A per-run visual_budget_preset_key override wins over the project
        default and is snapshotted (mirrors generation_preset_key)."""
        engine, pdir, settings = _setup_project_with_pool(tmp_path)
        with patched_register(evaluation_service_module):
            result = await start_evaluation_run(
                PID,
                visual_budget_preset_key="high_detail",
                settings=settings,
            )
        assert not isinstance(result, str), result
        # Project default is "balanced"; the override wins and is recorded.
        assert result["visual_budget_preset_key"] == "high_detail"
        with Session(engine) as s:
            run = s.query(RunRecord).filter_by(run_id=result["run_id"]).first()
            assert run.visual_budget_preset_key == "high_detail"

    @pytest.mark.asyncio
    async def test_invalid_visual_budget_preset_returns_error(self, tmp_path):
        """An unknown visual_budget_preset_key is rejected before a run is
        created (the router maps the error string to a 4xx)."""
        engine, pdir, settings = _setup_project_with_pool(tmp_path)
        result = await start_evaluation_run(
            PID,
            visual_budget_preset_key="ultra",
            settings=settings,
        )
        assert isinstance(result, str)
        assert "invalid visual_budget_preset_key" in result


class TestTargetOverrides:
    """The Student NIM benchmark lifecycle reuses this pipeline against a
    Student endpoint via ``target_model_config_id`` /
    ``target_inference_contract``; both default to None, which must preserve
    the plain Teacher behavior exactly."""

    STUDENT_MCID = "mc-student"

    def test_teacher_contract_constant(self):
        """TEACHER_CONTRACT is fixed at all/core_only/None.

        The Compare screen identifies its Teacher-baseline run by matching
        a run's ``inference_contract`` against this exact shape
        (``CompareBenchmarkPage.tsx``), so a change here silently breaks
        that selection — update the frontend predicate together with this
        constant.
        """
        from vlm_feedback_loop.schemas.inference_contract import TEACHER_CONTRACT

        assert TEACHER_CONTRACT.model_dump() == {
            "output_field_mode": "all",
            "icl_field_mode": "core_only",
            "icl_max_examples": None,
        }

    def _add_student_model_config(self, engine):
        with Session(engine) as s:
            add_model_config_row(
                s,
                PID,
                self.STUDENT_MCID,
                EID,
                model_name="student-model",
                eligible_roles=json.dumps(["student_inference"]),
            )
            s.commit()

    @pytest.mark.asyncio
    async def test_no_overrides_uses_teacher_and_teacher_contract(self, tmp_path):
        """With both overrides None the run snapshots the active Teacher and
        the fixed full-output/Core-only-ICL Teacher contract."""
        engine, pdir, settings = _setup_project_with_pool(tmp_path)
        with patched_register(evaluation_service_module):
            result = await start_evaluation_run(PID, settings=settings)
        assert not isinstance(result, str), result
        assert result["model_config_id"] == MCID
        with Session(engine) as s:
            run = s.query(RunRecord).filter_by(run_id=result["run_id"]).first()
            assert run.model_config_id == MCID
            assert run.inference_contract["output_field_mode"] == "all"
            assert run.inference_contract["icl_field_mode"] == "core_only"

    @pytest.mark.asyncio
    async def test_target_model_config_snapshotted_on_run_record(self, tmp_path):
        """A valid target_model_config_id is persisted on the RunRecord in
        place of the active Teacher."""
        engine, pdir, settings = _setup_project_with_pool(tmp_path)
        self._add_student_model_config(engine)
        with patched_register(evaluation_service_module):
            result = await start_evaluation_run(
                PID,
                icl_mode="disabled",
                target_model_config_id=self.STUDENT_MCID,
                settings=settings,
            )
        assert not isinstance(result, str), result
        assert result["model_config_id"] == self.STUDENT_MCID
        with Session(engine) as s:
            run = s.query(RunRecord).filter_by(run_id=result["run_id"]).first()
            assert run.model_config_id == self.STUDENT_MCID
            assert run.icl_mode == "disabled"

    @pytest.mark.asyncio
    async def test_unknown_target_returns_not_found_string(self, tmp_path):
        """An unknown target_model_config_id returns the standard not-found
        string so the router can map it to 404."""
        engine, pdir, settings = _setup_project_with_pool(tmp_path)
        result = await start_evaluation_run(
            PID,
            target_model_config_id="mc-missing",
            settings=settings,
        )
        assert isinstance(result, str)
        assert "not found: ModelConfig mc-missing" in result

    @pytest.mark.asyncio
    async def test_contract_override_persists_on_run_record(self, tmp_path):
        """A target_inference_contract replaces TEACHER_CONTRACT in the
        RunRecord snapshot, so a core_only-trained Student is evaluated
        under the same field-mode contract it was trained against."""
        engine, pdir, settings = _setup_project_with_pool(tmp_path)
        self._add_student_model_config(engine)
        contract = {
            "output_field_mode": "core_only",
            "icl_field_mode": "core_only",
            "icl_max_examples": None,
        }
        with patched_register(evaluation_service_module):
            result = await start_evaluation_run(
                PID,
                icl_mode="disabled",
                target_model_config_id=self.STUDENT_MCID,
                target_inference_contract=contract,
                settings=settings,
            )
        assert not isinstance(result, str), result
        with Session(engine) as s:
            run = s.query(RunRecord).filter_by(run_id=result["run_id"]).first()
            assert run.inference_contract["output_field_mode"] == "core_only"
            assert run.inference_contract["icl_field_mode"] == "core_only"


# ══════════════════════════════════════════════════════════════════════════════
# Section C: Pool Snapshot
# ══════════════════════════════════════════════════════════════════════════════


class TestPoolSnapshot:
    @pytest.mark.asyncio
    async def test_pool_snapshot_created_at_start(self, tmp_path):
        engine, pdir, settings = _setup_project_with_pool(tmp_path, n_pool=5)
        with patched_register(evaluation_service_module):
            result = await start_evaluation_run(PID, settings=settings)
        assert not isinstance(result, str)
        assert result["pool_version"] >= 1

        with Session(engine) as s:
            run = s.query(RunRecord).filter_by(run_id=result["run_id"]).first()
            pool = s.query(Pool).filter_by(pool_id=run.pool_version_id).first()
            assert pool is not None
            assert pool.member_count == 5


# ══════════════════════════════════════════════════════════════════════════════
# Section D: Supersession
# ══════════════════════════════════════════════════════════════════════════════


class TestSupersession:
    @pytest.mark.asyncio
    async def test_start_supersedes_running_eval(self, tmp_path):
        engine, pdir, settings = _setup_project_with_pool(tmp_path)
        # Create existing running eval
        with Session(engine) as s:
            s.add(
                RunRecord(
                    run_id="old-run",
                    project_id=PID,
                    run_type="evaluation_run",
                    status="running",
                    examples_total=5,
                )
            )
            s.commit()

        with patched_register(evaluation_service_module):
            result = await start_evaluation_run(PID, settings=settings)
        assert not isinstance(result, str)
        assert result["superseded_run_id"] == "old-run"

        # Old run should be canceling
        with Session(engine) as s:
            old = s.query(RunRecord).filter_by(run_id="old-run").first()
            assert old.status == "canceling"
            assert old.status_reason == "superseded_by_newer_evaluation"

    async def test_invalid_preset_does_not_supersede_running_eval(self, tmp_path):
        """A rejected request must not cancel the in-flight evaluation.

        Regression: the supersede step set the running run's in-memory cancel
        event before validating the request, and a DB rollback cannot undo
        evt.set() — so an invalid preset key stopped the running eval without
        starting a replacement. Validation now runs before supersede.
        """
        import asyncio

        from vlm_feedback_loop.services.evaluation_service import _cancel_events

        engine, pdir, settings = _setup_project_with_pool(tmp_path)
        with Session(engine) as s:
            s.add(
                RunRecord(
                    run_id="old-run",
                    project_id=PID,
                    run_type="evaluation_run",
                    status="running",
                    examples_total=5,
                )
            )
            s.commit()
        evt = asyncio.Event()
        _cancel_events["old-run"] = evt

        try:
            result = await start_evaluation_run(
                PID, settings=settings, generation_preset_key="not-a-real-preset"
            )
            assert isinstance(result, str)
            assert "invalid generation_preset_key" in result
            # The running eval is untouched: neither DB-canceled nor signaled.
            with Session(engine) as s:
                old = s.query(RunRecord).filter_by(run_id="old-run").first()
                assert old.status == "running"
                assert old.status_reason is None
            assert not evt.is_set()
        finally:
            _cancel_events.pop("old-run", None)

    @pytest.mark.asyncio
    async def test_benchmark_run_does_not_supersede_running_teacher_eval(
        self, tmp_path
    ):
        """A Student benchmark start must leave a running gate-basis eval alone.

        Newest-config-wins exists to kill stale-config gate-basis runs;
        benchmark runs snapshot immutable configs, so they never fire the
        supersede. Before the provenance scoping, a Student NIM rotation
        silently canceled a running Teacher baseline eval, wasting the
        whole run.
        """
        engine, pdir, settings = _setup_project_with_pool(tmp_path)
        with Session(engine) as s:
            s.add(
                RunRecord(
                    run_id="teacher-baseline",
                    project_id=PID,
                    run_type="evaluation_run",
                    status="running",
                    examples_total=5,
                )
            )
            s.commit()

        with patched_register(evaluation_service_module):
            result = await start_evaluation_run(
                PID, student_model_config_id="student-1", settings=settings
            )
        assert not isinstance(result, str)
        assert result["superseded_run_id"] is None

        with Session(engine) as s:
            baseline = s.query(RunRecord).filter_by(run_id="teacher-baseline").first()
            assert baseline.status == "running"
            assert baseline.status_reason is None

    @pytest.mark.asyncio
    async def test_gate_basis_start_supersedes_only_gate_basis_runs(self, tmp_path):
        """A new gate-basis eval cancels its gate-basis predecessor, never a benchmark.

        Benchmark runs end by completion, manual cancel, or restart
        recovery — a concurrent gate-basis start is none of those.
        """
        engine, pdir, settings = _setup_project_with_pool(tmp_path)
        with Session(engine) as s:
            s.add(
                RunRecord(
                    run_id="old-gate",
                    project_id=PID,
                    run_type="evaluation_run",
                    status="running",
                    examples_total=5,
                )
            )
            s.add(
                RunRecord(
                    run_id="student-benchmark",
                    project_id=PID,
                    run_type="evaluation_run",
                    status="running",
                    student_model_config_id="student-1",
                    examples_total=5,
                )
            )
            s.commit()

        with patched_register(evaluation_service_module):
            result = await start_evaluation_run(PID, settings=settings)
        assert not isinstance(result, str)
        assert result["superseded_run_id"] == "old-gate"

        with Session(engine) as s:
            old_gate = s.query(RunRecord).filter_by(run_id="old-gate").first()
            assert old_gate.status == "canceling"
            assert old_gate.status_reason == "superseded_by_newer_evaluation"
            benchmark = s.query(RunRecord).filter_by(run_id="student-benchmark").first()
            assert benchmark.status == "running"
            assert benchmark.status_reason is None


# ══════════════════════════════════════════════════════════════════════════════
# Section E: Get / List
# ══════════════════════════════════════════════════════════════════════════════


class TestGetList:
    def test_get_returns_full_detail(self, tmp_path):
        engine, pdir, settings = _setup_project_with_pool(tmp_path)
        with Session(engine) as s:
            s.add(
                RunRecord(
                    run_id="run-1",
                    project_id=PID,
                    run_type="evaluation_run",
                    status="completed",
                    examples_total=5,
                    guidance_id=GID,
                    model_config_id=MCID,
                    metrics={"overall": {"exact_match_rate": 0.8, "example_count": 5}},
                )
            )
            s.commit()
        result = get_evaluation_run(PID, "run-1", settings=settings)
        assert not isinstance(result, str)
        assert result["run_id"] == "run-1"
        assert result["status"] == "completed"
        assert result["metrics"] is not None

    def test_get_nonexistent_returns_error(self, tmp_path):
        engine, pdir, settings = _setup_project_with_pool(tmp_path)
        result = get_evaluation_run(PID, "nonexistent", settings=settings)
        assert isinstance(result, str)
        assert "not found" in result.lower()

    def test_list_returns_newest_first(self, tmp_path):
        engine, pdir, settings = _setup_project_with_pool(tmp_path)
        with Session(engine) as s:
            for i in range(3):
                s.add(
                    RunRecord(
                        run_id=f"run-{i}",
                        project_id=PID,
                        run_type="evaluation_run",
                        status="completed",
                        examples_total=5,
                    )
                )
            s.commit()
        items, cursor = list_evaluation_runs(PID, limit=10, settings=settings)
        assert len(items) == 3
        # Newest first
        assert items[0]["run_id"] in ("run-0", "run-1", "run-2")

    def test_list_with_status_filter(self, tmp_path):
        engine, pdir, settings = _setup_project_with_pool(tmp_path)
        with Session(engine) as s:
            s.add(
                RunRecord(
                    run_id="r-completed",
                    project_id=PID,
                    run_type="evaluation_run",
                    status="completed",
                    examples_total=5,
                )
            )
            s.add(
                RunRecord(
                    run_id="r-failed",
                    project_id=PID,
                    run_type="evaluation_run",
                    status="failed",
                    examples_total=5,
                )
            )
            s.commit()
        items, _ = list_evaluation_runs(
            PID,
            status_filter="completed",
            limit=10,
            settings=settings,
        )
        assert len(items) == 1
        assert items[0]["run_id"] == "r-completed"

    def test_list_basis_filter_scopes_by_provenance(self, tmp_path):
        """basis="gate" hides Student benchmarks; basis="benchmark" shows only them.

        The eval strip lists with basis="gate" so a running Student
        benchmark can never surface as the SME's "current evaluation".
        """
        engine, pdir, settings = _setup_project_with_pool(tmp_path)
        with Session(engine) as s:
            s.add(
                RunRecord(
                    run_id="r-gate",
                    project_id=PID,
                    run_type="evaluation_run",
                    status="running",
                    examples_total=5,
                )
            )
            s.add(
                RunRecord(
                    run_id="r-benchmark",
                    project_id=PID,
                    run_type="evaluation_run",
                    status="running",
                    student_model_config_id="student-1",
                    examples_total=5,
                )
            )
            s.commit()

        gate_items, _ = list_evaluation_runs(PID, basis="gate", settings=settings)
        assert [i["run_id"] for i in gate_items] == ["r-gate"]

        bench_items, _ = list_evaluation_runs(PID, basis="benchmark", settings=settings)
        assert [i["run_id"] for i in bench_items] == ["r-benchmark"]

        all_items, _ = list_evaluation_runs(PID, settings=settings)
        assert {i["run_id"] for i in all_items} == {"r-gate", "r-benchmark"}


# ══════════════════════════════════════════════════════════════════════════════
# Section F: Trigger Status
# ══════════════════════════════════════════════════════════════════════════════


def _add_completed_eval(s, run_id: str = "prev-eval", **overrides) -> None:
    """Stage a completed Teacher-contract evaluation whose config snapshot
    matches the fixture project's defaults; overrides adjust the fields a
    given trigger test cares about (the trigger reads only Teacher-contract
    runs, and config-change compares each snapshot field independently)."""
    row = {
        "run_id": run_id,
        "project_id": PID,
        "run_type": "evaluation_run",
        "status": "completed",
        "examples_total": 5,
        "evaluation_source": "nim",
        "model_config_id": MCID,
        "guidance_id": GID,
        "generation_preset_key": "precise",
        "thinking_mode_effective": "on",
        "visual_budget_preset_key": "balanced",
    }
    row.update(overrides)
    s.add(RunRecord(**row))


class TestTriggerStatus:
    def test_first_pool_inactive_below_threshold(self, tmp_path):
        """Pool < 5 → first_pool_threshold inactive."""
        engine, pdir, settings = _setup_project_with_pool(
            tmp_path, n_pool=3, n_nonpool=0
        )
        result = compute_trigger_status(PID, settings=settings)
        assert not isinstance(result, str)
        assert result["first_pool_threshold"]["is_active"] is False

    def test_first_pool_active_at_threshold(self, tmp_path):
        """Pool >= 5 → first_pool_threshold active (before any eval)."""
        engine, pdir, settings = _setup_project_with_pool(
            tmp_path, n_pool=5, n_nonpool=0
        )
        result = compute_trigger_status(PID, settings=settings)
        assert not isinstance(result, str)
        assert result["first_pool_threshold"]["is_active"] is True
        assert result["first_pool_threshold"]["dismissed"] is False

    def test_first_pool_active_message_is_exact_product_copy(self, tmp_path):
        """Active first-pool message MUST match the product copy exactly and
        MUST NOT leak raw thresholds (no-jargon rule). Regression guard
        against a past implementation that appended '(threshold: N)'."""
        engine, pdir, settings = _setup_project_with_pool(
            tmp_path, n_pool=5, n_nonpool=0
        )
        result = compute_trigger_status(PID, settings=settings)
        assert not isinstance(result, str)
        message = result["first_pool_threshold"]["message"]
        assert "Run an evaluation to measure quality" in message
        assert "threshold:" not in message

    def test_first_pool_dismissed_after_completed_eval(self, tmp_path):
        """After a completed evaluation, first_pool_threshold is dismissed."""
        engine, pdir, settings = _setup_project_with_pool(tmp_path, n_pool=5)
        with Session(engine) as s:
            _add_completed_eval(s, run_id="eval-done")
            s.commit()
        result = compute_trigger_status(PID, settings=settings)
        assert result["first_pool_threshold"]["dismissed"] is True
        assert result["first_pool_threshold"]["is_active"] is False

    def test_config_change_detects_model_change(self, tmp_path):
        """Changed teacher_model_config_id → config_change active."""
        engine, pdir, settings = _setup_project_with_pool(tmp_path)
        with Session(engine) as s:
            _add_completed_eval(s, model_config_id="different-mc")
            s.commit()
        result = compute_trigger_status(PID, settings=settings)
        assert result["configuration_change"]["is_active"] is True
        assert (
            "teacher_model"
            in result["configuration_change"]["context"]["changed_fields"]
        )

    def test_config_change_inactive_when_matching(self, tmp_path):
        """When all 5 fields match, config_change is inactive."""
        engine, pdir, settings = _setup_project_with_pool(tmp_path)
        with Session(engine) as s:
            _add_completed_eval(s)
            s.commit()
        result = compute_trigger_status(PID, settings=settings)
        assert result["configuration_change"]["is_active"] is False

    def test_config_change_ignores_tao_and_student_runs(self, tmp_path):
        """The config-change baseline is the last Teacher evaluation: a TAO
        rescoring run (evaluation_source="tao") and a Student NIM serving
        run (student_model_config_id set) both describe a different model
        and must not trigger a spurious "teacher_model changed" nudge."""
        engine, pdir, settings = _setup_project_with_pool(tmp_path)
        with Session(engine) as s:
            _add_completed_eval(
                s,
                run_id="eval-tao",
                evaluation_source="tao",
                model_config_id="tao-student-mc",
            )
            _add_completed_eval(
                s,
                run_id="eval-student-nim",
                student_model_config_id="student-mc-1",
                model_config_id="student-endpoint-mc",
            )
            s.commit()
        result = compute_trigger_status(PID, settings=settings)
        assert not isinstance(result, str)
        # Neither run qualifies as a baseline, so there is nothing to
        # compare against — the trigger must be inactive, not falsely fired.
        assert result["configuration_change"]["is_active"] is False
        assert (
            result["configuration_change"]["message"]
            == "No previous evaluation to compare against."
        )

    def test_config_change_reads_newest_teacher_run(self, tmp_path):
        """With two Teacher evaluations, the newest snapshot is the
        comparison basis (pins the created_at DESC ordering): the older
        run differs from the project config, the newer one matches, so
        the trigger stays inactive."""
        engine, pdir, settings = _setup_project_with_pool(tmp_path)
        with Session(engine) as s:
            for run_id, mc in (("eval-old", "different-mc"), ("eval-new", MCID)):
                _add_completed_eval(s, run_id=run_id, model_config_id=mc)
            s.commit()
            s.query(RunRecord).filter_by(run_id="eval-old").update(
                {"created_at": "2000-01-01T00:00:00Z"}
            )
            s.query(RunRecord).filter_by(run_id="eval-new").update(
                {"created_at": "2026-01-01T00:00:00Z"}
            )
            s.commit()
        result = compute_trigger_status(PID, settings=settings)
        assert not isinstance(result, str)
        assert result["configuration_change"]["is_active"] is False
        # Pins that a baseline WAS found (the newest run, matching the
        # project config) — inactive-because-nothing-matched would show
        # the no-previous-evaluation message instead.
        assert (
            result["configuration_change"]["message"]
            == "Settings match last evaluation."
        )

    @pytest.mark.parametrize(
        ("override", "expected_field"),
        [
            ({"guidance_id": "different-g"}, "guidance"),
            ({"generation_preset_key": "creative"}, "generation_preset"),
            ({"thinking_mode_effective": "off"}, "thinking"),
            ({"visual_budget_preset_key": "different-vb"}, "visual_budget"),
        ],
    )
    def test_config_change_detects_each_tracked_field(
        self, tmp_path, override, expected_field
    ):
        """Every tracked config field is compared independently against the
        last evaluation's snapshot: a prior run differing ONLY in that field
        activates the trigger and names exactly that field."""
        engine, pdir, settings = _setup_project_with_pool(tmp_path)
        with Session(engine) as s:
            if override.get("guidance_id"):
                # The era filter joins runs to real Guidance rows; a prior
                # run under a different (non-semantic) version needs one.
                add_fixture_guidance_row(
                    s, PID, override["guidance_id"], version_number=2
                )
            _add_completed_eval(s, **override)
            s.commit()
        result = compute_trigger_status(PID, settings=settings)
        assert result["configuration_change"]["is_active"] is True
        assert result["configuration_change"]["context"]["changed_fields"] == [
            expected_field
        ]

    def test_config_change_thinking_off_matches_off_snapshot(self, tmp_path):
        """A project with thinking OFF matches a prior "off" snapshot — pins
        the boolean→"on"/"off" mapping the comparison derives from
        thinking_default_on (a hardcoded "on" would false-positive here)."""
        engine, pdir, settings = _setup_project_with_pool(tmp_path)
        with Session(engine) as s:
            s.query(Project).filter_by(project_id=PID).update(
                {"thinking_default_on": False}
            )
            _add_completed_eval(s, thinking_mode_effective="off")
            s.commit()
        result = compute_trigger_status(PID, settings=settings)
        assert result["configuration_change"]["is_active"] is False

    def test_triggers_rebuild_from_zero_after_semantic_core_change(self, tmp_path):
        """After a semantic Core change, the Auto-Evaluate
        trigger counters rebuild from zero — a prior-era evaluation must
        not keep the first-pool nudge dismissed, must not serve as the
        config-change snapshot (spurious 'guidance changed'), and must
        not seed the icl_growth baseline."""
        engine, pdir, settings = _setup_project_with_pool(
            tmp_path, n_pool=5, n_nonpool=10
        )
        with Session(engine) as s:
            # Prior-era completed evaluation under GID (version 1).
            _add_completed_eval(
                s, run_id="old-era-eval", icl_eligible_count_at_completion=3
            )
            # Guidance v2 born from a semantic Core change; it becomes
            # the active guidance, starting a new schema era.
            add_fixture_guidance_row(
                s,
                PID,
                "guidance-v2",
                version_number=2,
                semantic_core_change_from_guidance_id=GID,
            )
            project = s.query(Project).filter_by(project_id=PID).first()
            assert project is not None
            project.active_guidance_id = "guidance-v2"
            s.commit()
        result = compute_trigger_status(PID, settings=settings)
        assert not isinstance(result, str)
        # first_pool: the old-era eval no longer dismisses the nudge.
        assert result["first_pool_threshold"]["dismissed"] is False
        assert result["first_pool_threshold"]["is_active"] is True
        # configuration_change: no baseline in the new era — no spurious
        # "guidance changed" nudge.
        assert result["configuration_change"]["is_active"] is False
        assert (
            result["configuration_change"]["message"]
            == "No previous evaluation to compare against."
        )
        # icl_growth: baseline rebuilt from zero. (The fixture's Edit
        # labels carry the old guidance_id, so icl_count is also 0 here;
        # the real pin is baseline_count == 0 — the old era's count of 3
        # no longer seeds the doubling baseline.)
        assert result["icl_growth"]["context"]["baseline_count"] == 0
        assert result["icl_growth"]["is_active"] is False

    def test_new_era_eval_at_floor_rearms_triggers(self, tmp_path):
        """Complement of the rebuild-from-zero test: an evaluation
        recorded under the semantic-change Guidance version itself (the
        era floor) re-dismisses first_pool and serves as the
        config-change baseline. Pins the >= boundary at the trigger
        sites and pins that the era filter is a version-floor join, not
        an active-guidance equality (the run's guidance v2 differs from
        the active v3 and must still qualify — firing the 'guidance'
        change nudge)."""
        engine, pdir, settings = _setup_project_with_pool(
            tmp_path, n_pool=5, n_nonpool=0
        )
        with Session(engine) as s:
            add_fixture_guidance_row(
                s,
                PID,
                "guidance-v2",
                version_number=2,
                semantic_core_change_from_guidance_id=GID,
            )
            add_fixture_guidance_row(s, PID, "guidance-v3", version_number=3)
            _add_completed_eval(s, run_id="new-era-eval", guidance_id="guidance-v2")
            project = s.query(Project).filter_by(project_id=PID).first()
            assert project is not None
            project.active_guidance_id = "guidance-v3"
            s.commit()
        result = compute_trigger_status(PID, settings=settings)
        assert not isinstance(result, str)
        assert result["first_pool_threshold"]["dismissed"] is True
        assert result["configuration_change"]["is_active"] is True
        assert "guidance" in result["configuration_change"]["context"]["changed_fields"]

    def test_icl_growth_when_doubled(self, tmp_path):
        """ICL count doubled from baseline → icl_growth active."""
        engine, pdir, settings = _setup_project_with_pool(
            tmp_path, n_pool=5, n_nonpool=10
        )
        with Session(engine) as s:
            # Previous eval completed with icl_count = 3
            _add_completed_eval(s, icl_eligible_count_at_completion=3)
            s.commit()
        # Current ICL eligible = 10 non-pool Edits >= 2*3 = 6
        result = compute_trigger_status(PID, settings=settings)
        assert result["icl_growth"]["is_active"] is True

    def test_icl_growth_inactive_below_double(self, tmp_path):
        """ICL count not doubled → inactive."""
        engine, pdir, settings = _setup_project_with_pool(
            tmp_path, n_pool=5, n_nonpool=2
        )
        with Session(engine) as s:
            _add_completed_eval(s, icl_eligible_count_at_completion=5)
            s.commit()
        # Current ICL = 2 < 2*5 = 10
        result = compute_trigger_status(PID, settings=settings)
        assert result["icl_growth"]["is_active"] is False

    def test_no_previous_eval_config_change_inactive(self, tmp_path):
        """No previous eval → config_change inactive."""
        engine, pdir, settings = _setup_project_with_pool(tmp_path)
        result = compute_trigger_status(PID, settings=settings)
        assert result["configuration_change"]["is_active"] is False


# ══════════════════════════════════════════════════════════════════════════════
# Section G: Trigger Dismissal
# ══════════════════════════════════════════════════════════════════════════════


class TestTriggerDismissal:
    def test_dismiss_icl_growth_updates_project(self, tmp_path):
        engine, pdir, settings = _setup_project_with_pool(tmp_path, n_nonpool=5)
        result = dismiss_trigger(PID, "icl_growth", settings=settings)
        assert not isinstance(result, str)
        assert result["dismissed"] is True

        with Session(engine) as s:
            project = s.query(Project).filter_by(project_id=PID).first()
            assert project.icl_recommendation_dismissed_at_count == 5

    def test_dismiss_first_pool_is_noop(self, tmp_path):
        engine, pdir, settings = _setup_project_with_pool(tmp_path)
        result = dismiss_trigger(PID, "first_pool_threshold", settings=settings)
        assert not isinstance(result, str)
        assert result["dismissed"] is True

    def test_dismiss_config_change_is_noop(self, tmp_path):
        engine, pdir, settings = _setup_project_with_pool(tmp_path)
        result = dismiss_trigger(PID, "configuration_change", settings=settings)
        assert not isinstance(result, str)
        assert result["dismissed"] is True


# ══════════════════════════════════════════════════════════════════════════════
# Section H: Metrics Serialization
# ══════════════════════════════════════════════════════════════════════════════


class TestMetrics:
    def test_serialize_metrics(self):
        agg = AggregateMetrics(
            overall_exact_match_rate=0.8,
            example_count=5,
            per_core_field_match_rate={"severity": 0.9, "damaged": 0.8},
            per_value_metrics={
                "severity": {
                    "high": PerValueMetrics(precision=0.9, recall=0.85, f1=0.874),
                    "low": PerValueMetrics(precision=0.8, recall=0.7, f1=0.746),
                },
            },
        )
        d = serialize_metrics_overall(agg)
        # Everything nests under the overall bucket — the same
        # EvaluationMetrics shape as bucketed NIM runs — so the scale-up
        # gate and the Compare page read one contract for every
        # evaluation_source.
        assert d["overall"]["exact_match_rate"] == 0.8
        assert d["overall"]["example_count"] == 5
        assert d["overall"]["per_field_match_rates"]["severity"] == 0.9
        assert "high" in d["overall"]["per_value_metrics"]["severity"]
        assert d["returning"] is None
        assert d["new"] is None
        assert "per_field_match_rates" not in d  # flat legacy shape is gone


# ══════════════════════════════════════════════════════════════════════════════
# Section I: Coverage Gaps
# ══════════════════════════════════════════════════════════════════════════════


class TestCoverageGaps:
    def test_detects_missing_enum_values(self):
        core_fields = [
            {
                "field_name": "severity",
                "type": "enum",
                "role": "core",
                "allowed_values": ["low", "medium", "high"],
            },
        ]
        ground_truth = {
            "ex1": {"severity": "high"},
            "ex2": {"severity": "high"},
        }
        gaps = _compute_coverage_gaps(core_fields, ground_truth, ["ex1", "ex2"])
        assert len(gaps) == 1
        assert set(gaps[0]["missing_values"]) == {"low", "medium"}

    def test_detects_missing_boolean_value(self):
        core_fields = [
            {"field_name": "damaged", "type": "boolean", "role": "core"},
        ]
        ground_truth = {"ex1": {"damaged": True}}
        gaps = _compute_coverage_gaps(core_fields, ground_truth, ["ex1"])
        assert len(gaps) == 1
        assert "false" in gaps[0]["missing_values"]

    def test_no_gaps_when_all_covered(self):
        core_fields = [
            {"field_name": "damaged", "type": "boolean", "role": "core"},
        ]
        ground_truth = {"ex1": {"damaged": True}, "ex2": {"damaged": False}}
        gaps = _compute_coverage_gaps(core_fields, ground_truth, ["ex1", "ex2"])
        assert len(gaps) == 0


# ══════════════════════════════════════════════════════════════════════════════
# Section I.5: Provider-aware concurrency (mirrors the CLIP embedding worker)
# ══════════════════════════════════════════════════════════════════════════════


class TestProviderAwareConcurrency:
    def test_hosted_endpoint_uses_hosted_default(self):
        s = make_stub_settings(
            EVAL_CONCURRENCY_HOSTED=1, EVAL_CONCURRENCY_SELF_HOSTED=8
        )
        assert _resolve_eval_concurrency("hosted", s) == 1

    def test_self_hosted_endpoint_uses_self_hosted_default(self):
        s = make_stub_settings(
            EVAL_CONCURRENCY_HOSTED=1, EVAL_CONCURRENCY_SELF_HOSTED=8
        )
        assert _resolve_eval_concurrency("self_hosted", s) == 8

    def test_local_system_managed_treated_as_self_hosted(self):
        s = make_stub_settings(
            EVAL_CONCURRENCY_HOSTED=1, EVAL_CONCURRENCY_SELF_HOSTED=8
        )
        # local_system_managed has no shared rate-limit either — treat as self-hosted
        assert _resolve_eval_concurrency("local_system_managed", s) == 8

    def test_explicit_override_wins_over_provider_default(self):
        s = make_stub_settings(
            EVAL_CONCURRENCY_HOSTED=1, EVAL_CONCURRENCY_SELF_HOSTED=8
        )
        assert _resolve_eval_concurrency("hosted", s, explicit_override=4) == 4
        assert _resolve_eval_concurrency("self_hosted", s, explicit_override=2) == 2

    def test_zero_override_falls_back_to_provider_default(self):
        s = make_stub_settings(
            EVAL_CONCURRENCY_HOSTED=1, EVAL_CONCURRENCY_SELF_HOSTED=8
        )
        # 0 isn't a sane concurrency; resolver ignores it and falls back
        assert _resolve_eval_concurrency("hosted", s, explicit_override=0) == 1


# ══════════════════════════════════════════════════════════════════════════════
# Section J: Restart Recovery
# ══════════════════════════════════════════════════════════════════════════════


class TestRestartRecovery:
    def test_running_eval_fails_on_recovery(self, tmp_path):
        """Existing main.py recovery transitions running eval → failed."""
        engine, pdir, settings = _setup_project_with_pool(tmp_path)
        with Session(engine) as s:
            s.add(
                RunRecord(
                    run_id="running-eval",
                    project_id=PID,
                    run_type="evaluation_run",
                    status="running",
                    examples_total=5,
                )
            )
            s.commit()

        # Simulate the recovery logic from main.py
        with Session(engine) as s:
            non_terminal = (
                s.query(RunRecord)
                .filter(
                    RunRecord.status.in_(["queued", "running", "canceling"]),
                )
                .all()
            )
            for rr in non_terminal:
                if rr.run_type == "evaluation_run":
                    rr.status = "failed"
                    rr.status_reason = "backend_restart_interrupted"
                    rr.completed_at = utc_now()
            s.commit()

        with Session(engine) as s:
            run = s.query(RunRecord).filter_by(run_id="running-eval").first()
            assert run.status == "failed"
            assert run.status_reason == "backend_restart_interrupted"


# ══════════════════════════════════════════════════════════════════════════════
# Section K: Background Execution (with mocked inference)
# ══════════════════════════════════════════════════════════════════════════════


class TestBackgroundExecution:
    @pytest.mark.asyncio
    async def test_successful_run_completes(self, tmp_path):
        """Full run with all examples passing → completed."""
        engine, pdir, settings = _setup_project_with_pool(
            tmp_path, n_pool=3, n_nonpool=0
        )

        # Mock _invoke_for_evaluation to return success for all examples
        async def _mock_invoke(project_id, run_id, example_key, **kwargs):
            return _make_success_result(example_key)

        with patch(
            "vlm_feedback_loop.services.evaluation_service._invoke_for_evaluation",
            side_effect=_mock_invoke,
        ):
            # Create run
            with patched_register(evaluation_service_module):
                result = await start_evaluation_run(PID, settings=settings)
            assert not isinstance(result, str)
            run_id = result["run_id"]

            # Execute the background task directly
            from vlm_feedback_loop.services.evaluation_service import (
                _execute_evaluation,
            )

            await _execute_evaluation(PID, run_id, settings)

        with Session(engine) as s:
            run = s.query(RunRecord).filter_by(run_id=run_id).first()
            assert run.status == "completed"
            assert run.metrics is not None
            assert run.metrics["overall"]["exact_match_rate"] == 1.0
            assert run.icl_eligible_count_at_completion is not None
            assert run.completed_at is not None

    @pytest.mark.asyncio
    async def test_finalizer_does_not_resurrect_terminalized_run(self, tmp_path):
        """A run terminalized mid-flight by another writer (schema
        evolution fails active runs with
        status_reason='schema_evolution_canceled') must stay failed: the
        finalizer refuses to overwrite terminal rows — resurrecting to
        'completed' would publish metrics computed under a Guidance whose
        labels were just wiped."""
        engine, pdir, settings = _setup_project_with_pool(
            tmp_path, n_pool=3, n_nonpool=0
        )

        async def _mock_invoke(project_id, run_id, example_key, **kwargs):
            # Simulate schema evolution landing mid-run: another writer
            # terminalizes the row while invocations are in flight.
            with Session(engine) as s:
                run = s.query(RunRecord).filter_by(run_id=run_id).one()
                if run.status != "failed":
                    run.status = "failed"
                    run.status_reason = "schema_evolution_canceled"
                    run.completed_at = utc_now()
                    s.commit()
            return _make_success_result(example_key)

        with patch(
            "vlm_feedback_loop.services.evaluation_service._invoke_for_evaluation",
            side_effect=_mock_invoke,
        ):
            with patched_register(evaluation_service_module):
                result = await start_evaluation_run(PID, settings=settings)
            assert not isinstance(result, str)
            run_id = result["run_id"]

            from vlm_feedback_loop.services.evaluation_service import (
                _execute_evaluation,
            )

            await _execute_evaluation(PID, run_id, settings)

        with Session(engine) as s:
            run = s.query(RunRecord).filter_by(run_id=run_id).first()
            assert run is not None
            assert run.status == "failed"
            assert run.status_reason == "schema_evolution_canceled"
            assert run.metrics is None

    @pytest.mark.asyncio
    async def test_partial_failure_transitions_to_incomplete(self, tmp_path):
        """Some examples fail after retry → incomplete."""
        engine, pdir, settings = _setup_project_with_pool(
            tmp_path, n_pool=3, n_nonpool=0
        )

        call_count = {}

        async def _mock_invoke(project_id, run_id, example_key, **kwargs):
            call_count.setdefault(example_key, 0)
            call_count[example_key] += 1
            if example_key == "pool_000":
                return _make_failure_result(example_key, "timeout")
            return _make_success_result(example_key)

        with patch(
            "vlm_feedback_loop.services.evaluation_service._invoke_for_evaluation",
            side_effect=_mock_invoke,
        ):
            with patched_register(evaluation_service_module):
                result = await start_evaluation_run(PID, settings=settings)
            run_id = result["run_id"]

            from vlm_feedback_loop.services.evaluation_service import (
                _execute_evaluation,
            )

            await _execute_evaluation(PID, run_id, settings)

        with Session(engine) as s:
            run = s.query(RunRecord).filter_by(run_id=run_id).first()
            assert run.status == "incomplete"
            # pool_000 was called twice (concurrent + retry)
            assert call_count.get("pool_000", 0) == 2

    @pytest.mark.asyncio
    async def test_unhandled_exception_transitions_to_failed(self, tmp_path):
        """Exception outside the inference gather → failed."""
        engine, pdir, settings = _setup_project_with_pool(
            tmp_path, n_pool=3, n_nonpool=0
        )

        with patched_register(evaluation_service_module):
            result = await start_evaluation_run(PID, settings=settings)
        run_id = result["run_id"]

        # Corrupt the pool_version_id to cause an exception during pool load
        with Session(engine) as s:
            run = s.query(RunRecord).filter_by(run_id=run_id).first()
            run.pool_version_id = "nonexistent-pool-id"
            s.commit()

        from vlm_feedback_loop.services.evaluation_service import _execute_evaluation

        await _execute_evaluation(PID, run_id, settings)

        with Session(engine) as s:
            run = s.query(RunRecord).filter_by(run_id=run_id).first()
            assert run.status == "failed"
            assert run.status_reason == "unhandled_exception"

    @pytest.mark.asyncio
    async def test_individual_inference_exception_causes_incomplete(self, tmp_path):
        """Per-example exception is handled gracefully → incomplete."""
        engine, pdir, settings = _setup_project_with_pool(
            tmp_path, n_pool=3, n_nonpool=0
        )

        async def _mock_invoke(project_id, run_id, example_key, **kwargs):
            if example_key == "pool_000":
                raise RuntimeError("Simulated per-example failure")
            return _make_success_result(example_key)

        with (
            patch(
                "vlm_feedback_loop.services.evaluation_service._invoke_for_evaluation",
                side_effect=_mock_invoke,
            ),
            patched_register(evaluation_service_module),
        ):
            result = await start_evaluation_run(PID, settings=settings)
        run_id = result["run_id"]

        with patch(
            "vlm_feedback_loop.services.evaluation_service._invoke_for_evaluation",
            side_effect=_mock_invoke,
        ):
            from vlm_feedback_loop.services.evaluation_service import (
                _execute_evaluation,
            )

            await _execute_evaluation(PID, run_id, settings)

        with Session(engine) as s:
            run = s.query(RunRecord).filter_by(run_id=run_id).first()
            assert run.status == "incomplete"

    @pytest.mark.asyncio
    async def test_returning_and_new_keys(self, tmp_path):
        """Second eval computes returning and new keys correctly."""
        engine, pdir, settings = _setup_project_with_pool(
            tmp_path, n_pool=3, n_nonpool=0
        )

        async def _mock_invoke(project_id, run_id, example_key, **kwargs):
            return _make_success_result(example_key)

        # First evaluation
        with (
            patch(
                "vlm_feedback_loop.services.evaluation_service._invoke_for_evaluation",
                side_effect=_mock_invoke,
            ),
            patched_register(evaluation_service_module),
        ):
            r1 = await start_evaluation_run(PID, settings=settings)
        run_id_1 = r1["run_id"]

        with patch(
            "vlm_feedback_loop.services.evaluation_service._invoke_for_evaluation",
            side_effect=_mock_invoke,
        ):
            from vlm_feedback_loop.services.evaluation_service import (
                _execute_evaluation,
            )

            await _execute_evaluation(PID, run_id_1, settings)

        # Add a new pool member
        with Session(engine) as s:
            _add_example(s, PID, "pool_new")
            _add_label(s, PID, "pool_new", pool_assignment="test_pool")
            s.commit()

        # Second evaluation
        with (
            patch(
                "vlm_feedback_loop.services.evaluation_service._invoke_for_evaluation",
                side_effect=_mock_invoke,
            ),
            patched_register(evaluation_service_module),
        ):
            r2 = await start_evaluation_run(PID, settings=settings)
        run_id_2 = r2["run_id"]

        with patch(
            "vlm_feedback_loop.services.evaluation_service._invoke_for_evaluation",
            side_effect=_mock_invoke,
        ):
            await _execute_evaluation(PID, run_id_2, settings)

        with Session(engine) as s:
            run2 = s.query(RunRecord).filter_by(run_id=run_id_2).first()
            assert run2.status == "completed"
            # Returning = original 3 pool members; New = pool_new
            assert run2.returning_example_keys is not None
            assert run2.new_example_keys is not None
            assert "pool_new" in run2.new_example_keys
            assert len(run2.returning_example_keys) == 3
            assert run2.previous_overall_exact_match is not None

    @pytest.mark.asyncio
    async def test_first_run_no_previous(self, tmp_path):
        """First evaluation has no returning/new keys."""
        engine, pdir, settings = _setup_project_with_pool(
            tmp_path, n_pool=3, n_nonpool=0
        )

        async def _mock_invoke(project_id, run_id, example_key, **kwargs):
            return _make_success_result(example_key)

        with (
            patch(
                "vlm_feedback_loop.services.evaluation_service._invoke_for_evaluation",
                side_effect=_mock_invoke,
            ),
            patched_register(evaluation_service_module),
        ):
            result = await start_evaluation_run(PID, settings=settings)
        run_id = result["run_id"]

        with patch(
            "vlm_feedback_loop.services.evaluation_service._invoke_for_evaluation",
            side_effect=_mock_invoke,
        ):
            from vlm_feedback_loop.services.evaluation_service import (
                _execute_evaluation,
            )

            await _execute_evaluation(PID, run_id, settings)

        with Session(engine) as s:
            run = s.query(RunRecord).filter_by(run_id=run_id).first()
            assert run.returning_example_keys is None
            assert run.new_example_keys is None
            assert run.previous_overall_exact_match is None

    @pytest.mark.asyncio
    async def test_incomplete_prior_run_not_used_as_baseline(self, tmp_path):
        """A prior ``incomplete`` eval MUST NOT be the Returning/New baseline.

        Regression guard: if an incomplete prior run with zero usable
        examples (overall exact_match_rate=0.0) is picked as the "previous"
        for a subsequent completed run, the UI shows a bogus
        "vs previous on same images" delta on what is effectively a first
        evaluation. The baseline is "the previous *completed* evaluation's
        snapshot"; only ``completed`` runs produce authoritative aggregate
        metrics.
        """
        engine, pdir, settings = _setup_project_with_pool(
            tmp_path, n_pool=3, n_nonpool=0
        )

        async def _mock_invoke(project_id, run_id, example_key, **kwargs):
            return _make_success_result(example_key)

        # Seed a prior *incomplete* eval directly — an earlier run that
        # finalized incomplete with a 0.0 overall metric.
        with Session(engine) as s:
            stale_pool_id = generate_uuid4()
            s.add(
                Pool(
                    pool_id=stale_pool_id,
                    project_id=PID,
                    pool_type="test_pool",
                    pool_version=1,
                    member_example_keys=["pool_0", "pool_1", "pool_2"],
                    member_count=3,
                    guidance_id=GID,
                    created_at=utc_now(),
                )
            )
            s.add(
                RunRecord(
                    run_id=generate_uuid4(),
                    project_id=PID,
                    run_type="evaluation_run",
                    status="incomplete",
                    guidance_id=GID,
                    model_config_id=MCID,
                    pool_version_id=stale_pool_id,
                    metrics={
                        "overall": {
                            "exact_match_rate": 0.0,
                            "example_count": 0,
                            "per_field_match_rates": {},
                            "per_value_metrics": {},
                        },
                        "returning": None,
                        "new": None,
                    },
                    created_at=utc_now(),
                    started_at=utc_now(),
                    completed_at=utc_now(),
                )
            )
            s.commit()

        # Now run a fresh eval — it should treat this as a first-eval
        # (no previous *completed* baseline).
        with (
            patch(
                "vlm_feedback_loop.services.evaluation_service._invoke_for_evaluation",
                side_effect=_mock_invoke,
            ),
            patched_register(evaluation_service_module),
        ):
            result = await start_evaluation_run(PID, settings=settings)
        run_id = result["run_id"]

        with patch(
            "vlm_feedback_loop.services.evaluation_service._invoke_for_evaluation",
            side_effect=_mock_invoke,
        ):
            from vlm_feedback_loop.services.evaluation_service import (
                _execute_evaluation,
            )

            await _execute_evaluation(PID, run_id, settings)

        with Session(engine) as s:
            run = s.query(RunRecord).filter_by(run_id=run_id).first()
            assert run.status == "completed"
            # No completed baseline exists — must render as first eval.
            assert run.returning_example_keys is None
            assert run.new_example_keys is None
            assert run.previous_overall_exact_match is None
            assert run.previous_pool_version is None


# ══════════════════════════════════════════════════════════════════════════════
# Section N: wire-mock e2e — real _invoke_for_evaluation against mocked NIM
# ══════════════════════════════════════════════════════════════════════════════
#
# These tests let the real ``_invoke_for_evaluation`` body run (not monkeypatched
# at the function level) and mock the lowest layer — ``nim_client.chat_completions``
# plus the image-transport helpers. They guard the real invoke pipeline:
# function-level mocks alone cannot catch a pipeline that persists zero
# OperationRecords on a live eval run.


def _patch_nim_pipeline(chat_return):
    """Patch nim_client.chat_completions + prepare_images.

    Accepts either a single return value (used for every call) or a
    side_effect-style list/callable for multi-call scenarios.
    """
    from unittest.mock import AsyncMock

    if callable(chat_return) or isinstance(chat_return, list):
        chat_mock = AsyncMock(side_effect=chat_return)
    else:
        chat_mock = AsyncMock(return_value=chat_return)

    return (
        patch(
            "vlm_feedback_loop.services.prompt_service.nim_client.chat_completions",
            new=chat_mock,
        ),
        patch(
            "vlm_feedback_loop.services.prompt_service.prepare_images",
            new=AsyncMock(return_value=fake_prepare_result(1)),
        ),
    )


class TestProfileBProductionPipeline:
    """The real _invoke_for_evaluation body, with NIM mocked at the wire level."""

    @pytest.mark.asyncio
    async def test_happy_path_all_examples_match_ground_truth(self, tmp_path):
        """Two pool members, NIM returns the exact ground truth → run
        completes with 100% Exact Match, one OperationRecord per member
        with purpose='evaluation' and evaluation_run_id set."""
        from vlm_feedback_loop.db.models.operation import OperationRecord
        from vlm_feedback_loop.services.evaluation_service import (
            _execute_evaluation,
        )

        engine, pdir, settings = _setup_project_with_pool(
            tmp_path,
            n_pool=2,
            n_nonpool=0,
        )

        # NIM returns the ground-truth JSON verbatim for every call.
        gt_json = '{"rationale_note":"ok","severity":"high","damaged":true}'
        patches = _patch_nim_pipeline(fake_nim_success(gt_json))

        with (
            patches[0],
            patches[1],
            patched_register(evaluation_service_module),
        ):
            result = await start_evaluation_run(PID, settings=settings)
            run_id = result["run_id"]
            await _execute_evaluation(PID, run_id, settings)

        with Session(engine) as s:
            run = s.query(RunRecord).filter_by(run_id=run_id).first()
            assert run.status == "completed", run.status_reason or "(no reason)"
            assert run.examples_total == 2
            assert run.metrics["overall"]["exact_match_rate"] == 1.0
            assert run.metrics["overall"]["example_count"] == 2
            # Per-status counter aggregation in Phase G. If these counters
            # stay at their 0 default, _compute_parseable_rate reads
            # 0/N = 0.0 and silently blocks failed → partial promotion on
            # live 8B NIM evals. Locking the contract here.
            assert run.examples_succeeded == 2
            assert run.examples_schema_invalid == 0
            assert run.examples_timeout == 0
            assert run.examples_endpoint_error == 0

            records = (
                s.query(OperationRecord)
                .filter_by(evaluation_run_id=run_id, purpose="evaluation")
                .all()
            )
            assert len(records) == 2
            for r in records:
                assert r.invocation_status == "success"
                assert r.schema_valid_core is True
                assert r.exact_match_pass is True
                assert r.model_name == "test-model"
                assert r.guidance_id == GID

    @pytest.mark.asyncio
    async def test_rest_progress_keeps_frozen_total_after_run(self, tmp_path):
        """REST exposes progress during a run and retains its frozen denominator.

        This lets reconnecting clients and terminal reports distinguish an
        incomplete result set from a smaller Test Pool.
        """
        from vlm_feedback_loop.services.evaluation_service import (
            _execute_evaluation,
            get_evaluation_run,
        )

        _engine, _pdir, settings = _setup_project_with_pool(
            tmp_path, n_pool=2, n_nonpool=0
        )

        gt_json = '{"rationale_note":"ok","severity":"high","damaged":true}'
        run_ids: list[str] = []
        mid_run_progress: list[dict | None] = []
        calls = {"n": 0}

        def chat_side_effect(*args, **kwargs):
            # At eval_concurrency=1 the second wire call happens strictly
            # after example 1's outcome landed, so the REST view captured
            # here is a genuine mid-run read.
            calls["n"] += 1
            if calls["n"] == 2:
                detail = get_evaluation_run(PID, run_ids[0], settings)
                assert not isinstance(detail, str)
                mid_run_progress.append(detail["progress"])
            return fake_nim_success(gt_json)

        patches = _patch_nim_pipeline(chat_side_effect)
        with (
            patches[0],
            patches[1],
            patched_register(evaluation_service_module),
        ):
            result = await start_evaluation_run(
                PID, eval_concurrency=1, settings=settings
            )
            run_ids.append(result["run_id"])
            await _execute_evaluation(PID, run_ids[0], settings, eval_concurrency=1)

        assert mid_run_progress == [{"processed": 1, "total": 2}]
        terminal = get_evaluation_run(PID, run_ids[0], settings)
        assert not isinstance(terminal, str)
        assert terminal["progress"] == {"processed": 2, "total": 2}

    @pytest.mark.asyncio
    async def test_schema_invalid_response_finalizes_incomplete(self, tmp_path):
        """NIM returns malformed JSON → every example has schema_valid_core=False
        and no field_matches, so all examples land in Phase D's
        ``persistently_failed_keys`` and the run finalizes as ``incomplete``
        Unlike timeouts (no answer exists to score), a
        schema-invalid response IS the model's answer, so it stays in the
        metric denominator as a scored all-fields miss — the same
        treatment the TAO rescoring path applies (normalized_pred={}), so
        Teacher-vs-Student accuracy on the Compare page is measured with one
        denominator definition."""
        from vlm_feedback_loop.db.models.operation import OperationRecord
        from vlm_feedback_loop.services.evaluation_service import (
            _execute_evaluation,
        )

        engine, pdir, settings = _setup_project_with_pool(
            tmp_path,
            n_pool=2,
            n_nonpool=0,
        )

        patches = _patch_nim_pipeline(fake_nim_success("not valid json at all"))

        with (
            patches[0],
            patches[1],
            patched_register(evaluation_service_module),
        ):
            result = await start_evaluation_run(PID, settings=settings)
            run_id = result["run_id"]
            await _execute_evaluation(PID, run_id, settings)

        with Session(engine) as s:
            run = s.query(RunRecord).filter_by(run_id=run_id).first()
            assert run.status == "incomplete", run.status_reason
            # schema_invalid examples must be counted on the run record so
            # _compute_parseable_rate can correctly compute
            # parseable_rate = succeeded / total and decide partial-promotion.
            # Live evidence: an 8B NIM eval with 83 success + 1 schema_invalid
            # + counters stuck at 0 left parseable_rate=0.0, blocking the
            # failed → partial promotion at the parseable-rate gate. Lock the
            # contract here.
            assert run.examples_succeeded == 0
            assert run.examples_schema_invalid == 2
            assert run.examples_timeout == 0
            assert run.examples_endpoint_error == 0
            records = s.query(OperationRecord).filter_by(evaluation_run_id=run_id).all()
            assert len(records) == 2
            for r in records:
                assert r.invocation_status == "schema_invalid"
                assert r.schema_valid_core is False
                assert r.exact_match_pass is None
            # Scored-miss inclusion: both schema-invalid examples count
            # in the denominator with a 0.0 exact-match rate — not excluded
            # as if they never ran.
            assert run.metrics is not None
            assert run.metrics["overall"]["example_count"] == 2
            assert run.metrics["overall"]["exact_match_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_timeout_triggers_sequential_retry(self, tmp_path):
        """First-pass NIM call times out → Phase C sequential retry kicks
        in → retry succeeds → run completes. Each pool example yields
        two OperationRecords (the failed attempt + the retry)."""
        from vlm_feedback_loop.db.models.operation import OperationRecord
        from vlm_feedback_loop.services.evaluation_service import (
            _execute_evaluation,
        )

        engine, pdir, settings = _setup_project_with_pool(
            tmp_path,
            n_pool=2,
            n_nonpool=0,
        )

        # First 2 calls (concurrent burst) time out; next 2 (sequential
        # retry) succeed with ground-truth JSON.
        gt_json = '{"rationale_note":"ok","severity":"high","damaged":true}'
        call_sequence = [
            fake_nim_failure("Request timed out", status_code=504),
            fake_nim_failure("Request timed out", status_code=504),
            fake_nim_success(gt_json),
            fake_nim_success(gt_json),
        ]
        patches = _patch_nim_pipeline(call_sequence)

        with (
            patches[0],
            patches[1],
            patched_register(evaluation_service_module),
        ):
            result = await start_evaluation_run(PID, settings=settings)
            run_id = result["run_id"]
            await _execute_evaluation(PID, run_id, settings)

        with Session(engine) as s:
            run = s.query(RunRecord).filter_by(run_id=run_id).first()
            # Expect completed — every example has a success after retry.
            assert run.status == "completed", (
                f"status={run.status}, reason={run.status_reason}"
            )
            assert run.examples_succeeded == 2
            assert run.examples_timeout == 0
            assert run.examples_endpoint_error == 0
            records = s.query(OperationRecord).filter_by(evaluation_run_id=run_id).all()
            # 2 timeouts + 2 retries = 4 records total
            assert len(records) == 4
            statuses = sorted(r.invocation_status for r in records)
            assert statuses == ["success", "success", "timeout", "timeout"]

    @pytest.mark.asyncio
    async def test_rate_limited_multipass_retry_completes_pool(self, tmp_path):
        """A 429 that survives the single Phase C pass is retried across the
        bounded Phase C2 multi-pass (backoff=0 in test) until it succeeds, so
        the pool finalizes COMPLETE instead of incomplete. This is the
        rate-limit-aware completion fix: 429 is transient, so don't let it
        truncate the regression pool."""
        from vlm_feedback_loop.services.evaluation_service import _execute_evaluation

        engine, pdir, _ = _setup_project_with_pool(tmp_path, n_pool=1, n_nonpool=0)
        settings = make_stub_settings(
            WORKSPACE_ROOT=str(tmp_path / "workspace"),
            EVAL_RATE_LIMIT_RETRY_MAX_PASSES=3,
            EVAL_RATE_LIMIT_RETRY_BACKOFF_S=0,
        )
        gt_json = '{"rationale_note":"ok","severity":"high","damaged":true}'
        rl = fake_nim_failure("Exhausted 3 retries. Last: HTTP 429")
        # burst 429 -> Phase C pass-1 429 -> Phase C2 extra pass succeeds.
        call_sequence = [rl, rl, fake_nim_success(gt_json)]
        patches = _patch_nim_pipeline(call_sequence)
        with (
            patches[0],
            patches[1],
            patched_register(evaluation_service_module),
        ):
            result = await start_evaluation_run(PID, settings=settings)
            run_id = result["run_id"]
            await _execute_evaluation(PID, run_id, settings)
        with Session(engine) as s:
            run = s.query(RunRecord).filter_by(run_id=run_id).first()
            assert run.status == "completed", (
                f"status={run.status}, reason={run.status_reason}"
            )

    @pytest.mark.asyncio
    async def test_rate_limited_multipass_bounded_finalizes_incomplete(self, tmp_path):
        """When the 429 block is sustained (every attempt rate-limited), the
        Phase C2 multi-pass is BOUNDED: after MAX_PASSES it stops and the run
        finalizes ``incomplete`` rather than hanging."""
        from vlm_feedback_loop.services.evaluation_service import _execute_evaluation

        engine, pdir, _ = _setup_project_with_pool(tmp_path, n_pool=1, n_nonpool=0)
        settings = make_stub_settings(
            WORKSPACE_ROOT=str(tmp_path / "workspace"),
            EVAL_RATE_LIMIT_RETRY_MAX_PASSES=2,
            EVAL_RATE_LIMIT_RETRY_BACKOFF_S=0,
        )
        # Every call 429s (constant). burst + Phase C pass + 2 bounded passes.
        rl = fake_nim_failure("Exhausted 3 retries. Last: HTTP 429")
        patches = _patch_nim_pipeline(rl)
        with (
            patches[0],
            patches[1],
            patched_register(evaluation_service_module),
        ):
            result = await start_evaluation_run(PID, settings=settings)
            run_id = result["run_id"]
            await _execute_evaluation(PID, run_id, settings)
        with Session(engine) as s:
            run = s.query(RunRecord).filter_by(run_id=run_id).first()
            assert run.status == "incomplete", (
                f"status={run.status}, reason={run.status_reason}"
            )
            assert run.examples_succeeded == 0
            assert run.examples_endpoint_error == 1

    @pytest.mark.asyncio
    async def test_partial_field_match_records_per_field_metrics(self, tmp_path):
        """NIM returns JSON that matches 'damaged' but not 'severity'. The
        per-field match rate reflects 100% for 'damaged' and 0% for
        'severity'; exact_match_pass=False on each record."""
        from vlm_feedback_loop.db.models.operation import OperationRecord
        from vlm_feedback_loop.services.evaluation_service import (
            _execute_evaluation,
        )

        engine, pdir, settings = _setup_project_with_pool(
            tmp_path,
            n_pool=2,
            n_nonpool=0,
        )

        wrong_severity = '{"rationale_note":"hm","severity":"low","damaged":true}'
        patches = _patch_nim_pipeline(fake_nim_success(wrong_severity))

        with (
            patches[0],
            patches[1],
            patched_register(evaluation_service_module),
        ):
            result = await start_evaluation_run(PID, settings=settings)
            run_id = result["run_id"]
            await _execute_evaluation(PID, run_id, settings)

        with Session(engine) as s:
            run = s.query(RunRecord).filter_by(run_id=run_id).first()
            assert run.status == "completed", run.status_reason
            overall = run.metrics["overall"]
            assert (
                overall["exact_match_rate"] == 0.0
            )  # neither example matches all core fields
            per_field = overall["per_field_match_rates"]
            assert per_field["damaged"] == 1.0
            assert per_field["severity"] == 0.0

            records = s.query(OperationRecord).filter_by(evaluation_run_id=run_id).all()
            assert len(records) == 2
            for r in records:
                assert r.invocation_status == "success"
                assert r.schema_valid_core is True
                assert r.exact_match_pass is False

    @pytest.mark.asyncio
    async def test_structured_gen_rejection_fails_run_under_auto_mode(self, tmp_path):
        """If a mid-run response_format 4xx rejection happens under
        ``auto`` mode, the whole run MUST fail with
        ``status_reason="structured_generation_rejected"`` — not
        ``incomplete`` — so the UI's "Restart with prompt-only" affordance
        can light up (EvaluationStrip.tsx)."""
        from vlm_feedback_loop.db.models.operation import OperationRecord
        from vlm_feedback_loop.services.evaluation_service import (
            _execute_evaluation,
        )

        engine, pdir, settings = _setup_project_with_pool(
            tmp_path,
            n_pool=2,
            n_nonpool=0,
        )

        # NIM returns a 400 with a response_format complaint for every call.
        # Both concurrent attempts hit the rejection; whichever arrives
        # first signals the run to fail.
        rejection = fake_nim_failure(
            "HTTP 400: response_format json_schema not supported"
        )
        patches = _patch_nim_pipeline(rejection)

        with (
            patches[0],
            patches[1],
            patched_register(evaluation_service_module),
        ):
            result = await start_evaluation_run(PID, settings=settings)
            run_id = result["run_id"]
            await _execute_evaluation(PID, run_id, settings)

        with Session(engine) as s:
            run = s.query(RunRecord).filter_by(run_id=run_id).first()
            assert run.status == "failed"
            assert run.status_reason == "structured_generation_rejected"
            # Metrics NOT persisted on a rejected run (partial aggregation
            # from an aborted burst is not meaningful).
            assert run.metrics is None

            # OperationRecords still persist for audit, carrying the
            # endpoint_error outcome.
            records = s.query(OperationRecord).filter_by(evaluation_run_id=run_id).all()
            assert len(records) >= 1
            for r in records:
                assert r.invocation_status == "endpoint_error"

    @pytest.mark.asyncio
    async def test_structured_gen_rejection_under_prompt_only_does_not_trigger(
        self, tmp_path
    ):
        """Under ``prompt_only`` mode, no response_format is sent, so the
        rejection heuristic MUST NOT trigger. A 400 from NIM under
        prompt_only should surface as plain endpoint_error → run
        finalizes ``incomplete``, not ``failed``."""
        from vlm_feedback_loop.services.evaluation_service import (
            _execute_evaluation,
        )

        engine, pdir, settings = _setup_project_with_pool(
            tmp_path,
            n_pool=2,
            n_nonpool=0,
        )

        rejection = fake_nim_failure(
            "HTTP 400: response_format json_schema not supported"
        )
        patches = _patch_nim_pipeline(rejection)

        with (
            patches[0],
            patches[1],
            patched_register(evaluation_service_module),
        ):
            result = await start_evaluation_run(
                PID,
                structured_generation_mode="prompt_only",
                settings=settings,
            )
            run_id = result["run_id"]
            await _execute_evaluation(PID, run_id, settings)

        with Session(engine) as s:
            run = s.query(RunRecord).filter_by(run_id=run_id).first()
            # prompt_only → structured_generation_attempted=False on the
            # invocation → rejection heuristic returns False → run goes
            # through the normal incomplete path.
            assert run.status == "incomplete"
            assert run.status_reason is None
