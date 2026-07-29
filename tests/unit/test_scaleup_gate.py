# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Scale-Up Readiness Gate, per-bucket Returning/New
metrics, coverage gaps, and log point 3."""

from __future__ import annotations

import copy
import logging

from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified

from conftest import (
    GID,
    MCID,
    PID,
    add_endpoint_and_model_rows,
    add_example_row,
    add_fixture_guidance_row,
    add_standard_project_row,
    make_stub_settings,
    setup_project_db,
)
from vlm_feedback_loop.db.base import generate_uuid4, utc_now
from vlm_feedback_loop.db.models.guidance import Guidance
from vlm_feedback_loop.db.models.label import Label
from vlm_feedback_loop.db.models.project import Project
from vlm_feedback_loop.db.models.run import RunRecord
from vlm_feedback_loop.services.evaluation_service import (
    _agg_to_dict,
    _find_previous_completed_eval,
    _log_gate_evaluation,
    _serialize_metrics_with_buckets,
    compute_accept_rate,
    compute_scaleup_gate,
)
from vlm_feedback_loop.services.exact_match_evaluator import (
    AggregateMetrics,
    FieldMatchResult,
    PerValueMetrics,
    compute_aggregate_metrics,
)

# ── Helpers ──────────────────────────────────────────────────────────────────


def _add_label(
    session,
    project_id,
    key,
    *,
    outcome="Accept",
    pool_assignment=None,
    guidance_id=GID,
    verified_at=None,
):
    session.add(
        Label(
            label_id=generate_uuid4(),
            project_id=project_id,
            example_key=key,
            label_status="verified",
            guidance_id=guidance_id,
            inference_invocation_id=generate_uuid4(),
            label_json={"rationale_note": "test", "severity": "high", "damaged": True},
            labeled_at=utc_now(),
            verified_outcome=outcome,
            verified_at=verified_at or utc_now(),
            edited_core_fields=[],
            edited_aux_fields=[],
            rationale_source="teacher_proposal"
            if outcome == "Accept"
            else "sme_edited",
            pool_assignment=pool_assignment,
        )
    )


def _add_example(session, project_id, key, state="Verified"):
    add_example_row(session, project_id, key, state=state)


def _make_completed_eval(session, project_id, run_id, metrics, pool_id=None, **kw):
    """Insert a completed evaluation RunRecord with the given metrics."""
    session.add(
        RunRecord(
            run_id=run_id,
            project_id=project_id,
            run_type="evaluation_run",
            status="completed",
            examples_total=kw.get("examples_total", 10),
            pool_version_id=pool_id,
            guidance_id=kw.get("guidance_id", GID),
            model_config_id=kw.get("model_config_id", MCID),
            generation_preset_key="precise",
            thinking_mode_effective="on",
            visual_budget_preset_key="balanced",
            evaluation_source=kw.get("evaluation_source", "nim"),
            metrics=metrics,
            icl_eligible_count_at_completion=kw.get(
                "icl_eligible_count_at_completion", 0
            ),
        )
    )


def _force_run_fields(session, run_id, **values):
    """Set RunRecord fields via bulk UPDATE after commit.

    created_at is stamped by the insert hook regardless of constructor
    kwargs, so tests that need deterministic ordering must force it
    post-insert; the bulk UPDATE bypasses the hook.
    """
    session.query(RunRecord).filter_by(run_id=run_id).update(values)
    session.commit()


def _full_gate_metrics(
    em=0.85,
    pf_severity=0.90,
    pf_damaged=0.85,
    pv_high_f1=0.90,
    pv_low_f1=0.80,
    pv_medium_f1=0.80,
    pv_true_f1=0.85,
    pv_false_f1=0.80,
):
    """Build a metrics dict that passes all gate criteria by default."""
    return {
        "overall": {
            "exact_match_rate": em,
            "example_count": 10,
            "per_field_match_rates": {
                "severity": pf_severity,
                "damaged": pf_damaged,
            },
            "per_value_metrics": {
                "severity": {
                    "high": {"precision": 0.90, "recall": 0.90, "f1": pv_high_f1},
                    "low": {"precision": 0.80, "recall": 0.80, "f1": pv_low_f1},
                    "medium": {"precision": 0.80, "recall": 0.80, "f1": pv_medium_f1},
                },
                "damaged": {
                    "true": {"precision": 0.85, "recall": 0.85, "f1": pv_true_f1},
                    "false": {"precision": 0.80, "recall": 0.80, "f1": pv_false_f1},
                },
            },
        },
        "returning": None,
        "new": None,
    }


def _setup_gate_ready(tmp_path, n_pool=25, n_accept=40, n_edit=10):
    """Set up a project that passes all 5 gate criteria."""
    engine, pdir = setup_project_db(tmp_path)
    settings = make_stub_settings(WORKSPACE_ROOT=str(tmp_path / "workspace"))

    with Session(engine) as s:
        add_standard_project_row(s, PID, pdir)
        add_fixture_guidance_row(s)
        add_endpoint_and_model_rows(s)
        # Pool members
        for i in range(n_pool):
            key = f"pool_{i:03d}"
            _add_example(s, PID, key)
            _add_label(s, PID, key, pool_assignment="test_pool", outcome="Accept")
        # Accept labels
        for i in range(n_accept):
            key = f"accept_{i:03d}"
            _add_example(s, PID, key)
            _add_label(s, PID, key, outcome="Accept")
        # Edit labels
        for i in range(n_edit):
            key = f"edit_{i:03d}"
            _add_example(s, PID, key)
            _add_label(s, PID, key, outcome="Edit")
        # Completed evaluation
        _make_completed_eval(s, PID, "eval-1", _full_gate_metrics())
        s.commit()

    return engine, pdir, settings


def _mutate_eval_metrics(engine, mutate_overall) -> None:
    """Rewrite the seeded eval-1 run's metrics: *mutate_overall* edits the
    ``metrics["overall"]`` dict in place. The JSON column needs the deepcopy
    + flag_modified dance — an in-place edit is invisible to the session."""
    with Session(engine) as s:
        run = s.query(RunRecord).filter_by(run_id="eval-1").first()
        assert run is not None
        metrics = copy.deepcopy(run.metrics)
        mutate_overall(metrics["overall"])
        run.metrics = metrics
        flag_modified(run, "metrics")
        s.commit()


# ══════════════════════════════════════════════════════════════════════════════
# Section A: Per-Bucket Metrics Serialization
# ══════════════════════════════════════════════════════════════════════════════


class TestPerBucketMetrics:
    def test_serialize_overall_only(self):
        """When returning/new are None, only overall is present."""
        agg = AggregateMetrics(
            overall_exact_match_rate=0.8,
            example_count=5,
            per_core_field_match_rate={"severity": 0.9},
            per_value_metrics={},
        )
        d = _serialize_metrics_with_buckets(agg, None, None)
        assert d["overall"]["exact_match_rate"] == 0.8
        assert d["returning"] is None
        assert d["new"] is None

    def test_serialize_with_all_buckets(self):
        """All three buckets present when returning and new have data."""
        overall = AggregateMetrics(0.8, 10, {"severity": 0.9}, {})
        returning = AggregateMetrics(0.85, 7, {"severity": 0.92}, {})
        new_agg = AggregateMetrics(0.6, 3, {"severity": 0.7}, {})
        d = _serialize_metrics_with_buckets(overall, returning, new_agg)
        assert d["overall"]["exact_match_rate"] == 0.8
        assert d["returning"]["exact_match_rate"] == 0.85
        assert d["returning"]["example_count"] == 7
        assert d["new"]["exact_match_rate"] == 0.6

    def test_each_bucket_has_full_fields(self):
        """Each bucket reports exact_match_rate, example_count, per_field, per_value."""
        pv = {"severity": {"high": PerValueMetrics(0.9, 0.85, 0.87)}}
        agg = AggregateMetrics(0.8, 5, {"severity": 0.9}, pv)
        d = _agg_to_dict(agg)
        assert "exact_match_rate" in d
        assert "example_count" in d
        assert "per_field_match_rates" in d
        assert "per_value_metrics" in d
        assert "high" in d["per_value_metrics"]["severity"]


# ══════════════════════════════════════════════════════════════════════════════
# Section B: Accept Rate
# ══════════════════════════════════════════════════════════════════════════════


class TestAcceptRate:
    def test_accept_rate_all_accept(self, tmp_path):
        engine, pdir = setup_project_db(tmp_path)
        with Session(engine) as s:
            add_standard_project_row(s, PID, pdir)
            add_fixture_guidance_row(s)
            for i in range(10):
                key = f"img_{i:03d}"
                _add_example(s, PID, key)
                _add_label(s, PID, key, outcome="Accept")
            s.commit()
        with Session(engine) as s:
            rate, denom = compute_accept_rate(s, PID, 50)
        assert rate == 1.0
        assert denom == 10

    def test_accept_rate_mixed(self, tmp_path):
        engine, pdir = setup_project_db(tmp_path)
        with Session(engine) as s:
            add_standard_project_row(s, PID, pdir)
            add_fixture_guidance_row(s)
            for i in range(8):
                key = f"a_{i:03d}"
                _add_example(s, PID, key)
                _add_label(s, PID, key, outcome="Accept")
            for i in range(2):
                key = f"e_{i:03d}"
                _add_example(s, PID, key)
                _add_label(s, PID, key, outcome="Edit")
            s.commit()
        with Session(engine) as s:
            rate, denom = compute_accept_rate(s, PID, 50)
        assert rate == 0.8
        assert denom == 10

    def test_accept_rate_no_labels(self, tmp_path):
        engine, pdir = setup_project_db(tmp_path)
        with Session(engine) as s:
            add_standard_project_row(s, PID, pdir)
            add_fixture_guidance_row(s)
            s.commit()
        with Session(engine) as s:
            rate, denom = compute_accept_rate(s, PID, 50)
        assert rate == 0.0
        assert denom == 0

    def test_accept_rate_window_smaller_than_total(self, tmp_path):
        """The rolling window selects the MOST RECENT labels by verified_at:
        with 5 older Edits and 5 newer Accepts, a window of 5 computes the
        rate over the Accepts only — 1.0, not the Edits' 0.0."""
        engine, pdir = setup_project_db(tmp_path)
        with Session(engine) as s:
            add_standard_project_row(s, PID, pdir)
            add_fixture_guidance_row(s)
            # Explicit distinct timestamps: Edits oldest, Accepts newest.
            for i in range(5):
                key = f"e_{i:03d}"
                _add_example(s, PID, key)
                _add_label(
                    s, PID, key, outcome="Edit", verified_at=f"2026-01-01T00:00:0{i}Z"
                )
            for i in range(5):
                key = f"a_{i:03d}"
                _add_example(s, PID, key)
                _add_label(
                    s, PID, key, outcome="Accept", verified_at=f"2026-01-02T00:00:0{i}Z"
                )
            s.commit()
        with Session(engine) as s:
            rate, denom = compute_accept_rate(s, PID, 5)
        assert rate == 1.0
        assert denom == 5

    def test_accept_rate_window_excludes_older_accepts(self, tmp_path):
        """Recency direction pinned from the other side: when the Edits are
        the NEWEST labels, a window of 5 computes 0.0 even though 5 older
        Accepts exist — the window must never reach back past newer labels."""
        engine, pdir = setup_project_db(tmp_path)
        with Session(engine) as s:
            add_standard_project_row(s, PID, pdir)
            add_fixture_guidance_row(s)
            for i in range(5):
                key = f"a_{i:03d}"
                _add_example(s, PID, key)
                _add_label(
                    s, PID, key, outcome="Accept", verified_at=f"2026-01-01T00:00:0{i}Z"
                )
            for i in range(5):
                key = f"e_{i:03d}"
                _add_example(s, PID, key)
                _add_label(
                    s, PID, key, outcome="Edit", verified_at=f"2026-01-02T00:00:0{i}Z"
                )
            s.commit()
        with Session(engine) as s:
            rate, denom = compute_accept_rate(s, PID, 5)
        assert rate == 0.0
        assert denom == 5


# ══════════════════════════════════════════════════════════════════════════════
# Section C: Scale-Up Readiness Gate — Criteria Logic
# ══════════════════════════════════════════════════════════════════════════════


class TestGateLogic:
    def test_all_criteria_pass_returns_ready(self, tmp_path):
        """When all 5 criteria pass, gate_status is 'ready'."""
        engine, pdir, settings = _setup_gate_ready(tmp_path)
        result = compute_scaleup_gate(PID, settings=settings)
        assert not isinstance(result, str)
        assert result["gate_status"] == "ready"
        assert all(c["passed"] for c in result["criteria"])

    def test_gate_ignores_tao_sourced_evaluations(self, tmp_path):
        """A newer TAO Student evaluation must not become the gate's quality
        basis: the gate measures the Teacher+Guidance+ICL setup, so
        only evaluation_source="nim" runs qualify."""
        engine, pdir, settings = _setup_gate_ready(tmp_path)
        with Session(engine) as s:
            _make_completed_eval(
                s,
                PID,
                "eval-tao",
                metrics={
                    "overall": {
                        "exact_match_rate": 0.10,
                        "example_count": 4,
                        "per_field_match_rates": {},
                        "per_value_metrics": {},
                    },
                    "returning": None,
                    "new": None,
                },
                evaluation_source="tao",
            )
            s.commit()
            # Force the TAO run strictly newer than eval-1 regardless of
            # insert-hook timestamp granularity (bulk update skips the hook).
            _force_run_fields(s, "eval-tao", created_at="2099-01-01T00:00:00Z")
        result = compute_scaleup_gate(PID, settings=settings)
        assert not isinstance(result, str)
        assert result["gate_status"] == "ready"
        em = next(
            c
            for c in result["criteria"]
            if c["criterion_name"] == "overall_exact_match"
        )
        # Quality basis is the NIM run's 85%, not the TAO run's 10%.
        assert em["current_value"] >= 0.8

    def test_gate_ignores_student_nim_serving_evaluations(self, tmp_path):
        """A newer Student NIM serving evaluation must not become the gate's
        quality basis: Student benchmark runs are written with
        evaluation_source="nim" too, so the Teacher discriminator is
        student_model_config_id IS NULL. Deploying and benchmarking a
        weak Student must not flip a ready gate to not_ready."""
        engine, pdir, settings = _setup_gate_ready(tmp_path)
        with Session(engine) as s:
            _make_completed_eval(
                s, PID, "eval-student", metrics=_full_gate_metrics(em=0.30)
            )
            s.commit()
            _force_run_fields(
                s,
                "eval-student",
                created_at="2099-01-01T00:00:00Z",
                student_model_config_id="student-mc-1",
            )
        result = compute_scaleup_gate(PID, settings=settings)
        assert not isinstance(result, str)
        assert result["gate_status"] == "ready"
        em = next(
            c
            for c in result["criteria"]
            if c["criterion_name"] == "overall_exact_match"
        )
        assert em["current_value"] >= 0.8

    def test_gate_uses_newest_completed_teacher_evaluation(self, tmp_path):
        """With two completed Teacher evaluations the gate reads the newest:
        a project that improved from a failing to a passing score must be
        judged on the passing run (pins the created_at DESC ordering)."""
        engine, pdir, settings = _setup_gate_ready(tmp_path)
        with Session(engine) as s:
            _make_completed_eval(
                s, PID, "eval-older-failing", metrics=_full_gate_metrics(em=0.10)
            )
            s.commit()
            _force_run_fields(
                s, "eval-older-failing", created_at="2000-01-01T00:00:00Z"
            )
        result = compute_scaleup_gate(PID, settings=settings)
        assert not isinstance(result, str)
        assert result["gate_status"] == "ready"
        em = next(
            c
            for c in result["criteria"]
            if c["criterion_name"] == "overall_exact_match"
        )
        assert em["current_value"] >= 0.8

    def test_no_eval_fails_overall(self, tmp_path):
        """No completed evaluation → overall_exact_match fails."""
        engine, pdir = setup_project_db(tmp_path)
        settings = make_stub_settings(WORKSPACE_ROOT=str(tmp_path / "workspace"))
        with Session(engine) as s:
            add_standard_project_row(s, PID, pdir)
            add_fixture_guidance_row(s)
            add_endpoint_and_model_rows(s)
            for i in range(25):
                key = f"pool_{i:03d}"
                _add_example(s, PID, key)
                _add_label(s, PID, key, pool_assignment="test_pool")
            s.commit()
        result = compute_scaleup_gate(PID, settings=settings)
        assert result["gate_status"] == "not_ready"
        em_criterion = next(
            c
            for c in result["criteria"]
            if c["criterion_name"] == "overall_exact_match"
        )
        assert em_criterion["passed"] is False
        assert "evaluation" in em_criterion["message"].lower()
        # Structural no-eval marker: the UI's "No evaluation run yet"
        # pending state keys on this flag, not on the message copy
        # (current_value=0.0 is ambiguous with a genuine 0% run).
        assert em_criterion["details"] == {"no_completed_run": True}

    def test_no_eval_marks_dependent_criteria_blocked_by(self, tmp_path):
        """With no completed evaluation, per_field_match and min_per_value_f1
        must be marked with ``details.blocked_by="overall_exact_match"`` so the
        frontend can filter them out of the actionable next-steps list while
        keeping them visible in the full gate-details expander."""
        engine, pdir = setup_project_db(tmp_path)
        settings = make_stub_settings(WORKSPACE_ROOT=str(tmp_path / "workspace"))
        with Session(engine) as s:
            add_standard_project_row(s, PID, pdir)
            add_fixture_guidance_row(s)
            add_endpoint_and_model_rows(s)
            for i in range(25):
                key = f"pool_{i:03d}"
                _add_example(s, PID, key)
                _add_label(s, PID, key, pool_assignment="test_pool")
            s.commit()
        result = compute_scaleup_gate(PID, settings=settings)

        em_criterion = next(
            c
            for c in result["criteria"]
            if c["criterion_name"] == "overall_exact_match"
        )
        # The root criterion keeps its actionable message (no blocked_by).
        assert em_criterion["details"] is None or (
            "blocked_by" not in em_criterion["details"]
        )
        assert "run an evaluation" in em_criterion["message"].lower()

        # Both dependent criteria are blocked by the root.
        pf = next(
            c for c in result["criteria"] if c["criterion_name"] == "per_field_match"
        )
        assert pf["passed"] is False
        assert pf["details"]["blocked_by"] == "overall_exact_match"
        assert pf["message"] == "Depends on evaluation results."

        pv = next(
            c for c in result["criteria"] if c["criterion_name"] == "min_per_value_f1"
        )
        assert pv["passed"] is False
        assert pv["details"]["blocked_by"] == "overall_exact_match"
        assert pv["message"] == "Depends on evaluation results."

    def test_overall_exact_match_fails(self, tmp_path):
        """Overall EM below threshold → fails."""
        engine, pdir, settings = _setup_gate_ready(tmp_path)
        _mutate_eval_metrics(engine, lambda o: o.update({"exact_match_rate": 0.70}))
        result = compute_scaleup_gate(PID, settings=settings)
        assert result["gate_status"] == "not_ready"
        em = next(
            c
            for c in result["criteria"]
            if c["criterion_name"] == "overall_exact_match"
        )
        assert em["passed"] is False
        assert em["current_value"] == 0.7

    def test_per_field_one_fails(self, tmp_path):
        """One core field below threshold → per_field_match fails."""
        engine, pdir, settings = _setup_gate_ready(tmp_path)
        _mutate_eval_metrics(
            engine, lambda o: o["per_field_match_rates"].update({"severity": 0.70})
        )
        result = compute_scaleup_gate(PID, settings=settings)
        pf = next(
            c for c in result["criteria"] if c["criterion_name"] == "per_field_match"
        )
        assert pf["passed"] is False
        assert len(pf["details"]["failing_fields"]) >= 1
        assert pf["details"]["failing_fields"][0]["field_name"] == "severity"

    def test_per_value_f1_one_fails(self, tmp_path):
        """One value F1 below threshold → min_per_value_f1 fails."""
        engine, pdir, settings = _setup_gate_ready(tmp_path)
        _mutate_eval_metrics(
            engine,
            lambda o: o["per_value_metrics"]["severity"]["low"].update({"f1": 0.60}),
        )
        result = compute_scaleup_gate(PID, settings=settings)
        pv = next(
            c for c in result["criteria"] if c["criterion_name"] == "min_per_value_f1"
        )
        assert pv["passed"] is False
        assert len(pv["details"]["failing_values"]) >= 1
        failing = pv["details"]["failing_values"][0]
        assert failing["value"] == "low"
        assert failing["f1"] == 0.6

    def test_per_field_empty_metrics_fails(self, tmp_path):
        """per_field_match must FAIL when the evaluation reports no per-field
        rates at all — an empty mapping means quality is unmeasured, not that
        every field passed."""
        engine, pdir, settings = _setup_gate_ready(tmp_path)
        _mutate_eval_metrics(engine, lambda o: o.update({"per_field_match_rates": {}}))
        result = compute_scaleup_gate(PID, settings=settings)
        pf = next(
            c for c in result["criteria"] if c["criterion_name"] == "per_field_match"
        )
        assert pf["passed"] is False
        assert pf["current_value"] == 0.0

    def test_per_value_empty_metrics_fails(self, tmp_path):
        """min_per_value_f1 must FAIL when the evaluation reports no
        per-value metrics — an empty mapping means quality is unmeasured,
        not that every value passed."""
        engine, pdir, settings = _setup_gate_ready(tmp_path)
        _mutate_eval_metrics(engine, lambda o: o.update({"per_value_metrics": {}}))
        result = compute_scaleup_gate(PID, settings=settings)
        pv = next(
            c for c in result["criteria"] if c["criterion_name"] == "min_per_value_f1"
        )
        assert pv["passed"] is False
        assert pv["current_value"] == 0.0

    def test_overall_exact_match_exactly_at_threshold_passes(self, tmp_path):
        """An exact-match rate exactly equal to the threshold passes the
        criterion — the comparison is at-or-above, not strictly above."""
        engine, pdir, settings = _setup_gate_ready(tmp_path)
        _mutate_eval_metrics(engine, lambda o: o.update({"exact_match_rate": 0.80}))
        result = compute_scaleup_gate(PID, settings=settings)
        em = next(
            c
            for c in result["criteria"]
            if c["criterion_name"] == "overall_exact_match"
        )
        assert em["passed"] is True
        assert em["current_value"] == 0.8
        assert em["threshold"] == 0.8

    def test_min_pool_size_exactly_at_threshold_passes(self, tmp_path):
        """A Test Pool exactly at min_test_pool_size passes the criterion —
        the comparison is at-or-above, not strictly above."""
        engine, pdir, settings = _setup_gate_ready(tmp_path, n_pool=20)
        result = compute_scaleup_gate(PID, settings=settings)
        ps = next(
            c for c in result["criteria"] if c["criterion_name"] == "min_test_pool_size"
        )
        assert ps["passed"] is True
        assert ps["current_value"] == 20
        assert ps["threshold"] == 20

    def test_accept_rate_exactly_at_threshold_passes(self, tmp_path):
        """An Accept rate exactly equal to the threshold passes the criterion
        — the comparison is at-or-above, not strictly above. 4 Accepts and
        1 Edit (window covers all 5) give exactly the 0.80 default."""
        engine, pdir = setup_project_db(tmp_path)
        settings = make_stub_settings(WORKSPACE_ROOT=str(tmp_path / "workspace"))
        with Session(engine) as s:
            add_standard_project_row(s, PID, pdir)
            add_fixture_guidance_row(s)
            add_endpoint_and_model_rows(s)
            for i in range(4):
                key = f"a_{i:03d}"
                _add_example(s, PID, key)
                _add_label(s, PID, key, outcome="Accept")
            _add_example(s, PID, "e_000")
            _add_label(s, PID, "e_000", outcome="Edit")
            s.commit()
        result = compute_scaleup_gate(PID, settings=settings)
        ar = next(c for c in result["criteria"] if c["criterion_name"] == "accept_rate")
        assert ar["passed"] is True
        assert ar["current_value"] == 0.8
        assert ar["threshold"] == 0.8

    def test_accept_rate_fails(self, tmp_path):
        """Accept rate below threshold → fails.

        All labels are Edit, so accept rate is 0%.
        """
        engine, pdir = setup_project_db(tmp_path)
        settings = make_stub_settings(WORKSPACE_ROOT=str(tmp_path / "workspace"))
        with Session(engine) as s:
            add_standard_project_row(s, PID, pdir)
            add_fixture_guidance_row(s)
            add_endpoint_and_model_rows(s)
            # Pool members (all Edit so they count against accept rate)
            for i in range(25):
                key = f"pool_{i:03d}"
                _add_example(s, PID, key)
                _add_label(s, PID, key, pool_assignment="test_pool", outcome="Edit")
            _make_completed_eval(s, PID, "eval-1", _full_gate_metrics())
            s.commit()
        result = compute_scaleup_gate(PID, settings=settings)
        ar = next(c for c in result["criteria"] if c["criterion_name"] == "accept_rate")
        assert ar["passed"] is False
        assert ar["current_value"] == 0.0

    def test_min_pool_size_fails(self, tmp_path):
        """Pool below min_test_pool_size → fails."""
        engine, pdir = setup_project_db(tmp_path)
        settings = make_stub_settings(WORKSPACE_ROOT=str(tmp_path / "workspace"))
        with Session(engine) as s:
            add_standard_project_row(s, PID, pdir, scaleup_min_test_pool_size=30)
            add_fixture_guidance_row(s)
            add_endpoint_and_model_rows(s)
            for i in range(10):
                key = f"pool_{i:03d}"
                _add_example(s, PID, key)
                _add_label(s, PID, key, pool_assignment="test_pool")
            _make_completed_eval(s, PID, "eval-1", _full_gate_metrics())
            s.commit()
        result = compute_scaleup_gate(PID, settings=settings)
        ps = next(
            c for c in result["criteria"] if c["criterion_name"] == "min_test_pool_size"
        )
        assert ps["passed"] is False
        assert ps["current_value"] == 10
        assert ps["threshold"] == 30

    def test_min_pool_criterion_discloses_growth_target(self, tmp_path):
        """The min_test_pool_size criterion's details carry the
        growth target floor(total_verified × test_pool_fraction) so the
        UI can disclose that verifying more labels will grow the holdout
        — the 'fixed benchmark' must never re-base silently."""
        engine, pdir = setup_project_db(tmp_path)
        settings = make_stub_settings(WORKSPACE_ROOT=str(tmp_path / "workspace"))
        with Session(engine) as s:
            # 10 pool members + 20 non-pool verified = 30 verified total;
            # fraction 0.4 → target floor(30 × 0.4) = 12 > pool_count 10.
            add_standard_project_row(
                s, PID, pdir, scaleup_min_test_pool_size=5, test_pool_fraction=0.4
            )
            add_fixture_guidance_row(s)
            add_endpoint_and_model_rows(s)
            for i in range(10):
                key = f"pool_{i:03d}"
                _add_example(s, PID, key)
                _add_label(s, PID, key, pool_assignment="test_pool")
            for i in range(20):
                key = f"train_{i:03d}"
                _add_example(s, PID, key)
                _add_label(s, PID, key, pool_assignment=None)
            _make_completed_eval(s, PID, "eval-1", _full_gate_metrics())
            s.commit()
        result = compute_scaleup_gate(PID, settings=settings)
        ps = next(
            c for c in result["criteria"] if c["criterion_name"] == "min_test_pool_size"
        )
        assert ps["details"] == {
            "pool_target": 12,
            "test_pool_fraction": 0.4,
            "total_verified": 30,
        }

    def test_five_criterion_names(self, tmp_path):
        """Gate response contains exactly 5 named criteria."""
        engine, pdir, settings = _setup_gate_ready(tmp_path)
        result = compute_scaleup_gate(PID, settings=settings)
        names = [c["criterion_name"] for c in result["criteria"]]
        assert set(names) == {
            "overall_exact_match",
            "per_field_match",
            "min_per_value_f1",
            "accept_rate",
            "min_test_pool_size",
        }

    def test_plain_language_messages(self, tmp_path):
        """Messages are plain language (no technical jargon)."""
        engine, pdir, settings = _setup_gate_ready(tmp_path)
        result = compute_scaleup_gate(PID, settings=settings)
        for c in result["criteria"]:
            assert isinstance(c["message"], str)
            assert len(c["message"]) > 10  # non-trivial message

    def test_thresholds_configurable(self, tmp_path):
        """Custom thresholds on the Project record are respected."""
        engine, pdir = setup_project_db(tmp_path)
        settings = make_stub_settings(WORKSPACE_ROOT=str(tmp_path / "workspace"))
        with Session(engine) as s:
            add_standard_project_row(
                s,
                PID,
                pdir,
                scaleup_exact_match_threshold=0.95,
                scaleup_per_field_match_threshold=0.95,
                scaleup_min_per_value_f1_threshold=0.95,
                scaleup_accept_rate_threshold=0.95,
                scaleup_min_test_pool_size=50,
            )
            add_fixture_guidance_row(s)
            add_endpoint_and_model_rows(s)
            for i in range(25):
                key = f"pool_{i:03d}"
                _add_example(s, PID, key)
                _add_label(s, PID, key, pool_assignment="test_pool")
            for i in range(50):
                _add_example(s, PID, f"a_{i}")
                _add_label(s, PID, f"a_{i}", outcome="Accept")
            _make_completed_eval(s, PID, "eval-1", _full_gate_metrics(em=0.85))
            s.commit()
        result = compute_scaleup_gate(PID, settings=settings)
        # All criteria should fail with stricter thresholds
        assert result["gate_status"] == "not_ready"
        em = next(
            c
            for c in result["criteria"]
            if c["criterion_name"] == "overall_exact_match"
        )
        assert em["threshold"] == 0.95

    def test_gate_lightweight_no_model_invocation(self, tmp_path):
        """Gate computation is just queries — no external calls needed."""
        engine, pdir, settings = _setup_gate_ready(tmp_path)
        # Just verify it works without any mock/external service
        result = compute_scaleup_gate(PID, settings=settings)
        assert isinstance(result, dict)
        assert "evaluated_at" in result

    def test_gate_reevaluates_on_each_call(self, tmp_path):
        """Each call recomputes — not cached."""
        engine, pdir, settings = _setup_gate_ready(tmp_path)
        r1 = compute_scaleup_gate(PID, settings=settings)
        # Change a metric to make it fail
        _mutate_eval_metrics(engine, lambda o: o.update({"exact_match_rate": 0.50}))
        r2 = compute_scaleup_gate(PID, settings=settings)
        assert r1["gate_status"] == "ready"
        assert r2["gate_status"] == "not_ready"


# ══════════════════════════════════════════════════════════════════════════════
# Section D: Log Point 3 (gate_evaluation)
# ══════════════════════════════════════════════════════════════════════════════


class TestLogPoint3:
    def test_gate_log_emits_info_with_criteria(self, caplog):
        """Log point 3: component=gate_evaluation, INFO level, criteria detail."""
        criteria = [
            {
                "criterion_name": "overall_exact_match",
                "passed": True,
                "current_value": 0.85,
                "threshold": 0.80,
                "message": "ok",
            },
            {
                "criterion_name": "accept_rate",
                "passed": False,
                "current_value": 0.70,
                "threshold": 0.80,
                "message": "low",
            },
        ]
        with caplog.at_level(
            logging.DEBUG, logger="vlm_feedback_loop.evaluation_service"
        ):
            _log_gate_evaluation(PID, "not_ready", criteria)
        assert len(caplog.records) == 1
        record = caplog.records[0]
        assert record.levelno == logging.INFO
        assert "gate_evaluation" in record.__dict__.get("component", "")
        details = record.__dict__.get("details", {})
        assert details["gate_status"] == "not_ready"
        assert len(details["criteria"]) == 2

    def test_gate_log_full_run(self, tmp_path):
        """compute_scaleup_gate emits log point 3 on every call."""
        engine, pdir, settings = _setup_gate_ready(tmp_path)
        # Just verify it runs without error and produces a result
        result = compute_scaleup_gate(PID, settings=settings)
        assert result["gate_status"] == "ready"
        assert "evaluated_at" in result


# ══════════════════════════════════════════════════════════════════════════════
# Section E: Returning vs New metric buckets (integration)
# ══════════════════════════════════════════════════════════════════════════════


class TestReturningNewBuckets:
    def test_first_eval_overall_only(self):
        """First evaluation: returning and new are None."""
        d = _serialize_metrics_with_buckets(
            AggregateMetrics(0.8, 5, {"severity": 0.9}, {}),
            None,
            None,
        )
        assert d["overall"] is not None
        assert d["returning"] is None
        assert d["new"] is None

    def test_returning_new_with_matching_fields(self):
        """compute_aggregate_metrics correctly produces per-bucket results."""
        core_fields = [
            {
                "field_name": "severity",
                "type": "enum",
                "role": "core",
                "allowed_values": ["low", "high"],
            },
        ]
        # 3 returning examples: 2 match, 1 mismatch
        returning_results = [
            [FieldMatchResult("severity", True, "high", "high")],
            [FieldMatchResult("severity", True, "high", "high")],
            [FieldMatchResult("severity", False, "low", "high")],
        ]
        # 2 new examples: 1 match, 1 mismatch
        new_results = [
            [FieldMatchResult("severity", True, "high", "high")],
            [FieldMatchResult("severity", False, "low", "high")],
        ]
        all_results = returning_results + new_results

        agg_overall = compute_aggregate_metrics(all_results, core_fields)
        agg_ret = compute_aggregate_metrics(returning_results, core_fields)
        agg_new = compute_aggregate_metrics(new_results, core_fields)

        d = _serialize_metrics_with_buckets(agg_overall, agg_ret, agg_new)

        assert d["overall"]["example_count"] == 5
        assert d["overall"]["exact_match_rate"] == 0.6  # 3/5
        assert d["returning"]["example_count"] == 3
        assert abs(d["returning"]["exact_match_rate"] - 2 / 3) < 0.01
        assert d["new"]["example_count"] == 2
        assert d["new"]["exact_match_rate"] == 0.5  # 1/2


# ══════════════════════════════════════════════════════════════════════════════
# Section F: Previous-evaluation baseline selection (Returning/New source)
# ══════════════════════════════════════════════════════════════════════════════


class TestPreviousEvalBaseline:
    """Pins which run qualifies as the Returning/New comparison baseline."""

    def _setup(self, tmp_path):
        engine, pdir = setup_project_db(tmp_path)
        settings = make_stub_settings(WORKSPACE_ROOT=str(tmp_path / "workspace"))
        with Session(engine) as s:
            add_standard_project_row(s, PID, pdir)
            add_fixture_guidance_row(s)
            add_endpoint_and_model_rows(s)
            s.commit()
        return engine, settings

    @staticmethod
    def _metrics(em):
        return {
            "overall": {
                "exact_match_rate": em,
                "example_count": 10,
                "per_field_match_rates": {},
                "per_value_metrics": {},
            },
            "returning": None,
            "new": None,
        }

    def test_previous_baseline_skips_student_serving_runs(self, tmp_path):
        """The Returning/New baseline is the Teacher's regression signal: a
        newer Student NIM serving run (same evaluation_source="nim" but
        student_model_config_id set) must not become the "Previous"
        comparison for the next Teacher evaluation."""
        engine, _ = self._setup(tmp_path)
        with Session(engine) as s:
            _make_completed_eval(
                s, PID, "eval-teacher", metrics=self._metrics(0.9), pool_id="pool-1"
            )
            _make_completed_eval(
                s, PID, "eval-student", metrics=self._metrics(0.3), pool_id="pool-2"
            )
            s.commit()
            _force_run_fields(s, "eval-teacher", created_at="2026-01-01T00:00:00Z")
            _force_run_fields(
                s,
                "eval-student",
                created_at="2099-01-01T00:00:00Z",
                student_model_config_id="student-mc-1",
            )
        prev_run, _pool = _find_previous_completed_eval(engine, PID, "next-run")
        assert prev_run is not None
        assert prev_run.run_id == "eval-teacher"

    def test_previous_baseline_uses_newest_teacher_run(self, tmp_path):
        """With two completed Teacher evaluations, the newest is the
        baseline (pins the created_at DESC ordering)."""
        engine, _ = self._setup(tmp_path)
        with Session(engine) as s:
            _make_completed_eval(
                s, PID, "eval-old", metrics=self._metrics(0.5), pool_id="pool-1"
            )
            _make_completed_eval(
                s, PID, "eval-new", metrics=self._metrics(0.9), pool_id="pool-2"
            )
            s.commit()
            _force_run_fields(s, "eval-old", created_at="2000-01-01T00:00:00Z")
            _force_run_fields(s, "eval-new", created_at="2026-01-01T00:00:00Z")
        prev_run, _pool = _find_previous_completed_eval(engine, PID, "next-run")
        assert prev_run is not None
        assert prev_run.run_id == "eval-new"

    def test_previous_baseline_resets_at_semantic_core_change(self, tmp_path):
        """A semantic Core change re-labels the project; comparing
        "same images, then vs now" across that boundary is meaningless, so
        the first evaluation of the new schema era has no baseline."""
        engine, _ = self._setup(tmp_path)
        with Session(engine) as s:
            _make_completed_eval(
                s, PID, "eval-old-era", metrics=self._metrics(0.9), pool_id="pool-1"
            )
            # Guidance v2 created by a semantic Core change from v1 (GID).
            add_fixture_guidance_row(
                s,
                PID,
                "guidance-v2",
                version_number=2,
                semantic_core_change_from_guidance_id=GID,
            )
            s.commit()
        prev_run, prev_pool = _find_previous_completed_eval(engine, PID, "next-run")
        assert prev_run is None
        assert prev_pool is None

    def test_previous_baseline_accepts_run_at_era_floor(self, tmp_path):
        """A run recorded under the era-floor Guidance itself (the version
        born from the semantic Core change) IS the new era and must
        qualify as the baseline — otherwise Returning/New would stay
        disabled for the entire post-change era."""
        engine, _ = self._setup(tmp_path)
        with Session(engine) as s:
            add_fixture_guidance_row(
                s,
                PID,
                "guidance-v2",
                version_number=2,
                semantic_core_change_from_guidance_id=GID,
            )
            _make_completed_eval(
                s,
                PID,
                "eval-new-era",
                metrics=self._metrics(0.8),
                pool_id="pool-2",
                guidance_id="guidance-v2",
            )
            s.commit()
        prev_run, _pool = _find_previous_completed_eval(engine, PID, "next-run")
        assert prev_run is not None
        assert prev_run.run_id == "eval-new-era"


# ══════════════════════════════════════════════════════════════════════════════
# Section G: Criterion 3 (min per-value F1) applicability
# ══════════════════════════════════════════════════════════════════════════════


class TestCriterion3Applicability:
    @staticmethod
    def _string_only_metrics(em=1.0):
        return {
            "overall": {
                "exact_match_rate": em,
                "example_count": 25,
                "per_field_match_rates": {"product_name": 1.0, "unit_count": 1.0},
                "per_value_metrics": {},
            },
            "returning": None,
            "new": None,
        }

    def test_gate_passes_criterion3_without_categorical_core_fields(self, tmp_path):
        """Gate criterion 3 quantifies over "every value of every
        categorical Core field" — a legal string/integer-only Core
        schema satisfies it vacuously. Previously such projects could
        never reach gate_status='ready' (the criterion failed forever)
        while its message simultaneously claimed 'Passed.'"""
        engine, pdir, settings = _setup_gate_ready(tmp_path)
        with Session(engine) as s:
            g = s.query(Guidance).filter_by(guidance_id=GID).first()
            assert g is not None
            schema = dict(g.schema)
            schema["fields"] = [
                {
                    "field_id": "f0",
                    "field_name": "rationale_note",
                    "type": "string",
                    "role": "aux",
                },
                {
                    "field_id": "f1",
                    "field_name": "product_name",
                    "type": "string",
                    "role": "core",
                },
                {
                    "field_id": "f2",
                    "field_name": "unit_count",
                    "type": "integer",
                    "role": "core",
                },
            ]
            g.schema = schema
            _make_completed_eval(
                s, PID, "eval-string-only", metrics=self._string_only_metrics()
            )
            s.commit()
            _force_run_fields(s, "eval-string-only", created_at="2099-01-01T00:00:00Z")
        result = compute_scaleup_gate(PID, settings=settings)
        assert not isinstance(result, str)
        c3 = next(
            c for c in result["criteria"] if c["criterion_name"] == "min_per_value_f1"
        )
        assert c3["passed"] is True
        assert "does not apply" in c3["message"]
        assert result["gate_status"] == "ready"

    def test_gate_criterion3_degenerate_eval_fails_without_passed_message(
        self, tmp_path
    ):
        """A categorical Core schema whose newest evaluation reported no
        per-value metrics is a degenerate run: the criterion fails, and
        the message must not contradict the flag by claiming 'Passed.'"""
        engine, pdir, settings = _setup_gate_ready(tmp_path)
        with Session(engine) as s:
            # Schema keeps its categorical Core fields (fixture default);
            # the eval carries empty per-value metrics.
            _make_completed_eval(
                s, PID, "eval-degenerate", metrics=self._string_only_metrics()
            )
            s.commit()
            _force_run_fields(s, "eval-degenerate", created_at="2099-01-01T00:00:00Z")
        result = compute_scaleup_gate(PID, settings=settings)
        assert not isinstance(result, str)
        c3 = next(
            c for c in result["criteria"] if c["criterion_name"] == "min_per_value_f1"
        )
        assert c3["passed"] is False
        assert "Passed." not in c3["message"]


class TestGateEraScope:
    def test_gate_does_not_read_prior_era_evaluations(self, tmp_path):
        """After a semantic Core change, the prior era's passing
        evaluation must not be the gate's quality basis: it scored
        labels the system deleted. Once the SME re-labels
        enough to refill the pool and the accept window, a stale-era
        pass would otherwise hard-unlock Batch Labeling with zero
        evaluations under the new schema."""
        engine, pdir, settings = _setup_gate_ready(tmp_path)
        with Session(engine) as s:
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
        result = compute_scaleup_gate(PID, settings=settings)
        assert not isinstance(result, str)
        assert result["gate_status"] == "not_ready"
        em = next(
            c
            for c in result["criteria"]
            if c["criterion_name"] == "overall_exact_match"
        )
        assert em["passed"] is False
        assert "No completed evaluation" in em["message"]
