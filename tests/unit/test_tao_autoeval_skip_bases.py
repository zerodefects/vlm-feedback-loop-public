# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Auto-skip TAO ``evaluate`` for known-broken base models.

Without the skip, affected chains auto-submit doomed ``evaluate`` jobs.
Cosmos-Reason2-8B hits an upstream weight-init loader gap; Cosmos 3 reasoners
return unusable predictions. The failed jobs waste TAO compute, leave Students
at ``quality_status="failed"``, and obscure the supported local NIM fallback.

The auto-skip routes around it: when the trained base is in
``Settings.TAO_AUTOEVAL_SKIP_BASES``, the polling service's chain-advance path
marks the planned ``evaluate`` TAOJob ``status="canceled"`` with a
``chain_halted_reason`` carrying the persisted auto-skip marker and
continues to the next eligible chain member. Quantize siblings (which parent
on the still-succeeded train, not on evaluate) remain eligible by the
chain-isolation rule. The Student lands at ``quality_status="pending"``,
cleanly routed through the cold-start NIM-eval-fallback branch instead of the
failure-branch pattern-match path.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.orm import Session

from conftest import (
    add_dataset_export_row,
    add_endpoint_row,
    add_guidance_row,
    add_model_config_row,
    add_project_row,
    add_tao_job_row,
    make_settings,
)
from vlm_feedback_loop.db.base import generate_uuid4
from vlm_feedback_loop.db.engine import open_project_db
from vlm_feedback_loop.db.models.model_config import ModelConfig
from vlm_feedback_loop.db.models.tao_job import TAOJob
from vlm_feedback_loop.db.models.training_suite import TrainingSuite
from vlm_feedback_loop.model_catalog_constants import (
    COSMOS3_NANO_REASONER,
    COSMOS_REASON2_2B,
    COSMOS_REASON2_8B,
)
from vlm_feedback_loop.services import student_model_service, tao_job_service
from vlm_feedback_loop.services.project_service import (
    set_project_engine,
)
from vlm_feedback_loop.services.tao_polling_service import (
    AUTO_SKIP_REASON_PREFIX,
    _advance_after_terminal,
    _migrate_legacy_lora_baseline_skips,
    _resolve_trained_base_name,
    _trained_base_is_blocklisted,
)

PID = "proj-autoskip"
GID = "g-autoskip"
MC_8B = "mc-8b"
MC_2B = "mc-2b"
MC_NANO = "mc-nano"
CHAIN = "chain-autoskip"


def _make_settings(workspace: Path, *, skip_bases: list[str] | None = None):
    overrides = {} if skip_bases is None else {"TAO_AUTOEVAL_SKIP_BASES": skip_bases}
    return make_settings(workspace, **overrides)


def _seed_project(tmp_path: Path):
    """Minimal project with two seeded student_base ModelConfigs (2B + 8B)."""
    workspace = tmp_path / "workspace"
    pdir = workspace / "projects" / PID
    pdir.mkdir(parents=True, exist_ok=True)
    engine = open_project_db(pdir)
    set_project_engine(PID, engine)
    with Session(engine) as s:
        add_project_row(s, PID, str(pdir))
        add_endpoint_row(s, PID, "ep-1")
        add_model_config_row(
            s,
            PID,
            MC_8B,
            "ep-1",
            model_name=COSMOS_REASON2_8B,
            eligible_roles=json.dumps(["student_base"]),
        )
        add_model_config_row(
            s,
            PID,
            MC_2B,
            "ep-1",
            model_name=COSMOS_REASON2_2B,
            eligible_roles=json.dumps(["student_base"]),
        )
        add_model_config_row(
            s,
            PID,
            MC_NANO,
            "ep-1",
            model_name=COSMOS3_NANO_REASONER,
            eligible_roles=json.dumps(["student_base"]),
        )
        add_guidance_row(s, PID, GID, {"fields": []})
        add_dataset_export_row(s, PID, "de-train", guidance_id=GID)
        s.commit()
    return engine, workspace, pdir


def _seed_chain(
    engine,
    *,
    chain_id: str,
    train_status: str,
    base_mc_id: str,
    add_quantize: bool = False,
    lora: bool = False,
):
    """Seed train (succeeded) + evaluate (not_started) [+ optional quantize].

    ``lora=True`` stamps the production ``job_config.lora_config`` shape
    (``enable_lora=true``) on the chain rows, as
    ``training_suite_service`` does for adapter-only suites.
    """
    train_id = generate_uuid4()
    eval_id = generate_uuid4()
    quant_id = generate_uuid4() if add_quantize else None
    suite_id = generate_uuid4()
    lora_job_config = (
        {"lora_config": {"enable_lora": True, "lora_rank": 16}} if lora else {}
    )
    with Session(engine) as s:
        s.add(
            TrainingSuite(
                training_suite_id=suite_id,
                project_id=PID,
                idempotency_key=f"idem-{chain_id}",
                guidance_id=GID,
                training_preset="standard",
                export_field_mode="all",
                include_auto_labeled=False,
                training_dataset_export_id="de-train",
                evaluation_dataset_export_id="de-train",
                selected_student_base_model_config_ids=[base_mc_id],
                quantization_schemes=[],
                chain_ids_ordered=[chain_id],
                status="running",
            )
        )
        add_tao_job_row(
            s,
            PID,
            train_id,
            action="train",
            status=train_status,
            student_base_model_config_id=base_mc_id,
            dataset_export_ids=["de-train"],
            job_config=dict(lora_job_config),
            tao_create_job_request={},
            outputs={},
            tao_external_job_id="ext-tr",
            chain_id=chain_id,
            chain_sequence=1,
        )
        add_tao_job_row(
            s,
            PID,
            eval_id,
            action="evaluate",
            student_base_model_config_id=base_mc_id,
            dataset_export_ids=["de-train"],
            job_config=dict(lora_job_config),
            tao_create_job_request={},
            outputs={},
            parent_tao_job_id=train_id,
            chain_id=chain_id,
            chain_sequence=2,
        )
        if add_quantize:
            add_tao_job_row(
                s,
                PID,
                quant_id,
                action="quantize",
                student_base_model_config_id=base_mc_id,
                dataset_export_ids=["de-train"],
                job_config={"quantization_method": "W8A16"},
                tao_create_job_request={},
                outputs={},
                parent_tao_job_id=train_id,
                chain_id=chain_id,
                chain_sequence=3,
            )
        s.commit()
    return train_id, eval_id, quant_id


class TestResolveAndBlocklistHelpers:
    def test_resolve_trained_base_name_returns_model_name(self, tmp_path):
        engine, _ws, _pdir = _seed_project(tmp_path)
        train_id, _, _ = _seed_chain(
            engine, chain_id=CHAIN, train_status="succeeded", base_mc_id=MC_8B
        )
        with Session(engine) as s:
            job = s.query(TAOJob).filter_by(tao_job_id=train_id).one()
            assert _resolve_trained_base_name(s, job) == COSMOS_REASON2_8B

    def test_resolve_returns_none_when_mc_missing(self, tmp_path):
        engine, _ws, _pdir = _seed_project(tmp_path)
        train_id, _, _ = _seed_chain(
            engine, chain_id=CHAIN, train_status="succeeded", base_mc_id=MC_8B
        )
        with Session(engine) as s:
            # Defensive: simulate a deleted ModelConfig row underneath an
            # in-flight chain. The skip helper must return None (treated as
            # not-blocklisted by the caller — fail-safe).
            s.query(ModelConfig).filter_by(model_config_id=MC_8B).delete()
            s.commit()
            job = s.query(TAOJob).filter_by(tao_job_id=train_id).one()
            assert _resolve_trained_base_name(s, job) is None

    def test_blocklist_match(self, tmp_path):
        engine, _ws, _pdir = _seed_project(tmp_path)
        train_id, _, _ = _seed_chain(
            engine, chain_id=CHAIN, train_status="succeeded", base_mc_id=MC_8B
        )
        with Session(engine) as s:
            job = s.query(TAOJob).filter_by(tao_job_id=train_id).one()
            assert _trained_base_is_blocklisted(s, job, frozenset({COSMOS_REASON2_8B}))
            assert not _trained_base_is_blocklisted(
                s, job, frozenset({COSMOS_REASON2_2B})
            )
            assert not _trained_base_is_blocklisted(s, job, frozenset())


class TestBaseBlocklistSkipPath:
    @pytest.mark.asyncio
    async def test_8b_evaluate_auto_skipped_when_default_settings(
        self, tmp_path, monkeypatch
    ):
        """Default behavior: the 8B chain's post-train evaluate is auto-canceled.

        Verifies (a) the evaluate TAOJob lands ``status="canceled"`` with a
        ``chain_halted_reason`` carrying the auto-skip marker, (b)
        ``submit_chain_job`` is NOT
        called for it (no doomed TAO POST), (c) the train Student is left
        at ``quality_status="pending"`` (cold-start path).
        """
        engine, workspace, _pdir = _seed_project(tmp_path)
        train_id, eval_id, _ = _seed_chain(
            engine, chain_id=CHAIN, train_status="succeeded", base_mc_id=MC_8B
        )
        settings = _make_settings(workspace)
        submit_mock = AsyncMock()
        monkeypatch.setattr(tao_job_service, "submit_chain_job", submit_mock)

        await _advance_after_terminal(
            PID,
            chain_id=CHAIN,
            chain_sequence=1,
            engine=engine,
            settings=settings,
        )

        with Session(engine) as s:
            eval_row = s.query(TAOJob).filter_by(tao_job_id=eval_id).one()
            assert eval_row.status == "canceled"
            assert (eval_row.chain_halted_reason or "").startswith(
                AUTO_SKIP_REASON_PREFIX
            )
            assert COSMOS_REASON2_8B in (eval_row.chain_halted_reason or "")
        submit_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_cr3_nano_evaluate_auto_skipped_when_default_settings(
        self, tmp_path, monkeypatch
    ):
        """CR3 reasoners ship in the default skip list: cosmos-rl evaluates
        them via the generic HFModel/qwen3_vl fallback whose freeform decode
        is unreliable (NVIDIA engineering ticket, 2026-07-14) — without the
        default, every CR3 chain wedges on a doomed evaluate and the FP8/W4A16
        quantize legs never produce artifacts."""
        engine, workspace, _pdir = _seed_project(tmp_path)
        train_id, eval_id, _ = _seed_chain(
            engine, chain_id=CHAIN, train_status="succeeded", base_mc_id=MC_NANO
        )
        settings = _make_settings(workspace)  # default skip list incl. CR3
        submit_mock = AsyncMock()
        monkeypatch.setattr(tao_job_service, "submit_chain_job", submit_mock)

        await _advance_after_terminal(
            PID,
            chain_id=CHAIN,
            chain_sequence=1,
            engine=engine,
            settings=settings,
        )

        with Session(engine) as s:
            eval_row = s.query(TAOJob).filter_by(tao_job_id=eval_id).one()
            assert eval_row.status == "canceled"
            assert (eval_row.chain_halted_reason or "").startswith(
                AUTO_SKIP_REASON_PREFIX
            )
            assert COSMOS3_NANO_REASONER in (eval_row.chain_halted_reason or "")
        submit_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_8b_evaluate_skip_finalizes_suite_completed_not_failed(
        self, tmp_path, monkeypatch
    ):
        """An auto-skipped evaluate must not fail the suite.

        Regression: the skip lands status="canceled", and _roll_up_suite_status
        counts canceled as a failure — so out of the box every 8B suite showed
        terminal "failed" even though train succeeded and the evaluate was
        skipped on purpose. Skip-canceled jobs are treated as
        success-equivalent in the roll-up, so a chain of
        train(succeeded)+evaluate(auto-skipped) finalizes "completed".
        """
        engine, workspace, _pdir = _seed_project(tmp_path)
        train_id, _eval_id, _ = _seed_chain(
            engine, chain_id=CHAIN, train_status="succeeded", base_mc_id=MC_8B
        )
        settings = _make_settings(workspace)
        monkeypatch.setattr(tao_job_service, "submit_chain_job", AsyncMock())

        await _advance_after_terminal(
            PID,
            chain_id=CHAIN,
            chain_sequence=1,
            engine=engine,
            settings=settings,
        )

        with Session(engine) as s:
            suite = s.query(TrainingSuite).filter_by(project_id=PID).one()
            assert suite.status == "completed"
            assert suite.completed_at is not None

    @pytest.mark.asyncio
    async def test_8b_chain_with_quantize_advances_to_quantize_not_evaluate(
        self, tmp_path, monkeypatch
    ):
        """When a quantize sibling exists, the auto-skip cancels the evaluate and
        submits the quantize instead (chain-isolation rule — quantize
        parents on train, not evaluate)."""
        engine, workspace, _pdir = _seed_project(tmp_path)
        train_id, eval_id, quant_id = _seed_chain(
            engine,
            chain_id=CHAIN,
            train_status="succeeded",
            base_mc_id=MC_8B,
            add_quantize=True,
        )
        settings = _make_settings(workspace)
        submit_mock = AsyncMock()
        monkeypatch.setattr(tao_job_service, "submit_chain_job", submit_mock)

        await _advance_after_terminal(
            PID,
            chain_id=CHAIN,
            chain_sequence=1,
            engine=engine,
            settings=settings,
        )

        with Session(engine) as s:
            eval_row = s.query(TAOJob).filter_by(tao_job_id=eval_id).one()
            assert eval_row.status == "canceled"
            quant_row = s.query(TAOJob).filter_by(tao_job_id=quant_id).one()
            # The skip must NOT touch the quantize — it stays not_started for the
            # chain-advance to submit.
            assert quant_row.status == "not_started"
        # The advance call submitted the quantize, NOT the evaluate.
        submit_mock.assert_called_once_with(PID, quant_id, settings=settings)

    @pytest.mark.asyncio
    async def test_2b_evaluate_NOT_skipped_under_default_settings(
        self, tmp_path, monkeypatch
    ):
        """The 2B base is NOT on the default skip list, so its evaluate
        gets submitted normally. Guards against accidental over-skipping."""
        engine, workspace, _pdir = _seed_project(tmp_path)
        train_id, eval_id, _ = _seed_chain(
            engine, chain_id=CHAIN, train_status="succeeded", base_mc_id=MC_2B
        )
        settings = _make_settings(workspace)
        submit_mock = AsyncMock()
        monkeypatch.setattr(tao_job_service, "submit_chain_job", submit_mock)

        await _advance_after_terminal(
            PID,
            chain_id=CHAIN,
            chain_sequence=1,
            engine=engine,
            settings=settings,
        )

        with Session(engine) as s:
            eval_row = s.query(TAOJob).filter_by(tao_job_id=eval_id).one()
            assert eval_row.status == "not_started"  # untouched by the auto-skip
        submit_mock.assert_called_once_with(PID, eval_id, settings=settings)

    @pytest.mark.asyncio
    async def test_empty_skip_list_disables_base_blocklist_skip(
        self, tmp_path, monkeypatch
    ):
        """Operator-cleared TAO_AUTOEVAL_SKIP_BASES = [] disables the skip
        (e.g., once TAO ships the upstream fix). 8B evaluate gets submitted."""
        engine, workspace, _pdir = _seed_project(tmp_path)
        train_id, eval_id, _ = _seed_chain(
            engine, chain_id=CHAIN, train_status="succeeded", base_mc_id=MC_8B
        )
        settings = _make_settings(workspace, skip_bases=[])
        submit_mock = AsyncMock()
        monkeypatch.setattr(tao_job_service, "submit_chain_job", submit_mock)

        await _advance_after_terminal(
            PID,
            chain_id=CHAIN,
            chain_sequence=1,
            engine=engine,
            settings=settings,
        )

        with Session(engine) as s:
            eval_row = s.query(TAOJob).filter_by(tao_job_id=eval_id).one()
            assert eval_row.status == "not_started"
        submit_mock.assert_called_once_with(PID, eval_id, settings=settings)


class TestAdapterOnlyBaselineEvaluation:
    """LoRA baselines are merged and evaluated by the local Student NIM."""

    @pytest.mark.asyncio
    async def test_lora_evaluate_runs_via_local_student_nim(
        self, tmp_path, monkeypatch
    ):
        engine, workspace, _pdir = _seed_project(tmp_path)
        _train_id, eval_id, _ = _seed_chain(
            engine,
            chain_id=CHAIN,
            train_status="succeeded",
            base_mc_id=MC_2B,
            lora=True,
        )
        settings = _make_settings(workspace)
        submit_mock = AsyncMock()
        monkeypatch.setattr(tao_job_service, "submit_chain_job", submit_mock)
        local_eval = AsyncMock(
            return_value={
                "success": True,
                "student_model_id": "student-1",
                "evaluation_run_id": "run-1",
                "quality_status": "validated",
                "serving_status": "validated",
            }
        )
        monkeypatch.setattr(
            student_model_service,
            "run_automatic_baseline_evaluation",
            local_eval,
        )

        await _advance_after_terminal(
            PID,
            chain_id=CHAIN,
            chain_sequence=1,
            engine=engine,
            settings=settings,
        )

        with Session(engine) as s:
            eval_row = s.query(TAOJob).filter_by(tao_job_id=eval_id).one()
            assert eval_row.status == "succeeded"
            assert eval_row.training_backend == "student_nim_local"
            assert eval_row.outputs["evaluation_source"] == "student_nim_local"
            assert eval_row.outputs["evaluation_run_id"] == "run-1"
            assert eval_row.completed_at is not None
        local_eval.assert_awaited_once()
        submit_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_lora_local_evaluation_survives_empty_blocklist(
        self, tmp_path, monkeypatch
    ):
        engine, workspace, _pdir = _seed_project(tmp_path)
        _train_id, eval_id, _ = _seed_chain(
            engine,
            chain_id=CHAIN,
            train_status="succeeded",
            base_mc_id=MC_2B,
            lora=True,
        )
        settings = _make_settings(workspace, skip_bases=[])
        submit_mock = AsyncMock()
        monkeypatch.setattr(tao_job_service, "submit_chain_job", submit_mock)
        monkeypatch.setattr(
            student_model_service,
            "run_automatic_baseline_evaluation",
            AsyncMock(
                return_value={
                    "success": True,
                    "student_model_id": "student-1",
                    "evaluation_run_id": "run-1",
                    "quality_status": "validated",
                    "serving_status": "validated",
                }
            ),
        )

        await _advance_after_terminal(
            PID,
            chain_id=CHAIN,
            chain_sequence=1,
            engine=engine,
            settings=settings,
        )

        with Session(engine) as s:
            eval_row = s.query(TAOJob).filter_by(tao_job_id=eval_id).one()
            assert eval_row.status == "succeeded"
        submit_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_lora_local_evaluation_finalizes_suite_completed(
        self, tmp_path, monkeypatch
    ):
        engine, workspace, _pdir = _seed_project(tmp_path)
        _train_id, _eval_id, _ = _seed_chain(
            engine,
            chain_id=CHAIN,
            train_status="succeeded",
            base_mc_id=MC_2B,
            lora=True,
        )
        settings = _make_settings(workspace)
        monkeypatch.setattr(tao_job_service, "submit_chain_job", AsyncMock())
        monkeypatch.setattr(
            student_model_service,
            "run_automatic_baseline_evaluation",
            AsyncMock(
                return_value={
                    "success": True,
                    "student_model_id": "student-1",
                    "evaluation_run_id": "run-1",
                    "quality_status": "validated",
                    "serving_status": "validated",
                }
            ),
        )

        await _advance_after_terminal(
            PID,
            chain_id=CHAIN,
            chain_sequence=1,
            engine=engine,
            settings=settings,
        )

        with Session(engine) as s:
            suite = s.query(TrainingSuite).filter_by(project_id=PID).one()
            assert suite.status == "completed"
            assert suite.completed_at is not None

    def test_legacy_adapter_only_skip_is_revived_once(self, tmp_path):
        engine, _workspace, _pdir = _seed_project(tmp_path)
        _train_id, eval_id, _ = _seed_chain(
            engine,
            chain_id=CHAIN,
            train_status="succeeded",
            base_mc_id=MC_2B,
            lora=True,
        )
        with Session(engine) as session:
            row = session.query(TAOJob).filter_by(tao_job_id=eval_id).one()
            row.status = "canceled"
            row.completed_at = "2026-07-15T00:00:00Z"
            row.chain_halted_reason = (
                f"{AUTO_SKIP_REASON_PREFIX} action=evaluate auto-skipped; "
                "trained checkpoint is adapter-only (enable_lora=true)"
            )
            session.commit()

        revived = _migrate_legacy_lora_baseline_skips(PID, engine=engine)
        assert len(revived) == 1
        assert revived[0][0] == eval_id
        assert _migrate_legacy_lora_baseline_skips(PID, engine=engine) == []
        with Session(engine) as session:
            row = session.query(TAOJob).filter_by(tao_job_id=eval_id).one()
            assert row.status == "not_started"
            assert row.completed_at is None
            assert row.chain_halted_reason is None
