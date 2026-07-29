# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for training_suite_service.

Coverage:
  * Validation (422): invalid preset, invalid quantization scheme,
    unknown student_base model, non-student_base role, missing project,
    missing guidance.
  * Idempotency replay: second POST with same key returns existing suite.
  * Phase-1d atomicity: a mid-flight chain failure rolls back the
    TrainingSuite + TAOJob rows; the Phase-1b DatasetExports stay
    committed (consistent with their archives on disk).
  * SQLite write discipline: no write transaction held across the
    S3 uploads (concurrent-writer probe).
  * Chain structure: per-model TAOJob rows with correct chain_id,
    chain_sequence, parent_tao_job_id, action ordering, status=not_started.
  * Phase 2 kickoff: first chain's first train job is submitted via
    submit_chain_job; on success → status=submitted + external id.
  * resolved_training_fields.policy.model_name_or_path persisted for
    the downstream LoRA-merge dependency.
  * force_create omitted from all chained tao_create_job_requests.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, Mock, call

import pytest
from sqlalchemy.orm import Session

from conftest import (
    add_endpoint_row,
    add_example_row,
    add_guidance_row,
    add_model_config_row,
    add_project_row,
    make_tao_settings,
    open_project_workspace,
)
from vlm_feedback_loop.db.base import generate_uuid4, utc_now
from vlm_feedback_loop.db.models.label import Label
from vlm_feedback_loop.db.models.model_config import ModelConfig
from vlm_feedback_loop.db.models.tao_job import TAOJob
from vlm_feedback_loop.db.models.training_suite import TrainingSuite
from vlm_feedback_loop.model_catalog_constants import (
    COSMOS3_NANO_REASONER,
    COSMOS3_SUPER_REASONER,
    COSMOS3_SUPER_REASONER_HF_PATH,
    COSMOS_REASON2_2B,
    COSMOS_REASON2_8B,
    COSMOS_REASON2_8B_HF_PATH,
)
from vlm_feedback_loop.services import tao_job_service, training_suite_service

PID = "test-proj"
GID = "guid-001"
MC_8B = "mc-cosmos-8b"
MC_2B = "mc-cosmos-2b"
EID = "ep-001"


# ── Fixtures ────────────────────────────────────────────────────────────────

FIXTURE_FIELDS = [
    {
        "field_id": "f0",
        "field_name": "rationale_note",
        "type": "string",
        "role": "aux",
        "display_order": -1,
    },
    {
        "field_id": "f1",
        "field_name": "severity",
        "type": "enum",
        "role": "core",
        "display_order": 1,
        "allowed_values": ["low", "high"],
    },
]
FIXTURE_SCHEMA = {
    "fields": FIXTURE_FIELDS,
    "generation_order": ["rationale_note", "severity"],
    "derived_json_schema": {},
    "schema_hash": "test",
}
SAMPLE_LABEL_JSON = {"rationale_note": "hi", "severity": "high"}


def _make_settings(workspace: Path, **overrides):
    return make_tao_settings(workspace, TAO_API_KEY="jwt-test", **overrides)


def _setup_project_db(tmp_path: Path, project_id: str = PID):
    return open_project_workspace(
        tmp_path, project_id, register_engine=True, subdirs=("artifacts", "exports")
    )


def _add_project(session, project_dir, **overrides):
    add_project_row(
        session,
        PID,
        str(project_dir),
        active_guidance_id=GID,
        teacher_model_config_id=MC_8B,
        **overrides,
    )


def _add_guidance(session):
    add_guidance_row(
        session, PID, GID, FIXTURE_SCHEMA, description="Classify severity."
    )


def _add_endpoint(session):
    add_endpoint_row(session, PID, EID, base_url="https://test/v1")


def _add_model(
    session,
    mc_id,
    model_name,
    roles,
    *,
    tao_base_experiment_id="be-test-uuid",
    local_deploy_metadata=None,
):
    """Insert a ModelConfig fixture row.

    Defaults to a non-null ``tao_base_experiment_id`` because it is
    required on every ``student_base`` ModelConfig that backs a
    training-suite chain (preflight enforces this gate before
    ``create_training_suite`` is called). Tests that need to exercise
    the null-id error path can pass ``tao_base_experiment_id=None``.

    ``local_deploy_metadata`` lets a test carry the per-project
    ``hf_model_path`` override that ``_hf_path_for_model`` consults — the
    mechanism by which a non-seeded base model (e.g. Cosmos 3) supplies
    its HF identifier without code-side map changes.
    """
    add_model_config_row(
        session,
        PID,
        mc_id,
        EID,
        model_name=model_name,
        eligible_roles=json.dumps(roles),
        thinking_toggle_mode="qwen_enable_thinking",
        thinking_toggle_support="supported",
        visual_budget_mode="mm_processor_size",
        visual_budget_support="supported",
        tao_base_experiment_id=tao_base_experiment_id,
        tao_base_experiment_pull_status=(
            "pull_complete" if tao_base_experiment_id else None
        ),
        local_deploy_metadata=local_deploy_metadata,
    )


def _add_example_with_image(session, tmp_path: Path, key: str, state="Verified"):
    images_dir = tmp_path / "images"
    images_dir.mkdir(exist_ok=True)
    img_path = images_dir / f"{key}.jpg"
    img_path.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
    add_example_row(session, PID, key, storage_ref=str(img_path), state=state)


def _add_verified_label(session, key, pool_assignment=None):
    session.add(
        Label(
            label_id=generate_uuid4(),
            project_id=PID,
            example_key=key,
            label_status="verified",
            guidance_id=GID,
            inference_invocation_id=generate_uuid4(),
            label_json=dict(SAMPLE_LABEL_JSON),
            labeled_at=utc_now(),
            verified_outcome="Accept",
            verified_at=utc_now(),
            edited_core_fields=[],
            edited_aux_fields=[],
            rationale_source="teacher_proposal",
            pool_assignment=pool_assignment,
        )
    )


def _bootstrap_tao_deployment_config(workspace: Path) -> None:
    """Flip the singleton TAODeploymentConfig to 'bootstrapped' state.

    ``create_training_suite`` is gated on the workspace being
    bootstrapped; this helper satisfies the gate for tests that don't
    exercise bootstrap itself.
    """
    from vlm_feedback_loop.db.deployment_models import TAODeploymentConfig
    from vlm_feedback_loop.db.engine import init_deployment_db

    engine = init_deployment_db(workspace)
    with Session(engine) as session:
        cfg = session.query(TAODeploymentConfig).first()
        assert cfg is not None
        cfg.tao_workspace_id = "ws-for-tests"
        cfg.tao_workspace_bucket = "test-bucket"
        cfg.tao_workspace_cloud_type = "seaweedfs"
        cfg.tao_workspace_s3_endpoint_url_internal = "http://seaweedfs-s3:8333"
        cfg.tao_workspace_s3_endpoint_url_external = "http://127.0.0.1:8333"
        cfg.bootstrap_status = "bootstrapped"
        session.commit()


async def _noop_upload(
    session,
    *,
    dataset_export,
    archive_path,
    deployment_config,
    s3_client,
    annotations_path,
    **_kwargs,
):
    """Default no-op upload used by the autouse fixture.

    Skips S3 and synthesises success. The S1-S4 upload-integration tests
    override this via the ``_upload_archive`` kwarg on
    ``create_training_suite``.
    """
    from vlm_feedback_loop.services.tao_dataset_upload_service import (
        UploadResult,
        build_s3_key,
        build_tao_spec_reference,
    )

    key = build_s3_key(
        project_id=dataset_export.project_id,
        dataset_export_id=dataset_export.dataset_export_id,
        archive_name=archive_path.name,
    )
    spec_reference = build_tao_spec_reference(
        deployment_config,
        bucket=deployment_config.tao_workspace_bucket,
        key=key,
    )
    annotation_key = build_s3_key(
        project_id=dataset_export.project_id,
        dataset_export_id=dataset_export.dataset_export_id,
        archive_name=annotations_path.name,
    )
    annotation_spec_reference = build_tao_spec_reference(
        deployment_config,
        bucket=deployment_config.tao_workspace_bucket,
        key=annotation_key,
    )
    dataset_export.dataset_upload_ref = key
    dataset_export.dataset_upload_uri = (
        f"s3://{deployment_config.tao_workspace_bucket}/{key}"
    )
    return UploadResult(
        success=True,
        dataset_export_id=dataset_export.dataset_export_id,
        bucket=deployment_config.tao_workspace_bucket,
        key=key,
        upload_uri=dataset_export.dataset_upload_uri,
        spec_reference=spec_reference,
        annotation_key=annotation_key,
        annotation_spec_reference=annotation_spec_reference,
        sha256="noop-sha",
        already_uploaded=False,
    )


@pytest.fixture(autouse=True)
def _autostub_upload_archive(monkeypatch):
    """Monkey-patch upload_dataset_archive + build_s3_client for every test.

    The S1-S4 upload-integration tests explicitly override via
    ``_upload_archive=`` + ``_s3_client=``. This autouse fixture covers
    the tests that don't care about upload plumbing.
    """
    from vlm_feedback_loop.services import tao_dataset_upload_service as _uploads
    from vlm_feedback_loop.services import training_suite_service as _tss

    monkeypatch.setattr(_uploads, "upload_dataset_archive", _noop_upload)
    monkeypatch.setattr(_tss, "build_s3_client", lambda _cfg: object())


@pytest.fixture()
def seeded(tmp_path):
    engine, project_dir, workspace = _setup_project_db(tmp_path)
    with Session(engine) as s:
        _add_project(s, project_dir=project_dir)
        _add_guidance(s)
        _add_endpoint(s)
        _add_model(s, MC_8B, COSMOS_REASON2_8B, ["teacher", "student_base"])
        _add_model(s, MC_2B, COSMOS_REASON2_2B, ["teacher", "student_base"])
        _add_model(s, "mc-mistral", "mistral-large-3", ["teacher"])
        # 3 non-pool Verified + 2 pool Verified so both exports produce content.
        for i in range(3):
            k = f"ver_{i}"
            _add_example_with_image(s, tmp_path, k)
            _add_verified_label(s, k)
        for i in range(2):
            k = f"pool_{i}"
            _add_example_with_image(s, tmp_path, k)
            _add_verified_label(s, k, pool_assignment="test_pool")
        s.commit()
    _bootstrap_tao_deployment_config(workspace)
    settings = _make_settings(workspace)
    yield engine, project_dir, settings


@pytest.fixture()
def mock_submit(monkeypatch):
    """AsyncMock for tao_job_service._submit_to_tao — returns success by default."""
    m = AsyncMock(
        return_value={
            "success": True,
            "tao_external_job_id": "ext-123",
            "error": None,
        }
    )
    monkeypatch.setattr(tao_job_service, "_submit_to_tao", m)
    return m


# ═══════════════════════════════════════════════════════════════════════════
# Training Jobs launch + conditional provisioning
# ═══════════════════════════════════════════════════════════════════════════


class TestTrainingJobsLaunch:
    @pytest.mark.asyncio
    async def test_missing_selected_bases_share_one_provisional_suite_step(
        self, seeded, monkeypatch
    ):
        """Nano/Super-style missing selections are provisioned together."""
        engine, _, settings = seeded
        with Session(engine) as session:
            for model_id, model_name in (
                (MC_8B, COSMOS3_NANO_REASONER),
                (MC_2B, COSMOS3_SUPER_REASONER),
            ):
                model = session.get(ModelConfig, model_id)
                assert model is not None
                model.model_name = model_name
                model.tao_base_experiment_id = None
                model.tao_base_experiment_pull_status = None
            session.commit()

        async def ready(**_kwargs):
            return {"status": "passed", "checks": [], "resolved_presets": {}}

        monkeypatch.setattr(
            training_suite_service.training_preflight_service,
            "run_training_preflight",
            ready,
        )
        provision = {
            "provisioning_run_id": "prov-both",
            "requested_model_names": [
                COSMOS3_NANO_REASONER,
                COSMOS3_SUPER_REASONER,
            ],
        }
        start = Mock(return_value=provision)
        monkeypatch.setattr(training_suite_service, "start_provisioning_run", start)

        captured: dict[str, str] = {}

        def capture(task_id, worker):
            captured["task_id"] = task_id
            worker.close()

        monkeypatch.setattr(
            training_suite_service.background_manager, "register", capture
        )

        result = await training_suite_service.launch_training_suite(
            PID,
            student_base_model_config_ids=[MC_8B, MC_2B],
            training_preset="standard",
            include_auto_labeled=True,
            export_field_mode="all",
            quantization_schemes=["FP8_DYNAMIC"],
            idempotency_key="launch-both",
            settings=settings,
        )

        assert isinstance(result, dict)
        assert result["status"] == "provisioning"
        assert result["chains"] == []
        assert result["provisioning_run_id"] == "prov-both"
        assert result["provisioning_model_names"] == [
            COSMOS3_NANO_REASONER,
            COSMOS3_SUPER_REASONER,
        ]
        start.assert_called_once_with(PID, [MC_8B, MC_2B], settings)
        assert captured["task_id"].startswith("training-suite-setup-")

    @pytest.mark.asyncio
    async def test_ready_bases_skip_provisioning(self, seeded, monkeypatch):
        """Already-ready selections use the existing suite path with no setup step."""

        async def ready(**_kwargs):
            return {"status": "passed", "checks": [], "resolved_presets": {}}

        monkeypatch.setattr(
            training_suite_service.training_preflight_service,
            "run_training_preflight",
            ready,
        )
        create = AsyncMock(return_value={"training_suite_id": "suite-ready"})
        monkeypatch.setattr(training_suite_service, "create_training_suite", create)
        start = Mock()
        monkeypatch.setattr(training_suite_service, "start_provisioning_run", start)
        _, _, settings = seeded

        result = await training_suite_service.launch_training_suite(
            PID,
            student_base_model_config_ids=[MC_8B, MC_2B],
            training_preset="standard",
            include_auto_labeled=True,
            export_field_mode="all",
            quantization_schemes=["FP8_DYNAMIC"],
            idempotency_key="launch-ready",
            settings=settings,
        )

        assert result == {"training_suite_id": "suite-ready"}
        start.assert_not_called()
        create.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_server_readiness_failure_starts_no_transfer(
        self, seeded, monkeypatch
    ):
        """Removing the visible Preflight section never removes fail-closed safety."""

        async def blocked(**_kwargs):
            return {
                "status": "failed",
                "checks": [
                    {
                        "check_name": "tao_job_timeout_supported",
                        "passed": False,
                        "message": "TAO timeout override is unavailable.",
                    }
                ],
                "resolved_presets": {},
            }

        monkeypatch.setattr(
            training_suite_service.training_preflight_service,
            "run_training_preflight",
            blocked,
        )
        start = Mock()
        monkeypatch.setattr(training_suite_service, "start_provisioning_run", start)
        _, _, settings = seeded

        result = await training_suite_service.launch_training_suite(
            PID,
            student_base_model_config_ids=[MC_8B],
            training_preset="standard",
            include_auto_labeled=True,
            export_field_mode="all",
            quantization_schemes=["FP8_DYNAMIC"],
            idempotency_key="launch-blocked",
            settings=settings,
        )

        assert isinstance(result, str)
        assert result.startswith("tao_unreachable:")
        start.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════════
# Validation (422)
# ═══════════════════════════════════════════════════════════════════════════


class TestValidation:
    @pytest.mark.asyncio
    async def test_invalid_preset_returns_validation_error(self, seeded, mock_submit):
        _, _, settings = seeded
        result = await training_suite_service.create_training_suite(
            PID,
            student_base_model_config_ids=[MC_8B],
            training_preset="ludicrous",
            include_auto_labeled=False,
            export_field_mode="all",
            quantization_schemes=["FP8_DYNAMIC"],
            idempotency_key="k1",
            settings=settings,
        )
        assert isinstance(result, str)
        assert "validation" in result.lower()
        assert "training_preset" in result
        mock_submit.assert_not_called()

    @pytest.mark.asyncio
    async def test_invalid_export_field_mode(self, seeded, mock_submit):
        _, _, settings = seeded
        result = await training_suite_service.create_training_suite(
            PID,
            student_base_model_config_ids=[MC_8B],
            training_preset="standard",
            include_auto_labeled=False,
            export_field_mode="gibberish",
            quantization_schemes=["FP8_DYNAMIC"],
            idempotency_key="k2",
            settings=settings,
        )
        assert isinstance(result, str)
        assert "validation" in result.lower()
        assert "export_field_mode" in result

    @pytest.mark.asyncio
    async def test_invalid_quantization_scheme(self, seeded, mock_submit):
        _, _, settings = seeded
        result = await training_suite_service.create_training_suite(
            PID,
            student_base_model_config_ids=[MC_8B],
            training_preset="standard",
            include_auto_labeled=False,
            export_field_mode="all",
            quantization_schemes=["INT128_HYPE"],
            idempotency_key="k3",
            settings=settings,
        )
        assert isinstance(result, str)
        assert "validation" in result.lower()

    @pytest.mark.asyncio
    async def test_model_not_found(self, seeded, mock_submit):
        _, _, settings = seeded
        result = await training_suite_service.create_training_suite(
            PID,
            student_base_model_config_ids=["mc-does-not-exist"],
            training_preset="standard",
            include_auto_labeled=False,
            export_field_mode="all",
            quantization_schemes=["FP8_DYNAMIC"],
            idempotency_key="k4",
            settings=settings,
        )
        assert isinstance(result, str)
        assert "not found" in result.lower()

    @pytest.mark.asyncio
    async def test_non_student_base_role_rejected(self, seeded, mock_submit):
        _, _, settings = seeded
        result = await training_suite_service.create_training_suite(
            PID,
            student_base_model_config_ids=["mc-mistral"],
            training_preset="standard",
            include_auto_labeled=False,
            export_field_mode="all",
            quantization_schemes=["FP8_DYNAMIC"],
            idempotency_key="k5",
            settings=settings,
        )
        assert isinstance(result, str)
        assert "student_base" in result

    @pytest.mark.asyncio
    async def test_unknown_project(self, tmp_path, mock_submit):
        workspace = tmp_path / "workspace"
        workspace.mkdir(parents=True)
        settings = _make_settings(workspace)
        result = await training_suite_service.create_training_suite(
            "does-not-exist",
            student_base_model_config_ids=[MC_8B],
            training_preset="standard",
            include_auto_labeled=False,
            export_field_mode="all",
            quantization_schemes=["FP8_DYNAMIC"],
            idempotency_key="k6",
            settings=settings,
        )
        assert isinstance(result, str)
        assert "not found" in result.lower()


# ═══════════════════════════════════════════════════════════════════════════
# Happy path: chain structure + kickoff
# ═══════════════════════════════════════════════════════════════════════════


class TestHappyPathSingleModel:
    @pytest.mark.asyncio
    async def test_creates_full_chain_structure_single_model(self, seeded, mock_submit):
        engine, _, settings = seeded
        result = await training_suite_service.create_training_suite(
            PID,
            student_base_model_config_ids=[MC_8B],
            training_preset="standard",
            include_auto_labeled=False,
            export_field_mode="all",
            quantization_schemes=["FP8_DYNAMIC", "W4A16"],
            idempotency_key="happy-1",
            settings=settings,
        )
        assert not isinstance(result, str), result
        assert result["status"] == "running"
        assert len(result["chains"]) == 1
        chain = result["chains"][0]
        assert chain["student_base_model_config_id"] == MC_8B
        assert chain["base_model_name"] == COSMOS_REASON2_8B
        # 1 train + 1 baseline eval + 2 × (quantize + eval) = 6 jobs.
        assert len(chain["jobs"]) == 6
        actions = [j["action"] for j in chain["jobs"]]
        assert actions == [
            "train",
            "evaluate",
            "quantize",
            "evaluate",
            "quantize",
            "evaluate",
        ]
        # chain_sequence starts at 1 and increments.
        seqs = [j["chain_sequence"] for j in chain["jobs"]]
        assert seqs == [1, 2, 3, 4, 5, 6]

        # The pinned TAO / Cosmos-RL versions are persisted on the job for
        # reproducibility. This suite runs with default Settings for these
        # keys, so it fails if the shipped defaults drift.
        with Session(engine) as s:
            train_row = (
                s.query(TAOJob)
                .filter_by(project_id=PID, chain_id=chain["chain_id"], action="train")
                .one()
            )
            assert train_row.job_config["tao_release_version"] == "6.26.3"
            assert train_row.job_config["cosmos_rl_container_tag"] == "6.26.3-cosmos-rl"

    @pytest.mark.asyncio
    async def test_parent_tao_job_id_linkage(self, seeded, mock_submit):
        engine, _, settings = seeded
        result = await training_suite_service.create_training_suite(
            PID,
            student_base_model_config_ids=[MC_8B],
            training_preset="quick",
            include_auto_labeled=False,
            export_field_mode="all",
            quantization_schemes=["FP8_DYNAMIC"],
            idempotency_key="happy-2",
            settings=settings,
        )
        assert not isinstance(result, str)
        chain_id = result["chains"][0]["chain_id"]
        with Session(engine) as s:
            rows = (
                s.query(TAOJob)
                .filter_by(project_id=PID, chain_id=chain_id)
                .order_by(TAOJob.chain_sequence.asc())
                .all()
            )
            train, baseline_eval, quant, quant_eval = rows
        assert train.parent_tao_job_id is None
        assert baseline_eval.parent_tao_job_id == train.tao_job_id
        assert quant.parent_tao_job_id == train.tao_job_id
        assert quant_eval.parent_tao_job_id == quant.tao_job_id

    @pytest.mark.asyncio
    async def test_first_train_job_submitted_via_chain_job(self, seeded, mock_submit):
        engine, _, settings = seeded
        result = await training_suite_service.create_training_suite(
            PID,
            student_base_model_config_ids=[MC_8B],
            training_preset="standard",
            include_auto_labeled=False,
            export_field_mode="all",
            quantization_schemes=["FP8_DYNAMIC"],
            idempotency_key="kickoff-1",
            settings=settings,
        )
        assert not isinstance(result, str)
        # First train job transitioned to submitted with the mocked external id.
        chain_id = result["chains"][0]["chain_id"]
        with Session(engine) as s:
            train = (
                s.query(TAOJob)
                .filter_by(project_id=PID, chain_id=chain_id, chain_sequence=1)
                .first()
            )
        assert train.status == "submitted"
        assert train.tao_external_job_id == "ext-123"
        # Only the first train job was submitted (chain 0 seq 1). Subsequent
        # jobs remain not_started.
        mock_submit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_resolved_training_fields_persisted(self, seeded, mock_submit):
        engine, _, settings = seeded
        result = await training_suite_service.create_training_suite(
            PID,
            student_base_model_config_ids=[MC_8B],
            training_preset="standard",
            include_auto_labeled=False,
            export_field_mode="all",
            quantization_schemes=[],
            idempotency_key="resolved-1",
            settings=settings,
        )
        assert not isinstance(result, str)
        chain_id = result["chains"][0]["chain_id"]
        with Session(engine) as s:
            train = (
                s.query(TAOJob)
                .filter_by(project_id=PID, chain_id=chain_id, chain_sequence=1)
                .first()
            )
        resolved = (
            train.job_config.get("resolved_training_fields", {})
            .get("policy", {})
            .get("model_name_or_path")
        )
        assert resolved == COSMOS_REASON2_8B_HF_PATH

    @pytest.mark.asyncio
    async def test_cosmos3_hf_path_override_flows_to_spec(self, seeded, mock_submit):
        # A non-seeded student_base (``nvidia/cosmos3-nano``) supplies
        # its HF identifier via the
        # per-project ``local_deploy_metadata.hf_model_path`` override that
        # ``_hf_path_for_model`` consults — no code-side map change. The
        # cosmos-rl spec the Blueprint emits carries ONLY
        # ``policy.model_name_or_path = hf_model://<id>`` (no architecture-
        # override slot), so this is the full extent of what the app side
        # can express for CR3 today. Asserts the override threads through to
        # the emitted train spec end-to-end.
        engine, _, settings = seeded
        MC_CR3 = "mc-cosmos3-nano"
        with Session(engine) as s:
            _add_model(
                s,
                MC_CR3,
                "nvidia/cosmos3-nano",
                ["student_base"],
                local_deploy_metadata={"hf_model_path": "nvidia/Cosmos3-Nano"},
            )
            s.commit()

        result = await training_suite_service.create_training_suite(
            PID,
            student_base_model_config_ids=[MC_CR3],
            training_preset="standard",
            include_auto_labeled=False,
            export_field_mode="all",
            quantization_schemes=[],
            idempotency_key="cosmos3-1",
            settings=settings,
        )
        assert not isinstance(result, str), result
        chain_id = result["chains"][0]["chain_id"]
        with Session(engine) as s:
            train = (
                s.query(TAOJob)
                .filter_by(project_id=PID, chain_id=chain_id, chain_sequence=1)
                .first()
            )
        # Emitted cosmos-rl spec: hf_model:// scheme wrapping the override.
        policy = train.tao_create_job_request["specs"]["policy"]
        assert policy["model_name_or_path"] == "hf_model://nvidia/Cosmos3-Nano"
        # No architecture-override key exists in the spec the Blueprint
        # emits — reasoner-vs-base is delegated to the TAO base-experiment
        # checkpoint config, NOT settable from the app side.
        assert "model_architecture" not in policy
        # Lineage field keeps the bare HF id for checkpoint re-pull.
        resolved = (
            train.job_config.get("resolved_training_fields", {})
            .get("policy", {})
            .get("model_name_or_path")
        )
        assert resolved == "nvidia/Cosmos3-Nano"

    def test_cosmos3_catalog_names_resolve_via_central_hf_map(self):
        # First-class path: a CR3 student_base whose model_name is
        # the catalog-namespaced reasoner id, with NO local_deploy_metadata
        # override, resolves to the correct ``hf_model://`` reasoner repo via
        # the canonical model_catalog_constants.HF_MODEL_PATHS roster — so
        # seeded CR3 catalog rows train
        # without needing a per-project override.
        from vlm_feedback_loop.services.training_suite_service import _hf_model_url

        nano = ModelConfig(model_name=COSMOS3_NANO_REASONER, local_deploy_metadata=None)
        sup = ModelConfig(model_name=COSMOS3_SUPER_REASONER, local_deploy_metadata=None)
        assert _hf_model_url(nano) == "hf_model://nvidia/Cosmos3-Nano-Reasoner"
        assert _hf_model_url(sup) == "hf_model://nvidia/Cosmos3-Super-Reasoner"

    @pytest.mark.asyncio
    async def test_per_model_parallelism_and_train_overrides_emitted(
        self, seeded, mock_submit
    ):
        # Large tiers (Cosmos 3 Super-Reasoner ~30B) OOM at the default tp=1
        # AND need a bf16 optimizer master to fit 8×80 GB (tp=1/tp=4
        # OOM; tp=8 + master_dtype=bfloat16 works).
        # Per-model ``tao_train_parallelism`` →
        # ``policy.parallelism`` (shards at load) and ``tao_train_overrides``
        # → ``train.*`` (e.g. master_dtype). Small tiers omit both (TAO
        # auto-injects tp=1 — covered by
        # test_dataset_bindings_use_correct_action_keys asserting no
        # parallelism key).
        engine, _, settings = seeded
        MC_SUPER = "mc-cosmos3-super"
        with Session(engine) as s:
            _add_model(
                s,
                MC_SUPER,
                COSMOS3_SUPER_REASONER,
                ["student_base"],
                local_deploy_metadata={
                    "hf_model_path": COSMOS3_SUPER_REASONER_HF_PATH,
                    "tao_train_parallelism": {"tp_size": 8, "dp_shard_size": 1},
                    "tao_train_overrides": {"master_dtype": "bfloat16"},
                },
            )
            s.commit()
        result = await training_suite_service.create_training_suite(
            PID,
            student_base_model_config_ids=[MC_SUPER],
            training_preset="quick",
            include_auto_labeled=False,
            export_field_mode="all",
            quantization_schemes=[],
            idempotency_key="parallelism-1",
            settings=settings,
        )
        assert not isinstance(result, str), result
        chain_id = result["chains"][0]["chain_id"]
        with Session(engine) as s:
            train = (
                s.query(TAOJob)
                .filter_by(project_id=PID, chain_id=chain_id, chain_sequence=1)
                .first()
            )
        specs = train.tao_create_job_request["specs"]
        assert specs["policy"]["parallelism"] == {"tp_size": 8, "dp_shard_size": 1}
        # train-section override wins, and the SFT defaults are still present.
        assert specs["train"]["master_dtype"] == "bfloat16"
        assert specs["train"]["train_policy"]["type"] == "sft"
        policy = specs["policy"]
        assert (
            policy["model_name_or_path"] == "hf_model://nvidia/Cosmos3-Super-Reasoner"
        )

    @pytest.mark.asyncio
    async def test_timeout_minutes_emitted_on_every_suite_job(
        self, seeded, mock_submit
    ):
        """Every job carries the configured stale-heartbeat ceiling.

        cosmos-rl does not heartbeat during training, so omitting the field
        silently restores FTMS's unsafe 60-minute default.
        """
        engine, _, settings = seeded
        settings = settings.model_copy(update={"TAO_JOB_TIMEOUT_MINUTES": 1440})
        result = await training_suite_service.create_training_suite(
            PID,
            student_base_model_config_ids=[MC_8B],
            training_preset="standard",
            include_auto_labeled=False,
            export_field_mode="all",
            quantization_schemes=["FP8_DYNAMIC"],
            idempotency_key="timeout-wire-1",
            settings=settings,
        )
        assert not isinstance(result, str), result
        chain_id = result["chains"][0]["chain_id"]
        with Session(engine) as s:
            chain_jobs = (
                s.query(TAOJob)
                .filter_by(project_id=PID, chain_id=chain_id)
                .order_by(TAOJob.chain_sequence)
                .all()
            )
        assert {j.action for j in chain_jobs} == {"train", "evaluate", "quantize"}
        for job in chain_jobs:
            assert job.tao_create_job_request["timeout_minutes"] == 1440, (
                f"{job.action} request missing timeout_minutes"
            )

    @pytest.mark.asyncio
    async def test_default_timeout_prevents_ftms_sixty_minute_ceiling(
        self, seeded, mock_submit
    ):
        """The shipped default is a generous dead-job reaper, not one hour."""
        engine, _, settings = seeded
        assert settings.TAO_JOB_TIMEOUT_MINUTES == 1440
        result = await training_suite_service.create_training_suite(
            PID,
            student_base_model_config_ids=[MC_8B],
            training_preset="standard",
            include_auto_labeled=False,
            export_field_mode="all",
            quantization_schemes=["FP8_DYNAMIC"],
            idempotency_key="timeout-wire-2",
            settings=settings,
        )
        assert not isinstance(result, str), result
        chain_id = result["chains"][0]["chain_id"]
        with Session(engine) as s:
            chain_jobs = (
                s.query(TAOJob)
                .filter_by(project_id=PID, chain_id=chain_id)
                .order_by(TAOJob.chain_sequence)
                .all()
            )
        for job in chain_jobs:
            assert job.tao_create_job_request["timeout_minutes"] == 1440, job.action

    @pytest.mark.asyncio
    async def test_lora_default_reaches_the_wire(self, seeded, mock_submit):
        """The LoRA-first default trains with LoRA, not just records it.

        job_config.lora_config was persisted on every training job while
        the cosmos-rl spec never carried a ``policy.lora`` block — so every
        "LoRA" student was actually a full fine-tune (live-verified: the
        trainer's parameter table logged every module TRAINABLE and an 8B
        train OOM'd on full-model optimizer state). The persisted record
        and the wire must agree.
        """
        engine, _, settings = seeded
        result = await training_suite_service.create_training_suite(
            PID,
            student_base_model_config_ids=[MC_8B],
            training_preset="standard",
            include_auto_labeled=False,
            export_field_mode="all",
            quantization_schemes=[],
            idempotency_key="lora-wire-1",
            settings=settings,
        )
        assert not isinstance(result, str), result
        chain_id = result["chains"][0]["chain_id"]
        with Session(engine) as s:
            train = (
                s.query(TAOJob)
                .filter_by(project_id=PID, chain_id=chain_id, chain_sequence=1)
                .first()
            )
        assert train.job_config["lora_config"]["enable_lora"] is True
        lora = train.tao_create_job_request["specs"]["policy"]["lora"]
        # cosmos-rl LoraConfig field names (r / lora_alpha / lora_dropout /
        # target_modules / modules_to_save) mapped from the Blueprint's
        # lora_config shape.
        assert lora == {
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "target_modules": ["q_proj", "v_proj"],
            "modules_to_save": None,
        }

    @pytest.mark.asyncio
    async def test_lora_chain_quantize_jobs_carry_merge_flags(
        self, seeded, mock_submit
    ):
        """LoRA chains produce adapter-only train checkpoints, so every
        quantize job in the chain must carry the in-container merge flags
        (``enable_lora`` + the model's resolved BARE HuggingFace id) or the
        cosmos-rl-quantize container dies with "base_model_path is required
        when enable_lora is True" — observed live on both quantize legs of
        a LoRA chain, FTMS 6.26.3, 2026-07-15."""
        engine, _, settings = seeded
        result = await training_suite_service.create_training_suite(
            PID,
            student_base_model_config_ids=[MC_8B],
            training_preset="standard",
            include_auto_labeled=False,
            export_field_mode="all",
            quantization_schemes=["FP8_DYNAMIC", "W4A16"],
            idempotency_key="lora-quant-wire-1",
            settings=settings,
        )
        assert not isinstance(result, str), result
        chain_id = result["chains"][0]["chain_id"]
        with Session(engine) as s:
            quant_jobs = (
                s.query(TAOJob)
                .filter_by(project_id=PID, chain_id=chain_id, action="quantize")
                .all()
            )
        assert len(quant_jobs) == 2
        for job in quant_jobs:
            specs = job.tao_create_job_request["specs"]
            assert specs["enable_lora"] is True
            assert specs["base_model_path"] == COSMOS_REASON2_8B_HF_PATH

    @pytest.mark.asyncio
    async def test_enable_lora_false_keeps_full_weight_wire_shape(
        self, seeded, mock_submit
    ):
        """``enable_lora=false`` opts the suite into full-weight training:
        no ``policy.lora`` on the wire, and every chain job's persisted
        lora_config records the opt-out (evaluate/quantize records must not
        claim a LoRA default the chain isn't using)."""
        engine, _, settings = seeded
        result = await training_suite_service.create_training_suite(
            PID,
            student_base_model_config_ids=[MC_8B],
            training_preset="standard",
            include_auto_labeled=False,
            export_field_mode="all",
            quantization_schemes=["FP8_DYNAMIC"],
            enable_lora=False,
            idempotency_key="lora-optout-1",
            settings=settings,
        )
        assert not isinstance(result, str), result
        chain_id = result["chains"][0]["chain_id"]
        with Session(engine) as s:
            chain_jobs = (
                s.query(TAOJob)
                .filter_by(project_id=PID, chain_id=chain_id)
                .order_by(TAOJob.chain_sequence)
                .all()
            )
            assert len(chain_jobs) >= 3
            train = chain_jobs[0]
            assert "lora" not in train.tao_create_job_request["specs"]["policy"]
            for job in chain_jobs:
                assert job.job_config["lora_config"]["enable_lora"] is False
                if job.action == "quantize":
                    # Full-weight chains keep the plain quantize wire
                    # shape — no in-container merge flags.
                    specs = job.tao_create_job_request["specs"]
                    assert "enable_lora" not in specs
                    assert "base_model_path" not in specs

    @pytest.mark.asyncio
    async def test_force_create_omitted_from_chained_requests(
        self, seeded, mock_submit
    ):
        engine, _, settings = seeded
        result = await training_suite_service.create_training_suite(
            PID,
            student_base_model_config_ids=[MC_8B],
            training_preset="standard",
            include_auto_labeled=False,
            export_field_mode="all",
            quantization_schemes=["FP8_DYNAMIC"],
            idempotency_key="force-create-1",
            settings=settings,
        )
        assert not isinstance(result, str)
        chain_id = result["chains"][0]["chain_id"]
        with Session(engine) as s:
            rows = s.query(TAOJob).filter_by(project_id=PID, chain_id=chain_id).all()
        for job in rows:
            assert "force_create" not in (job.tao_create_job_request or {})

    @pytest.mark.asyncio
    async def test_dataset_bindings_use_correct_action_keys(self, seeded, mock_submit):
        engine, _, settings = seeded
        result = await training_suite_service.create_training_suite(
            PID,
            student_base_model_config_ids=[MC_8B],
            training_preset="quick",
            include_auto_labeled=False,
            export_field_mode="all",
            quantization_schemes=["FP8_DYNAMIC"],
            idempotency_key="binding-1",
            settings=settings,
        )
        assert not isinstance(result, str)
        chain_id = result["chains"][0]["chain_id"]
        with Session(engine) as s:
            rows = (
                s.query(TAOJob)
                .filter_by(project_id=PID, chain_id=chain_id)
                .order_by(TAOJob.chain_sequence.asc())
                .all()
            )
        train, baseline_eval, quant, quant_eval = rows
        # train uses custom.train_dataset.media_path — Cosmos-RL's
        # CustomConfig.train_dataset field name (a bare top-level
        # "dataset" key on train is rejected by TAO with
        # ValidationError).
        assert "custom" in train.tao_create_job_request["specs"]
        assert (
            "media_path"
            in train.tao_create_job_request["specs"]["custom"]["train_dataset"]
        )
        # evaluate uses top-level ``dataset.media_dir`` because cosmos-rl
        # ``ITSEvaluator`` reads ``config["dataset"]`` (the train-side
        # ``custom.val_dataset`` shape crashes evaluate with
        # ``KeyError: 'dataset'``).
        assert "media_dir" in baseline_eval.tao_create_job_request["specs"]["dataset"]
        # quantize uses top-level dataset.media_dir
        # (cosmos-rl-quantize accepts --media_dir and rejects
        # the train-side --media_path).
        assert "media_dir" in quant.tao_create_job_request["specs"]["dataset"]

        # Every chain job must carry the TAO-required job
        # metadata, otherwise FTMS POST /jobs returns HTTP 400. Lock in the
        # required fields + the deterministic name pattern across train,
        # eval-baseline, quantize, eval-quantized. ``timeout_minutes`` is
        # required so cosmos-rl's quiet training loop cannot fall back to
        # FTMS's unsafe 60-minute stale-heartbeat ceiling.
        for job in (train, baseline_eval, quant, quant_eval):
            req = job.tao_create_job_request
            assert req["network_arch"] == "cosmos-rl", job.action
            assert req["base_experiment_ids"] == ["be-test-uuid"], job.action
            assert req["workspace"] == "ws-for-tests", job.action
            assert isinstance(req["name"], str) and req["name"], job.action
            assert req["name"].startswith(f"vlm-fb-{job.action}-"), job.action
            assert req["timeout_minutes"] == 1440, job.action

            specs = req["specs"]
            # Cosmos-RL SFT runtime overrides apply only to train.
            # quantize MUST NOT emit them — the
            # cosmos-rl-quantize CLI rejects --compile, --type,
            # --dataloader_drop_last, --save_mode, --enable as unknown
            # arguments. Evaluate inherits them (cosmos-rl
            # ITSEvaluator reuses the same SFT trainer init path).
            if job.action in ("train", "evaluate"):
                assert specs["train"]["compile"] is False, job.action
                assert specs["train"]["train_policy"]["type"] == "sft", job.action
                assert (
                    specs["train"]["train_policy"]["dataloader_drop_last"] is False
                ), job.action
                assert specs["train"]["ckpt"]["save_mode"] == "sync", job.action
                assert specs["validation"]["enable"] is False, job.action
            else:
                assert "train" not in specs, (
                    f"quantize spec MUST NOT carry train.* (cosmos-rl-quantize "
                    f"CLI rejects --compile, --type, --dataloader_drop_last, "
                    f"--save_mode); got {specs!r}"
                )
                assert "validation" not in specs, (
                    f"quantize spec MUST NOT carry validation.enable "
                    f"(cosmos-rl-quantize CLI rejects --enable); got {specs!r}"
                )

            # num_gpu and policy.parallelism are intentionally omitted
            # so TAO's auto-injection (tp=1, dp_shard=NUM_GPU_PER_NODE)
            # uses every visible GPU on the rental.
            assert "num_gpu" not in req, job.action
            policy = specs.get("policy") or {}
            assert "parallelism" not in policy, job.action
            # train emits policy.model_name_or_path with the hf_model://
            # scheme so cosmos-rl knows which base experiment to load.
            # evaluate intentionally OMITS the field so FTMS's
            # ``parent_job_id`` resolution can substitute the trained
            # checkpoint folder; sending an explicit value overrides the
            # resolution and forces the worker to load the base model
            # (an explicit base model on eval produces verbatim
            # base-model output even after SFT). quantize ALSO omits
            # policy.model_name_or_path — cosmos-rl-quantize's CLI accepts
            # ``--model_path`` (auto-injected from parent_tao_job_id), NOT
            # ``--model_name_or_path``; the latter is rejected as
            # unknown.
            if job.action in ("evaluate", "quantize"):
                assert "model_name_or_path" not in policy, job.action
            else:
                assert (
                    policy["model_name_or_path"]
                    == "hf_model://nvidia/Cosmos-Reason2-8B"
                ), job.action

            # quantize-specific: the `quantization_scheme` spec key
            # name MUST match cosmos-rl-quantize's `--quantization_scheme`
            # CLI flag. A `quantization_method` key (mirroring TAO
            # FTMS's user-facing quantization-method abstraction) is
            # rejected as `unrecognized arguments: --quantization_method`.
            if job.action == "quantize":
                assert specs.get("quantization_scheme") == "FP8_DYNAMIC", specs
                assert "quantization_method" not in specs, specs

    @pytest.mark.asyncio
    async def test_create_fails_when_model_missing_tao_base_experiment_id(
        self, tmp_path
    ):
        """Negative path: a student_base ModelConfig with null
        ``tao_base_experiment_id`` MUST cause atomic rollback before any
        TAO POST. Preflight is the primary gate but this guard is
        the last line of defence.
        """
        engine, project_dir, workspace = _setup_project_db(tmp_path)
        with Session(engine) as session:
            _add_project(session, project_dir)
            _add_guidance(session)
            _add_endpoint(session)
            _add_model(
                session,
                MC_8B,
                COSMOS_REASON2_8B,
                ["student_base"],
                tao_base_experiment_id=None,
            )
            for i in range(3):
                _add_example_with_image(session, tmp_path, f"ex-{i}")
                _add_verified_label(session, f"ex-{i}")
            session.commit()
        _bootstrap_tao_deployment_config(workspace)

        settings = _make_settings(workspace)
        result = await training_suite_service.create_training_suite(
            PID,
            student_base_model_config_ids=[MC_8B],
            training_preset="quick",
            include_auto_labeled=False,
            export_field_mode="all",
            quantization_schemes=[],
            idempotency_key="missing-be-1",
            settings=settings,
        )
        # A resolvable precondition (base experiment not provisioned) is a
        # returned validation error (→ 400), not a 500. Rollback still holds.
        assert isinstance(result, str)
        assert "tao_base_experiment_id" in result
        from vlm_feedback_loop.services.errors import map_service_error

        assert map_service_error(result).status_code == 400
        # No TAOJob rows persisted.
        with Session(engine) as session:
            assert session.query(TAOJob).count() == 0


# ═══════════════════════════════════════════════════════════════════════════
# Multi-model (sequential chains)
# ═══════════════════════════════════════════════════════════════════════════


class TestMultiModel:
    @pytest.mark.asyncio
    async def test_two_chains_only_first_kicked_off(self, seeded, mock_submit):
        engine, _, settings = seeded
        result = await training_suite_service.create_training_suite(
            PID,
            student_base_model_config_ids=[MC_8B, MC_2B],
            training_preset="standard",
            include_auto_labeled=False,
            export_field_mode="all",
            quantization_schemes=["FP8_DYNAMIC"],
            idempotency_key="multi-1",
            settings=settings,
        )
        assert not isinstance(result, str)
        assert len(result["chains"]) == 2
        # chain_ids_ordered preserves request order.
        assert result["chain_ids_ordered"] == [c["chain_id"] for c in result["chains"]]
        assert result["chains"][0]["student_base_model_config_id"] == MC_8B
        assert result["chains"][1]["student_base_model_config_id"] == MC_2B
        # Only ONE submission — the first chain's seq=1 train job.
        mock_submit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_2b_uses_2b_epoch_table(self, seeded, mock_submit):
        engine, _, settings = seeded
        result = await training_suite_service.create_training_suite(
            PID,
            student_base_model_config_ids=[MC_2B],
            training_preset="high_quality",
            include_auto_labeled=False,
            export_field_mode="all",
            quantization_schemes=[],
            idempotency_key="2b-hq-1",
            settings=settings,
        )
        assert not isinstance(result, str)
        chain_id = result["chains"][0]["chain_id"]
        with Session(engine) as s:
            train = (
                s.query(TAOJob)
                .filter_by(project_id=PID, chain_id=chain_id, chain_sequence=1)
                .first()
            )
        # Model-aware preset table: 2B high_quality = 12 epochs.
        assert train.job_config["hyperparameters"]["train"]["epoch"] == 12


# ═══════════════════════════════════════════════════════════════════════════
# Idempotency
# ═══════════════════════════════════════════════════════════════════════════


class TestIdempotency:
    @pytest.mark.asyncio
    async def test_replay_returns_existing_suite_without_new_writes(
        self, seeded, mock_submit
    ):
        engine, _, settings = seeded
        first = await training_suite_service.create_training_suite(
            PID,
            student_base_model_config_ids=[MC_8B],
            training_preset="standard",
            include_auto_labeled=False,
            export_field_mode="all",
            quantization_schemes=["FP8_DYNAMIC"],
            idempotency_key="idem-key-1",
            settings=settings,
        )
        assert not isinstance(first, str)

        second = await training_suite_service.create_training_suite(
            PID,
            student_base_model_config_ids=[MC_8B],
            training_preset="standard",
            include_auto_labeled=False,
            export_field_mode="all",
            quantization_schemes=["FP8_DYNAMIC"],
            idempotency_key="idem-key-1",
            settings=settings,
        )
        assert not isinstance(second, str)
        assert first["training_suite_id"] == second["training_suite_id"]
        # Only the FIRST call should have triggered submission.
        mock_submit.assert_awaited_once()

        # Exactly one TrainingSuite row in the DB with this key.
        with Session(engine) as s:
            rows = (
                s.query(TrainingSuite)
                .filter_by(project_id=PID, idempotency_key="idem-key-1")
                .all()
            )
        assert len(rows) == 1

    @pytest.mark.asyncio
    async def test_different_keys_create_distinct_suites(self, seeded, mock_submit):
        engine, _, settings = seeded
        a = await training_suite_service.create_training_suite(
            PID,
            student_base_model_config_ids=[MC_8B],
            training_preset="standard",
            include_auto_labeled=False,
            export_field_mode="all",
            quantization_schemes=[],
            idempotency_key="key-A",
            settings=settings,
        )
        b = await training_suite_service.create_training_suite(
            PID,
            student_base_model_config_ids=[MC_8B],
            training_preset="standard",
            include_auto_labeled=False,
            export_field_mode="all",
            quantization_schemes=[],
            idempotency_key="key-B",
            settings=settings,
        )
        assert not isinstance(a, str) and not isinstance(b, str)
        assert a["training_suite_id"] != b["training_suite_id"]
        assert mock_submit.await_count == 2


# ═══════════════════════════════════════════════════════════════════════════
# Atomic rollback
# ═══════════════════════════════════════════════════════════════════════════


class TestAtomicity:
    @pytest.mark.asyncio
    async def test_rollback_on_mid_flight_failure(
        self, seeded, mock_submit, monkeypatch
    ):
        engine, _, settings = seeded
        # Force chain creation to explode AFTER the DatasetExports commit
        # but BEFORE the TrainingSuite insert commits. The chain + suite
        # rows are all-or-nothing (Phase 1d single transaction); the
        # DatasetExport rows deliberately survive — they committed in
        # Phase 1b, matching the archives already on disk and the shape
        # the standalone exports API produces (write discipline forbids holding
        # one write transaction across the archive builds and S3 uploads
        # that separate the two phases).
        real_fn = training_suite_service._create_chain_rows_in_session

        def _boom(*args, **kwargs):  # noqa: ARG001
            raise RuntimeError("simulated mid-Phase-1 DB failure")

        monkeypatch.setattr(
            training_suite_service,
            "_create_chain_rows_in_session",
            _boom,
        )
        # An unexpected internal failure is NOT a client validation error:
        # it propagates (→ 500 at the router) rather than being flattened
        # into a 400 string. The rollback guarantee still holds — the `with
        # Session` context manager rolls back on the way out.
        try:
            with pytest.raises(RuntimeError, match="simulated mid-Phase-1"):
                await training_suite_service.create_training_suite(
                    PID,
                    student_base_model_config_ids=[MC_8B],
                    training_preset="standard",
                    include_auto_labeled=False,
                    export_field_mode="all",
                    quantization_schemes=["FP8_DYNAMIC"],
                    idempotency_key="rollback-1",
                    settings=settings,
                )
        finally:
            monkeypatch.setattr(
                training_suite_service,
                "_create_chain_rows_in_session",
                real_fn,
            )

        # No TAOJob rows, no TrainingSuite rows after rollback; the two
        # committed DatasetExports (training + evaluation) remain,
        # consistent with their archives on disk.
        with Session(engine) as s:
            assert s.query(TrainingSuite).count() == 0
            assert s.query(TAOJob).count() == 0
            from vlm_feedback_loop.db.models.dataset_export import DatasetExport

            assert s.query(DatasetExport).count() == 2
        mock_submit.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════════
# SQLite write discipline
# ═══════════════════════════════════════════════════════════════════════════


class TestWriteDiscipline:
    @pytest.mark.asyncio
    async def test_no_write_transaction_held_across_uploads(self, seeded, mock_submit):
        """Write discipline: while the dataset archives stream to the TAO
        workspace S3 (a long network operation), the project database
        must accept writes from other workers — the interactive labeling
        loop keeps saving labels during suite creation. A write
        transaction held open across the upload surfaces here as
        'database is locked' on an independent probe connection."""
        import sqlite3

        engine, project_dir, settings = seeded
        db_path = project_dir / "project.db"
        probe_results: list[str] = []

        async def probing_upload(
            session,
            *,
            dataset_export,
            archive_path,
            deployment_config,
            s3_client,
            **kw,
        ):
            # Simulate "mid-upload": try a write from a second connection
            # with a short busy timeout. DELETE acquires the write lock
            # even when no row matches, so no data changes either way.
            conn = sqlite3.connect(str(db_path), timeout=0.5)
            try:
                conn.execute("DELETE FROM labels WHERE label_id = 'no-such-row'")
                conn.commit()
                probe_results.append("ok")
            except sqlite3.OperationalError as exc:
                probe_results.append(f"locked: {exc}")
            finally:
                conn.close()
            return await _noop_upload(
                session,
                dataset_export=dataset_export,
                archive_path=archive_path,
                deployment_config=deployment_config,
                s3_client=s3_client,
                **kw,
            )

        result = await training_suite_service.create_training_suite(
            PID,
            student_base_model_config_ids=[MC_8B],
            training_preset="standard",
            include_auto_labeled=False,
            export_field_mode="all",
            quantization_schemes=["FP8_DYNAMIC"],
            idempotency_key="write-discipline-1",
            settings=settings,
            _upload_archive=probing_upload,
            _s3_client=object(),
        )
        assert not isinstance(result, str), result
        assert probe_results == ["ok", "ok"]


# ═══════════════════════════════════════════════════════════════════════════
# Guidance pinning across phases (identical human turn)
# ═══════════════════════════════════════════════════════════════════════════


class TestGuidancePinning:
    @pytest.mark.asyncio
    async def test_guidance_change_mid_flight_is_refused(
        self, seeded, mock_submit, monkeypatch
    ):
        """Both dataset exports are pinned to the Guidance that was active
        when suite creation began — training and evaluation exports MUST
        render the identical human turn. If the SME activates a
        different Guidance while the archives build and upload, the suite
        is refused with a conflict (retry uses the new Guidance) instead
        of silently training on a stale or split pair."""
        from vlm_feedback_loop.db.models.dataset_export import DatasetExport
        from vlm_feedback_loop.db.models.project import Project

        engine, project_dir, settings = seeded

        real_create = training_suite_service.create_dataset_export
        flipped = {"done": False}

        def flipping_create(*args, **kwargs):
            # Between the two export builds (the first committed, the
            # second not yet started — no write lock held), the SME
            # activates a new Guidance version.
            if kwargs.get("dataset_intent") == "evaluation" and not flipped["done"]:
                flipped["done"] = True
                with Session(engine) as flip:
                    add_guidance_row(
                        flip,
                        PID,
                        "guid-002",
                        FIXTURE_SCHEMA,
                        version_number=2,
                        description="Changed mid-flight.",
                    )
                    proj = flip.query(Project).filter_by(project_id=PID).first()
                    assert proj is not None
                    proj.active_guidance_id = "guid-002"
                    flip.commit()
            return real_create(*args, **kwargs)

        monkeypatch.setattr(
            training_suite_service,
            "create_dataset_export",
            flipping_create,
        )
        result = await training_suite_service.create_training_suite(
            PID,
            student_base_model_config_ids=[MC_8B],
            training_preset="standard",
            include_auto_labeled=False,
            export_field_mode="all",
            quantization_schemes=["FP8_DYNAMIC"],
            idempotency_key="guidance-flip-1",
            settings=settings,
        )

        assert isinstance(result, str), result
        assert result.startswith("conflict:")
        with Session(engine) as s:
            assert s.query(TrainingSuite).count() == 0
            assert s.query(TAOJob).count() == 0
            # Both exports were built under the ORIGINAL pinned Guidance,
            # not split across versions.
            exports = s.query(DatasetExport).all()
            assert len(exports) == 2
            assert {e.guidance_id for e in exports} == {GID}


class TestGuidanceEditStraddle:
    @pytest.mark.asyncio
    async def test_edit_committing_inside_phase_1d_window_conflicts(
        self, seeded, mock_submit
    ):
        """Every read before Phase 1d's transaction runs on autocommit, so
        an edit committing just before the suite's first write previously
        persisted a TrainingSuite pinned to a retired Guidance — hours of
        TAO training against a schema the SME just retired. The post-flush
        re-read under the write lock must catch it and refuse with a
        conflict. The listener commits the edit at the transaction's first
        TAO write, inside the exact window."""
        from sqlalchemy import event, text

        engine, project_dir, settings = seeded
        flipped = {"done": False}

        def _flip_at_first_tao_write(
            conn, cursor, statement, parameters, context, executemany
        ):
            if flipped["done"] or not statement.startswith("INSERT INTO tao"):
                return
            flipped["done"] = True
            with engine.begin() as c2:
                c2.execute(
                    text(
                        "INSERT INTO guidances (guidance_id, project_id, "
                        "version_number, description, schema, rules, "
                        "created_at) SELECT 'g2-mid-suite', project_id, 2, "
                        "'v2', schema, rules, created_at FROM guidances "
                        "WHERE guidance_id = :g"
                    ),
                    {"g": GID},
                )
                c2.execute(
                    text(
                        "UPDATE projects SET active_guidance_id = "
                        "'g2-mid-suite' WHERE project_id = :p"
                    ),
                    {"p": PID},
                )

        event.listen(engine, "before_cursor_execute", _flip_at_first_tao_write)
        try:
            result = await training_suite_service.create_training_suite(
                PID,
                student_base_model_config_ids=[MC_8B],
                training_preset="standard",
                include_auto_labeled=False,
                export_field_mode="all",
                quantization_schemes=["FP8_DYNAMIC"],
                idempotency_key="k-straddle",
                settings=settings,
            )
        finally:
            event.remove(engine, "before_cursor_execute", _flip_at_first_tao_write)

        assert flipped["done"], "the mid-suite edit never fired"
        assert isinstance(result, str)
        assert result.startswith("conflict:")
        assert "active Guidance changed" in result
        with Session(engine) as s:
            assert s.query(TrainingSuite).count() == 0


# ═══════════════════════════════════════════════════════════════════════════
# Workspace S3 dataset-upload wiring
# ═══════════════════════════════════════════════════════════════════════════


class TestWorkspaceS3UploadWiring:
    """Dataset archives are uploaded to the TAO workspace's cloud storage
    after the DatasetExports are written and before any chain rows are
    created: job specs reference the uploaded objects (never local
    paths), a failed upload rolls everything back, and an idempotent
    replay never re-uploads."""

    @pytest.mark.asyncio
    async def test_upload_runs_after_export_and_before_chain_rows(
        self, seeded, mock_submit
    ):
        """Upload runs AFTER DatasetExports exist and BEFORE chain rows."""
        engine, _, settings = seeded

        call_log: list[str] = []

        async def recording_upload(
            session,
            *,
            dataset_export,
            archive_path,
            deployment_config,
            s3_client,
            **kw,
        ):
            # At upload time, the DatasetExport row exists (must have an id).
            assert dataset_export.dataset_export_id
            # But NO TAOJob chain rows yet.
            tao_count = session.query(TAOJob).count()
            call_log.append(
                f"upload:{dataset_export.dataset_intent}:tao_count={tao_count}"
            )
            return await _noop_upload(
                session,
                dataset_export=dataset_export,
                archive_path=archive_path,
                deployment_config=deployment_config,
                s3_client=s3_client,
                **kw,
            )

        result = await training_suite_service.create_training_suite(
            PID,
            student_base_model_config_ids=[MC_8B],
            training_preset="standard",
            include_auto_labeled=False,
            export_field_mode="all",
            quantization_schemes=["FP8_DYNAMIC"],
            idempotency_key="upload-order-k",
            settings=settings,
            _upload_archive=recording_upload,
            _s3_client=object(),
        )
        assert isinstance(result, dict)

        # Both uploads ran, both BEFORE chain creation (tao_count == 0).
        assert call_log == [
            "upload:training:tao_count=0",
            "upload:evaluation:tao_count=0",
        ]

        # Chain rows exist after the run completes.
        with Session(engine) as s:
            assert (
                s.query(TAOJob).count() >= 3
            )  # at least train + eval + 1 quantize pair

    @pytest.mark.asyncio
    async def test_job_specs_reference_s3_not_local_paths(self, seeded, mock_submit):
        """TAOJob specs reference the S3 spec_reference, not local paths."""
        engine, _, settings = seeded

        result = await training_suite_service.create_training_suite(
            PID,
            student_base_model_config_ids=[MC_8B],
            training_preset="quick",
            include_auto_labeled=False,
            export_field_mode="all",
            quantization_schemes=["FP8_DYNAMIC"],
            idempotency_key="spec-ref-k",
            settings=settings,
            _upload_archive=_noop_upload,
            _s3_client=object(),
        )
        assert isinstance(result, dict)

        # Inspect the persisted TAOJob rows: every spec path must be the
        # s3:// spec reference, never a local filesystem path.
        with Session(engine) as s:
            jobs = s.query(TAOJob).all()
            assert jobs
            for job in jobs:
                specs = job.tao_create_job_request.get("specs", {})
                # Paths live in different places per action; flatten all.
                # train → custom.train_dataset.{media_path,
                # annotation_path} (cosmos-rl CustomConfig). evaluate +
                # quantize → top-level dataset.{media_dir, annotation_path}
                # (cosmos-rl-quantize accepts --media_dir;
                # --media_path is rejected as an unknown arg).
                all_paths: list[str] = []
                custom = specs.get("custom") or {}
                for key in ("train_dataset", "val_dataset"):
                    ds = custom.get(key) or {}
                    all_paths.append(str(ds.get("media_path", "")))
                    all_paths.append(str(ds.get("annotation_path", "")))
                if "dataset" in specs:
                    ds = specs["dataset"]
                    all_paths.append(str(ds.get("media_dir", "")))
                    all_paths.append(str(ds.get("annotation_path", "")))
                for p in all_paths:
                    if p:
                        # Test fixture has cloud_type="seaweedfs" so the
                        # spec reference must use the seaweedfs:// scheme
                        # per the workspace's cloud_type (TAO dispatches
                        # on URL scheme; a mismatch surfaces as
                        # KeyError: 'aws').
                        assert p.startswith(("s3://", "seaweedfs://", "azure://")), (
                            f"spec path not a workspace-storage reference: "
                            f"{p!r} on {job.action} job"
                        )
                        assert "/exports/" not in p, (
                            f"Blueprint-local path leaked into TAO spec: {p!r}"
                        )

    @pytest.mark.asyncio
    async def test_upload_failure_rolls_back_without_orphan_chains(
        self, seeded, mock_submit
    ):
        """A failed S3 upload aborts Phase 1 — no TAOJobs + no TrainingSuite."""
        engine, _, settings = seeded

        from vlm_feedback_loop.services.tao_dataset_upload_service import UploadResult

        async def failing_upload(
            session,
            *,
            dataset_export,
            archive_path,
            deployment_config,
            s3_client,
            **kw,
        ):
            return UploadResult(
                success=False,
                dataset_export_id=dataset_export.dataset_export_id,
                error="S3 upload failed: 403 Forbidden",
            )

        result = await training_suite_service.create_training_suite(
            PID,
            student_base_model_config_ids=[MC_8B],
            training_preset="quick",
            include_auto_labeled=False,
            export_field_mode="all",
            quantization_schemes=["FP8_DYNAMIC"],
            idempotency_key="upload-fail-k",
            settings=settings,
            _upload_archive=failing_upload,
            _s3_client=object(),
        )
        # A workspace-S3 upload failure is infra, not client input: the
        # service must classify it tao_unreachable (→ 503 via
        # map_service_error), not as a 400 validation error.
        assert isinstance(result, str)
        assert result.startswith("tao_unreachable:")
        from vlm_feedback_loop.services.errors import map_service_error

        assert map_service_error(result).status_code == 503
        # No TrainingSuite and no TAOJob rows persist. The two Phase-1b
        # DatasetExports remain — they committed before the upload began
        # (the upload runs outside any write transaction) and
        # match the archives already on disk.
        with Session(engine) as s:
            assert s.query(TrainingSuite).count() == 0
            assert s.query(TAOJob).count() == 0
            from vlm_feedback_loop.db.models.dataset_export import DatasetExport

            assert s.query(DatasetExport).count() == 2
        mock_submit.assert_not_called()

    @pytest.mark.asyncio
    async def test_idempotent_retry_skips_upload(self, seeded, mock_submit):
        """Re-POST with the same idempotency key does not re-invoke upload."""
        engine, _, settings = seeded

        call_count = {"n": 0}

        async def counting_upload(
            session,
            *,
            dataset_export,
            archive_path,
            deployment_config,
            s3_client,
            **kw,
        ):
            call_count["n"] += 1
            return await _noop_upload(
                session,
                dataset_export=dataset_export,
                archive_path=archive_path,
                deployment_config=deployment_config,
                s3_client=s3_client,
                **kw,
            )

        first = await training_suite_service.create_training_suite(
            PID,
            student_base_model_config_ids=[MC_8B],
            training_preset="quick",
            include_auto_labeled=False,
            export_field_mode="all",
            quantization_schemes=["FP8_DYNAMIC"],
            idempotency_key="retry-shared",
            settings=settings,
            _upload_archive=counting_upload,
            _s3_client=object(),
        )
        assert isinstance(first, dict)
        first_n = call_count["n"]
        assert first_n == 2  # train + eval

        # Second POST with the SAME idempotency key.
        second = await training_suite_service.create_training_suite(
            PID,
            student_base_model_config_ids=[MC_8B],
            training_preset="quick",
            include_auto_labeled=False,
            export_field_mode="all",
            quantization_schemes=["FP8_DYNAMIC"],
            idempotency_key="retry-shared",
            settings=settings,
            _upload_archive=counting_upload,
            _s3_client=object(),
        )
        assert isinstance(second, dict)
        # No new upload calls — the replay path short-circuited.
        assert call_count["n"] == first_n
        # Suite ID is stable.
        assert first["training_suite_id"] == second["training_suite_id"]


class TestQuantizePayloadSpec:
    def test_quantize_spec_sets_vlm_adequate_max_sequence_length(self):
        """cosmos-rl-quantize tokenizes calibration samples with
        truncation=True at max_sequence_length (CLI default 2048). VLM
        calibration images expand to thousands of image tokens (a
        4032x3024 photo is ~11.8k), so leaving the default truncates the
        <image> expansion and every calibration batch fails with
        "Mismatch in `image` token count" — observed live on FTMS 6.26.3
        (2026-07-14). The Blueprint must pin an adequate ceiling."""
        from vlm_feedback_loop.services.training_suite_service import (
            _build_quantize_payload,
        )

        job_config, request = _build_quantize_payload(
            quantization_method="FP8_DYNAMIC",
            training_archive_path="seaweedfs://bucket/exports/e1/e1.tar.gz",
            training_annotation_path="seaweedfs://bucket/exports/e1/a.json",
            parent_tao_job_id="parent-ext-id",
            tao_release_version="6.26.3",
            cosmos_rl_container_tag="6.26.3-cosmos-rl",
            training_preset="quick",
            workspace_id="ws-1",
            base_experiment_id="be-1",
            job_name="vlm-fb-quantize-test",
        )

        specs = request["specs"]
        assert specs["max_sequence_length"] == 16384
        assert specs["quantization_scheme"] == "FP8_DYNAMIC"
        # Train-specific blocks stay out of the quantize spec (the
        # cosmos-rl-quantize CLI rejects their flag names).
        assert "train" not in specs and "policy" not in specs

    def test_lora_chain_quantize_spec_carries_merge_flags(self):
        """A quantize on an adapter-only parent must tell the container to
        merge: cosmos-rl-quantize's loader raises "base_model_path is
        required when enable_lora is True" when the LoRA flag arrives
        (FTMS injects it from the parent's train spec) without a base
        model path — observed live on FTMS 6.26.3 (2026-07-15), both
        GT-nano quantize legs. The spec must carry ``enable_lora`` and a
        BARE HuggingFace id (the container feeds it straight to
        ``from_pretrained``; ``hf_model://`` is a train/eval-side FTMS
        convention it does not understand)."""
        from vlm_feedback_loop.services.training_suite_service import (
            _build_quantize_payload,
        )

        _, request = _build_quantize_payload(
            quantization_method="FP8_DYNAMIC",
            training_archive_path="seaweedfs://bucket/exports/e1/e1.tar.gz",
            training_annotation_path="seaweedfs://bucket/exports/e1/a.json",
            parent_tao_job_id="parent-ext-id",
            tao_release_version="6.26.3",
            cosmos_rl_container_tag="6.26.3-cosmos-rl",
            training_preset="quick",
            workspace_id="ws-1",
            base_experiment_id="be-1",
            job_name="vlm-fb-quantize-test",
            enable_lora=True,
            base_model_path="nvidia/Cosmos3-Nano-Reasoner",
        )

        specs = request["specs"]
        assert specs["enable_lora"] is True
        assert specs["base_model_path"] == "nvidia/Cosmos3-Nano-Reasoner"
        assert not specs["base_model_path"].startswith("hf_model://")

    def test_full_weight_chain_quantize_spec_omits_merge_flags(self):
        """Full-weight chains keep the plain quantize wire shape: no
        ``enable_lora`` / ``base_model_path`` keys at all (the checkpoint
        already contains full model shards; sending merge flags would make
        the container attempt a pointless adapter merge)."""
        from vlm_feedback_loop.services.training_suite_service import (
            _build_quantize_payload,
        )

        _, request = _build_quantize_payload(
            quantization_method="FP8_DYNAMIC",
            training_archive_path="seaweedfs://bucket/exports/e1/e1.tar.gz",
            training_annotation_path="seaweedfs://bucket/exports/e1/a.json",
            parent_tao_job_id="parent-ext-id",
            tao_release_version="6.26.3",
            cosmos_rl_container_tag="6.26.3-cosmos-rl",
            training_preset="quick",
            workspace_id="ws-1",
            base_experiment_id="be-1",
            job_name="vlm-fb-quantize-test",
        )

        specs = request["specs"]
        assert "enable_lora" not in specs
        assert "base_model_path" not in specs

    def test_lora_without_base_model_path_is_rejected_at_build_time(self):
        """Building a LoRA-chain quantize spec without a base model path
        must fail at build time, not as a live container crash 10 minutes
        into an FTMS job."""
        from vlm_feedback_loop.services.training_suite_service import (
            _build_quantize_payload,
        )

        with pytest.raises(ValueError, match="base_model_path"):
            _build_quantize_payload(
                quantization_method="FP8_DYNAMIC",
                training_archive_path="seaweedfs://bucket/exports/e1/e1.tar.gz",
                training_annotation_path="seaweedfs://bucket/exports/e1/a.json",
                parent_tao_job_id="parent-ext-id",
                tao_release_version="6.26.3",
                cosmos_rl_container_tag="6.26.3-cosmos-rl",
                training_preset="quick",
                workspace_id="ws-1",
                base_experiment_id="be-1",
                job_name="vlm-fb-quantize-test",
                enable_lora=True,
            )


# ═══════════════════════════════════════════════════════════════════════════
# Suite-level best-effort cancellation
# ═══════════════════════════════════════════════════════════════════════════


def _seed_suite_for_bulk_cancel(engine) -> None:
    with Session(engine) as session:
        session.add(
            TrainingSuite(
                training_suite_id="suite-bulk-cancel",
                project_id=PID,
                idempotency_key="bulk-cancel",
                guidance_id=GID,
                training_preset="standard",
                export_field_mode="all",
                include_auto_labeled=False,
                training_dataset_export_id="de-train",
                evaluation_dataset_export_id="de-eval",
                selected_student_base_model_config_ids=[MC_8B, MC_2B],
                quantization_schemes=["FP8_DYNAMIC"],
                chain_ids_ordered=["cancel-chain-a", "cancel-chain-b"],
                provisioning_run_id="provision-1",
                provisioning_model_names=[COSMOS_REASON2_8B],
                status="running",
                started_at=utc_now(),
            )
        )
        for job_id, chain_id, sequence, status, external_id in (
            ("done", "cancel-chain-a", 1, "succeeded", "ext-done"),
            ("running", "cancel-chain-a", 2, "running", "ext-running"),
            ("waiting", "cancel-chain-a", 3, "not_started", None),
            ("submitting", "cancel-chain-b", 1, "submitting", None),
        ):
            session.add(
                TAOJob(
                    tao_job_id=job_id,
                    project_id=PID,
                    student_base_model_config_id=MC_8B,
                    dataset_export_ids=["de-train"],
                    action="train" if sequence == 1 else "evaluate",
                    status=status,
                    training_backend="cosmos_rl_tao_vlm",
                    training_policy_type="sft",
                    job_config={},
                    tao_create_job_request={},
                    tao_external_job_id=external_id,
                    chain_id=chain_id,
                    chain_sequence=sequence,
                    completed_at=utc_now() if status == "succeeded" else None,
                )
            )
        session.commit()


class TestCancelTrainingSuite:
    @pytest.mark.asyncio
    async def test_cancels_every_remaining_job_and_releases_suite(
        self, seeded, monkeypatch
    ):
        engine, _, settings = seeded
        _seed_suite_for_bulk_cancel(engine)

        remote_cancel = AsyncMock(return_value={"success": True, "error": None})
        cancel_task = AsyncMock(side_effect=[True, True])
        monkeypatch.setattr(
            training_suite_service.tao_job_service,
            "request_tao_job_cancel",
            remote_cancel,
        )
        monkeypatch.setattr(
            training_suite_service.background_manager,
            "cancel_task",
            cancel_task,
        )
        monkeypatch.setattr(training_suite_service.sse_manager, "emit", AsyncMock())

        result = await training_suite_service.cancel_training_suite(
            PID,
            "suite-bulk-cancel",
            settings=settings,
        )

        assert not isinstance(result, str), result
        assert result["training_suite"]["status"] == "canceled"
        assert result["jobs_canceled"] == 3
        assert result["jobs_already_terminal"] == 1
        assert result["setup_tasks_canceled"] == 2
        assert result["remote_cancel_failures"] == []
        remote_cancel.assert_awaited_once_with("ext-running", settings=settings)
        cancel_task.assert_has_awaits(
            [
                call("training-suite-setup-suite-bulk-cancel"),
                call("tao-base-provision-provision-1"),
            ]
        )

        with Session(engine) as session:
            suite = session.get(TrainingSuite, "suite-bulk-cancel")
            assert suite is not None
            assert suite.status == "canceled"
            assert suite.completed_at is not None
            statuses = {
                job.tao_job_id: job.status
                for job in session.query(TAOJob)
                .filter(TAOJob.chain_id.in_(["cancel-chain-a", "cancel-chain-b"]))
                .all()
            }
        assert statuses == {
            "done": "succeeded",
            "running": "canceled",
            "waiting": "canceled",
            "submitting": "canceled",
        }

    @pytest.mark.asyncio
    async def test_remote_failure_is_reported_but_local_suite_still_releases(
        self, seeded, monkeypatch
    ):
        engine, _, settings = seeded
        _seed_suite_for_bulk_cancel(engine)
        monkeypatch.setattr(
            training_suite_service.tao_job_service,
            "request_tao_job_cancel",
            AsyncMock(
                return_value={
                    "success": False,
                    "error": "TAO cancel failed: endpoint unavailable",
                }
            ),
        )
        monkeypatch.setattr(
            training_suite_service.background_manager,
            "cancel_task",
            AsyncMock(return_value=False),
        )
        monkeypatch.setattr(training_suite_service.sse_manager, "emit", AsyncMock())

        result = await training_suite_service.cancel_training_suite(
            PID,
            "suite-bulk-cancel",
            settings=settings,
        )

        assert not isinstance(result, str), result
        assert result["training_suite"]["status"] == "canceled"
        assert result["remote_cancel_failures"] == [
            {
                "tao_job_id": "running",
                "error": "TAO cancel failed: endpoint unavailable",
            }
        ]
        with Session(engine) as session:
            running = session.get(TAOJob, "running")
            assert running is not None
            assert running.status == "canceled"
            assert "suite_cancel_unconfirmed" in (running.poll_error_ref or "")
