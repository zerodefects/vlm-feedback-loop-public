# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``deployment_handoff`` Action Request generator.

Covers:
  - ``409 INFERENCE_CONTRACT_MISMATCH`` when the training Contract differs
    from the evaluation Contract.
  - Dual-gate behavior + happy path.
  - Contract snapshot + Teacher contract constants.
"""

from __future__ import annotations

import pytest
from sqlalchemy.orm import Session

from conftest import make_settings
from vlm_feedback_loop.db.base import generate_uuid4
from vlm_feedback_loop.db.models.dataset_export import DatasetExport
from vlm_feedback_loop.db.models.guidance import Guidance
from vlm_feedback_loop.db.models.local_nim_deployment import LocalNimDeployment
from vlm_feedback_loop.db.models.model_config import ModelConfig
from vlm_feedback_loop.db.models.run import RunRecord
from vlm_feedback_loop.db.models.student_model import StudentModel
from vlm_feedback_loop.db.models.tao_job import TAOJob
from vlm_feedback_loop.model_catalog_constants import (
    COSMOS3_SUPER_REASONER,
    COSMOS_REASON2_2B,
    COSMOS_REASON2_2B_NIM_IMAGE,
    COSMOS_REASON2_8B,
    COSMOS_REASON2_8B_NIM_IMAGE,
)
from vlm_feedback_loop.services import deployment_handoff_generator, local_nim_service

# ── Helper: seed a Student in a fully-validated state ───────────────────────


def _seed_validated_student(
    settings,
    *,
    training_contract: dict,
    serving_contract: dict,
    quality_status: str = "validated",
    serving_status: str = "validated",
    base_model_name: str = "nvidia/cosmos-reason2-2b-suite",
    quantization_method: str = "fp8",
    nim_container_image: str = COSMOS_REASON2_2B_NIM_IMAGE,
    nim_model_profile_selected: str | None = "vllm-tp1-fp8",
    profile_tp: int = 1,
    project_id_override: str | None = None,
) -> tuple[str, str]:
    """Create a project + Student in the requested gate state.

    Returns ``(project_id, student_model_id)``.
    """
    from vlm_feedback_loop.services import project_service

    if project_id_override is not None:
        project_id = project_id_override
    else:
        project = project_service.create_project("handoff-test", None, settings)
        project_id = project.project_id
    engine = project_service.get_project_engine(project_id, settings.WORKSPACE_ROOT)

    student_model_id = generate_uuid4()
    base_mc_id = generate_uuid4()
    quality_run_id = generate_uuid4()
    serving_run_id = generate_uuid4()
    de_train_id = generate_uuid4()
    de_test_id = generate_uuid4()
    guidance_id = generate_uuid4()
    tao_job_id = generate_uuid4()
    now = "2026-04-29T00:00:00Z"

    with Session(engine) as session:
        # Reuse seeded endpoint; create dedicated base mc, guidance, exports
        from vlm_feedback_loop.db.models.nim_endpoint import NimEndpoint

        endpoint_id = (
            session.query(NimEndpoint).filter_by(project_id=project_id).first()
        ).endpoint_id
        session.add(
            ModelConfig(
                model_config_id=base_mc_id,
                project_id=project_id,
                endpoint_id=endpoint_id,
                model_name=base_model_name,
                context_window_tokens=256000,
                eligible_roles=["student_base"],
                supports_image_input=True,
            )
        )
        # Compute the next version so seeding a second Student into the same
        # project (the paired 2B/8B fixture) doesn't collide on the
        # UNIQUE(project_id, version_number) constraint.
        from sqlalchemy import func, select

        existing_max = session.execute(
            select(func.max(Guidance.version_number)).where(
                Guidance.project_id == project_id
            )
        ).scalar()
        session.add(
            Guidance(
                project_id=project_id,
                guidance_id=guidance_id,
                version_number=(existing_max or 0) + 1,
                description="d",
                schema={"core_fields": [], "aux_fields": []},
                rules="",
                created_at=now,
            )
        )
        session.add(
            DatasetExport(
                dataset_export_id=de_train_id,
                project_id=project_id,
                dataset_intent="training",
                export_field_mode=training_contract["output_field_mode"],
                guidance_id=guidance_id,
                label_tier_filter="verified",
                selection_definition_snapshot={},
                artifact_refs={"checksum_sha256": "train-sha"},
                manifest_ref="m1",
                example_count=10,
            )
        )
        session.add(
            DatasetExport(
                dataset_export_id=de_test_id,
                project_id=project_id,
                dataset_intent="testing",
                export_field_mode=training_contract["output_field_mode"],
                guidance_id=guidance_id,
                label_tier_filter="verified",
                selection_definition_snapshot={},
                artifact_refs={"checksum_sha256": "test-pool-sha"},
                manifest_ref="m2",
                example_count=20,
            )
        )
        session.add(
            TAOJob(
                tao_job_id=tao_job_id,
                project_id=project_id,
                student_base_model_config_id=base_mc_id,
                dataset_export_ids=[de_train_id],
                training_backend="cosmos_rl_tao_vlm",
                action="train",
                status="succeeded",
                tao_status_raw="Done",
                job_config={},
                tao_create_job_request={},
                created_at=now,
            )
        )
        session.add(
            RunRecord(
                run_id=quality_run_id,
                project_id=project_id,
                run_type="evaluation_run",
                status="completed",
                created_at=now,
                started_at=now,
                completed_at=now,
                evaluation_source="tao",
                model_config_id=base_mc_id,
                inference_contract=training_contract,
                metrics={"overall": {"exact_match_rate": 0.9}},
                rescored_metrics={"overall": {"exact_match_rate": 0.9}},
            )
        )
        session.add(
            RunRecord(
                run_id=serving_run_id,
                project_id=project_id,
                run_type="evaluation_run",
                status="completed",
                created_at=now,
                started_at=now,
                completed_at=now,
                evaluation_source="nim",
                model_config_id=base_mc_id,
                inference_contract=serving_contract,
                metrics={"overall": {"exact_match_rate": 0.85}},
                visual_budget_preset_key="balanced",
                generation_preset_key="precise",
                thinking_mode_effective="on",
            )
        )
        session.add(
            StudentModel(
                student_model_id=student_model_id,
                project_id=project_id,
                student_base_model_config_id=base_mc_id,
                tao_job_id=tao_job_id,
                guidance_id=guidance_id,
                dataset_export_ids=[de_train_id, de_test_id],
                training_preset="standard",
                lora_config={"enable_lora": True},
                created_at=now,
                checkpoint_packaging_status="validated",
                nim_checkpoint_ref="/tmp/ckpt",
                quality_status=quality_status,
                quality_evaluation_run_id=quality_run_id,
                serving_status=serving_status,
                serving_evaluation_run_id=serving_run_id,
                nim_endpoint_url="http://localhost:8000",
                nim_model_profile_requested="vllm-tp1",
                nim_model_profile_selected=nim_model_profile_selected,
                nim_profile_metadata={
                    "backend": "vllm",
                    "precision": "fp8",
                    "tp": profile_tp,
                },
                gpu_type="A100",
                gpu_count=1,
                quantization_method=quantization_method,
                training_inference_contract=training_contract,
            )
        )
        # Seed a LocalNimDeployment representing the post-:deploy_nim state.
        # The handoff success path requires it (the canonical docker run is
        # built from this row's fields, not StudentModel's). Without it, the
        # generator emits a "(deployment not yet recorded …)" placeholder —
        # correct, but not what the success-path tests are exercising.
        if serving_status == "validated":
            session.add(
                LocalNimDeployment(
                    local_nim_deployment_id=generate_uuid4(),
                    project_id=project_id,
                    model_config_id=base_mc_id,
                    role="student",
                    nim_container_image=nim_container_image,
                    container_name=f"vlm-student-{project_id[:8]}-{student_model_id[:8]}",
                    container_id="abcdef012345",
                    host_port=8000,
                    endpoint_url="http://localhost:8000/v1",
                    gpu_assignment="device=0",
                    status="running",
                    student_model_id=student_model_id,
                    checkpoint_mount_path="/tmp/ckpt",
                    nim_served_model_name=f"student-{student_model_id[:8]}",
                    nim_model_name_path="/opt/checkpoints/student",
                    precision_method=quantization_method,
                )
            )
        session.commit()

    return project_id, student_model_id


@pytest.fixture()
def settings_in_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    return make_settings(workspace)


# ── Tests ───────────────────────────────────────────────────────────────────


_ALL_FIELDS = {
    "output_field_mode": "all",
    "icl_field_mode": "all",
    "icl_max_examples": None,
}
_CORE_ONLY = {
    "output_field_mode": "core_only",
    "icl_field_mode": "core_only",
    "icl_max_examples": None,
}


def test_handoff_blocks_on_quality_pending(settings_in_workspace):
    """409 quality_status_not_validated."""
    project_id, student_id = _seed_validated_student(
        settings_in_workspace,
        training_contract=_ALL_FIELDS,
        serving_contract=_ALL_FIELDS,
        quality_status="pending",
    )
    result = deployment_handoff_generator.generate_deployment_handoff_for_student(
        project_id=project_id,
        student_model_id=student_id,
        settings=settings_in_workspace,
    )
    assert isinstance(result, str), result
    assert "quality_status_not_validated" in result


def test_handoff_blocks_on_quality_partial_with_dedicated_conflict_code(
    settings_in_workspace,
):
    """``quality_status="partial"`` returns 409 with body
    ``conflict: quality_status_partial`` (distinct from
    ``conflict: quality_status_not_validated``) so the frontend can
    render partial-specific guidance. Even with serving=validated, the
    handoff dual gate refuses partial."""
    project_id, student_id = _seed_validated_student(
        settings_in_workspace,
        training_contract=_ALL_FIELDS,
        serving_contract=_ALL_FIELDS,
        quality_status="partial",
    )
    result = deployment_handoff_generator.generate_deployment_handoff_for_student(
        project_id=project_id,
        student_model_id=student_id,
        settings=settings_in_workspace,
    )
    assert isinstance(result, str), result
    assert "quality_status_partial" in result
    # Distinct conflict code — must NOT match the not_validated body.
    assert "quality_status_not_validated" not in result


def test_handoff_partial_quality_takes_precedence_over_serving_pending(
    settings_in_workspace,
):
    """When quality=partial AND serving=pending, the partial
    conflict fires first (so the SME's first remediation step is
    'rerun NIM eval to clear partial', not 'benchmark for serving').
    Documents the precedence the generator's branch order produces."""
    project_id, student_id = _seed_validated_student(
        settings_in_workspace,
        training_contract=_ALL_FIELDS,
        serving_contract=_ALL_FIELDS,
        quality_status="partial",
        serving_status="pending",
    )
    result = deployment_handoff_generator.generate_deployment_handoff_for_student(
        project_id=project_id,
        student_model_id=student_id,
        settings=settings_in_workspace,
    )
    assert isinstance(result, str), result
    assert "quality_status_partial" in result


def test_handoff_blocks_on_serving_failed(settings_in_workspace):
    """409 serving_status_not_validated."""
    project_id, student_id = _seed_validated_student(
        settings_in_workspace,
        training_contract=_ALL_FIELDS,
        serving_contract=_ALL_FIELDS,
        serving_status="failed",
    )
    result = deployment_handoff_generator.generate_deployment_handoff_for_student(
        project_id=project_id,
        student_model_id=student_id,
        settings=settings_in_workspace,
    )
    assert isinstance(result, str), result
    assert "serving_status_not_validated" in result


def test_handoff_blocks_on_contract_mismatch(settings_in_workspace):
    """409 INFERENCE_CONTRACT_MISMATCH when the contracts differ."""
    project_id, student_id = _seed_validated_student(
        settings_in_workspace,
        training_contract=_CORE_ONLY,
        serving_contract=_ALL_FIELDS,
    )
    result = deployment_handoff_generator.generate_deployment_handoff_for_student(
        project_id=project_id,
        student_model_id=student_id,
        settings=settings_in_workspace,
    )
    assert isinstance(result, str), result
    assert "INFERENCE_CONTRACT_MISMATCH" in result


def test_handoff_success(settings_in_workspace):
    """200 with technical_requirements + current_environment + rendered_text."""
    project_id, student_id = _seed_validated_student(
        settings_in_workspace,
        training_contract=_ALL_FIELDS,
        serving_contract=_ALL_FIELDS,
    )
    result = deployment_handoff_generator.generate_deployment_handoff_for_student(
        project_id=project_id,
        student_model_id=student_id,
        settings=settings_in_workspace,
    )
    assert isinstance(result, dict), result
    assert "technical_requirements" in result
    assert "current_environment" in result
    assert "rendered_text" in result
    assert result["request_type"] == "deployment_handoff"

    tech = result["technical_requirements"]
    assert tech["quantization_method"] == "fp8"
    assert tech["nim_profile_metadata"]["backend"] == "vllm"
    assert tech["inference_contract"] == _ALL_FIELDS

    env = result["current_environment"]
    assert env["quality_status"] == "validated"
    assert env["serving_status"] == "validated"
    # Quality + serving evaluation summaries persisted
    assert env["quality_evaluation_overall_exact_match"] == 0.9
    assert env["serving_evaluation_overall_exact_match"] == 0.85
    assert env["dataset_manifest_sha256"] == "test-pool-sha"

    # Rendered text: docker run line + key fields
    text = result["rendered_text"]
    assert "Deployment Handoff Request" in text
    assert "docker run" in text
    assert "fp8" in text


def test_contracts_equivalent_helper():
    """Only valid, equal canonical contracts pass the handoff gate."""
    from vlm_feedback_loop.services.deployment_handoff_generator import (
        _contracts_equivalent,
    )

    a = dict(_CORE_ONLY)
    b = dict(_CORE_ONLY)
    assert _contracts_equivalent(a, b) is True

    c = dict(_CORE_ONLY)
    c["output_field_mode"] = "all"
    assert _contracts_equivalent(a, c) is False

    with_extra = dict(_CORE_ONLY)
    with_extra["removed_field"] = None
    assert _contracts_equivalent(a, with_extra) is False

    # None/None backfill case — both empty → True (defensive)
    assert _contracts_equivalent(None, None) is True
    assert _contracts_equivalent({}, {}) is True
    # One None, one populated → False
    assert _contracts_equivalent(None, _ALL_FIELDS) is False


def test_generator_registered_at_import():
    """The deployment_handoff generator is registered with the AR registry."""
    from vlm_feedback_loop.services.action_requests import _generators

    assert "deployment_handoff" in _generators


def test_handoff_blocks_on_serving_run_missing(settings_in_workspace):
    """Edge case: 409 serving_evaluation_run_missing if FK is None."""
    project_id, student_id = _seed_validated_student(
        settings_in_workspace,
        training_contract=_ALL_FIELDS,
        serving_contract=_ALL_FIELDS,
    )
    # Null out serving_evaluation_run_id post-seed.
    from vlm_feedback_loop.services.project_service import get_project_engine

    engine = get_project_engine(project_id, settings_in_workspace.WORKSPACE_ROOT)
    with Session(engine) as session:
        student = (
            session.query(StudentModel).filter_by(student_model_id=student_id).first()
        )
        student.serving_evaluation_run_id = None
        session.commit()

    result = deployment_handoff_generator.generate_deployment_handoff_for_student(
        project_id=project_id,
        student_model_id=student_id,
        settings=settings_in_workspace,
    )
    assert isinstance(result, str)
    assert "serving_evaluation_run_missing" in result


def test_handoff_recovers_contract_from_lineage_on_null_snapshot(
    settings_in_workspace,
):
    """An incomplete Student snapshot is recovered from DatasetExport lineage."""
    project_id, student_id = _seed_validated_student(
        settings_in_workspace,
        training_contract=_CORE_ONLY,  # both will match
        serving_contract=_CORE_ONLY,
    )
    # Null out the snapshot — generator must derive from DatasetExports.
    from vlm_feedback_loop.services.project_service import get_project_engine

    engine = get_project_engine(project_id, settings_in_workspace.WORKSPACE_ROOT)
    with Session(engine) as session:
        student = (
            session.query(StudentModel).filter_by(student_model_id=student_id).first()
        )
        student.training_inference_contract = None
        session.commit()

    result = deployment_handoff_generator.generate_deployment_handoff_for_student(
        project_id=project_id,
        student_model_id=student_id,
        settings=settings_in_workspace,
    )
    # Recomputed contract still matches serving → 200 success
    assert isinstance(result, dict), result
    assert result["technical_requirements"]["inference_contract"] == _CORE_ONLY


# ── 2B vs 8B handoff differentiation ────────────────────────────────────
#
# Four handoff fields MUST differentiate per
# base-model size: ``nim_model_profile_recommended``, ``gpu_requirements``,
# ``tensor_parallelism``, ``nim_env_vars_recommended``. These unit tests
# exercise the differentiation in isolation, without any hardware dependency,
# by seeding paired 2B + 8B Students under different base-model names +
# container images + profile selections, then calling the generator twice and
# asserting on the per-field differences.


def _seed_paired_2b_and_8b(
    settings,
) -> tuple[str, str, str]:
    """Seed a project containing one 2B and one 8B Student, both validated.

    Returns ``(project_id, student_2b_id, student_8b_id)``.
    """
    from vlm_feedback_loop.services import project_service

    project = project_service.create_project("c21-paired", None, settings)
    project_id = project.project_id

    _, student_2b = _seed_validated_student(
        settings,
        training_contract=_ALL_FIELDS,
        serving_contract=_ALL_FIELDS,
        base_model_name=COSMOS_REASON2_2B,
        quantization_method="none",
        nim_container_image=COSMOS_REASON2_2B_NIM_IMAGE,
        nim_model_profile_selected="vllm-cosmos-reason2-2b-bf16",
        profile_tp=1,
        project_id_override=project_id,
    )
    _, student_8b = _seed_validated_student(
        settings,
        training_contract=_ALL_FIELDS,
        serving_contract=_ALL_FIELDS,
        base_model_name=COSMOS_REASON2_8B,
        quantization_method="none",
        nim_container_image=COSMOS_REASON2_8B_NIM_IMAGE,
        nim_model_profile_selected="vllm-cosmos-reason2-8b-bf16",
        profile_tp=1,
        project_id_override=project_id,
    )
    return project_id, student_2b, student_8b


def _generate_paired_handoffs(settings) -> tuple[dict, dict]:
    project_id, student_2b, student_8b = _seed_paired_2b_and_8b(settings)
    handoff_2b = deployment_handoff_generator.generate_deployment_handoff_for_student(
        project_id=project_id, student_model_id=student_2b, settings=settings
    )
    handoff_8b = deployment_handoff_generator.generate_deployment_handoff_for_student(
        project_id=project_id, student_model_id=student_8b, settings=settings
    )
    assert isinstance(handoff_2b, dict), handoff_2b
    assert isinstance(handoff_8b, dict), handoff_8b
    return handoff_2b, handoff_8b


def test_nim_model_profile_recommended_differs(settings_in_workspace):
    """``nim_model_profile_recommended`` differs between 2B and 8B."""
    h2b, h8b = _generate_paired_handoffs(settings_in_workspace)
    profile_2b = h2b["technical_requirements"]["nim_model_profile_recommended"]
    profile_8b = h8b["technical_requirements"]["nim_model_profile_recommended"]
    assert profile_2b is not None
    assert profile_8b is not None
    assert profile_2b != profile_8b, (profile_2b, profile_8b)


def test_gpu_memory_floor_unknown_base_never_zero(settings_in_workspace):
    """The handoff's former mirror returned 0 for any base that wasn't a
    Cosmos Reason2 2B/8B — cosmos3 Students shipped handoffs with NO
    memory floor. The shared policy falls back to the catalog's
    published minimum, then conservatively to the 8B BF16 floor."""
    from vlm_feedback_loop.services.gpu_memory_floor import (
        resolve_gpu_memory_floor_gb,
    )

    settings = settings_in_workspace

    # Catalog-published fallback wins for an unknown base.
    assert (
        resolve_gpu_memory_floor_gb(
            base_model_name=COSMOS3_SUPER_REASONER,
            quantization_method=None,
            base_local_deploy_metadata={"nim_gpu_memory_minimum_gb": 96},
            settings=settings,
        )
        == 96
    )
    # No metadata → conservative 8B BF16 floor, never 0.
    floor = resolve_gpu_memory_floor_gb(
        base_model_name=COSMOS3_SUPER_REASONER,
        quantization_method=None,
        base_local_deploy_metadata=None,
        settings=settings,
    )
    assert floor == settings.NIM_GPU_MEMORY_8B_BF16_GB
    assert floor > 0


def test_gpu_requirements_differs(settings_in_workspace):
    """``gpu_requirements`` differs (memory hint reflects base-model min)."""
    h2b, h8b = _generate_paired_handoffs(settings_in_workspace)
    req_2b = h2b["technical_requirements"]["gpu_requirements"]
    req_8b = h8b["technical_requirements"]["gpu_requirements"]
    assert req_2b != req_8b, (req_2b, req_8b)
    # Single-GPU deploys land "1× A100" on both — the hint is what
    # differentiates. 2B BF16 = 36 GB, 8B BF16 = 56 GB (per Settings).
    assert "36" in req_2b
    assert "56" in req_8b


def test_tensor_parallelism_present_and_int(settings_in_workspace):
    """``tensor_parallelism`` is populated as int on every handoff.

    Single-GPU deploys land tp=1 for BOTH 2B and 8B — the
    differentiation requirement is satisfied by population, not strict
    inequality on every field.
    """
    h2b, h8b = _generate_paired_handoffs(settings_in_workspace)
    tp_2b = h2b["technical_requirements"]["tensor_parallelism"]
    tp_8b = h8b["technical_requirements"]["tensor_parallelism"]
    assert isinstance(tp_2b, int) and tp_2b >= 1
    assert isinstance(tp_8b, int) and tp_8b >= 1


def test_nim_env_vars_recommended_differs(settings_in_workspace):
    """``nim_env_vars_recommended`` content differs (per-Student NIM_*)."""
    h2b, h8b = _generate_paired_handoffs(settings_in_workspace)
    env_2b = h2b["technical_requirements"]["nim_env_vars_recommended"]
    env_8b = h8b["technical_requirements"]["nim_env_vars_recommended"]
    assert env_2b != env_8b, (env_2b, env_8b)
    # NGC_API_KEY + NIM_MODEL_NAME at minimum.
    for env in (env_2b, env_8b):
        assert "NGC_API_KEY" in env
        assert "NIM_MODEL_NAME" in env
    # Per-Student served-model-name must differ (Student-id-derived).
    assert env_2b["NIM_SERVED_MODEL_NAME"] != env_8b["NIM_SERVED_MODEL_NAME"]
    # Profile recommendation flows into env vars when set.
    assert env_2b["NIM_MODEL_PROFILE"] != env_8b["NIM_MODEL_PROFILE"]


def test_handoff_emits_canonical_docker_run(settings_in_workspace):
    """``docker_run_command`` + ``docker_run_args`` are populated.

    ``docker_run_args`` is machine-parseable for direct ``subprocess.run``
    re-execution; the rendered_text uses the display form.
    """
    project_id, student_id = _seed_validated_student(
        settings_in_workspace,
        training_contract=_ALL_FIELDS,
        serving_contract=_ALL_FIELDS,
    )
    result = deployment_handoff_generator.generate_deployment_handoff_for_student(
        project_id=project_id,
        student_model_id=student_id,
        settings=settings_in_workspace,
    )
    assert isinstance(result, dict)
    tech = result["technical_requirements"]
    assert tech["docker_run_command"], tech["docker_run_command"]
    assert tech["docker_run_command"].startswith("docker run")
    args = tech["docker_run_args"]
    assert isinstance(args, list) and len(args) > 0
    assert args[0] == "docker"
    assert "run" in args
    # Canonical builder details — `-u $(id -u)` and `:ro` on checkpoint mount.
    assert "-u" in args
    # The :ro flag is appended to the checkpoint mount value.
    assert any(":ro" in str(a) for a in args)
    # rendered_text echoes the display command.
    assert "docker run" in result["rendered_text"]


def test_handoff_and_public_builder_use_name_only_secret_forwarding(
    settings_in_workspace,
):
    """Every Student argv consumer gets one always-safe Docker shape."""
    sentinel = "SENTINEL_HANDOFF_NGC"
    settings_in_workspace.NGC_API_KEY = sentinel
    project_id, student_id = _seed_validated_student(
        settings_in_workspace,
        training_contract=_ALL_FIELDS,
        serving_contract=_ALL_FIELDS,
    )
    result = deployment_handoff_generator.generate_deployment_handoff_for_student(
        project_id=project_id,
        student_model_id=student_id,
        settings=settings_in_workspace,
    )
    tech = result["technical_requirements"]
    from vlm_feedback_loop.services import project_service

    engine = project_service.get_project_engine(
        project_id, settings_in_workspace.WORKSPACE_ROOT
    )
    with Session(engine) as session:
        deployment = (
            session.query(LocalNimDeployment)
            .filter_by(student_model_id=student_id)
            .one()
        )
        session.expunge(deployment)

    public_args = local_nim_service.build_student_docker_run_args(
        deployment=deployment,
        settings=settings_in_workspace,
    )
    for args in (tech["docker_run_args"], public_args):
        ngc_index = args.index("NGC_API_KEY")
        assert args[ngc_index - 1] == "-e"
        assert not any(str(token).startswith("NGC_API_KEY=") for token in args)
        flat = " ".join(args)
        assert sentinel not in flat
        assert "$NGC_API_KEY" not in flat

    for rendered in (tech["docker_run_command"], result["rendered_text"]):
        assert "-e NGC_API_KEY" in rendered
        assert "NGC_API_KEY=" not in rendered
        assert sentinel not in rendered
        assert "$NGC_API_KEY" not in rendered
