# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Training suite orchestration service.

``POST /v1/projects/{project_id}/training_suites`` atomically creates:

  1. Training DatasetExport (Verified non-pool + optional Auto-Labeled).
  2. Test-Pool evaluation DatasetExport.
  3. Per selected student_base model: a TAOJob chain pre-created with
     ``status="not_started"`` — ``train`` → ``evaluate`` baseline → for
     each quantization scheme: ``quantize`` → ``evaluate`` quantized.
     Chain linkage via ``chain_id``, ``chain_sequence``,
     ``parent_tao_job_id``.
  4. A TrainingSuite record capturing suite-level lineage + the ordered
     chain list.

Then Phase 2 (outside the DB transaction) submits the first chain's
``train`` job via :func:`tao_job_service.submit_chain_job`, which runs the
standard 4-step submission protocol. The polling loop
(``tao_polling_service``) picks up the rest — chain advancement after
each success, chain halt on failure.

Idempotency: ``(project_id, idempotency_key)`` is UNIQUE on TrainingSuite.
A retry POST with the same key returns the existing suite's response
without new writes, without re-kickoff.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, cast

from sqlalchemy.orm import Session

from vlm_feedback_loop.config import Settings
from vlm_feedback_loop.db.base import generate_uuid4, utc_now
from vlm_feedback_loop.db.deployment_models import TAODeploymentConfig
from vlm_feedback_loop.db.engine import DatabaseMigrationError, open_project_db
from vlm_feedback_loop.db.models.dataset_export import DatasetExport
from vlm_feedback_loop.db.models.model_config import ModelConfig
from vlm_feedback_loop.db.models.project import Project
from vlm_feedback_loop.db.models.tao_job import TAOJob
from vlm_feedback_loop.db.models.training_suite import TrainingSuite
from vlm_feedback_loop.model_catalog_constants import HF_MODEL_PATHS
from vlm_feedback_loop.services import (
    tao_dataset_upload_service,
    tao_job_service,
    training_preflight_service,
)
from vlm_feedback_loop.services.background import background_manager
from vlm_feedback_loop.services.dataset_export_service import (
    create_dataset_export,
)
from vlm_feedback_loop.services.pagination import (
    InvalidCursorError,
    after_position_asc,
    decode_cursor,
    encode_cursor,
)
from vlm_feedback_loop.services.project_service import get_project_engine
from vlm_feedback_loop.services.sse import sse_manager
from vlm_feedback_loop.services.tao_base_experiment_provisioning_run_service import (
    ACTIVE_STATUSES as PROVISIONING_ACTIVE_STATUSES,
)
from vlm_feedback_loop.services.tao_base_experiment_provisioning_run_service import (
    get_provisioning_run,
    start_provisioning_run,
)
from vlm_feedback_loop.services.tao_bootstrap_service import iter_project_dirs
from vlm_feedback_loop.services.tao_dataset_upload_service import (
    UploadResult,
    build_s3_client,
)
from vlm_feedback_loop.services.tao_job_service import (
    apply_dataset_binding,
    compute_request_checksum,
)
from vlm_feedback_loop.services.tao_workspace_service import read_tao_deployment_config
from vlm_feedback_loop.services.training_preset import (
    TRAINING_PRESETS,
    resolve_training_preset,
)

# Type alias for the injectable upload hook used by tests.
# Signature matches ``tao_dataset_upload_service.upload_dataset_archive``.
UploadArchiveFn = Callable[..., Awaitable[UploadResult]]

logger = logging.getLogger("vlm_feedback_loop.training_suite_service")


# ── Constants ───────────────────────────────────────────────────────────────

# Valid quantization schemes per the TAO Cosmos-RL `quantize` action.
VALID_QUANTIZATION_SCHEMES: frozenset[str] = frozenset(
    {"FP8_DYNAMIC", "W8A8", "W8A16", "W4A16"}
)

# The single training policy type v1 supports. Used both for the
# persisted ``training_policy_type`` lineage fields (mirrors the
# ``Literal["sft"]`` contract in ``schemas/tao_job.py``) and for the
# cosmos-rl ``train.train_policy.type`` spec key — without it cosmos-rl
# defaults to GRPO and the SFT helper exits.
_TRAINING_POLICY_TYPE_SFT = "sft"

# TAO POST /jobs requires `network_arch`, `base_experiment_ids`,
# `workspace`, and `name` — FTMS 6.26.3 answers HTTP 400 from the
# create-job endpoint when any is omitted.
#
# ``timeout_minutes`` (measured 2026-07-15): FTMS's timeout monitor kills
# any job whose status heartbeat goes stale for ``timeout_minutes`` —
# DEFAULT 60 WHEN THE FIELD IS ABSENT — and cosmos-rl emits no status
# updates during training. The v8 TAO install guide patches FTMS 6.26.3's
# v2 schema to accept the field. Training preflight verifies that capability;
# every suite job carries the configured generous stale-job ceiling.
_TAO_NETWORK_ARCH = "cosmos-rl"

# ``policy.model_name_or_path`` is emitted with the ``hf_model://``
# scheme prefix (e.g. ``hf_model://nvidia/Cosmos-Reason2-2B``). FTMS
# resolves that prefix at job-prep time: the registered base experiment
# triggers an air-gapped pull (or, in offline mode, points at the
# pre-staged checkpoint directory in the workspace). Sending the bare
# HF identifier (``nvidia/Cosmos-Reason2-2B``) makes cosmos-rl call
# HuggingFace directly and hit gated-repo HTTP 401; sending a literal
# local path (``/ptm/huggingface_models``) makes from_pretrained crash
# with ``Can't load configuration`` because the path isn't materialized
# without the URL-scheme trigger. Only the ``hf_model://`` shape works.
_TAO_HF_MODEL_SCHEME = "hf_model://"


def _hf_model_url(mc: ModelConfig) -> str:
    """Return ``hf_model://<hf-identifier>`` for use in cosmos-rl specs."""
    return f"{_TAO_HF_MODEL_SCHEME}{_hf_path_for_model(mc)}"


# Parallelism is intentionally left blank in the spec. The TAO patch
# auto-injects ``policy.parallelism.tp_size = 1`` and
# ``dp_shard_size = NUM_GPU_PER_NODE`` (= the host's full GPU count,
# e.g. 8 on a Profile-D 8×A100 box) when no block is provided.
# Explicit values would be respected but would cap the job to fewer
# GPUs; jobs should use every available rank. Same for ``num_gpu`` on
# the request body — omitting it lets the docker_handler default of
# ``-1`` (= all GPUs) take effect.


def _build_job_name(
    *, action: str, project_id: str, chain_id: str, chain_sequence: int
) -> str:
    """Generate a deterministic, human-readable TAO job name.

    Pattern: ``vlm-fb-{action}-{project_short}-{chain_short}-{chain_seq:02d}``
    where the short ids are the leading 8 hex chars of the UUIDs. Stays
    well under FTMS's 100-char `name` cap and embeds enough provenance for
    operators to grep TAO job listings without cross-referencing the
    Blueprint DB.
    """
    project_short = project_id.split("-")[0][:8] if project_id else "unknown"
    chain_short = chain_id.split("-")[0][:8] if chain_id else "unknown"
    return f"vlm-fb-{action}-{project_short}-{chain_short}-{chain_sequence:02d}"


# Valid export field modes.
_VALID_EXPORT_FIELD_MODES: frozenset[str] = frozenset(
    {"all", "aux_and_core", "core_only"}
)

# LoRA default config — attention-only target modules are the
# safe, NIM-mergeable default for Cosmos Reason2. Users who need visual
# encoder fine-tuning can customize post-v1 by editing the TAOJob payload.
_DEFAULT_LORA_CONFIG: dict[str, Any] = {
    "enable_lora": True,
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_target_modules": ["q_proj", "v_proj"],
    "modules_to_save": None,
}


def _hf_path_for_model(mc: ModelConfig) -> str:
    """Resolve the effective ``policy.model_name_or_path`` for a model config.

    The resolved value lands in the Cosmos-RL training spec; checkpoint
    packaging (``student_model_service``) reads it back from
    ``job_config.resolved_training_fields.policy.model_name_or_path``
    for LoRA adapter merge on the full-precision baseline path.

    Prefers ``local_deploy_metadata.hf_model_path`` when operators have
    set it (custom deployments); falls back to the canonical
    ``model_catalog_constants.HF_MODEL_PATHS`` roster; final fallback is
    the model's display name so TAO at least receives a non-empty value
    (TAO will reject unknown paths — that failure will surface through
    the standard polling path).
    """
    if mc.local_deploy_metadata:
        override = mc.local_deploy_metadata.get("hf_model_path")
        if isinstance(override, str) and override:
            return override
    return HF_MODEL_PATHS.get(mc.model_name.strip().lower(), mc.model_name)


# ── Validation helpers ──────────────────────────────────────────────────────


def _validate_request(
    *,
    training_preset: str,
    export_field_mode: str,
    quantization_schemes: list[str],
) -> str | None:
    """Return None on success or an error string for the router layer."""
    if training_preset not in TRAINING_PRESETS:
        return (
            f"validation: invalid training_preset {training_preset!r}; "
            f"expected one of {sorted(TRAINING_PRESETS)}"
        )
    if export_field_mode not in _VALID_EXPORT_FIELD_MODES:
        return (
            f"validation: invalid export_field_mode {export_field_mode!r}; "
            f"expected one of {sorted(_VALID_EXPORT_FIELD_MODES)}"
        )
    unknown = [s for s in quantization_schemes if s not in VALID_QUANTIZATION_SCHEMES]
    if unknown:
        return (
            f"validation: invalid quantization scheme(s) {unknown}; "
            f"expected subset of {sorted(VALID_QUANTIZATION_SCHEMES)}"
        )
    return None


def _load_student_base_models(
    session: Session,
    project_id: str,
    model_config_ids: list[str],
) -> tuple[list[ModelConfig], str | None]:
    """Load + role-validate every student_base model config.

    Returns a ``(models, error)`` tuple. Models are returned in the order
    the caller supplied ``model_config_ids`` so ``chain_ids_ordered`` and
    the chain-kickoff order match the SME's selection order in the
    training setup UI.
    """
    models: list[ModelConfig] = []
    for mc_id in model_config_ids:
        mc = (
            session.query(ModelConfig)
            .filter_by(project_id=project_id, model_config_id=mc_id)
            .first()
        )
        if mc is None:
            return (
                [],
                f"not found: ModelConfig {mc_id} not found in project {project_id}",
            )
        # eligible_roles is a JSON list on disk; tests seed it as json.dumps(...)
        # but the catalog seeder stores a native list.
        roles = mc.eligible_roles
        if isinstance(roles, str):
            try:
                import json as _json

                roles = _json.loads(roles)
            except (ValueError, TypeError):
                roles = []
        if not isinstance(roles, list) or "student_base" not in roles:
            return (
                [],
                f"validation: ModelConfig {mc_id} does not have student_base role "
                f"(eligible_roles={roles})",
            )
        models.append(mc)
    return models, None


# ── Payload construction ────────────────────────────────────────────────────


def _base_job_config(
    *,
    training_preset: str,
    tao_release_version: str,
    cosmos_rl_container_tag: str,
    track_artifacts: list[str],
    hyperparameters: dict[str, Any] | None = None,
    dataset_refs: dict[str, str] | None = None,
    resolved_training_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Shared ``job_config`` shape persisted on every TAOJob row.

    ``num_gpus_per_node`` stays None — the TAO patch reads
    NUM_GPU_PER_NODE from the host when not specified.
    ``intended_outputs`` is Blueprint-internal audit metadata only
    (cosmos-rl does NOT consume it): an advisory record of what the
    operator expected to track at submission time; actual artifact
    retrieval is action-aware and discovers the cosmos-rl layout
    dynamically via ``:list_files`` + workspace S3.
    """
    return {
        "training_backend": "cosmos_rl_tao_vlm",
        "training_preset": training_preset,
        "training_policy_type": _TRAINING_POLICY_TYPE_SFT,
        "lora_config": dict(_DEFAULT_LORA_CONFIG),
        "hyperparameters": hyperparameters or {},
        "parallelism_config": None,
        "num_nodes": 1,
        "num_gpus_per_node": None,
        "redis_config": None,
        "tao_release_version": tao_release_version,
        "cosmos_rl_container_tag": cosmos_rl_container_tag,
        "dataset_refs": dataset_refs or {},
        "intended_outputs": {
            "track_logs": True,
            "track_metrics": True,
            "track_artifacts": track_artifacts,
        },
        "resolved_training_fields": resolved_training_fields or {},
    }


def _build_train_payload(
    *,
    mc: ModelConfig,
    training_preset: str,
    hyperparameters: dict[str, Any],
    training_dataset_export_id: str,
    training_archive_path: str,
    training_annotation_path: str,
    tao_release_version: str,
    cosmos_rl_container_tag: str,
    workspace_id: str,
    base_experiment_id: str,
    job_name: str,
    enable_lora: bool = True,
    timeout_minutes: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build (job_config, tao_create_job_request) for a train action.

    The resulting ``tao_create_job_request`` MUST carry
    the TAO-required job metadata (``network_arch``, ``base_experiment_ids``,
    ``workspace``, ``name``) alongside ``kind``/``action``/``specs``.
    Omitting any of those surfaces as HTTP 400 from FTMS 6.26.3.

    The cosmos-rl SFT runtime requires several additional spec keys to
    actually train (verified against cosmos-rl's ``tao_sft_example.py``
    + reference ``spec.toml``):

    * ``policy.model_name_or_path = "hf_model://<hf-identifier>"`` —
      the URL-scheme trigger that lets FTMS substitute the registered
      base experiment's airgapped checkpoint at job-prep time. See the
      ``_TAO_HF_MODEL_SCHEME`` block above; this is a single canonical
      shape — bare HF ids and bare local paths both fail.
    * ``train.train_policy.type = "sft"`` — without this cosmos-rl
      defaults to GRPO and the SFT helper exits.
    * ``train.train_policy.dataloader_drop_last = false`` — small
      datasets otherwise yield 0 batches, the trainer treats step 0 as
      ``is_last_step``, and checkpointing crashes on ``NoneType
      .state_dict`` (optimizer/scheduler aren't built yet).
    * ``train.compile = false`` — torch.compile is unsupported for
      ``HFModel`` and the worker dies with an AssertionError.
    * ``train.ckpt.save_mode = "sync"`` — async save races with the
      step-0 checkpoint path on small datasets.
    * ``validation.enable = false`` — the Blueprint exports no
      validation dataset; cosmos-rl asserts on the missing val_dataset
      otherwise.

    Parallelism + ``num_gpu`` are NOT emitted: the TAO patch
    auto-injects ``tp_size=1, dp_shard_size=NUM_GPU_PER_NODE`` and
    allocates all visible GPUs to the container, which is what we
    want.

    ``resolved_training_fields.policy.model_name_or_path`` keeps the
    bare HF identifier (no scheme) because checkpoint packaging
    (``services/student_model_service.register_from_tao_terminal``)
    uses that field for LoRA-merge fallback when the training output is
    adapter-only. Spec key (``hf_model://...``) vs. lineage key (bare
    HF id) are deliberately separate.
    """
    hf_path = _hf_path_for_model(mc)

    # resolved_training_fields keeps the HF identifier so checkpoint
    # packaging can re-pull the base model when merging adapter-only outputs.
    job_config = _base_job_config(
        training_preset=training_preset,
        tao_release_version=tao_release_version,
        cosmos_rl_container_tag=cosmos_rl_container_tag,
        track_artifacts=["best_model", "latest_model", "training_config"],
        hyperparameters=hyperparameters,
        dataset_refs={"training_dataset_export_id": training_dataset_export_id},
        resolved_training_fields={"policy": {"model_name_or_path": hf_path}},
    )

    # Inline the preset patch into specs and bind the dataset.
    specs = dict(hyperparameters)

    # Cosmos-RL SFT spec overrides — see docstring for the failure mode
    # each one prevents.
    train_section = dict(specs.get("train") or {})
    train_section["compile"] = False
    train_section.setdefault("train_policy", {})
    train_section["train_policy"] = {
        **train_section["train_policy"],
        "type": _TRAINING_POLICY_TYPE_SFT,
        # Cosmos-RL defaults to ``dataloader_drop_last=True`` which
        # silently drops the final partial batch. On small datasets
        # (e.g. a 3-example sanity project) this leaves
        # the dataloader with 0 batches per epoch; the trainer then
        # treats step 0 as ``is_last_step`` and tries to checkpoint
        # before the optimizer/scheduler have been built — surfaces as
        # ``AttributeError: 'NoneType' object has no attribute
        # 'state_dict'``. Setting drop_last=False keeps the partial
        # batch so the optimizer steps at least once before save.
        "dataloader_drop_last": False,
    }
    # NVIDIA's reference cosmos-rl SFT spec uses ``save_mode = "sync"``;
    # the default ``"async"`` triggers a pre-training save_at_step_0 that
    # crashes with ``AttributeError: 'NoneType' object has no attribute
    # 'state_dict'`` because the optimizer isn't constructed yet.
    # Sync mode defers the first save until after the optimizer step.
    ckpt_section = dict(train_section.get("ckpt") or {})
    ckpt_section.setdefault("save_mode", "sync")
    train_section["ckpt"] = ckpt_section

    # Per-model train-section overrides (top-level ``train.*`` keys), carried
    # on ``ModelConfig.local_deploy_metadata.tao_train_overrides`` and applied
    # last so they win over the SFT defaults above. The motivating lever is
    # ``master_dtype="bfloat16"`` for the large tier: a ~30B model with fp32
    # AdamW master + m/v state is ~16 bytes/param (~460 GB), and that state is
    # sharded by ``dp_shard`` (not ``tp``) — so tensor parallelism alone does
    # NOT relieve it. Halving the master weights to bf16 is the biggest
    # steady-state memory win (Cosmos 3 Super-Reasoner OOMs at both tp=1
    # and tp=4 during init/steady-state without it).
    train_overrides = (mc.local_deploy_metadata or {}).get("tao_train_overrides")
    if isinstance(train_overrides, dict) and train_overrides:
        train_section.update(cast("dict[str, Any]", train_overrides))

    specs["train"] = train_section

    policy_section = dict(specs.get("policy") or {})
    policy_section["model_name_or_path"] = _hf_model_url(mc)
    # parallelism is normally left blank — TAO auto-injects tp_size=1,
    # dp_shard_size=NUM_GPU_PER_NODE, which fits the small tiers (CR2 2B/8B,
    # Cosmos 3 Nano-Reasoner). Large tiers need tensor parallelism so the
    # model is sharded across GPUs *at load*, not just data-parallel FSDP:
    # Cosmos 3 Super-Reasoner (~30B) OOMs GPU 0 at tp=1 on 80 GB cards
    # (full model + fp32 master weights materialize before FSDP shards
    # them). A per-model override carried
    # on ``ModelConfig.local_deploy_metadata.tao_train_parallelism`` (e.g.
    # ``{"tp_size": 4, "dp_shard_size": 2}`` for Super) is emitted verbatim
    # as ``policy.parallelism``. Scoped per-model on purpose — bumping the
    # global TAO default would needlessly fragment the small tiers.
    parallelism = (mc.local_deploy_metadata or {}).get("tao_train_parallelism")
    if isinstance(parallelism, dict) and parallelism:
        policy_section["parallelism"] = dict(cast("dict[str, Any]", parallelism))

    # Measured 2026-07-14: the Spec's LoRA-first default (§2, §9.7.3.2) never
    # reached the wire — job_config persisted a lora_config while cosmos-rl
    # trained full-weight (``policy.lora`` defaulted to None; live-verified:
    # the trainer's parameter table logged every module TRAINABLE, and an
    # 8B "LoRA" train OOM'd an A100-80GB on full-model optimizer state).
    # Map the persisted lora_config onto cosmos-rl's ``policy.lora``
    # (``LoraConfig``: r / lora_alpha / lora_dropout / target_modules /
    # modules_to_save) so records and training agree. Nested policy dicts
    # pass through TAO's spec mapper (``policy.parallelism`` uses the same
    # path). ``enable_lora=false`` keeps the legacy full-weight wire shape
    # and records the opt-out honestly.
    job_config["lora_config"]["enable_lora"] = enable_lora
    if enable_lora:
        lora_cfg = job_config["lora_config"]
        policy_section["lora"] = {
            "r": lora_cfg["lora_rank"],
            "lora_alpha": lora_cfg["lora_alpha"],
            "lora_dropout": lora_cfg["lora_dropout"],
            "target_modules": list(lora_cfg["lora_target_modules"]),
            "modules_to_save": lora_cfg["modules_to_save"],
        }
    specs["policy"] = policy_section

    specs.setdefault("validation", {})["enable"] = False

    tao_create_job_request: dict[str, Any] = {
        "kind": "experiment",
        "action": "train",
        "name": job_name,
        "network_arch": _TAO_NETWORK_ARCH,
        "base_experiment_ids": [base_experiment_id],
        "workspace": workspace_id,
        "specs": specs,
    }
    # Required safety ceiling. Training preflight confirms the TAO server's
    # v2 schema accepts it before the UI enables suite creation.
    if timeout_minutes is not None:
        tao_create_job_request["timeout_minutes"] = timeout_minutes
    tao_create_job_request = apply_dataset_binding(
        tao_create_job_request,
        action="train",
        annotation_path=training_annotation_path,
        media_root=training_archive_path,
    )

    job_config["tao_create_job_request_checksum"] = compute_request_checksum(
        tao_create_job_request
    )

    return job_config, tao_create_job_request


def _build_evaluate_payload(
    *,
    eval_dataset_export_id: str,
    eval_archive_path: str,
    eval_annotation_path: str,
    parent_tao_job_id: str,
    tao_release_version: str,
    cosmos_rl_container_tag: str,
    training_preset: str,
    workspace_id: str,
    base_experiment_id: str,
    job_name: str,
    quantization_method: str | None = None,
    timeout_minutes: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build (job_config, tao_create_job_request) for an evaluate action.

    ``parent_tao_job_id`` is recorded in both the TAOJob column and the
    job_config (for downstream lineage tools that only read the JSON blob).

    Adds the TAO-required `network_arch` / `base_experiment_ids` /
    `workspace` / `name` fields TAO requires plus the cosmos-rl
    runtime overrides described on ``_build_train_payload``.
    """
    job_config = _base_job_config(
        training_preset=training_preset,
        tao_release_version=tao_release_version,
        cosmos_rl_container_tag=cosmos_rl_container_tag,
        track_artifacts=["metrics", "per_sample_predictions"],
        dataset_refs={"evaluation_dataset_export_id": eval_dataset_export_id},
    )
    job_config["parent_tao_job_id_lineage"] = parent_tao_job_id
    if quantization_method is not None:
        job_config["quantization_method"] = quantization_method

    # ``policy.model_name_or_path`` is intentionally NOT emitted on
    # evaluate. FTMS leaves ``model.model_name`` unpopulated when
    # ``parent_id`` is None (job prep crashes with ``TypeError:
    # expected str, bytes or os.PathLike object, not NoneType``);
    # ``submit_chain_job`` injects top-level ``parent_job_id``, and
    # FTMS's ``infer_parent_model_folder`` resolves the trained
    # checkpoint folder onto cosmos-rl's spec at job-prep time. Sending
    # an explicit ``model_name_or_path`` here OVERRIDES the resolution
    # and causes the worker to load the base experiment instead of the
    # fine-tuned checkpoint — even a 12-epoch SFT job's eval produces
    # verbatim base-model output because the spec pins the base model.
    # Parallelism is omitted so TAO auto-injects
    # tp=1, dp_shard=NUM_GPU_PER_NODE.
    specs: dict[str, Any] = {
        "train": {
            "compile": False,
            "train_policy": {
                "type": _TRAINING_POLICY_TYPE_SFT,
                "dataloader_drop_last": False,
            },
            "ckpt": {"save_mode": "sync"},
        },
        "validation": {"enable": False},
    }
    tao_create_job_request: dict[str, Any] = {
        "kind": "experiment",
        "action": "evaluate",
        "name": job_name,
        "network_arch": _TAO_NETWORK_ARCH,
        "base_experiment_ids": [base_experiment_id],
        "workspace": workspace_id,
        "specs": specs,
    }
    # Required heartbeat-timeout ceiling — see _build_train_payload.
    if timeout_minutes is not None:
        tao_create_job_request["timeout_minutes"] = timeout_minutes
    tao_create_job_request = apply_dataset_binding(
        tao_create_job_request,
        action="evaluate",
        annotation_path=eval_annotation_path,
        media_root=eval_archive_path,
    )
    job_config["tao_create_job_request_checksum"] = compute_request_checksum(
        tao_create_job_request
    )
    return job_config, tao_create_job_request


def _build_quantize_payload(
    *,
    quantization_method: str,
    training_archive_path: str,
    training_annotation_path: str,
    parent_tao_job_id: str,
    tao_release_version: str,
    cosmos_rl_container_tag: str,
    training_preset: str,
    workspace_id: str,
    base_experiment_id: str,
    job_name: str,
    timeout_minutes: int | None = None,
    calibration_samples: int = 128,
    enable_lora: bool = False,
    base_model_path: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build (job_config, tao_create_job_request) for a quantize action.

    TAO ``quantize`` auto-merges LoRA when ``enable_lora=true`` and the
    base model path is provided. The training chain's checkpoint
    flows through ``parent_tao_job_id``.

    ``enable_lora`` / ``base_model_path``: when the chain's train
    job produced an adapter-only checkpoint (``enable_lora=true``), the
    cosmos-rl-quantize container must merge the adapter before
    calibration, and its ``load_model_and_processor`` hard-requires
    ``--base_model_path`` in that mode (``lora_utils.py`` raises
    "base_model_path is required when enable_lora is True" before its
    own adapter-config inference fallback can run — observed live on
    FTMS 6.26.3, 2026-07-15, jobs ``ba502a27``/``9f19d324``). The spec
    keys map 1:1 onto the CLI flags (same mechanism as
    ``max_sequence_length``): ``--enable_lora True --base_model_path
    <bare HF id>``. ``base_model_path`` is the BARE HuggingFace
    identifier (e.g. ``nvidia/Cosmos3-Nano-Reasoner``) — the quantize
    container feeds it straight to ``from_pretrained()``; the
    ``hf_model://`` scheme is a train/eval-side FTMS convention and is
    NOT understood here. Gated-repo auth reaches the container via the
    uniform ``docker_env_vars.HF_TOKEN`` injection (or the v8 guide's
    FTMS-side passthrough patch).

    Adds the TAO-required `network_arch` / `base_experiment_ids` /
    `workspace` / `name` fields TAO requires.

    The cosmos-rl-quantize CLI does NOT accept the train action's
    argument set. Given the train shape, TAO submits the call with
    rejected args:
    ``--quantization_method`` (correct: ``--quantization_scheme``),
    ``--model_name_or_path`` (rejected entirely — quantize uses
    ``--model_path`` from ``parent_tao_job_id``), ``--media_path`` (correct:
    ``--media_dir``), plus the train-specific SFT overrides (``--compile``,
    ``--type``, ``--dataloader_drop_last``, ``--save_mode``, ``--enable``).
    The Blueprint-controllable subset is fixed here:

    - ``quantization_scheme`` not ``quantization_method`` at spec key
      AND in ``job_config`` (the persisted field name aligns with the
      cosmos-rl CLI flag for audit clarity).
    - ``policy.model_name_or_path`` is omitted entirely. The trained
      checkpoint flows in via ``parent_tao_job_id``; quantize doesn't
      re-load the base model.
    - The SFT spec overrides (`train.compile`,
      ``train.train_policy.*``, ``train.ckpt.*``, ``validation.enable``)
      are train-specific. The cosmos-rl-quantize CLI rejects every one
      of those flag names; they MUST NOT appear in the quantize spec.
    - Dataset binding switches to top-level ``specs.dataset.{media_dir,
      annotation_path}`` (matching evaluate's shape). cosmos-rl-quantize's
      CLI accepts ``--media_dir``, NOT ``--media_path``.

    Known TAO-side residual gap: TAO's cosmos-rl entrypoint mapper
    auto-injects parallelism args (``--n_init_replicas``, ``--tp_size``,
    ``--cp_size``, ``--dp_shard_size``, ``--dp_replicate_size``,
    ``--pp_size``) for every action including quantize, but cosmos-rl-quantize
    rejects them. That's outside the Blueprint's control — fixing it
    requires a TAO-side patch. This Blueprint-side fix removes our
    contribution to the rejection set; whether quantize succeeds end-to-end
    depends on the TAO-side patch landing too.
    """
    job_config = _base_job_config(
        training_preset=training_preset,
        tao_release_version=tao_release_version,
        cosmos_rl_container_tag=cosmos_rl_container_tag,
        track_artifacts=["quantized_model"],
    )
    # The persisted job_config field keeps the ``quantization_method`` name
    # to match StudentModel.quantization_method (the user-facing concept).
    # The ``--quantization_scheme`` rename applies only to the
    # cosmos-rl-quantize CLI spec key (see ``specs`` below) where TAO maps
    # spec keys to CLI flags 1:1.
    job_config["quantization_method"] = quantization_method
    job_config["parent_tao_job_id_lineage"] = parent_tao_job_id
    # Minimal quantize spec. ``policy``, ``train``, and ``validation``
    # blocks are train-specific and rejected by cosmos-rl-quantize.
    # Dataset binding flows in via ``apply_dataset_binding(action="quantize")``
    # below, which produces top-level ``dataset.{media_dir,
    # annotation_path}`` per the cosmos-rl-quantize CLI signature.
    #
    # ``max_sequence_length``: cosmos-rl-quantize tokenizes calibration
    # samples with ``truncation=True`` at this length (CLI default 2048).
    # VLM calibration images expand to thousands of image tokens — a
    # 4032×3024 photo is ~11.8k — so the default truncates the ``<image>``
    # expansion and every calibration batch dies with "Mismatch in `image`
    # token count between text and `input_ids`" (observed live on FTMS
    # 6.26.3, 2026-07-14; retry with 16384 passed). 16384 covers the
    # largest single-image expansion we ship plus the answer text.
    specs: dict[str, Any] = {
        "quantization_scheme": quantization_method,
        "max_sequence_length": 16384,
        # Cosmos-RL defaults to 512 and materializes every preprocessed
        # VLM sample into one Arrow ListArray. High-resolution images can
        # push that array beyond PyArrow's signed 32-bit (2 GiB) offset
        # ceiling after the final sample, even though all samples mapped
        # successfully. Cap the calibration set before materialization.
        "num_calibration_samples": calibration_samples,
    }
    # Adapter-only parents need the in-container LoRA merge.
    # Emitted explicitly rather than relying on FTMS's inherited
    # ``--enable_lora`` injection so the wire shape is deterministic and
    # auditable from the persisted request alone.
    if enable_lora:
        if not base_model_path:
            raise ValueError(
                "base_model_path is required when building a quantize spec "
                "for a LoRA training chain"
            )
        specs["enable_lora"] = True
        specs["base_model_path"] = base_model_path
    tao_create_job_request: dict[str, Any] = {
        "kind": "experiment",
        "action": "quantize",
        "name": job_name,
        "network_arch": _TAO_NETWORK_ARCH,
        "base_experiment_ids": [base_experiment_id],
        "workspace": workspace_id,
        "specs": specs,
    }
    # Required heartbeat-timeout ceiling — see _build_train_payload (FP8
    # calibration at high resolutions has legitimately exceeded an hour too).
    if timeout_minutes is not None:
        tao_create_job_request["timeout_minutes"] = timeout_minutes
    tao_create_job_request = apply_dataset_binding(
        tao_create_job_request,
        action="quantize",
        annotation_path=training_annotation_path,
        media_root=training_archive_path,
    )
    job_config["tao_create_job_request_checksum"] = compute_request_checksum(
        tao_create_job_request
    )
    return job_config, tao_create_job_request


# ── Chain construction ──────────────────────────────────────────────────────


def _create_chain_rows_in_session(
    session: Session,
    *,
    project_id: str,
    mc: ModelConfig,
    training_preset: str,
    hyperparameters: dict[str, Any],
    training_dataset_export_id: str,
    training_archive_path: str,
    training_annotation_path: str,
    eval_dataset_export_id: str,
    eval_archive_path: str,
    eval_annotation_path: str,
    quantization_schemes: list[str],
    tao_release_version: str,
    cosmos_rl_container_tag: str,
    workspace_id: str,
    base_experiment_id: str,
    enable_lora: bool = True,
    timeout_minutes: int | None = None,
    quantization_calibration_samples: int = 128,
) -> tuple[str, list[TAOJob]]:
    """Pre-create one model's full TAO chain with status=not_started.

    Returns ``(chain_id, [ordered TAOJob rows])``. All rows are added to
    the session but not yet committed — the caller owns the transaction.

    ``workspace_id`` and ``base_experiment_id`` are threaded into every
    payload builder so the resulting ``tao_create_job_request`` carries
    the TAO-required job metadata and FTMS POST /jobs accepts it.
    """
    chain_id = generate_uuid4()
    jobs: list[TAOJob] = []

    # chain_sequence = 1: train
    train_job_config, train_request = _build_train_payload(
        mc=mc,
        training_preset=training_preset,
        hyperparameters=hyperparameters,
        training_dataset_export_id=training_dataset_export_id,
        training_archive_path=training_archive_path,
        training_annotation_path=training_annotation_path,
        tao_release_version=tao_release_version,
        cosmos_rl_container_tag=cosmos_rl_container_tag,
        workspace_id=workspace_id,
        base_experiment_id=base_experiment_id,
        enable_lora=enable_lora,
        timeout_minutes=timeout_minutes,
        job_name=_build_job_name(
            action="train",
            project_id=project_id,
            chain_id=chain_id,
            chain_sequence=1,
        ),
    )
    train_job = TAOJob(
        tao_job_id=generate_uuid4(),
        project_id=project_id,
        student_base_model_config_id=mc.model_config_id,
        dataset_export_ids=[training_dataset_export_id],
        action="train",
        status="not_started",
        training_backend="cosmos_rl_tao_vlm",
        training_policy_type=_TRAINING_POLICY_TYPE_SFT,
        job_config=train_job_config,
        tao_create_job_request=train_request,
        chain_id=chain_id,
        chain_sequence=1,
        parent_tao_job_id=None,
    )
    session.add(train_job)
    session.flush()  # so later jobs can reference train_job.tao_job_id
    jobs.append(train_job)

    # chain_sequence = 2: evaluate baseline (parent = train)
    eval_job_config, eval_request = _build_evaluate_payload(
        eval_dataset_export_id=eval_dataset_export_id,
        eval_archive_path=eval_archive_path,
        eval_annotation_path=eval_annotation_path,
        parent_tao_job_id=train_job.tao_job_id,
        tao_release_version=tao_release_version,
        cosmos_rl_container_tag=cosmos_rl_container_tag,
        training_preset=training_preset,
        workspace_id=workspace_id,
        base_experiment_id=base_experiment_id,
        job_name=_build_job_name(
            action="evaluate",
            project_id=project_id,
            chain_id=chain_id,
            chain_sequence=2,
        ),
        timeout_minutes=timeout_minutes,
    )
    eval_baseline = TAOJob(
        tao_job_id=generate_uuid4(),
        project_id=project_id,
        student_base_model_config_id=mc.model_config_id,
        dataset_export_ids=[eval_dataset_export_id],
        action="evaluate",
        status="not_started",
        training_backend="cosmos_rl_tao_vlm",
        training_policy_type=None,
        job_config=eval_job_config,
        tao_create_job_request=eval_request,
        chain_id=chain_id,
        chain_sequence=2,
        parent_tao_job_id=train_job.tao_job_id,
    )
    session.add(eval_baseline)
    session.flush()
    jobs.append(eval_baseline)

    # Per scheme: quantize + evaluate quantized
    seq = 3
    for scheme in quantization_schemes:
        quant_job_config, quant_request = _build_quantize_payload(
            quantization_method=scheme,
            training_archive_path=training_archive_path,
            training_annotation_path=training_annotation_path,
            parent_tao_job_id=train_job.tao_job_id,
            tao_release_version=tao_release_version,
            cosmos_rl_container_tag=cosmos_rl_container_tag,
            training_preset=training_preset,
            workspace_id=workspace_id,
            base_experiment_id=base_experiment_id,
            job_name=_build_job_name(
                action="quantize",
                project_id=project_id,
                chain_id=chain_id,
                chain_sequence=seq,
            ),
            timeout_minutes=timeout_minutes,
            calibration_samples=quantization_calibration_samples,
            enable_lora=enable_lora,
            base_model_path=_hf_path_for_model(mc) if enable_lora else None,
        )
        quant_job = TAOJob(
            tao_job_id=generate_uuid4(),
            project_id=project_id,
            student_base_model_config_id=mc.model_config_id,
            dataset_export_ids=[training_dataset_export_id],
            action="quantize",
            status="not_started",
            training_backend="cosmos_rl_tao_vlm",
            training_policy_type=None,
            job_config=quant_job_config,
            tao_create_job_request=quant_request,
            chain_id=chain_id,
            chain_sequence=seq,
            parent_tao_job_id=train_job.tao_job_id,
        )
        session.add(quant_job)
        session.flush()
        jobs.append(quant_job)
        seq += 1

        quant_eval_job_config, quant_eval_request = _build_evaluate_payload(
            eval_dataset_export_id=eval_dataset_export_id,
            eval_archive_path=eval_archive_path,
            eval_annotation_path=eval_annotation_path,
            parent_tao_job_id=quant_job.tao_job_id,
            tao_release_version=tao_release_version,
            cosmos_rl_container_tag=cosmos_rl_container_tag,
            training_preset=training_preset,
            workspace_id=workspace_id,
            base_experiment_id=base_experiment_id,
            job_name=_build_job_name(
                action="evaluate",
                project_id=project_id,
                chain_id=chain_id,
                chain_sequence=seq,
            ),
            quantization_method=scheme,
            timeout_minutes=timeout_minutes,
        )
        quant_eval = TAOJob(
            tao_job_id=generate_uuid4(),
            project_id=project_id,
            student_base_model_config_id=mc.model_config_id,
            dataset_export_ids=[eval_dataset_export_id],
            action="evaluate",
            status="not_started",
            training_backend="cosmos_rl_tao_vlm",
            training_policy_type=None,
            job_config=quant_eval_job_config,
            tao_create_job_request=quant_eval_request,
            chain_id=chain_id,
            chain_sequence=seq,
            parent_tao_job_id=quant_job.tao_job_id,
        )
        session.add(quant_eval)
        session.flush()
        jobs.append(quant_eval)
        seq += 1

    # Record honesty across the whole chain: every job's persisted
    # lora_config reflects the suite's effective training mode (the train
    # payload already sets it; evaluate/quantize records must not claim a
    # LoRA default the chain isn't using).
    for job in jobs:
        cfg = dict(job.job_config)
        lora_record = dict(cfg.get("lora_config") or {})
        lora_record["enable_lora"] = enable_lora
        cfg["lora_config"] = lora_record
        job.job_config = cfg

    return chain_id, jobs


# ── Response construction ───────────────────────────────────────────────────


def _suite_to_response(
    suite: TrainingSuite,
    *,
    chains_data: list[dict[str, Any]],
) -> dict[str, Any]:
    """Convert a TrainingSuite row + chain payloads into the public response dict."""
    return {
        "training_suite_id": suite.training_suite_id,
        "project_id": suite.project_id,
        "idempotency_key": suite.idempotency_key,
        "guidance_id": suite.guidance_id,
        "training_preset": suite.training_preset,
        "export_field_mode": suite.export_field_mode,
        "include_auto_labeled": suite.include_auto_labeled,
        "quantization_schemes": list(suite.quantization_schemes or []),
        "training_dataset_export_id": suite.training_dataset_export_id,
        "evaluation_dataset_export_id": suite.evaluation_dataset_export_id,
        "selected_student_base_model_config_ids": list(
            suite.selected_student_base_model_config_ids or []
        ),
        "chain_ids_ordered": list(suite.chain_ids_ordered or []),
        "chains": chains_data,
        "provisioning_run_id": suite.provisioning_run_id,
        "provisioning_model_names": list(suite.provisioning_model_names or []),
        "setup_error_ref": suite.setup_error_ref,
        "status": suite.status,
        "created_at": suite.created_at,
        "started_at": suite.started_at,
        "completed_at": suite.completed_at,
    }


def _build_chains_snapshot(
    session: Session,
    project_id: str,
    *,
    chain_ids_ordered: list[str],
    models_by_id: dict[str, ModelConfig],
) -> list[dict[str, Any]]:
    """Read TAOJob rows and assemble the ``chains[]`` response section."""
    chains: list[dict[str, Any]] = []
    for chain_id in chain_ids_ordered:
        rows = (
            session.query(TAOJob)
            .filter_by(project_id=project_id, chain_id=chain_id)
            .order_by(TAOJob.chain_sequence.asc())
            .all()
        )
        if not rows:
            continue
        head = rows[0]
        mc = models_by_id.get(head.student_base_model_config_id)
        base_model_name = mc.model_name if mc is not None else ""
        chains.append(
            {
                "chain_id": chain_id,
                "student_base_model_config_id": head.student_base_model_config_id,
                "base_model_name": base_model_name,
                "jobs": [
                    {
                        "tao_job_id": job.tao_job_id,
                        "action": job.action,
                        "chain_sequence": job.chain_sequence or 0,
                        "status": job.status,
                        "tao_external_job_id": job.tao_external_job_id,
                        "chain_halted_reason": job.chain_halted_reason,
                    }
                    for job in rows
                ],
            }
        )
    return chains


def _suite_snapshot_response(
    session: Session,
    project_id: str,
    suite: TrainingSuite,
) -> dict[str, Any]:
    """Full response dict for one suite: models lookup → chains → response.

    The single per-suite response assembly shared by create (fresh +
    idempotent replay), get, and list.
    """
    models_by_id: dict[str, ModelConfig] = {}
    for mc_id in suite.selected_student_base_model_config_ids or []:
        mc = (
            session.query(ModelConfig)
            .filter_by(project_id=project_id, model_config_id=mc_id)
            .first()
        )
        if mc is not None:
            models_by_id[mc_id] = mc

    chains = _build_chains_snapshot(
        session,
        project_id,
        chain_ids_ordered=list(suite.chain_ids_ordered or []),
        models_by_id=models_by_id,
    )
    return _suite_to_response(suite, chains_data=chains)


# ── Public API ──────────────────────────────────────────────────────────────


def resolve_training_presets_for_models(
    project_id: str,
    *,
    student_base_model_config_ids: list[str],
    settings: Settings,
) -> dict[str, Any] | str:
    """Resolve every preset for selected Student bases without running setup checks."""
    engine = get_project_engine(project_id, settings.WORKSPACE_ROOT)
    if engine is None:
        return f"not found: Project {project_id}"
    with Session(engine) as session:
        models, error = _load_student_base_models(
            session, project_id, list(student_base_model_config_ids)
        )
        if error is not None:
            return error
        return {
            "resolved_presets": {
                model.model_config_id: {
                    preset: resolve_training_preset(preset, model.model_name)
                    for preset in sorted(TRAINING_PRESETS)
                }
                for model in models
            }
        }


def _set_suite_setup_status(
    project_id: str,
    training_suite_id: str,
    settings: Settings,
    *,
    status: str,
    error: str | None = None,
) -> None:
    """Persist one pre-chain setup transition on a provisional suite."""
    engine = get_project_engine(project_id, settings.WORKSPACE_ROOT)
    if engine is None:
        return
    with Session(engine) as session:
        suite = session.get(TrainingSuite, training_suite_id)
        if suite is None:
            return
        if suite.status == "canceled":
            return
        suite.status = status
        suite.setup_error_ref = error[:4096] if error else None
        if status == "failed":
            suite.completed_at = utc_now()
        session.commit()


async def _complete_suite_after_provisioning(
    project_id: str,
    training_suite_id: str,
    *,
    student_base_model_config_ids: list[str],
    training_preset: str,
    include_auto_labeled: bool,
    export_field_mode: str,
    quantization_schemes: list[str],
    enable_lora: bool,
    idempotency_key: str,
    settings: Settings,
) -> None:
    """Wait for the selected bases, then materialize and start the suite."""
    engine = get_project_engine(project_id, settings.WORKSPACE_ROOT)
    if engine is None:
        return
    with Session(engine) as session:
        suite = session.get(TrainingSuite, training_suite_id)
        provisioning_run_id = suite.provisioning_run_id if suite else None
    if provisioning_run_id is None:
        _set_suite_setup_status(
            project_id,
            training_suite_id,
            settings,
            status="failed",
            error="Training setup lost its TAO provisioning run reference.",
        )
        return

    try:
        while True:
            run = get_provisioning_run(project_id, provisioning_run_id, settings)
            if isinstance(run, str):
                _set_suite_setup_status(
                    project_id,
                    training_suite_id,
                    settings,
                    status="failed",
                    error=run,
                )
                return
            if run["status"] not in PROVISIONING_ACTIVE_STATUSES:
                break
            await asyncio.sleep(2)

        if run["status"] != "succeeded":
            detail = run.get("error_ref")
            failures = run.get("failures")
            if (
                not isinstance(detail, str)
                and isinstance(failures, list)
                and failures
                and isinstance(failures[0], dict)
            ):
                failure = cast("dict[str, Any]", failures[0])
                failure_error = failure.get("error")
                detail = failure_error if isinstance(failure_error, str) else None
            if not isinstance(detail, str):
                detail = "TAO base provisioning failed."
            _set_suite_setup_status(
                project_id,
                training_suite_id,
                settings,
                status="failed",
                error=f"Base provisioning failed: {detail}",
            )
            return

        _set_suite_setup_status(
            project_id,
            training_suite_id,
            settings,
            status="preparing",
        )
        result = await create_training_suite(
            project_id,
            student_base_model_config_ids=student_base_model_config_ids,
            training_preset=training_preset,
            include_auto_labeled=include_auto_labeled,
            export_field_mode=export_field_mode,
            quantization_schemes=quantization_schemes,
            enable_lora=enable_lora,
            idempotency_key=idempotency_key,
            settings=settings,
            _provisioning_suite_id=training_suite_id,
        )
        if isinstance(result, str):
            _set_suite_setup_status(
                project_id,
                training_suite_id,
                settings,
                status="failed",
                error=f"Training job preparation failed: {result}",
            )
    except asyncio.CancelledError:
        _set_suite_setup_status(
            project_id,
            training_suite_id,
            settings,
            status="failed",
            error=(
                "Backend shutdown interrupted Training Jobs setup. "
                "Return to Student Training and start a new suite."
            ),
        )
        raise
    except Exception as exc:
        logger.exception(
            "Training suite %s setup failed after provisioning", training_suite_id
        )
        _set_suite_setup_status(
            project_id,
            training_suite_id,
            settings,
            status="failed",
            error=f"Training job preparation failed: {type(exc).__name__}: {exc}",
        )


async def launch_training_suite(
    project_id: str,
    *,
    student_base_model_config_ids: list[str],
    training_preset: str,
    include_auto_labeled: bool,
    export_field_mode: str,
    quantization_schemes: list[str],
    idempotency_key: str,
    settings: Settings,
    enable_lora: bool = True,
) -> dict[str, Any] | str:
    """Validate and launch Training Jobs, provisioning selected bases if needed.

    Ready bases use the normal suite path and never render a provisioning step.
    If one or more selected bases are missing, a provisional TrainingSuite is
    returned immediately and owns one aggregate provisioning stage.
    """
    engine = get_project_engine(project_id, settings.WORKSPACE_ROOT)
    if engine is None:
        return f"not found: Project {project_id}"

    error = _validate_request(
        training_preset=training_preset,
        export_field_mode=export_field_mode,
        quantization_schemes=list(quantization_schemes),
    )
    if error is not None:
        return error

    with Session(engine) as session:
        existing = (
            session.query(TrainingSuite)
            .filter_by(project_id=project_id, idempotency_key=idempotency_key)
            .first()
        )
        if existing is not None:
            return _suite_snapshot_response(session, project_id, existing)

    # This is intentionally server-side. The SME no longer sees or waits on a
    # separate Preflight section, but the safety contract still fails closed
    # before a provisioning transfer or TAO job can start.
    readiness = await training_preflight_service.run_training_preflight(
        project_id=project_id,
        student_base_model_config_ids=list(student_base_model_config_ids),
        settings=settings,
        include_auto_labeled=include_auto_labeled,
        enable_lora=enable_lora,
    )
    failed_checks = [check for check in readiness["checks"] if not check["passed"]]
    if failed_checks:
        first = failed_checks[0]
        prefix = (
            "tao_unreachable"
            if first["check_name"]
            in {
                "tao_reachable",
                "tao_job_timeout_supported",
                "tao_workspace_reachable",
            }
            else "validation"
        )
        return f"{prefix}: {first['message']}"

    with Session(engine) as session:
        project = session.get(Project, project_id)
        if project is None:
            return f"not found: Project {project_id}"
        if not project.active_guidance_id:
            return "validation: No active Guidance configured"
        models, error = _load_student_base_models(
            session, project_id, list(student_base_model_config_ids)
        )
        if error is not None:
            return error
        missing = [
            model
            for model in models
            if not (
                model.tao_base_experiment_id
                and model.tao_base_experiment_pull_status == "pull_complete"
            )
        ]
        active_guidance_id = project.active_guidance_id

    if not missing:
        return await create_training_suite(
            project_id,
            student_base_model_config_ids=student_base_model_config_ids,
            training_preset=training_preset,
            include_auto_labeled=include_auto_labeled,
            export_field_mode=export_field_mode,
            quantization_schemes=quantization_schemes,
            enable_lora=enable_lora,
            idempotency_key=idempotency_key,
            settings=settings,
        )

    provisioning = start_provisioning_run(
        project_id,
        list(student_base_model_config_ids),
        settings,
    )
    if isinstance(provisioning, str):
        return provisioning

    suite_id = generate_uuid4()
    now = utc_now()
    with Session(engine) as session:
        suite = TrainingSuite(
            training_suite_id=suite_id,
            project_id=project_id,
            idempotency_key=idempotency_key,
            guidance_id=active_guidance_id,
            training_preset=training_preset,
            export_field_mode=export_field_mode,
            include_auto_labeled=include_auto_labeled,
            training_dataset_export_id=None,
            evaluation_dataset_export_id=None,
            selected_student_base_model_config_ids=list(student_base_model_config_ids),
            quantization_schemes=list(quantization_schemes),
            chain_ids_ordered=[],
            provisioning_run_id=provisioning["provisioning_run_id"],
            provisioning_model_names=list(provisioning["requested_model_names"]),
            setup_error_ref=None,
            status="provisioning",
            created_at=now,
            started_at=now,
        )
        session.add(suite)
        session.commit()
        response = _suite_snapshot_response(session, project_id, suite)

    worker = _complete_suite_after_provisioning(
        project_id,
        suite_id,
        student_base_model_config_ids=list(student_base_model_config_ids),
        training_preset=training_preset,
        include_auto_labeled=include_auto_labeled,
        export_field_mode=export_field_mode,
        quantization_schemes=list(quantization_schemes),
        enable_lora=enable_lora,
        idempotency_key=idempotency_key,
        settings=settings,
    )
    try:
        background_manager.register(f"training-suite-setup-{suite_id}", worker)
    except RuntimeError as exc:
        worker.close()
        _set_suite_setup_status(
            project_id,
            suite_id,
            settings,
            status="failed",
            error=f"Could not start Training Jobs setup: {exc}",
        )
        return get_training_suite(project_id, suite_id, settings=settings)
    return response


async def create_training_suite(
    project_id: str,
    *,
    student_base_model_config_ids: list[str],
    training_preset: str,
    include_auto_labeled: bool,
    export_field_mode: str,
    quantization_schemes: list[str],
    idempotency_key: str,
    settings: Settings,
    enable_lora: bool = True,
    _upload_archive: UploadArchiveFn | None = None,
    _s3_client: Any | None = None,
    _provisioning_suite_id: str | None = None,
) -> dict[str, Any] | str:
    """Create a training suite end-to-end.

    Behavior:
      * Replays idempotent retry: returns existing suite without re-writes.
      * Validates request (preset, field mode, quantization schemes).
      * Validates student_base role + project-scoping for every model.
      * Phase 1a — read-only validation (project, Guidance, models).
      * Phase 1b — builds + commits the training and eval DatasetExports.
        The gzip-tar builds run in a worker thread, each with its own
        short-lived session (SQLite write discipline). A later failure
        leaves standalone DatasetExport rows — the same shape the exports
        API produces — consistent with the archives already on disk.
      * Phase 1c — uploads both archives to the TAO workspace S3 with no
        write transaction open; lineage commits per upload.
      * Phase 1d — single short write transaction: pre-creates every chain
        job with ``status="not_started"`` and inserts the TrainingSuite
        record (these stay all-or-nothing).
      * Phase 2 — outside any transaction: kicks off the first chain's
        first train job via :func:`tao_job_service.submit_chain_job`.
        If kickoff fails, the suite's first train job is persisted as
        ``failed`` with a sanitized ``error_ref``; the suite response
        status becomes ``"failed"`` and chain advancement is blocked for
        that chain (the polling loop's chain-halted logic takes over).

    Returns the full suite response dict (shaped for the Training Job
    Monitor pre-render) or an error string mapped to HTTP codes by the router.
    """
    engine = get_project_engine(project_id, settings.WORKSPACE_ROOT)
    if engine is None:
        return f"not found: Project {project_id}"

    # ── Validate request shape (422 on failure) ─────────────────────────
    err = _validate_request(
        training_preset=training_preset,
        export_field_mode=export_field_mode,
        quantization_schemes=list(quantization_schemes),
    )
    if err is not None:
        return err

    # ── Idempotency replay ─────────────────────────────────────────────
    with Session(engine) as session:
        existing = (
            session.query(TrainingSuite)
            .filter_by(project_id=project_id, idempotency_key=idempotency_key)
            .first()
        )
        if existing is not None and not (
            _provisioning_suite_id == existing.training_suite_id
            and existing.status in {"provisioning", "preparing"}
        ):
            # Snapshot the response without any writes.
            return _suite_snapshot_response(session, project_id, existing)

    # ── Phase 1a: validate inputs (read-only) ──────────────────────────
    suite_id: str
    chain_ids_ordered: list[str]
    first_chain_first_job_id: str | None = None
    with Session(engine) as session:
        project = session.query(Project).filter_by(project_id=project_id).first()
        if project is None:
            return f"not found: Project {project_id}"
        if not project.active_guidance_id:
            return "No active Guidance configured"
        active_guidance_id = project.active_guidance_id

        # Validate all selected models before the expensive export builds.
        _models, err = _load_student_base_models(
            session, project_id, list(student_base_model_config_ids)
        )
        if err is not None:
            return err

    # TAO deployment preconditions are pure reads — check them before
    # minutes of archive building, not after.
    tao_deployment_config = read_tao_deployment_config(settings)
    if tao_deployment_config is None:
        return (
            "validation: TAO deployment config missing — run "
            "`vlm-feedback-loop tao-bootstrap` first."
        )
    if tao_deployment_config.bootstrap_status != "bootstrapped":
        return (
            "validation: TAO workspace is not bootstrapped — run "
            "`vlm-feedback-loop tao-bootstrap` first."
        )

    try:
        # ── Phase 1b: build + persist the dataset exports ───────────────
        # create_dataset_export gzip-tars every training/eval image and
        # hashes the finished archive — CPU + disk I/O proportional to
        # total image bytes — so each call runs in a worker thread and
        # commits in its own short-lived session (write discipline: no
        # write transaction across long work; no long work on the event
        # loop). The export rows commit here: a later upload or chain
        # failure leaves standalone DatasetExport rows, the same shape
        # the exports API produces, keeping the DB consistent with the
        # archive files already on disk. selection_filters pins the
        # Phase-1a Guidance snapshot: both exports MUST render the
        # identical human turn — without it, each build re-reads
        # active_guidance_id and a mid-flight Guidance activation could
        # split the pair.
        training_tier = "combined" if include_auto_labeled else "verified_only"

        def _build_export(intent: str, tier: str) -> dict[str, Any] | str:
            return create_dataset_export(
                project_id,
                dataset_intent=intent,
                label_tier_filter=tier,
                export_field_mode=export_field_mode,
                selection_filters={"guidance_id": active_guidance_id},
                settings=settings,
            )

        train_export = await asyncio.to_thread(_build_export, "training", training_tier)
        if isinstance(train_export, str):
            return train_export
        eval_export = await asyncio.to_thread(
            _build_export, "evaluation", "verified_only"
        )
        if isinstance(eval_export, str):
            return eval_export

        # Sidecar annotations.json paths — required by the cosmos-rl
        # spec contract (annotation_path must be a separate JSON URL
        # from media_path; see _upload_export_archive docstring).
        training_annotations_local = train_export["artifact_refs"].get(
            "annotations_path"
        )
        eval_annotations_local = eval_export["artifact_refs"].get("annotations_path")
        if not training_annotations_local or not eval_annotations_local:
            raise RuntimeError(
                "Dataset export returned no annotations_path for "
                "training or evaluation — refusing to build the chain"
            )

        # ── Phase 1c: upload exports to the TAO workspace S3 ────────────
        # No write transaction is open while archive bytes stream to S3;
        # each upload persists its lineage fields in its own short commit.
        # Chains reference the TAO-readable spec URI, never the
        # Blueprint-local archive path.
        async def _upload_one(
            export: dict[str, Any], annotations_local: str | None, kind: str
        ) -> UploadResult | str:
            with Session(engine) as upload_session:
                export_row = (
                    upload_session.query(DatasetExport)
                    .filter_by(dataset_export_id=export["dataset_export_id"])
                    .one()
                )
                upload = await _upload_export_archive(
                    session=upload_session,
                    export_row=export_row,
                    archive_path=Path(export["artifact_refs"]["archive_path"]),
                    deployment_config=tao_deployment_config,
                    upload_archive=_upload_archive,
                    s3_client=_s3_client,
                    annotations_path=(
                        Path(annotations_local) if annotations_local else None
                    ),
                )
                if not upload.success:
                    # Reaching the TAO workspace S3 failed — an infra
                    # problem, not the SME's input. Return the
                    # tao_unreachable token (→ 503) instead of raising into
                    # the catch-all that would mislabel it a 400.
                    logger.error("%s dataset upload failed: %s", kind, upload.error)
                    return (
                        f"tao_unreachable: {kind.lower()} dataset upload "
                        f"failed: {upload.error}"
                    )
                upload_session.commit()
                return upload

        train_upload = await _upload_one(
            train_export, training_annotations_local, "Training"
        )
        if isinstance(train_upload, str):
            return train_upload
        eval_upload = await _upload_one(
            eval_export, eval_annotations_local, "Evaluation"
        )
        if isinstance(eval_upload, str):
            return eval_upload

        # cosmos-rl needs annotation_path as a separate JSON URL.
        # ``media_path`` points at the PARENT directory of the
        # .tar.gz, not the .tar.gz itself. TAO's path-substitution
        # logic (``download_from_user_storage``) does
        # ``dict[key] = local_path.replace(".tar.gz", "")`` but
        # extracts the tarball into ``os.path.dirname(local_path)``.
        # The two paths differ by exactly one component when the
        # value is a .tar.gz URL — cosmos-rl's media_path then
        # points one level deeper than where files actually land,
        # surfacing as ``FileNotFoundError`` on every dataloader
        # ``open()``. When the value is
        # a directory URL TAO uses ``download_folder`` + walks for
        # tarballs to extract; the substituted path equals the
        # extraction destination, so cosmos-rl's joins resolve.
        def _strip_tar_basename(url: str | None) -> str | None:
            if not url or not url.endswith(".tar.gz"):
                return url
            # Drop the trailing ``/{export_id}.tar.gz`` segment so
            # the remaining URL points at the dataset_exports/
            # directory containing both the tar and the sidecar.
            return url.rsplit("/", 1)[0] + "/"

        training_archive_path_opt = _strip_tar_basename(train_upload.spec_reference)
        training_annotation_path_opt = train_upload.annotation_spec_reference
        eval_archive_path_opt = _strip_tar_basename(eval_upload.spec_reference)
        eval_annotation_path_opt = eval_upload.annotation_spec_reference
        # The dataset upload step above sets spec_reference for both
        # train and eval; either being None here means the upload
        # service silently misbehaved.
        if (
            training_archive_path_opt is None
            or training_annotation_path_opt is None
            or eval_archive_path_opt is None
            or eval_annotation_path_opt is None
        ):
            raise RuntimeError(
                "Dataset upload returned no spec reference for "
                "training or eval — refusing to build the chain"
            )
        training_archive_path = training_archive_path_opt
        training_annotation_path = training_annotation_path_opt
        eval_archive_path = eval_archive_path_opt
        eval_annotation_path = eval_annotation_path_opt

        # Build per-model chains. Each TAO job payload
        # carries `workspace` and `base_experiment_ids`; both have
        # already been validated by the training preflight
        # (`tao_workspace_reachable` + `tao_base_experiment_ready`) by
        # the time the suite is created, so we treat null values here
        # as a programming error rather than a runtime failure.
        workspace_id = tao_deployment_config.tao_workspace_id
        if not workspace_id:
            raise RuntimeError(
                "TAODeploymentConfig.tao_workspace_id is null even "
                "though bootstrap_status='bootstrapped' — preflight "
                "should have rejected this; refusing to build a "
                "malformed TAO request"
            )
        # ── Phase 1d: chains + suite (single short write transaction) ───
        with Session(engine) as session:
            models, err = _load_student_base_models(
                session, project_id, list(student_base_model_config_ids)
            )
            if err is not None:
                return err
            chain_ids_ordered = []
            first_chain_train_job: TAOJob | None = None
            for mc in models:
                base_experiment_id = mc.tao_base_experiment_id
                if not base_experiment_id:
                    # A resolvable precondition (the base experiment isn't
                    # provisioned), not an internal bug — return a validation
                    # error string (→ 400) rather than raising into the 500
                    # path. Preflight (tao_base_experiment_ready) is the
                    # spec-mandated gate; this is the last line of defence.
                    # Returning exits the `with Session` with nothing
                    # committed, so no chain rows are persisted.
                    logger.error(
                        "Training suite: ModelConfig %s (%s) has null "
                        "tao_base_experiment_id",
                        mc.model_config_id,
                        mc.model_name,
                    )
                    return (
                        f"validation: ModelConfig {mc.model_config_id} "
                        f"({mc.model_name}) has null tao_base_experiment_id "
                        "— run training preflight to provision the base "
                        "experiment first"
                    )
                hyperparameters = resolve_training_preset(
                    training_preset, mc.model_name
                )
                chain_id, jobs = _create_chain_rows_in_session(
                    session,
                    project_id=project_id,
                    mc=mc,
                    training_preset=training_preset,
                    hyperparameters=hyperparameters,
                    training_dataset_export_id=train_export["dataset_export_id"],
                    training_archive_path=training_archive_path,
                    training_annotation_path=training_annotation_path,
                    eval_dataset_export_id=eval_export["dataset_export_id"],
                    eval_archive_path=eval_archive_path,
                    eval_annotation_path=eval_annotation_path,
                    quantization_schemes=list(quantization_schemes),
                    tao_release_version=settings.TAO_RELEASE_VERSION,
                    cosmos_rl_container_tag=settings.COSMOS_RL_CONTAINER_TAG,
                    workspace_id=workspace_id,
                    base_experiment_id=base_experiment_id,
                    enable_lora=enable_lora,
                    timeout_minutes=settings.TAO_JOB_TIMEOUT_MINUTES,
                    quantization_calibration_samples=(
                        settings.TAO_QUANTIZATION_CALIBRATION_SAMPLES
                    ),
                )
                chain_ids_ordered.append(chain_id)
                if first_chain_train_job is None:
                    # chain_sequence=1 is the train job; jobs[0] by construction.
                    first_chain_train_job = jobs[0]
            if _provisioning_suite_id is not None:
                suite_id = _provisioning_suite_id
                suite = session.get(TrainingSuite, suite_id)
                if suite is None or suite.project_id != project_id:
                    return f"not found: TrainingSuite {suite_id}"
                if suite.status not in {"provisioning", "preparing"}:
                    return (
                        f"conflict: TrainingSuite {suite_id} is no longer "
                        "waiting for setup"
                    )
                suite.guidance_id = active_guidance_id
                suite.training_dataset_export_id = train_export["dataset_export_id"]
                suite.evaluation_dataset_export_id = eval_export["dataset_export_id"]
                suite.chain_ids_ordered = chain_ids_ordered
                suite.setup_error_ref = None
                suite.status = "initialized"
            else:
                suite_id = generate_uuid4()
                suite = TrainingSuite(
                    training_suite_id=suite_id,
                    project_id=project_id,
                    idempotency_key=idempotency_key,
                    guidance_id=active_guidance_id,
                    training_preset=training_preset,
                    export_field_mode=export_field_mode,
                    include_auto_labeled=include_auto_labeled,
                    training_dataset_export_id=train_export["dataset_export_id"],
                    evaluation_dataset_export_id=eval_export["dataset_export_id"],
                    selected_student_base_model_config_ids=list(
                        student_base_model_config_ids
                    ),
                    quantization_schemes=list(quantization_schemes),
                    chain_ids_ordered=chain_ids_ordered,
                    provisioning_run_id=None,
                    provisioning_model_names=None,
                    setup_error_ref=None,
                    status="initialized",
                )
                session.add(suite)

            first_chain_first_job_id = (
                first_chain_train_job.tao_job_id if first_chain_train_job else None
            )

            # The exports were pinned to the Phase-1a Guidance snapshot,
            # and every read since ran on autocommit — an edit could
            # commit any time before this transaction's first write. The
            # flush takes the write lock, so the re-read below is
            # post-edit truth: a mismatch means the suite would train on
            # prompts the active setup no longer produces, so it is
            # abandoned (the session rolls back on return) and the SME
            # retries under the new Guidance.
            session.flush()
            active_gid_now = (
                session.query(Project.active_guidance_id)
                .filter_by(project_id=project_id)
                .scalar()
            )
            if active_gid_now != active_guidance_id:
                return (
                    "conflict: the active Guidance changed while the "
                    "training datasets were being built — start the "
                    "training suite again to use the new Guidance."
                )

            session.commit()
    except ValueError as exc:
        # Genuine input/validation error (the export and upload services
        # raise ValueError for malformed inputs) → 400.
        logger.exception("Training suite creation rejected input")
        return f"validation: training suite creation failed: {exc}"
    except Exception:
        # A transport/DB/programming failure is NOT a client validation
        # error. Let it propagate so FastAPI returns 500 (the `with
        # Session` context manager rolls back on the way out) instead of
        # mislabeling every failure a 400. Upload/reachability failures are
        # already returned as tao_unreachable (503) above.
        logger.exception("Training suite creation failed unexpectedly")
        raise

    # ── Phase 2: kickoff first chain's first train job ─────────────────
    suite_status = "running"
    if first_chain_first_job_id is not None:
        # advance_on_failure=False: a first-job submission failure fails the
        # whole suite here (below); it must NOT cross-advance to the next chain
        # the way a mid-chain failure does.
        kickoff = await tao_job_service.submit_chain_job(
            project_id,
            first_chain_first_job_id,
            settings=settings,
            advance_on_failure=False,
        )
        if kickoff == "failed":
            suite_status = "failed"
        elif kickoff.startswith(("not found", "conflict", "validation")):
            # Should not happen for a freshly created not_started job, but
            # log defensively. We do NOT roll back — the TrainingSuite is
            # persisted; the polling loop will retry via chain recovery.
            logger.warning(
                "Training suite %s: kickoff returned %r — polling recovery will retry",
                suite_id,
                kickoff,
            )

    # ── Update suite status post-kickoff ───────────────────────────────
    now = utc_now()
    with Session(engine) as session:
        suite = (
            session.query(TrainingSuite).filter_by(training_suite_id=suite_id).first()
        )
        if suite is not None:
            suite.status = suite_status
            suite.started_at = now
            if suite_status == "failed":
                suite.completed_at = now
            session.commit()

    # ── Build response ─────────────────────────────────────────────────
    with Session(engine) as session:
        suite = (
            session.query(TrainingSuite).filter_by(training_suite_id=suite_id).first()
        )
        if suite is None:
            return f"not found: TrainingSuite {suite_id}"

        return _suite_snapshot_response(session, project_id, suite)


def get_training_suite(
    project_id: str,
    training_suite_id: str,
    *,
    settings: Settings,
) -> dict[str, Any] | str:
    """Return the suite with a live snapshot of each chain's TAOJob statuses."""
    engine = get_project_engine(project_id, settings.WORKSPACE_ROOT)
    if engine is None:
        return f"not found: Project {project_id}"
    with Session(engine) as session:
        suite = (
            session.query(TrainingSuite)
            .filter_by(project_id=project_id, training_suite_id=training_suite_id)
            .first()
        )
        if suite is None:
            return f"not found: TrainingSuite {training_suite_id}"

        return _suite_snapshot_response(session, project_id, suite)


async def cancel_training_suite(
    project_id: str,
    training_suite_id: str,
    *,
    settings: Settings,
) -> dict[str, Any] | str:
    """Best-effort cancel every remaining operation in a Training Suite.

    The suite is terminalized first so project re-entry stops resuming the
    monitor immediately. Setup/provisioning tasks are canceled, every known
    TAO-side job receives a cancellation request, and all remaining local
    TAOJob rows become ``canceled`` even when TAO cannot confirm the remote
    request. Remote failures remain on the job audit rows and in the response.
    """
    engine = get_project_engine(project_id, settings.WORKSPACE_ROOT)
    if engine is None:
        return f"not found: Project {project_id}"

    now = utc_now()
    with Session(engine) as session:
        suite = (
            session.query(TrainingSuite)
            .filter_by(
                project_id=project_id,
                training_suite_id=training_suite_id,
            )
            .first()
        )
        if suite is None:
            return f"not found: TrainingSuite {training_suite_id}"
        if suite.status in {"completed", "failed"}:
            return (
                "conflict: cannot cancel TrainingSuite in terminal status "
                f"{suite.status!r}"
            )

        # Persist the authoritative exit state before touching external work.
        # Background completion handlers and the polling roll-up both preserve
        # this status, preventing late events from restoring the redirect.
        suite.status = "canceled"
        suite.completed_at = suite.completed_at or now
        provisioning_run_id = suite.provisioning_run_id
        session.commit()

    setup_task_ids = [f"training-suite-setup-{training_suite_id}"]
    if provisioning_run_id:
        setup_task_ids.append(f"tao-base-provision-{provisioning_run_id}")
    setup_tasks_canceled = 0
    for task_id in setup_task_ids:
        if await background_manager.cancel_task(task_id):
            setup_tasks_canceled += 1

    # Reload after setup tasks settle: a worker may have materialized chain
    # rows immediately before observing cancellation.
    with Session(engine) as session:
        suite = session.get(TrainingSuite, training_suite_id)
        assert suite is not None
        jobs = (
            session.query(TAOJob)
            .filter(
                TAOJob.project_id == project_id,
                TAOJob.chain_id.in_(list(suite.chain_ids_ordered or [])),
            )
            .all()
            if suite.chain_ids_ordered
            else []
        )
        remote_targets = [
            (job.tao_job_id, job.tao_external_job_id)
            for job in jobs
            if job.status not in tao_job_service.TERMINAL_STATUSES
            and job.tao_external_job_id
        ]

    remote_calls = [
        tao_job_service.request_tao_job_cancel(external_id, settings=settings)
        for _, external_id in remote_targets
    ]
    remote_results = (
        await asyncio.gather(*remote_calls, return_exceptions=True)
        if remote_calls
        else []
    )

    failure_by_job: dict[str, str] = {}
    for (tao_job_id, _), result in zip(remote_targets, remote_results, strict=True):
        if isinstance(result, BaseException):
            detail = f"{type(result).__name__}: {result}"
        elif not result.get("success"):
            detail = str(result.get("error") or "TAO cancel was not confirmed")
        else:
            continue
        failure_by_job[tao_job_id] = (
            tao_job_service.sanitize_error(detail) or "TAO cancel was not confirmed"
        )

    jobs_canceled = 0
    jobs_already_terminal = 0
    with Session(engine) as session:
        suite = session.get(TrainingSuite, training_suite_id)
        assert suite is not None
        jobs = (
            session.query(TAOJob)
            .filter(
                TAOJob.project_id == project_id,
                TAOJob.chain_id.in_(list(suite.chain_ids_ordered or [])),
            )
            .all()
            if suite.chain_ids_ordered
            else []
        )
        for job in jobs:
            if job.status in tao_job_service.TERMINAL_STATUSES:
                jobs_already_terminal += 1
                continue
            job.status = "canceled"
            job.completed_at = job.completed_at or now
            job.chain_halted_reason = (
                job.chain_halted_reason or "Training Suite canceled by SME"
            )
            if job.tao_job_id in failure_by_job:
                job.poll_error_ref = (
                    f"suite_cancel_unconfirmed: {failure_by_job[job.tao_job_id]}"
                )
            else:
                job.poll_error_ref = None
            jobs_canceled += 1

        # A late worker/poller may have touched the suite while remote calls
        # were in flight. Cancellation remains the authoritative parent state.
        suite.status = "canceled"
        suite.completed_at = suite.completed_at or now
        session.commit()
        snapshot = _suite_snapshot_response(session, project_id, suite)

    failures = [
        {"tao_job_id": tao_job_id, "error": error}
        for tao_job_id, error in failure_by_job.items()
    ]
    response = {
        "training_suite": snapshot,
        "jobs_canceled": jobs_canceled,
        "jobs_already_terminal": jobs_already_terminal,
        "setup_tasks_canceled": setup_tasks_canceled,
        "remote_cancel_failures": failures,
    }

    try:
        await sse_manager.emit(
            project_id,
            "training_suite_canceled",
            {
                "run_id": training_suite_id,
                "run_type": "training_suite",
                "status": "canceled",
                "jobs_canceled": jobs_canceled,
                "remote_cancel_failures": len(failures),
            },
        )
    except Exception:  # pragma: no cover - SSE is a hint channel
        logger.exception("SSE emit failed after Training Suite cancellation")

    return response


def list_training_suites(
    project_id: str,
    *,
    cursor: str | None = None,
    limit: int = 20,
    settings: Settings,
) -> tuple[list[dict[str, Any]], str | None] | str:
    """List training suites newest-first with cursor pagination."""
    engine = get_project_engine(project_id, settings.WORKSPACE_ROOT)
    if engine is None:
        return f"not found: Project {project_id}"
    with Session(engine) as session:
        stmt = (
            session.query(TrainingSuite)
            .filter_by(project_id=project_id)
            .order_by(
                TrainingSuite.created_at.desc(), TrainingSuite.training_suite_id.asc()
            )
        )
        if cursor:
            try:
                cur_created_at, cur_id = decode_cursor(cursor)
            except InvalidCursorError:
                return "validation: invalid cursor"
            stmt = stmt.filter(
                after_position_asc(
                    TrainingSuite.created_at,
                    TrainingSuite.training_suite_id,
                    cur_created_at,
                    cur_id,
                )
            )

        rows = stmt.limit(limit + 1).all()

        next_cursor: str | None = None
        if len(rows) > limit:
            rows = rows[:limit]
            tail = rows[-1]
            next_cursor = encode_cursor(tail.created_at, tail.training_suite_id)

        return [
            _suite_snapshot_response(session, project_id, suite) for suite in rows
        ], next_cursor


def recover_interrupted_training_suite_setups(settings: Settings) -> int:
    """Fail provisional suites whose in-process setup task cannot survive restart."""
    recovered = 0
    for project_dir in iter_project_dirs(Path(settings.WORKSPACE_ROOT)):
        try:
            engine = open_project_db(project_dir)
        except DatabaseMigrationError as exc:
            # A pre-public or corrupt project database must not prevent the
            # backend from starting for every healthy/new project. Other
            # lifespan recovery scans apply the same per-project isolation.
            logger.warning(
                "Skipping Training Suite setup recovery for %s (%s: %s)",
                project_dir.name,
                type(exc).__name__,
                exc,
            )
            continue
        with Session(engine) as session:
            suites = (
                session.query(TrainingSuite)
                .filter(TrainingSuite.status.in_({"provisioning", "preparing"}))
                .all()
            )
            for suite in suites:
                suite.status = "failed"
                suite.setup_error_ref = (
                    "Backend restart interrupted Training Jobs setup. "
                    "Return to Student Training and start a new suite."
                )
                suite.completed_at = utc_now()
                recovered += 1
            if suites:
                session.commit()
    return recovered


__all__ = [
    "create_training_suite",
    "cancel_training_suite",
    "get_training_suite",
    "launch_training_suite",
    "list_training_suites",
    "recover_interrupted_training_suite_setups",
    "resolve_training_presets_for_models",
]


# ── Workspace S3 upload helpers ─────────────────────────────────────────────


async def _upload_export_archive(
    *,
    session: Session,
    export_row: DatasetExport,
    archive_path: Path,
    deployment_config: TAODeploymentConfig,
    upload_archive: UploadArchiveFn | None,
    s3_client: Any | None,
    annotations_path: Path | None = None,
) -> UploadResult:
    """Drive ``upload_dataset_archive`` with an optional test-supplied hook.

    ``annotations_path`` is a sidecar JSON file alongside the .tar.gz that
    Cosmos-RL needs as a separate URL for ``annotation_path``: pointing
    annotation_path at the same .tar.gz URL as media_path crashes the
    cosmos-rl worker with FileNotFoundError because TAO's URL→path
    mapping resolves both to the extracted directory rather than the
    JSON file.
    """
    fn: UploadArchiveFn = (
        upload_archive
        if upload_archive is not None
        else tao_dataset_upload_service.upload_dataset_archive
    )
    if s3_client is None and upload_archive is None:
        # Production path: build the real boto3 S3 client. Only done
        # when neither the upload hook nor client is injected.
        s3_client = build_s3_client(deployment_config)
    # Guard above guarantees a client when no override hook is set; the
    # hook path supplies its own client (used by tests / mocks).
    assert s3_client is not None or upload_archive is not None

    return await fn(
        session,
        dataset_export=export_row,
        archive_path=archive_path,
        deployment_config=deployment_config,
        s3_client=s3_client,  # pyright: ignore[reportArgumentType] — narrowed by the assert above; pyright cannot prove the upload_archive branch
        annotations_path=annotations_path,
    )
