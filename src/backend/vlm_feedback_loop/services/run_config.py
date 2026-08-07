# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build immutable inference inputs for evaluation and Batch executors.

New runs persist the mutable model, endpoint, and request-shaping settings they
consume. Before each Phase-B loop, this module validates that snapshot and
combines it with the run's Guidance and Example paths. Credentials, filesystem
authorization, timeouts, retries, and scheduling remain live operational
boundaries and are deliberately excluded.
"""

from __future__ import annotations

import copy
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy.orm import Session

from vlm_feedback_loop.config import Settings
from vlm_feedback_loop.db.models.example import Example
from vlm_feedback_loop.db.models.guidance import Guidance
from vlm_feedback_loop.db.models.model_config import ModelConfig
from vlm_feedback_loop.db.models.nim_endpoint import NimEndpoint
from vlm_feedback_loop.db.models.project import Project
from vlm_feedback_loop.db.models.run import RunRecord
from vlm_feedback_loop.schemas.model_config import (
    ThinkingToggleMode,
    VisualBudgetMode,
)
from vlm_feedback_loop.schemas.nim import AuthMode, EndpointMode
from vlm_feedback_loop.services.image_cap_resolver import (
    resolve_max_images_per_request,
)
from vlm_feedback_loop.services.prompt_service import (
    ModelConfigInput,
    resolve_generation_params,
    resolve_visual_budget,
)

CapabilitySupport = Literal["unknown", "supported", "unsupported"]


class SamplingParamsSnapshot(BaseModel):
    """Concrete sampling values selected for one run."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    temperature: float = Field(ge=0.0)
    top_p: float = Field(gt=0.0, le=1.0)


class VisualBudgetParamsSnapshot(BaseModel):
    """Concrete processor arguments selected for one run."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    mm_processor_kwargs: dict[str, Any]


class InferenceSettingsSnapshot(BaseModel):
    """Credential-free Settings values that can change request semantics."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    generation_preset_key: str = Field(min_length=1)
    sampling_params: SamplingParamsSnapshot
    visual_budget_preset_key: str = Field(min_length=1)
    visual_budget_params_effective: VisualBudgetParamsSnapshot | None
    base_output_tokens_floor: int = Field(ge=1)
    json_structural_overhead: int = Field(ge=0)
    max_output_fraction: float = Field(gt=0.0, le=1.0)
    rationale_note_estimate: int = Field(ge=1)
    default_unbounded_string_budget: int = Field(ge=1)
    model_reasoning_headroom_tokens: int = Field(ge=4096)
    runtime_prompt_output_max_tokens_override: int | None = Field(default=None, ge=1)
    token_safety_margin: float = Field(gt=0.0, le=1.0)
    icl_max_examples: int | None = Field(default=None, ge=1)
    icl_candidate_limit: int | None = Field(default=None, ge=1)
    icl_sim_gap: float | None = Field(default=None, ge=0.0)
    icl_abs_threshold: float | None = Field(default=None, ge=-1.0, le=1.0)
    image_transport_max_longest_edge: int | None = Field(default=None, ge=1)


class _RuntimeConfigSnapshotBase(BaseModel):
    """Model and endpoint fields shared by both persisted snapshot versions."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    project_id: str = Field(min_length=1)
    model_config_id: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    endpoint_id: str = Field(min_length=1)
    endpoint_base_url: str = Field(min_length=1)
    endpoint_mode: EndpointMode
    endpoint_auth_mode: AuthMode
    context_window_tokens: int = Field(ge=1)
    thinking_toggle_mode: ThinkingToggleMode
    thinking_toggle_support: CapabilitySupport
    visual_budget_mode: VisualBudgetMode
    visual_budget_support: CapabilitySupport
    structured_generation_support: CapabilitySupport
    max_images_per_request: int = Field(ge=1)
    default_icl_max_examples: int | None = Field(default=None, ge=1)


class RuntimeConfigSnapshotV1(_RuntimeConfigSnapshotBase):
    """Legacy model/endpoint-only shape written by schema revision v1_0003."""

    version: Literal[1] = 1


class RuntimeConfigSnapshot(_RuntimeConfigSnapshotBase):
    """Complete, immutable, credential-free inference inputs for one run."""

    version: Literal[2] = 2
    inference_settings: InferenceSettingsSnapshot


def _create_inference_settings_snapshot(
    settings: Settings,
    *,
    generation_preset_key: str,
    visual_budget_preset_key: str,
    visual_budget_mode: str,
    visual_budget_support: str,
    icl_max_examples: int | None,
    icl_candidate_limit: int | None,
    icl_sim_gap: float | None,
    icl_abs_threshold: float | None,
) -> InferenceSettingsSnapshot:
    """Resolve the concrete request-shaping values selected at run creation."""
    sampling_params = resolve_generation_params(
        generation_preset_key,
        settings.LABELING_PRESETS,
    )
    visual_budget = resolve_visual_budget(
        visual_budget_preset_key,
        visual_budget_mode,
        visual_budget_support,
        settings.VISUAL_BUDGET_PRESETS,
    )
    visual_params = visual_budget["visual_budget_params_effective"]

    return InferenceSettingsSnapshot(
        generation_preset_key=generation_preset_key,
        sampling_params=SamplingParamsSnapshot.model_validate(sampling_params),
        visual_budget_preset_key=visual_budget_preset_key,
        visual_budget_params_effective=(
            VisualBudgetParamsSnapshot.model_validate(copy.deepcopy(visual_params))
            if visual_params is not None
            else None
        ),
        base_output_tokens_floor=settings.BASE_OUTPUT_TOKENS_FLOOR,
        json_structural_overhead=settings.JSON_STRUCTURAL_OVERHEAD_TOKENS,
        max_output_fraction=settings.MAX_OUTPUT_FRACTION,
        rationale_note_estimate=settings.RATIONALE_NOTE_ESTIMATE_TOKENS,
        default_unbounded_string_budget=settings.DEFAULT_UNBOUNDED_STRING_BUDGET,
        model_reasoning_headroom_tokens=settings.MODEL_REASONING_HEADROOM_TOKENS,
        runtime_prompt_output_max_tokens_override=(
            settings.RUNTIME_PROMPT_OUTPUT_MAX_TOKENS_OVERRIDE
        ),
        token_safety_margin=settings.RUNTIME_PROMPT_TOKEN_SAFETY_MARGIN,
        icl_max_examples=icl_max_examples,
        icl_candidate_limit=icl_candidate_limit,
        icl_sim_gap=icl_sim_gap,
        icl_abs_threshold=icl_abs_threshold,
        image_transport_max_longest_edge=(settings.IMAGE_TRANSPORT_MAX_LONGEST_EDGE),
    )


def create_runtime_config_snapshot(
    session: Session,
    project_id: str,
    model_config_id: str,
    *,
    settings: Settings,
    generation_preset_key: str,
    visual_budget_preset_key: str,
    icl_max_examples: int | None,
    icl_candidate_limit: int | None,
    icl_sim_gap: float | None,
    icl_abs_threshold: float | None,
) -> dict[str, Any]:
    """Capture every mutable, result-shaping inference input a run consumes."""
    model_config = (
        session.query(ModelConfig)
        .filter_by(project_id=project_id, model_config_id=model_config_id)
        .one_or_none()
    )
    if model_config is None:
        raise RuntimeError(f"ModelConfig {model_config_id} not found for project")
    endpoint = (
        session.query(NimEndpoint)
        .filter_by(project_id=project_id, endpoint_id=model_config.endpoint_id)
        .one_or_none()
    )
    if endpoint is None:
        raise RuntimeError(
            f"NimEndpoint {model_config.endpoint_id} not found for model config"
        )

    return RuntimeConfigSnapshot.model_validate(
        {
            "project_id": project_id,
            "model_config_id": model_config.model_config_id,
            "model_name": model_config.model_name,
            "endpoint_id": endpoint.endpoint_id,
            "endpoint_base_url": endpoint.base_url,
            "endpoint_mode": endpoint.endpoint_mode,
            "endpoint_auth_mode": endpoint.auth_mode,
            "context_window_tokens": model_config.context_window_tokens,
            "thinking_toggle_mode": model_config.thinking_toggle_mode or "none",
            "thinking_toggle_support": (
                model_config.thinking_toggle_support or "unknown"
            ),
            "visual_budget_mode": model_config.visual_budget_mode or "none",
            "visual_budget_support": model_config.visual_budget_support or "unknown",
            "structured_generation_support": (
                model_config.structured_generation_support or "unknown"
            ),
            "max_images_per_request": resolve_max_images_per_request(
                model_config=model_config,
                nim_endpoint=endpoint,
            ),
            "default_icl_max_examples": model_config.default_icl_max_examples,
            "inference_settings": _create_inference_settings_snapshot(
                settings,
                generation_preset_key=generation_preset_key,
                visual_budget_preset_key=visual_budget_preset_key,
                visual_budget_mode=model_config.visual_budget_mode or "none",
                visual_budget_support=model_config.visual_budget_support or "unknown",
                icl_max_examples=icl_max_examples,
                icl_candidate_limit=icl_candidate_limit,
                icl_sim_gap=icl_sim_gap,
                icl_abs_threshold=icl_abs_threshold,
            ),
        }
    ).model_dump()


def _legacy_icl_max_examples(run: RunRecord, settings: Settings) -> int | None:
    """Recover a v1 run's persisted contract cap when one exists."""
    contract = run.inference_contract or {}
    contract_cap = contract.get("icl_max_examples")
    return contract_cap if contract_cap is not None else settings.ICL_MAX_EXAMPLES


def ensure_runtime_config_snapshot(
    session: Session,
    project_id: str,
    run: RunRecord,
    settings: Settings,
) -> RuntimeConfigSnapshot:
    """Validate a v2 snapshot or materialize one legacy run exactly once."""
    raw: Any = run.runtime_config_snapshot
    if raw is None:
        if run.model_config_id is None:
            raise RuntimeError(f"Run {run.run_id} has no model configuration")
        raw = create_runtime_config_snapshot(
            session,
            project_id,
            run.model_config_id,
            settings=settings,
            generation_preset_key=run.generation_preset_key or "precise",
            visual_budget_preset_key=run.visual_budget_preset_key or "high_detail",
            icl_max_examples=_legacy_icl_max_examples(run, settings),
            icl_candidate_limit=None,
            icl_sim_gap=settings.ICL_SIM_GAP,
            icl_abs_threshold=settings.ICL_ABS_THRESHOLD,
        )
        run.runtime_config_snapshot = raw
        session.flush()

    try:
        if not isinstance(raw, dict):
            raise ValueError("snapshot is not an object")
        raw_dict = cast("dict[str, Any]", raw)
        version: Any = raw_dict.get("version")
        if version == 1:
            legacy = RuntimeConfigSnapshotV1.model_validate(raw_dict)
            inference_settings = _create_inference_settings_snapshot(
                settings,
                generation_preset_key=run.generation_preset_key or "precise",
                visual_budget_preset_key=(
                    run.visual_budget_preset_key or "high_detail"
                ),
                visual_budget_mode=legacy.visual_budget_mode,
                visual_budget_support=legacy.visual_budget_support,
                icl_max_examples=_legacy_icl_max_examples(run, settings),
                icl_candidate_limit=None,
                icl_sim_gap=settings.ICL_SIM_GAP,
                icl_abs_threshold=settings.ICL_ABS_THRESHOLD,
            )
            snapshot = RuntimeConfigSnapshot(
                **legacy.model_dump(exclude={"version"}),
                inference_settings=inference_settings,
            )
            run.runtime_config_snapshot = snapshot.model_dump()
            session.flush()
        elif version == 2:
            snapshot = RuntimeConfigSnapshot.model_validate(raw_dict)
        else:
            raise ValueError(f"unsupported snapshot version {version!r}")
    except (TypeError, ValueError, ValidationError) as exc:
        raise RuntimeError(
            f"Run {run.run_id} has an invalid runtime configuration snapshot"
        ) from exc
    if (
        snapshot.project_id != project_id
        or snapshot.model_config_id != run.model_config_id
    ):
        raise RuntimeError(
            f"Run {run.run_id} runtime configuration lineage does not match the run"
        )
    inference_settings = snapshot.inference_settings
    if inference_settings.generation_preset_key != (
        run.generation_preset_key or "precise"
    ) or inference_settings.visual_budget_preset_key != (
        run.visual_budget_preset_key or "high_detail"
    ):
        raise RuntimeError(
            f"Run {run.run_id} runtime preset lineage does not match the run"
        )
    return snapshot


def snapshot_run_config(
    session: Session,
    project_id: str,
    run: RunRecord,
    *,
    example_keys: list[str],
    settings: Settings,
) -> dict[str, Any]:
    """Snapshot everything the run's invoke path needs as plain scalars.

    Reads the run's immutable model/endpoint snapshot plus its referenced
    Guidance and Examples, then returns the shared ``run_config`` dict; each
    executor adds its pipeline-specific keys on top.
    """
    guidance_id = run.guidance_id
    sgm_effective = run.structured_generation_mode_effective or "auto"

    project = session.query(Project).filter_by(project_id=project_id).first()
    if project is None:
        raise RuntimeError(f"Project {project_id} vanished during execute")
    project_dir = project.project_dir

    # Guidance envelope
    guidance = (
        session.query(Guidance)
        .filter_by(
            project_id=project_id,
            guidance_id=guidance_id,
        )
        .first()
    )
    if guidance is None or not guidance.schema:
        raise RuntimeError(f"Guidance {guidance_id} missing or schemaless")
    guidance_schema: dict[str, Any] = guidance.schema or {}
    guidance_fields_raw: Any = guidance_schema.get("fields", []) or []
    guidance_fields: list[dict[str, Any]] = (
        cast("list[dict[str, Any]]", guidance_fields_raw)
        if isinstance(guidance_fields_raw, list)
        else []
    )
    generation_order_raw: Any = guidance_schema.get("generation_order", []) or []
    generation_order: list[str] = (
        cast("list[str]", generation_order_raw)
        if isinstance(generation_order_raw, list)
        else []
    )
    derived_json_schema_raw: Any = guidance_schema.get("derived_json_schema", {}) or {}
    derived_json_schema: dict[str, Any] = (
        cast("dict[str, Any]", derived_json_schema_raw)
        if isinstance(derived_json_schema_raw, dict)
        else {}
    )

    runtime = ensure_runtime_config_snapshot(session, project_id, run, settings)
    inference_settings = runtime.inference_settings

    mc_input = ModelConfigInput(
        context_window_tokens=runtime.context_window_tokens,
        thinking_toggle_mode=runtime.thinking_toggle_mode,
        thinking_toggle_support=runtime.thinking_toggle_support,
        visual_budget_mode=runtime.visual_budget_mode,
        visual_budget_support=runtime.visual_budget_support,
        structured_generation_support=(
            # sgm_effective="prompt_only" forces ICL only — suppress
            # structured generation for the whole run regardless of
            # the model's actual capability (run-level control).
            "unsupported"
            if sgm_effective == "prompt_only"
            else runtime.structured_generation_support
        ),
        max_images_per_request=runtime.max_images_per_request,
        default_icl_max_examples=runtime.default_icl_max_examples,
    )

    # Per-example image paths
    storage_refs: dict[str, str] = {}
    examples = (
        session.query(Example)
        .filter(
            Example.project_id == project_id,
            Example.example_key.in_(example_keys),
        )
        .all()
    )
    for ex in examples:
        storage_refs[ex.example_key] = ex.storage_ref

    visual_budget_preset: dict[str, Any] = {}
    if inference_settings.visual_budget_params_effective is not None:
        visual_budget_preset[runtime.visual_budget_mode] = copy.deepcopy(
            inference_settings.visual_budget_params_effective.mm_processor_kwargs
        )

    invoke_settings = {
        "labeling_presets": {
            inference_settings.generation_preset_key: (
                inference_settings.sampling_params.model_dump()
            )
        },
        "visual_budget_presets": {
            inference_settings.visual_budget_preset_key: visual_budget_preset
        },
        "base_output_tokens_floor": inference_settings.base_output_tokens_floor,
        "json_structural_overhead": inference_settings.json_structural_overhead,
        "max_output_fraction": inference_settings.max_output_fraction,
        "rationale_note_estimate": inference_settings.rationale_note_estimate,
        "default_unbounded_string_budget": (
            inference_settings.default_unbounded_string_budget
        ),
        "model_reasoning_headroom_tokens": (
            inference_settings.model_reasoning_headroom_tokens
        ),
        "runtime_prompt_output_max_tokens_override": (
            inference_settings.runtime_prompt_output_max_tokens_override
        ),
        "token_safety_margin": inference_settings.token_safety_margin,
    }

    return {
        "guidance_id": guidance_id,
        "guidance_description": guidance.description or "",
        "guidance_rules": guidance.rules or "",
        "guidance_fields": guidance_fields,
        "generation_order": generation_order,
        "derived_json_schema": derived_json_schema,
        "model_config_id": runtime.model_config_id,
        "model_name": runtime.model_name,
        "mc_input": mc_input,
        "endpoint_id": runtime.endpoint_id,
        "endpoint_base_url": runtime.endpoint_base_url,
        "endpoint_mode": runtime.endpoint_mode,
        "endpoint_auth_mode": runtime.endpoint_auth_mode,
        "gen_preset_key": run.generation_preset_key or "precise",
        "thinking_on": (run.thinking_mode_effective or "on") == "on",
        "vb_preset_key": run.visual_budget_preset_key or "high_detail",
        "sgm_effective": sgm_effective,
        "inference_contract": dict(run.inference_contract or {}),
        "invoke_settings": invoke_settings,
        "icl_max_examples": inference_settings.icl_max_examples,
        "icl_candidate_limit": inference_settings.icl_candidate_limit,
        "icl_sim_gap": inference_settings.icl_sim_gap,
        "icl_abs_threshold": inference_settings.icl_abs_threshold,
        "image_transport_max_longest_edge": (
            inference_settings.image_transport_max_longest_edge
        ),
        "project_dir": project_dir,
        "storage_refs": storage_refs,
    }
