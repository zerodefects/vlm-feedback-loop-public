# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Phase-A run-config snapshot shared by the evaluation and batch-label executors.

Both pipelines snapshot the same project state (guidance envelope, model
config, endpoint, generation controls) into a plain-scalar dict before their
Phase-B loops so no ORM objects cross the async boundary and mid-run project
edits cannot affect an in-flight run. One implementation, used by both.
"""

from __future__ import annotations

from typing import Any, cast

from sqlalchemy.orm import Session

from vlm_feedback_loop.db.models.example import Example
from vlm_feedback_loop.db.models.guidance import Guidance
from vlm_feedback_loop.db.models.model_config import ModelConfig
from vlm_feedback_loop.db.models.nim_endpoint import NimEndpoint
from vlm_feedback_loop.db.models.project import Project
from vlm_feedback_loop.db.models.run import RunRecord
from vlm_feedback_loop.services.image_cap_resolver import (
    resolve_max_images_per_request,
)
from vlm_feedback_loop.services.prompt_service import ModelConfigInput


def snapshot_run_config(
    session: Session,
    project_id: str,
    run: RunRecord,
    *,
    example_keys: list[str],
) -> dict[str, Any]:
    """Snapshot everything the run's invoke path needs as plain scalars.

    Reads the run's snapshotted configuration fields plus the referenced
    Guidance / ModelConfig / NimEndpoint rows and returns the shared
    ``run_config`` dict; each executor adds its pipeline-specific keys on top.
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

    # Model config + endpoint
    model_config_row = (
        session.query(ModelConfig)
        .filter_by(
            project_id=project_id,
            model_config_id=run.model_config_id,
        )
        .first()
    )
    if model_config_row is None:
        raise RuntimeError(f"ModelConfig {run.model_config_id} not found for project")
    endpoint_id = model_config_row.endpoint_id

    # Load the NimEndpoint BEFORE building ModelConfigInput so the
    # ``max_images_per_request`` resolver can pick up any
    # per-endpoint override (services/image_cap_resolver.py).
    endpoint = (
        session.query(NimEndpoint)
        .filter_by(
            project_id=project_id,
            endpoint_id=endpoint_id,
        )
        .first()
    )
    if endpoint is None:
        raise RuntimeError(f"NimEndpoint {endpoint_id} not found for model config")

    mc_input = ModelConfigInput(
        context_window_tokens=model_config_row.context_window_tokens,
        thinking_toggle_mode=model_config_row.thinking_toggle_mode or "none",
        thinking_toggle_support=model_config_row.thinking_toggle_support or "unknown",
        visual_budget_mode=model_config_row.visual_budget_mode or "none",
        visual_budget_support=model_config_row.visual_budget_support or "unknown",
        structured_generation_support=(
            # sgm_effective="prompt_only" forces ICL only — suppress
            # structured generation for the whole run regardless of
            # the model's actual capability (run-level control).
            "unsupported"
            if sgm_effective == "prompt_only"
            else (model_config_row.structured_generation_support or "unknown")
        ),
        max_images_per_request=resolve_max_images_per_request(
            model_config=model_config_row, nim_endpoint=endpoint
        ),
        default_icl_max_examples=model_config_row.default_icl_max_examples,
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

    return {
        "guidance_id": guidance_id,
        "guidance_description": guidance.description or "",
        "guidance_rules": guidance.rules or "",
        "guidance_fields": guidance_fields,
        "generation_order": generation_order,
        "derived_json_schema": derived_json_schema,
        "model_config_id": model_config_row.model_config_id,
        "model_name": model_config_row.model_name,
        "mc_input": mc_input,
        "endpoint_id": endpoint_id,
        "endpoint_base_url": endpoint.base_url,
        "endpoint_mode": endpoint.endpoint_mode,
        "endpoint_auth_mode": endpoint.auth_mode,
        "gen_preset_key": run.generation_preset_key or "precise",
        "thinking_on": (run.thinking_mode_effective or "on") == "on",
        "vb_preset_key": run.visual_budget_preset_key or "high_detail",
        "sgm_effective": sgm_effective,
        "project_dir": project_dir,
        "storage_refs": storage_refs,
    }
