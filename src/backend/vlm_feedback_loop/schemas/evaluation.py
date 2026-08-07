# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pydantic schemas for evaluation run endpoints and trigger status."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# ── Evaluation Run Create ───────────────────────────────────────────────────


class EvaluationRunCreateRequest(BaseModel):
    """Request body for ``POST .../evaluation_runs``."""

    model_config = ConfigDict(extra="forbid")

    icl_mode: Literal["enabled", "disabled"] = "enabled"
    structured_generation_mode: Literal["auto", "prompt_only"] | None = None
    # Diagnostic per-run override for the deployment's ICL_MAX_EXAMPLES
    # setting. The effective value is persisted in runtime_config_snapshot so
    # delayed execution cannot observe a later process-config change.
    icl_max_examples: int | None = Field(default=None, ge=1)
    # Diagnostic per-run override that restricts the ICL candidate POOL (not
    # the injected K) to the first ``icl_candidate_limit`` Edits in stable
    # temporal (loop) order — i.e. the first N Edits the SME verified. This
    # simulates pool GROWTH: at limit=p, selection (relevance top-K etc.) runs
    # over only the oldest p candidates. ``icl_max_examples`` then caps how many
    # of those p are injected. When None (default), the full candidate pool is
    # used. Applied UPSTREAM of selection in
    # _invoke_for_evaluation (first-N by ``Label.labeled_at`` ascending).
    # Persisted in runtime_config_snapshot with the other semantic controls.
    icl_candidate_limit: int | None = Field(default=None, ge=1)
    # Diagnostic per-run override for the provider-aware eval concurrency
    # (EVAL_CONCURRENCY_HOSTED / EVAL_CONCURRENCY_SELF_HOSTED). The provider-
    # aware default already picks 1 for hosted endpoints (under
    # the shared per-account RPM cap) and 8 for self-hosted (saturate GPU);
    # this override is for explicit diagnostic control. Runtime-only; not
    # persisted.
    eval_concurrency: int | None = Field(default=None, ge=1)
    # Diagnostic per-run knobs for adaptive per-query ICL depth (similarity-gap
    # stopping). After relevance ranking, neighbor i (i>=2) is kept iff it is
    # close enough to the query: its CLIP cosine sim_i satisfies BOTH (when
    # set) sim_i >= sim_1 - icl_sim_gap AND sim_i >= icl_abs_threshold; stop
    # at the first failure, always keep >=1, then cap at icl_max_examples.
    # Large effective K when the pool has many close same-class neighbors,
    # small when it doesn't. None/None (default) = the deployment defaults
    # (settings ICL_SIM_GAP / ICL_ABS_THRESHOLD). The effective controls are
    # persisted in runtime_config_snapshot. The effective per-query K is
    # recorded as ``icl_images_attached_count`` on each OperationRecord, so the
    # sweep harness can aggregate avg-K from there.
    icl_sim_gap: float | None = Field(default=None, ge=0.0)
    icl_abs_threshold: float | None = Field(default=None, ge=-1.0, le=1.0)
    # Diagnostic per-run override for the generation preset (Output Stability:
    # precise/explore -> temperature/top_p). When None, the run inherits the
    # project's labeling_generation_preset_key. Validated against
    # settings.LABELING_PRESETS in start_evaluation_run (the preset set is
    # operator-configurable, so this is a str, not a Literal). Unlike the other
    # diagnostic controls, the key also has a dedicated RunRecord column so
    # the audit trail and configuration-change trigger expose it directly.
    generation_preset_key: str | None = None
    # Diagnostic per-run override for the Thinking toggle. When None, the
    # run inherits the project default (thinking_default_on). Like
    # generation_preset_key, this also has a dedicated
    # RunRecord.thinking_mode_effective column so the audit trail and the
    # config-change trigger expose it directly. Models whose
    # thinking_toggle_mode is "none"/"always_on_reasoning" ignore it at resolve
    # time (prompt_service.resolve_thinking_fields no-ops) — safe to pass.
    thinking_on: bool | None = None
    # Diagnostic per-run override for the Visual Budget preset. When None,
    # inherits project.visual_budget_preset_key. Validated against
    # settings.VISUAL_BUDGET_PRESETS in start_evaluation_run (operator-
    # configurable, so str not Literal). Snapshotted onto
    # RunRecord.visual_budget_preset_key (audit + config-change trigger).
    # Models without visual-budget support no-op at resolve time.
    visual_budget_preset_key: str | None = None


class EvaluationRunCreateResponse(BaseModel):
    """Response from ``POST .../evaluation_runs``."""

    run_id: str
    run_type: str
    status: str
    pool_version: int
    guidance_id: str | None
    model_config_id: str | None
    generation_preset_key: str | None
    thinking_mode_effective: str | None
    visual_budget_preset_key: str | None
    structured_generation_mode_effective: str | None
    evaluation_source: str | None
    icl_mode: str | None
    created_at: str
    superseded_run_id: str | None = None


# ── Evaluation Run Detail ───────────────────────────────────────────────────


class EvaluationRunResponse(BaseModel):
    """Full evaluation run detail for ``GET .../evaluation_runs/{run_id}``."""

    run_id: str
    run_type: str
    status: str
    status_reason: str | None = None

    # Pool
    pool_version_id: str | None = None

    # Config snapshot
    guidance_id: str | None = None
    model_config_id: str | None = None
    icl_mode: str | None = None
    evaluation_source: str | None = None
    generation_preset_key: str | None = None
    thinking_mode_effective: str | None = None
    visual_budget_preset_key: str | None = None
    structured_generation_mode_effective: str | None = None
    inference_contract: dict[str, Any] | None = None
    icl_eligible_count_at_start: int | None = None
    icl_eligible_count_at_completion: int | None = None

    # Provenance: set on Student evaluation runs (quality/serving), null
    # on Teacher runs — the Compare screen's Teacher-baseline
    # discriminator.
    student_model_config_id: str | None = None

    # Progress (retained on terminal records so the frozen total remains visible)
    progress: dict[str, int] | None = None  # {processed, total}

    # Metrics
    metrics: dict[str, Any] | None = None

    # Returning vs New
    previous_pool_version: int | None = None
    returning_example_keys: list[str] | None = None
    new_example_keys: list[str] | None = None
    previous_overall_exact_match: float | None = None
    coverage_gaps: list[dict[str, Any]] | None = None

    # Timestamps
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None


# ── Evaluation Run List ─────────────────────────────────────────────────────


class EvaluationRunListResponse(BaseModel):
    """Response from ``GET .../evaluation_runs``."""

    items: list[EvaluationRunResponse]
    next_cursor: str | None = None


# ── Cancel ──────────────────────────────────────────────────────────────────


class EvaluationRunCancelResponse(BaseModel):
    """Response from ``POST .../evaluation_runs/{run_id}:cancel``."""

    run_id: str
    status: str
    cancel_requested_at: str


# ── Trigger Status ──────────────────────────────────────────────────────────


class TriggerInfo(BaseModel):
    """Status of a single evaluation trigger."""

    is_active: bool
    dismissed: bool
    message: str
    context: dict[str, Any] | None = None


class TriggerStatusResponse(BaseModel):
    """Response from ``GET .../evaluation_trigger_status``."""

    auto_evaluate_enabled: bool
    first_pool_threshold: TriggerInfo
    configuration_change: TriggerInfo
    icl_growth: TriggerInfo
    updated_at: str


class TriggerDismissRequest(BaseModel):
    """Request body for ``POST .../evaluation_trigger_status:dismiss``."""

    model_config = ConfigDict(extra="forbid")

    trigger_type: Literal["first_pool_threshold", "configuration_change", "icl_growth"]


class TriggerDismissResponse(BaseModel):
    """Response from trigger dismiss."""

    trigger_type: str
    dismissed: bool


# ── Scale-Up Readiness Gate ───────────────────────────────


class GateCriterion(BaseModel):
    """Status of a single gate criterion."""

    criterion_name: str
    passed: bool
    current_value: float | int
    threshold: float | int
    message: str
    details: dict[str, Any] | None = None


class ScaleUpGateResponse(BaseModel):
    """Response from ``GET .../scaleup_gate``."""

    gate_status: Literal["not_ready", "ready"]
    criteria: list[GateCriterion]
    evaluated_at: str
