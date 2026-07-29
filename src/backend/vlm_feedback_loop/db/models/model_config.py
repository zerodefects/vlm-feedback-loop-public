# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ModelConfig record."""

from __future__ import annotations

from typing import Any

from sqlalchemy import Boolean, Integer, String
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.orm import Mapped, mapped_column

from vlm_feedback_loop.db.base import ProjectBase, created_at_col, uuid_pk


class ModelConfig(ProjectBase):
    __tablename__ = "model_configs"

    model_config_id: Mapped[str] = uuid_pk()
    project_id: Mapped[str] = mapped_column(String(36), nullable=False)
    endpoint_id: Mapped[str] = mapped_column(String(36), nullable=False)
    model_name: Mapped[str] = mapped_column(String, nullable=False)
    context_window_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    eligible_roles: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    supports_image_input: Mapped[bool] = mapped_column(Boolean, nullable=False)
    structured_generation_support: Mapped[str] = mapped_column(
        String, nullable=False, default="unknown"
    )
    thinking_toggle_mode: Mapped[str] = mapped_column(
        String, nullable=False, default="none"
    )
    thinking_toggle_support: Mapped[str] = mapped_column(
        String, nullable=False, default="unknown"
    )
    visual_budget_mode: Mapped[str] = mapped_column(
        String, nullable=False, default="none"
    )
    visual_budget_support: Mapped[str] = mapped_column(
        String, nullable=False, default="unknown"
    )
    # Per-model cap on the number of image content parts a single
    # chat/completions request may carry (query image + ICL example images).
    # Used by `prompt_service.invoke_teacher` to decide whether inline
    # ICL image injection fits the model's budget; when not, ICL
    # falls back to text-only labels with a one-line warning. Values are
    # live-probed via ``scripts/probe_hosted_image_caps.py`` and seeded
    # into the catalog; unknown future models are seeded conservatively
    # at 5 and refined by the runtime ``image_cap_support`` probe.
    max_images_per_request: Mapped[int] = mapped_column(
        Integer, nullable=False, default=5
    )
    image_cap_support: Mapped[str] = mapped_column(
        String, nullable=False, default="unknown"
    )
    # Per-model default cap on ICL example count ("depth"). The July 2026
    # cross-model depth studies established that the useful ICL depth is a
    # property of the MODEL FAMILY, not the task: Nemotron VL overshoots past
    # ~2 shots on every task measured, Cosmos CR3 is monotonic-up through ~8,
    # CR2-2B collapses at 16, CR2-8B keeps gaining at 16, and ceiling-class
    # teachers (Mistral Large) are depth-insensitive so 2 is the cost floor.
    # Consumed by `prompt_service.invoke_teacher` as the selection cap when
    # no explicit `icl_max_examples` override is provided (per-run API
    # override or a non-null `ICL_MAX_EXAMPLES` setting wins); adaptive-K
    # still trims per query within the cap, and the image budget still
    # prunes after selection. NULL = no model default (unmeasured model);
    # behavior is then identical to before this column existed.
    default_icl_max_examples: Mapped[int | None] = mapped_column(Integer, nullable=True)
    model_quantization: Mapped[str | None] = mapped_column(String, nullable=True)
    nim_model_profile: Mapped[str | None] = mapped_column(String, nullable=True)
    nim_profile_metadata: Mapped[dict[str, Any] | None] = mapped_column(
        JSON, nullable=True
    )
    local_deploy_metadata: Mapped[dict[str, Any] | None] = mapped_column(
        JSON, nullable=True
    )
    # Whether this model is invocable on the build.nvidia.com hosted NIM
    # endpoint with a standard NVIDIA_API_KEY (no NVCF function subscription
    # needed). Mistral / Qwen / Nemotron are True; Cosmos Reason2 8B/2B are
    # False because hosted lists them in GET /v1/models but inference is
    # NVCF-gated and 404s for accounts without that subscription. Only
    # consulted when the bound endpoint's mode is "hosted"; self-hosted and
    # local endpoints ignore this flag.
    hosted_compatible: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True
    )
    # TAO base-experiment lineage. Populated by first-use provisioning or the
    # operator CLIs for `student_base`-eligible entries.
    # pull_status ∈ {unknown, starting, in_progress, pulling, pull_complete,
    #                invalid_pull, failed}.
    tao_base_experiment_id: Mapped[str | None] = mapped_column(String, nullable=True)
    tao_base_experiment_pull_status: Mapped[str | None] = mapped_column(
        String, nullable=True
    )
    created_at: Mapped[str] = created_at_col()
