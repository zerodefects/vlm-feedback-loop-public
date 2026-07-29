# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NimEndpoint record."""

from __future__ import annotations

from sqlalchemy import Boolean, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from vlm_feedback_loop.db.base import (
    ProjectBase,
    created_at_col,
    updated_at_col,
    uuid_pk,
)


class NimEndpoint(ProjectBase):
    __tablename__ = "nim_endpoints"

    endpoint_id: Mapped[str] = uuid_pk()
    project_id: Mapped[str] = mapped_column(String(36), nullable=False)
    display_name: Mapped[str] = mapped_column(String, nullable=False)
    endpoint_mode: Mapped[str] = mapped_column(
        String, nullable=False
    )  # hosted | self_hosted | local_system_managed
    base_url: Mapped[str] = mapped_column(String, nullable=False)
    api_format: Mapped[str] = mapped_column(
        String, nullable=False, default="openai_compatible"
    )
    auth_mode: Mapped[str] = mapped_column(
        String, nullable=False, default="none"
    )  # bearer (hosted, NVIDIA_API_KEY) | none (self-hosted / local, trusted network)
    # Paths are relative to base_url (which already includes /v1).
    # E.g. base_url="https://integrate.api.nvidia.com/v1" + models_path="/models"
    #   → "https://integrate.api.nvidia.com/v1/models"
    models_path: Mapped[str] = mapped_column(String, nullable=False, default="/models")
    health_ready_path: Mapped[str] = mapped_column(
        String, nullable=False, default="/health/ready"
    )
    health_live_path: Mapped[str | None] = mapped_column(
        String, nullable=True, default="/health/live"
    )
    metrics_path: Mapped[str | None] = mapped_column(
        String, nullable=True, default="/metrics"
    )
    is_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    last_probe_at: Mapped[str | None] = mapped_column(String(24), nullable=True)
    last_probe_status: Mapped[str] = mapped_column(
        String, nullable=False, default="unknown"
    )  # unknown | healthy | unhealthy | auth_failed | unreachable
    last_probe_error_ref: Mapped[str | None] = mapped_column(String, nullable=True)
    source_kind: Mapped[str] = mapped_column(
        String, nullable=False
    )  # seeded_hosted | user_configured | auto_registered_local
    local_nim_deployment_id: Mapped[str | None] = mapped_column(
        String(36), nullable=True
    )
    # Per-endpoint overrides for the image-content-parts cap. When null,
    # the runtime falls back to ``ModelConfig.max_images_per_request`` /
    # ``ModelConfig.image_cap_support``. The override exists because the
    # cap is a property of the **endpoint** (gateway clamp / NIM
    # container / model layer), not the model alone — hosted Mistral
    # caps at 8 (build.nvidia.com gateway), local cosmos-reason2 NIM
    # caps at 999 (vLLM container), and the same model can run on both
    # endpoints with different caps. See
    # ``services/image_cap_resolver.py`` for the resolution helper.
    max_images_per_request: Mapped[int | None] = mapped_column(Integer, nullable=True)
    image_cap_support: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[str] = created_at_col()
    updated_at: Mapped[str] = updated_at_col()
