# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deployment-scoped models (stored in workspace deployment.db)."""

from __future__ import annotations

from typing import Any

from sqlalchemy import Integer, String
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.orm import Mapped, mapped_column

from vlm_feedback_loop.db.base import (
    DeploymentBase,
    created_at_col,
    updated_at_col,
    uuid_pk,
)


class EmbeddingDeploymentConfig(DeploymentBase):
    """Embedding NIM service configuration — singleton per deployment.

    Default model is NeMo Retriever VL 1B v2 (`nvidia/llama-nemotron-embed-vl-1b-v2`).
    Operators can override to NV-CLIP via ``EMBEDDING_MODEL_ID`` /
    ``EMBEDDING_DIM`` / ``EMBEDDING_INPUT_TYPE`` overrides.

    Deployment-scoped (shared across all projects), not project-scoped.
    """

    __tablename__ = "embedding_deployment_configs"

    embedding_deployment_config_id: Mapped[str] = uuid_pk()
    provider: Mapped[str] = mapped_column(
        String, nullable=False, default="none"
    )  # hosted_nvclip | self_hosted_nvclip | local_nvclip | none
    model_name: Mapped[str] = mapped_column(String, nullable=False)
    embedding_dim: Mapped[int] = mapped_column(Integer, nullable=False)
    endpoint_url: Mapped[str | None] = mapped_column(String, nullable=True)
    nim_container_image: Mapped[str] = mapped_column(String, nullable=False)
    preferred_host_port: Mapped[int] = mapped_column(
        Integer, nullable=False, default=8001
    )
    gpu_memory_minimum_gb: Mapped[int] = mapped_column(
        Integer, nullable=False, default=16
    )
    gpu_assignment: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[str] = created_at_col()
    updated_at: Mapped[str] = updated_at_col()


class TAODeploymentConfig(DeploymentBase):
    """TAO workspace + bootstrap state — singleton per deployment.

    Deployment-scoped (shared across all projects). Persists workspace
    identity and bootstrap lifecycle; S3 credentials live only in .env —
    this record holds reference labels, not credential values.

    `bootstrap_status` ∈ {"not_bootstrapped", "in_progress", "bootstrapped",
    "failed"}. Initial state is "not_bootstrapped"; flips to "bootstrapped"
    after `vlm-feedback-loop tao-bootstrap` succeeds.
    """

    __tablename__ = "tao_deployment_configs"

    tao_deployment_config_id: Mapped[str] = uuid_pk()
    tao_workspace_id: Mapped[str | None] = mapped_column(String, nullable=True)
    tao_workspace_name: Mapped[str | None] = mapped_column(String, nullable=True)
    tao_workspace_cloud_type: Mapped[str | None] = mapped_column(
        String, nullable=True
    )  # seaweedfs | aws | azure | self_hosted | huggingface | lepton | slurm
    tao_workspace_bucket: Mapped[str | None] = mapped_column(String, nullable=True)
    tao_workspace_s3_endpoint_url_internal: Mapped[str | None] = mapped_column(
        String, nullable=True
    )
    tao_workspace_s3_endpoint_url_external: Mapped[str | None] = mapped_column(
        String, nullable=True
    )
    tao_workspace_s3_access_key_ref: Mapped[str | None] = mapped_column(
        String, nullable=True
    )
    tao_workspace_s3_secret_key_ref: Mapped[str | None] = mapped_column(
        String, nullable=True
    )
    bootstrap_status: Mapped[str] = mapped_column(
        String, nullable=False, default="not_bootstrapped"
    )
    bootstrap_last_run_at: Mapped[str | None] = mapped_column(String(24), nullable=True)
    bootstrap_error_ref: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[str] = created_at_col()
    updated_at: Mapped[str] = updated_at_col()


class TAOBaseExperimentProvisioningRun(DeploymentBase):
    """One tracked, deployment-scoped Student-base provisioning attempt.

    Base experiments belong to the shared TAO workspace, not to a project.
    ``project_id`` and the selected project-local ModelConfig ids retain the
    SME request that started the attempt; canonical model names let the worker
    patch the corresponding seeded rows across every project.

    ``status`` ∈ {"queued", "running", "succeeded", "failed"}. Provisioning
    is an in-process background task. Startup recovery marks an interrupted
    active row failed so the UI can offer a safe retry.
    """

    __tablename__ = "tao_base_experiment_provisioning_runs"

    provisioning_run_id: Mapped[str] = uuid_pk()
    project_id: Mapped[str] = mapped_column(String(36), nullable=False)
    requested_model_config_ids: Mapped[list[str]] = mapped_column(
        JSON, nullable=False, default=list
    )
    requested_model_names: Mapped[list[str]] = mapped_column(
        JSON, nullable=False, default=list
    )
    status: Mapped[str] = mapped_column(String, nullable=False, default="queued")
    registered: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    already_registered: Mapped[list[str]] = mapped_column(
        JSON, nullable=False, default=list
    )
    failures: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON, nullable=False, default=list
    )
    error_ref: Mapped[str | None] = mapped_column(String, nullable=True)
    started_at: Mapped[str | None] = mapped_column(String(24), nullable=True)
    completed_at: Mapped[str | None] = mapped_column(String(24), nullable=True)
    created_at: Mapped[str] = created_at_col()
    updated_at: Mapped[str] = updated_at_col()
