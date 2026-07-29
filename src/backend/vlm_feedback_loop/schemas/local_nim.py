# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pydantic schemas for local NIM deployment."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

# ── Request schemas ───────────────────────────────────────────────────────────


class LocalNimDeployRequest(BaseModel):
    """Request body for deploying a local NIM container."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["teacher", "embedding"]
    model_config_id: str | None = None  # required for teacher; ignored for embedding
    nim_container_image: str | None = None  # override; defaults from catalog/config
    gpu_assignment: str | None = None  # e.g., "device=0"; None = auto-place
    preferred_port: int | None = None  # override; defaults from settings
    # One-NIM-per-GPU invariant: when True and the target GPU has an
    # active resident NIM, the orchestrator stops the resident before
    # starting this deployment (the *replace* semantics). When False
    # (default) and the target GPU is occupied, the deploy returns
    # ``409 gpu_occupied``. The FTUE keeps its client-side
    # skip-embedding-on-single-GPU branch as defense-in-depth; the
    # Student NIM lifecycle passes ``replace_resident=true``
    # automatically.
    replace_resident: bool = False
    # Teacher-only: select this ModelConfig after the new or reused NIM is
    # verified healthy. The activation intent is persisted on queued
    # deployments so a cold start can finish safely after a backend restart.
    activate_on_success: bool = False


class LocalNimPreflightRequest(BaseModel):
    """Request body for running a preflight check only (dry run)."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["teacher", "embedding"]
    model_config_id: str | None = None
    nim_container_image: str | None = None
    gpu_assignment: str | None = None


# ── Response schemas ──────────────────────────────────────────────────────────


class PreflightCheckSchema(BaseModel):
    """A single preflight check result."""

    check_name: str
    passed: bool
    diagnostic: str


class PreflightResponse(BaseModel):
    """Result of running all preflight checks."""

    all_passed: bool
    checks: list[PreflightCheckSchema]
    docker_run_command: str | None = None
    resolved_port: int | None = None
    gpu_assignment: str | None = None


class LocalNimDeploymentResponse(BaseModel):
    """Response representing a LocalNimDeployment record."""

    model_config = ConfigDict(from_attributes=True)

    local_nim_deployment_id: str
    project_id: str
    model_config_id: str
    role: str
    nim_container_image: str
    container_name: str
    container_id: str | None
    host_port: int
    endpoint_url: str
    gpu_assignment: str
    status: str
    status_reason: str | None
    activate_on_success: bool = False
    deployed_at: str | None
    stopped_at: str | None
    created_at: str
    # One-NIM-per-GPU displacement audit.
    displaced_by_deployment_id: str | None = None
    displaced_at: str | None = None
    # Whether the project's ACTIVE config for this deployment's role
    # still references this deployment's model config. False on a teacher
    # deploy whose Teacher the SME has since switched away from — the UI
    # suppresses the stale failure banner on that signal. Computed at
    # response time (matches_active_role_config); always True for
    # non-teacher roles.
    matches_active_role_config: bool = True


class LocalNimDeploymentListResponse(BaseModel):
    """List response for local NIM deployments."""

    items: list[LocalNimDeploymentResponse]


class ActiveNimResidentResponse(BaseModel):
    """Non-secret summary of a Blueprint-managed host GPU resident."""

    project_id: str
    project_name: str
    local_nim_deployment_id: str
    role: str
    model_name: str | None
    nim_container_image: str
    gpu_assignment: str
    status: str


class LocalNimDeployResponse(BaseModel):
    """Response from deploy endpoint — includes deployment record and preflight."""

    # Null when disposition="reused": the resident deployment belongs to the
    # owning project and the caller must not poll or stop it as if it were new.
    deployment: LocalNimDeploymentResponse | None
    preflight: PreflightResponse
    disposition: Literal["queued", "reused"] = "queued"
    resident: ActiveNimResidentResponse | None = None
