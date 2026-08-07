# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pydantic schemas for NIM endpoint management and environment assessment.

Covers: NimEndpoint CRUD, transient connection test, EmbeddingDeploymentConfig,
and environment assessment response.
"""

from __future__ import annotations

from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, computed_field

from vlm_feedback_loop.schemas.local_nim import ActiveNimResidentResponse

# Shared Literal types for enum-like fields
EndpointMode = Literal["hosted", "self_hosted", "local_system_managed"]
AuthMode = Literal["bearer", "none"]
SourceKind = Literal["seeded_hosted", "user_configured", "auto_registered_local"]
EndpointUsagePolicy = Literal["evaluation_only", "operator_managed"]
ProbeKind = Literal["models", "embeddings"]
RecommendedMode = Literal["hosted", "local", "none"]
EmbeddingProvider = Literal[
    "hosted_nvclip", "self_hosted_nvclip", "local_nvclip", "none"
]


# ── NimEndpoint CRUD ────────────────────────────────────────────────────────


class NimEndpointCreate(BaseModel):
    """Request body for creating a NimEndpoint record."""

    model_config = ConfigDict(extra="forbid")

    display_name: str
    endpoint_mode: EndpointMode
    base_url: str
    api_format: str = "openai_compatible"
    auth_mode: AuthMode = "none"
    # Path defaults assume base_url already carries the version prefix (/v1)
    # per the NimEndpoint convention. See db/models/nim_endpoint.py and
    # project_service._seed_endpoint() for the matching seeded values.
    models_path: str = "/models"
    health_ready_path: str = "/health/ready"
    health_live_path: str | None = "/health/live"
    metrics_path: str | None = "/metrics"
    source_kind: SourceKind = "user_configured"


class NimEndpointUpdate(BaseModel):
    """Partial update for a NimEndpoint record."""

    model_config = ConfigDict(extra="forbid")

    display_name: str | None = None
    base_url: str | None = None
    auth_mode: AuthMode | None = None
    is_enabled: bool | None = None
    # Per-endpoint overrides for image-cap fields. Set to None to clear
    # the override and revert to the ModelConfig fallback.
    max_images_per_request: int | None = None
    image_cap_support: str | None = None


class NimEndpointResponse(BaseModel):
    """Full NimEndpoint record for API responses."""

    model_config = ConfigDict(from_attributes=True)

    endpoint_id: str
    project_id: str
    display_name: str
    endpoint_mode: str
    base_url: str
    api_format: str
    auth_mode: str
    models_path: str
    health_ready_path: str
    health_live_path: str | None
    metrics_path: str | None
    is_enabled: bool
    last_probe_at: str | None
    last_probe_status: str
    last_probe_error_ref: str | None
    source_kind: str
    local_nim_deployment_id: str | None
    # Per-endpoint overrides — null means fall back to ModelConfig
    # (gateway-correct value of e.g. 8 for hosted). Set to e.g. 32 on
    # auto-registered local NIM endpoints to lift the artificial cap.
    # See ``services/image_cap_resolver.py``.
    max_images_per_request: int | None = None
    image_cap_support: str | None = None
    created_at: str
    updated_at: str

    @computed_field
    @property
    def usage_policy(self) -> EndpointUsagePolicy:
        """Classify whether the Blueprint knows this endpoint is trial-only.

        NVIDIA's API Catalog host is governed by the API Trial Terms. Other
        endpoints are supplied or licensed by the operator, so
        the Blueprint deliberately makes no production-entitlement claim.
        """

        try:
            hostname = (urlsplit(self.base_url).hostname or "").rstrip(".").lower()
        except ValueError:
            hostname = ""
        if self.endpoint_mode == "hosted" and hostname == "integrate.api.nvidia.com":
            return "evaluation_only"
        return "operator_managed"


class NimEndpointListResponse(BaseModel):
    """List of NimEndpoint records."""

    items: list[NimEndpointResponse]


class SelfHostedTeacherConfigureRequest(BaseModel):
    """Apply one verified self-hosted endpoint to a cataloged Teacher."""

    model_config = ConfigDict(extra="forbid")

    base_url: str
    model_config_id: str


class SelfHostedTeacherConfigureResponse(BaseModel):
    """Durable result of configuring a self-hosted Teacher endpoint."""

    endpoint: NimEndpointResponse
    model_config_id: str
    model_name: str
    structured_generation_support: str
    thinking_toggle_support: str
    visual_budget_support: str


# ── Transient connection test ───────────────────────────────────────────────


class ConnectionTestRequest(BaseModel):
    """Request body for POST /v1/nim/test_connection.

    ``credential_transient`` is held in memory only for the probe request,
    then discarded.  It is NEVER persisted to any durable store.
    """

    model_config = ConfigDict(extra="forbid")

    base_url: str
    auth_mode: AuthMode = "none"
    credential_transient: str | None = None
    probe_kind: ProbeKind = "models"


class ConnectionTestResponse(BaseModel):
    """Result of a transient connection test."""

    success: bool
    models: list[str] | None = None
    error: str | None = None


class CredentialTestRequest(BaseModel):
    """Request body for POST /v1/nim/test_ngc_credential and
    /v1/nim/test_nvidia_credential.

    Two modes:

    * ``credential_transient`` is non-empty → probe THAT key. Held in
      request memory only and discarded once the function returns;
      never written to any durable store.
    * ``credential_transient`` is null/empty → probe the currently
      effective key resolved from the runtime secrets layer (the
      ``.env`` file plus any in-process overrides). Used by the setup
      gate to validate a key that a prior session left persisted, so an
      invalid one doesn't sail past the gate just because the SME didn't
      re-paste it.
    """

    model_config = ConfigDict(extra="forbid")

    credential_transient: str | None = None


# ── EmbeddingDeploymentConfig ───────────────────────────────────────────────


class EmbeddingDeploymentConfigUpdate(BaseModel):
    """Partial update for the deployment-scoped EmbeddingDeploymentConfig."""

    model_config = ConfigDict(extra="forbid")

    provider: EmbeddingProvider | None = None
    endpoint_url: str | None = None
    gpu_assignment: str | None = None


class SelfHostedEmbeddingConfigureRequest(BaseModel):
    """Live-verify and apply one external embedding NIM endpoint."""

    model_config = ConfigDict(extra="forbid")

    base_url: str


class EmbeddingDeploymentConfigResponse(BaseModel):
    """Full EmbeddingDeploymentConfig record."""

    model_config = ConfigDict(from_attributes=True)

    embedding_deployment_config_id: str
    provider: str
    model_name: str
    embedding_dim: int
    endpoint_url: str | None
    nim_container_image: str
    preferred_host_port: int
    gpu_memory_minimum_gb: int
    gpu_assignment: str | None
    created_at: str
    updated_at: str


# ── Environment assessment ──────────────────────────────────────────────────


class GpuInfoSchema(BaseModel):
    """Detected GPU from nvidia-smi."""

    name: str
    memory_total_gb: float
    compute_capability: float | None = None


class LocalDeployableModelSchema(BaseModel):
    """A seeded model that supports local NIM deployment."""

    model_name: str
    nim_container_image: str
    gpu_memory_minimum_gb: int
    compute_capability_minimum: float | None = None
    fits: bool


class MissingPrerequisiteSchema(BaseModel):
    """A missing prerequisite for local NIM deployment."""

    check: str
    install_hint: str


class EmbeddingDeploymentSummary(BaseModel):
    """Embedding NIM deployment metadata from EmbeddingDeploymentConfig.

    Default model is NeMo Retriever VL 1B v2; operators can override to
    NV-CLIP via the embedding-related Settings overrides.
    """

    model_name: str
    nim_container_image: str
    gpu_memory_minimum_gb: int
    fits: bool
    provider: str


class EnvironmentResponse(BaseModel):
    """Response for GET /v1/environment."""

    hosted_nim_available: bool
    local_deploy_available: bool
    docker_available: bool
    nvidia_toolkit_available: bool
    nvidia_api_key_configured: bool
    ngc_api_key_configured: bool
    gpus: list[GpuInfoSchema]
    local_deployable_models: list[LocalDeployableModelSchema]
    embedding_deployment: EmbeddingDeploymentSummary
    missing_prerequisites: list[MissingPrerequisiteSchema]
    recommended_teacher_mode: RecommendedMode
    recommended_embedding_mode: RecommendedMode
    # Deployment-config values the frontend needs verbatim (backend is the
    # single source of truth — the UI must not hardcode its own copies):
    # the NIM startup budget shown in benchmark stage labels, and the
    # benchmark concurrency sweep shown as Compare-table columns.
    nim_startup_timeout_s: int
    student_latency_test_concurrencies: list[int]
    # The seeded hosted default Teacher (settings.DEFAULT_TEACHER_MODEL). The
    # Confirm Defaults screen preselects this model when a project's stored
    # teacher id is null/orphaned — the UI reads it from here rather than
    # hardcoding a model name, so a config.yaml override or a default reseat
    # takes effect without a frontend change.
    default_teacher_model_name: str
    # Populated when a teacher-eligible locally-deployable model fits
    # the detected GPU (quality-first: Omni on supported cc>=9.0 / >=80 GB
    # GPUs, CR3-Nano at >=56 GB, CR2-2B at 36–55 GB). Null when no GPU or
    # no teacher fits. The
    # frontend uses these to render the NIM setup gate's local primary
    # card or hybrid peer card.
    recommended_local_teacher_model_name: str | None = None
    recommended_local_teacher_image: str | None = None
    recommended_local_teacher_gpu_memory_minimum_gb: int | None = None
    # Blueprint-managed NIMs currently claiming host GPUs. The FTUE uses this
    # to say when its recommended Teacher is already running and reusable
    # instead of presenting it as a new deployment.
    active_local_nim_residents: list[ActiveNimResidentResponse] = Field(
        default_factory=lambda: list[ActiveNimResidentResponse]()
    )
    # When False, the NIM Configuration UI hides
    # the "Save to ~/.vlm_feedback_loop/.env" checkbox and POST
    # /v1/secrets:set returns 403 for ``persist=true``. Production /
    # container deployments typically set this to False (the .env file
    # is managed externally).
    allow_secret_persist: bool
