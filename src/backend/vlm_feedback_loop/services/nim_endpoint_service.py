# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NimEndpoint CRUD service with auto-probe on create/update.

NimEndpoint records persist endpoint configurations (base URL, auth mode,
health state) — never API key values.  Probe credentials come from
deployment-level Settings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from sqlalchemy import select
from sqlalchemy.orm import Session

from vlm_feedback_loop.config import Settings
from vlm_feedback_loop.db.base import generate_uuid4, utc_now
from vlm_feedback_loop.db.models.model_config import ModelConfig
from vlm_feedback_loop.db.models.nim_endpoint import NimEndpoint
from vlm_feedback_loop.db.models.project import Project
from vlm_feedback_loop.services import nim_client
from vlm_feedback_loop.services.project_service import get_project_engine
from vlm_feedback_loop.services.runtime_secrets import get_effective_secret


@dataclass
class NimEndpointConfigurationError(Exception):
    """Safe, structured configuration failure for router translation."""

    status_code: int
    code: str
    message: str


def normalize_self_hosted_base_url(raw: str) -> str:
    """Validate and normalize a credential-free HTTP(S) NIM API root."""
    value = raw.strip().rstrip("/")
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise NimEndpointConfigurationError(
            400,
            "invalid_base_url",
            "Base URL must be an absolute http:// or https:// URL including /v1.",
        )
    if parsed.username is not None or parsed.password is not None:
        raise NimEndpointConfigurationError(
            400,
            "embedded_credentials_forbidden",
            "Do not include credentials in a self-hosted NIM URL.",
        )
    if parsed.query or parsed.fragment:
        raise NimEndpointConfigurationError(
            400,
            "invalid_base_url",
            "Base URL must not include a query string or fragment.",
        )
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", ""))


# ── Probe helper ────────────────────────────────────────────────────────────


def resolve_probe_credential(endpoint: NimEndpoint, settings: Settings) -> str | None:
    """Resolve the credential for probing a NimEndpoint.

    Hosted endpoints use the deployment-level NVIDIA_API_KEY.
    Self-hosted and local endpoints use no auth (trusted network).
    """
    if endpoint.auth_mode == "bearer":
        # The runtime secret override takes precedence over the
        # .env-loaded value so probes run with the UI-applied key.
        return get_effective_secret("NVIDIA_API_KEY", settings)
    return None


async def _run_endpoint_probe(
    endpoint: NimEndpoint,
    settings: Settings,
) -> tuple[str, str | None]:
    """Probe a NimEndpoint and return (status, error_ref).

    Maps: success → "healthy", 401/403 → "auth_failed",
    connection error → "unreachable", other → "unhealthy".
    """
    try:
        credential = resolve_probe_credential(endpoint, settings)
        auth_headers = nim_client.build_auth_headers(
            auth_mode=endpoint.auth_mode,
            credential=credential,
        )
    except ValueError:
        # Missing credential for bearer (hosted) — can't probe
        return ("unknown", "Cannot probe: missing credential for auth_mode")

    result = await nim_client.list_models(
        base_url=endpoint.base_url,
        auth_headers=auth_headers,
        deadline_s=settings.HTTP_DEADLINE_INTERACTIVE_S,
        models_path=endpoint.models_path,
        max_retries=1,  # single attempt for probes
    )

    if result.success:
        return ("healthy", None)

    if result.status_code in (401, 403):
        return ("auth_failed", result.error)

    # No status_code means the request never got an HTTP response
    # (timeout or connection failure) — the endpoint is unreachable.
    if result.status_code is None:
        return ("unreachable", result.error)

    return ("unhealthy", result.error)


# ── CRUD ────────────────────────────────────────────────────────────────────


async def create_nim_endpoint(
    project_id: str,
    data: dict[str, Any],
    workspace_root: str,
    settings: Settings,
) -> NimEndpoint | None:
    """Create a NimEndpoint record and auto-run a connection probe."""
    engine = get_project_engine(project_id, workspace_root)
    if engine is None:
        return None

    endpoint = NimEndpoint(
        endpoint_id=generate_uuid4(),
        project_id=project_id,
        display_name=data["display_name"],
        endpoint_mode=data["endpoint_mode"],
        base_url=data["base_url"],
        api_format=data.get("api_format", "openai_compatible"),
        auth_mode=data.get("auth_mode", "none"),
        models_path=data.get("models_path", "/models"),
        health_ready_path=data.get("health_ready_path", "/health/ready"),
        health_live_path=data.get("health_live_path", "/health/live"),
        metrics_path=data.get("metrics_path", "/metrics"),
        source_kind=data.get("source_kind", "user_configured"),
    )

    probe_status, probe_error = await _run_endpoint_probe(endpoint, settings)
    endpoint.last_probe_status = probe_status
    endpoint.last_probe_error_ref = probe_error
    endpoint.last_probe_at = utc_now()

    with Session(engine) as session:
        session.add(endpoint)
        session.commit()
        session.refresh(endpoint)
        # Detach from session for return
        session.expunge(endpoint)

    return endpoint


def list_nim_endpoints(
    project_id: str,
    workspace_root: str,
) -> list[NimEndpoint]:
    """List all NimEndpoint records for a project."""
    engine = get_project_engine(project_id, workspace_root)
    if engine is None:
        return []

    with Session(engine) as session:
        stmt = (
            select(NimEndpoint)
            .where(NimEndpoint.project_id == project_id)
            .order_by(NimEndpoint.created_at)
        )
        endpoints = list(session.execute(stmt).scalars().all())
        for ep in endpoints:
            session.expunge(ep)
        return endpoints


def get_nim_endpoint(
    project_id: str,
    endpoint_id: str,
    workspace_root: str,
) -> NimEndpoint | None:
    """Retrieve a single NimEndpoint by ID."""
    engine = get_project_engine(project_id, workspace_root)
    if engine is None:
        return None

    with Session(engine) as session:
        stmt = select(NimEndpoint).where(
            NimEndpoint.project_id == project_id,
            NimEndpoint.endpoint_id == endpoint_id,
        )
        endpoint = session.execute(stmt).scalar_one_or_none()
        if endpoint is not None:
            session.expunge(endpoint)
        return endpoint


async def update_nim_endpoint(
    project_id: str,
    endpoint_id: str,
    updates: dict[str, Any],
    workspace_root: str,
    settings: Settings,
) -> NimEndpoint | None:
    """Update a NimEndpoint record. Re-probes if base_url or auth_mode changed."""
    engine = get_project_engine(project_id, workspace_root)
    if engine is None:
        return None

    with Session(engine) as session:
        stmt = select(NimEndpoint).where(
            NimEndpoint.project_id == project_id,
            NimEndpoint.endpoint_id == endpoint_id,
        )
        endpoint = session.execute(stmt).scalar_one_or_none()
        if endpoint is None:
            return None

        # Track whether probe-triggering fields changed
        needs_reprobe = False
        for key, value in updates.items():
            if key in ("base_url", "auth_mode") and getattr(endpoint, key) != value:
                needs_reprobe = True
            setattr(endpoint, key, value)

        if needs_reprobe:
            probe_status, probe_error = await _run_endpoint_probe(endpoint, settings)
            endpoint.last_probe_status = probe_status
            endpoint.last_probe_error_ref = probe_error
            endpoint.last_probe_at = utc_now()

        session.commit()
        session.refresh(endpoint)
        session.expunge(endpoint)

    return endpoint


async def configure_self_hosted_teacher(
    project_id: str,
    model_config_id: str,
    base_url: str,
    workspace_root: str,
    settings: Settings,
) -> tuple[NimEndpoint, ModelConfig]:
    """Probe, persist, bind, and capability-check a self-hosted Teacher.

    The network probe completes before the transaction. The transaction then
    idempotently reuses an endpoint with the same URL, binds the exact
    cataloged vision Teacher, and selects it on the Project. Capability probes
    run afterward against the newly bound endpoint, so support learned from a
    previous hosted/local runtime is never carried across silently.
    """
    normalized_url = normalize_self_hosted_base_url(base_url)
    engine = get_project_engine(project_id, workspace_root)
    if engine is None:
        raise NimEndpointConfigurationError(
            404, "project_not_found", "Project not found"
        )

    # Validate the requested identity and active-use guard without holding a
    # transaction across the outbound /models probe.
    from vlm_feedback_loop.services import model_config_service

    with Session(engine) as session:
        project = session.get(Project, project_id)
        model_config = session.get(ModelConfig, model_config_id)
        if project is None:
            raise NimEndpointConfigurationError(
                404, "project_not_found", "Project not found"
            )
        if model_config is None or model_config.project_id != project_id:
            raise NimEndpointConfigurationError(
                404, "model_config_not_found", "Teacher model configuration not found"
            )
        if (
            "teacher" not in (model_config.eligible_roles or [])
            or not model_config.supports_image_input
        ):
            raise NimEndpointConfigurationError(
                400,
                "not_a_vision_teacher",
                "The selected model is not a vision-capable Teacher.",
            )
        if model_config_service.is_model_in_active_use(
            session, project_id, model_config_id
        ):
            raise NimEndpointConfigurationError(
                409,
                "model_in_active_use",
                "Cannot change this Teacher endpoint while it is used by an active run.",
            )
        model_name = model_config.model_name

    probe = await nim_client.list_models(
        base_url=normalized_url,
        auth_headers={},
        deadline_s=float(settings.HTTP_DEADLINE_INTERACTIVE_S),
        max_retries=1,
    )
    if not probe.success:
        raise NimEndpointConfigurationError(
            400,
            "endpoint_probe_failed",
            probe.error or "Could not connect to the self-hosted NIM endpoint.",
        )
    served_models = {name for name in (probe.models or []) if name}
    if model_name not in served_models:
        raise NimEndpointConfigurationError(
            400,
            "model_not_served",
            f"The endpoint does not report the selected Teacher model {model_name}.",
        )

    now = utc_now()
    with Session(engine) as session:
        project = session.get(Project, project_id)
        model_config = session.get(ModelConfig, model_config_id)
        if (
            project is None
            or model_config is None
            or model_config.project_id != project_id
            or model_config.model_name != model_name
            or "teacher" not in (model_config.eligible_roles or [])
            or not model_config.supports_image_input
        ):
            raise NimEndpointConfigurationError(
                409,
                "configuration_changed",
                "The project changed while the endpoint was being tested. Try again.",
            )
        if model_config_service.is_model_in_active_use(
            session, project_id, model_config_id
        ):
            raise NimEndpointConfigurationError(
                409,
                "model_in_active_use",
                "Cannot change this Teacher endpoint while it is used by an active run.",
            )

        endpoint = (
            session.execute(
                select(NimEndpoint)
                .where(
                    NimEndpoint.project_id == project_id,
                    NimEndpoint.endpoint_mode == "self_hosted",
                    NimEndpoint.base_url == normalized_url,
                )
                .order_by(NimEndpoint.created_at, NimEndpoint.endpoint_id)
            )
            .scalars()
            .first()
        )
        if endpoint is None:
            endpoint = NimEndpoint(
                endpoint_id=generate_uuid4(),
                project_id=project_id,
                display_name=f"Self-hosted Teacher ({normalized_url})",
                endpoint_mode="self_hosted",
                base_url=normalized_url,
                api_format="openai_compatible",
                auth_mode="none",
                models_path="/models",
                health_ready_path="/health/ready",
                health_live_path="/health/live",
                metrics_path="/metrics",
                source_kind="user_configured",
            )
            session.add(endpoint)
        endpoint.display_name = f"Self-hosted Teacher ({normalized_url})"
        endpoint.api_format = "openai_compatible"
        endpoint.auth_mode = "none"
        endpoint.models_path = "/models"
        endpoint.health_ready_path = "/health/ready"
        endpoint.health_live_path = "/health/live"
        endpoint.metrics_path = "/metrics"
        endpoint.is_enabled = True
        endpoint.last_probe_at = now
        endpoint.last_probe_status = "healthy"
        endpoint.last_probe_error_ref = None
        endpoint.source_kind = "user_configured"
        endpoint.local_nim_deployment_id = None

        model_config.endpoint_id = endpoint.endpoint_id
        model_config.structured_generation_support = "unknown"
        model_config.thinking_toggle_support = "unknown"
        model_config.visual_budget_support = "unknown"
        model_config.image_cap_support = "unknown"
        project.teacher_model_config_id = model_config_id
        session.commit()
        endpoint_id = endpoint.endpoint_id

    probed = await model_config_service.reprobe_model_config(
        project_id=project_id,
        model_config_id=model_config_id,
        workspace_root=workspace_root,
        settings=settings,
    )
    if probed is None or isinstance(probed, str):
        raise NimEndpointConfigurationError(
            409,
            "capability_probe_interrupted",
            "The endpoint was saved, but its capability probe could not complete. Re-probe it before use.",
        )
    endpoint = get_nim_endpoint(project_id, endpoint_id, workspace_root)
    if endpoint is None:
        raise NimEndpointConfigurationError(
            500,
            "endpoint_missing_after_save",
            "The endpoint was saved but could not be reloaded.",
        )
    return endpoint, probed
