# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NimEndpoint CRUD service with auto-probe on create/update.

NimEndpoint records persist endpoint configurations (base URL, auth mode,
health state) — never API key values.  Probe credentials come from
deployment-level Settings.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from vlm_feedback_loop.config import Settings
from vlm_feedback_loop.db.base import generate_uuid4, utc_now
from vlm_feedback_loop.db.models.nim_endpoint import NimEndpoint
from vlm_feedback_loop.services import nim_client
from vlm_feedback_loop.services.project_service import get_project_engine
from vlm_feedback_loop.services.runtime_secrets import get_effective_secret

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
