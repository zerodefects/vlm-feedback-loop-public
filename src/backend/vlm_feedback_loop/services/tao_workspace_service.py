# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TAO workspace + base-experiment orchestration.

This service is the Blueprint's client to the TAO FTMS workspace APIs:

- `create_or_get_workspace(...)` — adopts the workspace referenced by
  the persisted ``tao_workspace_id``, else adopts an existing FTMS
  workspace by name, else creates one; persists the resulting identity
  on the deployment-scoped ``TAODeploymentConfig`` singleton.
- `get_workspace(...)` — read-only variant used by
  ``training_preflight_service`` to verify the workspace is still
  reachable.
- `find_base_experiment_by_arch(...)` — discovers the UUID of an
  already-indexed base experiment. Stock FTMS exposes no client-driven
  pull endpoint (live OpenAPI probes, 6.25.11 and 6.26.3); base
  experiments enter the workspace via the self-service airgapped load
  (``tao_base_experiment_provisioning_service``) or admin-side pulls.

All TAO calls flow through ``resilient_request`` with ``_transport``
injectable for unit tests (no live HTTP required).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import httpx
from sqlalchemy.orm import Session

from vlm_feedback_loop.config import Settings
from vlm_feedback_loop.db.deployment_models import TAODeploymentConfig
from vlm_feedback_loop.db.engine import init_deployment_db
from vlm_feedback_loop.services.http_client import resilient_request
from vlm_feedback_loop.services.logging_config import redact_exact_secrets

# ``tao_auth_headers`` moved to ``tao_auth``; the redundant alias marks it
# as deliberately re-exported here so existing importers
# (``tao_base_experiment_provisioning_service``, tests) keep working.
from vlm_feedback_loop.services.tao_auth import (
    tao_auth_headers as tao_auth_headers,
)
from vlm_feedback_loop.services.tao_auth import tao_base_url

logger = logging.getLogger("vlm_feedback_loop.services.tao_workspace")


# ── Result types ─────────────────────────────────────────────────────────────


@dataclass
class WorkspaceResult:
    """Outcome of a workspace-service call.

    ``success=True``: ``workspace_id`` + ``workspace_detail`` populated.
    ``success=False``: ``error`` contains a plain-language message.
    ``already_provisioned``: True when idempotent re-call found an existing
    workspace and skipped the POST.
    """

    success: bool
    workspace_id: str | None = None
    workspace_detail: dict[str, Any] | None = None
    already_provisioned: bool = False
    error: str | None = None
    status_code: int | None = None


# ── TAO endpoint helpers ─────────────────────────────────────────────────────


def _require_tao_settings(settings: Settings) -> str | None:
    """Return the first missing setting name, or None if all present."""
    for name in ("TAO_API_BASE_URL", "TAO_API_KEY", "TAO_ORG_NAME"):
        if not getattr(settings, name):
            return name
    return None


def _workspaces_url(settings: Settings, workspace_id: str | None = None) -> str:
    root = f"{tao_base_url(settings)}/orgs/{settings.TAO_ORG_NAME}/workspaces"
    return f"{root}/{workspace_id}" if workspace_id else root


def _build_create_workspace_body(
    *,
    name: str,
    cloud_type: str,
    bucket: str,
    endpoint_url_internal: str,
    access_key: str,
    secret_key: str,
) -> dict[str, Any]:
    """Build the POST body for TAO workspace creation.

    FTMS 6.26.3 discriminates ``cloud_specific_details`` using a second
    ``cloud_type`` field inside that object. The workspace bucket field is
    named ``cloud_bucket_name`` (not ``cloud_bucket``), and ``endpoint_url``
    is the URL TAO's own containers use. The Blueprint-facing external URL
    is persisted separately and must never be sent in its place.
    """
    cloud_details: dict[str, Any] = {
        "cloud_type": cloud_type,
        "cloud_bucket_name": bucket,
        "endpoint_url": endpoint_url_internal,
        "access_key": access_key,
        "secret_key": secret_key,
    }
    if cloud_type == "seaweedfs":
        # SeaweedFS implements the S3 API without a physical AWS region,
        # but FTMS uses the conventional S3-compatible region value.
        cloud_details["cloud_region"] = "us-east-1"

    return {
        "name": name,
        "cloud_type": cloud_type,
        "cloud_specific_details": cloud_details,
    }


def _redact_workspace_credentials(
    text: str,
    *,
    access_key: str,
    secret_key: str,
) -> str:
    """Remove S3 credentials that an FTMS diagnostic may echo."""
    return redact_exact_secrets(text, (access_key, secret_key))


# ── Workspace: create or get ────────────────────────────────────────────────


def _persist_workspace_identity(
    workspace_root: str | Path,
    *,
    workspace_id: str,
    workspace_name: str,
    cloud_type: str,
    bucket: str,
    endpoint_url_internal: str,
    endpoint_url_external: str,
) -> None:
    """Write the workspace identity onto the TAODeploymentConfig singleton.

    Shared by the create path and the adopt-by-name path so an adopted
    workspace leaves deployment.db in exactly the state a create would.
    """
    engine = init_deployment_db(workspace_root)
    with Session(engine) as session:
        cfg = session.query(TAODeploymentConfig).first()
        assert cfg is not None  # seeded by init_deployment_db
        cfg.tao_workspace_id = workspace_id
        cfg.tao_workspace_name = workspace_name
        cfg.tao_workspace_cloud_type = cloud_type
        cfg.tao_workspace_bucket = bucket
        cfg.tao_workspace_s3_endpoint_url_internal = endpoint_url_internal
        cfg.tao_workspace_s3_endpoint_url_external = endpoint_url_external
        cfg.tao_workspace_s3_access_key_ref = "TAO_WORKSPACE_S3_ACCESS_KEY"
        cfg.tao_workspace_s3_secret_key_ref = "TAO_WORKSPACE_S3_SECRET_KEY"
        session.commit()


async def _list_workspaces(
    settings: Settings,
    *,
    _transport: httpx.AsyncBaseTransport | None = None,
) -> list[dict[str, Any]] | None:
    """GET the org's workspaces; ``None`` on any transport/endpoint error.

    A ``None`` (listing unavailable) deliberately does NOT block the
    create path — callers fall through to POST, preserving pre-listing
    behavior when FTMS can't answer the query.
    """
    url = _workspaces_url(settings)
    result = await resilient_request(
        "GET",
        url,
        deadline_s=float(settings.HTTP_DEADLINE_INTERACTIVE_S),
        max_retries=1,
        headers=await tao_auth_headers(settings),
        _transport=_transport,
    )
    if result.error_class is not None:
        logger.warning("TAO workspace listing failed: %s", result.error_detail)
        return None
    body_any: Any = result.body
    raw_entries: list[Any] = []
    if isinstance(body_any, dict):
        entries_any: Any = cast("dict[str, Any]", body_any).get("workspaces")
        if isinstance(entries_any, list):
            raw_entries = cast("list[Any]", entries_any)
    elif isinstance(body_any, list):
        raw_entries = cast("list[Any]", body_any)
    entries: list[dict[str, Any]] = []
    for entry_any in raw_entries:
        if isinstance(entry_any, dict):
            entries.append(cast("dict[str, Any]", entry_any))
    return entries


def _get_tao_deployment_config(workspace_root: str | Path) -> TAODeploymentConfig:
    """Load the singleton TAODeploymentConfig (seeded by init_deployment_db)."""
    engine = init_deployment_db(workspace_root)
    with Session(engine) as session:
        config = session.query(TAODeploymentConfig).first()
    if config is None:  # pragma: no cover — init_deployment_db always seeds it
        raise RuntimeError("TAODeploymentConfig singleton not found in deployment.db")
    return config


def read_tao_deployment_config(settings: Settings) -> TAODeploymentConfig | None:
    """Load the deployment-scoped TAODeploymentConfig singleton, or ``None``.

    ``deployment.db`` is the sole source of truth for non-secret TAO
    workspace state (workspace_id, bucket, endpoint URLs);
    secrets live in ``.env``. Returns ``None`` when ``WORKSPACE_ROOT``
    is unset. The canonical reader shared by training preflight, the
    training suite, TAO polling, and base-experiment provisioning
    (each previously carried its own copy).
    """
    if not settings.WORKSPACE_ROOT:
        return None
    engine = init_deployment_db(settings.WORKSPACE_ROOT)
    with Session(engine) as session:
        return session.query(TAODeploymentConfig).first()


async def create_or_get_workspace(
    settings: Settings,
    *,
    workspace_name: str,
    cloud_type: str,
    bucket: str,
    endpoint_url_internal: str,
    endpoint_url_external: str,
    access_key: str,
    secret_key: str,
    _transport: httpx.AsyncBaseTransport | None = None,
) -> WorkspaceResult:
    """Idempotently ensure a TAO workspace exists for this deployment.

    Resolution order (first hit wins):

    1. **Adopt by persisted id** — ``TAODeploymentConfig.tao_workspace_id``
       set (any ``bootstrap_status``): GET the workspace to confirm it is
       still reachable; return ``already_provisioned=True`` — no POST. A
       404 falls through to the next step; any other failure is returned
       as-is (never create when confirmation merely failed).
    2. **Adopt by name** — a workspace named ``workspace_name`` already
       exists on FTMS (admin-provisioned, or deployment.db was lost):
       persist its identity onto the singleton and return
       ``already_provisioned=True`` — no POST. A cloud_type mismatch is
       an error, not an adoption.
    3. **Create** — POST to ``/orgs/{org}/workspaces``, persist the
       returned identity onto the singleton, and return success.
    """
    missing = _require_tao_settings(settings)
    if missing:
        return WorkspaceResult(
            success=False,
            error=f"{missing} is not configured",
        )

    if not settings.WORKSPACE_ROOT:
        return WorkspaceResult(
            success=False,
            error="WORKSPACE_ROOT is not configured",
        )

    workspace_root = Path(settings.WORKSPACE_ROOT)
    config = _get_tao_deployment_config(workspace_root)

    # Idempotent path: a persisted workspace id → GET-confirm and adopt.
    # Deliberately NOT gated on ``bootstrap_status == "bootstrapped"``:
    # the CLI stamps ``in_progress`` before calling (and a failed prior
    # run leaves ``failed``), so a status gate makes every re-run take
    # the POST branch — which FTMS rejects with an opaque HTTP 400 when
    # the workspace name already exists (found live 2026-07-14).
    if config.tao_workspace_id:
        existing = await get_workspace(
            settings,
            workspace_id=config.tao_workspace_id,
            _transport=_transport,
        )
        if existing.success:
            existing.already_provisioned = True
            return existing
        if existing.status_code != 404:
            # Transient/auth failure — refuse to create: a blind POST
            # here could produce a duplicate workspace once TAO recovers.
            return existing
        logger.warning(
            "Persisted TAO workspace %s no longer exists on FTMS; "
            "attempting adoption by name before creating.",
            config.tao_workspace_id,
        )

    # Adopt-by-name: the workspace may already exist FTMS-side without
    # this deployment knowing its UUID (admin-provisioned install, or a
    # prior bootstrap whose deployment.db was lost). FTMS enforces name
    # uniqueness, so check the listing before POSTing.
    listed = await _list_workspaces(settings, _transport=_transport)
    if listed is not None:
        match = next((w for w in listed if w.get("name") == workspace_name), None)
        if match is not None:
            matched_cloud = match.get("cloud_type")
            if matched_cloud and matched_cloud != cloud_type:
                return WorkspaceResult(
                    success=False,
                    error=(
                        f"TAO workspace '{workspace_name}' already exists with "
                        f"cloud_type '{matched_cloud}' (requested '{cloud_type}'). "
                        "Adopt it with matching --cloud-type, or choose a "
                        "different --workspace-name."
                    ),
                )
            adopted_id = str(match.get("id") or "")
            if adopted_id:
                _persist_workspace_identity(
                    workspace_root,
                    workspace_id=adopted_id,
                    workspace_name=workspace_name,
                    cloud_type=cloud_type,
                    bucket=bucket,
                    endpoint_url_internal=endpoint_url_internal,
                    endpoint_url_external=endpoint_url_external,
                )
                logger.info(
                    "Adopted existing TAO workspace name=%s id=%s (no POST issued)",
                    workspace_name,
                    adopted_id,
                )
                return WorkspaceResult(
                    success=True,
                    workspace_id=adopted_id,
                    workspace_detail=match,
                    already_provisioned=True,
                    status_code=200,
                )

    url = _workspaces_url(settings)
    body = _build_create_workspace_body(
        name=workspace_name,
        cloud_type=cloud_type,
        bucket=bucket,
        endpoint_url_internal=endpoint_url_internal,
        access_key=access_key,
        secret_key=secret_key,
    )

    result = await resilient_request(
        "POST",
        url,
        deadline_s=float(settings.HTTP_DEADLINE_INTERACTIVE_S),
        max_retries=1,
        headers=await tao_auth_headers(settings),
        json_body=body,
        _transport=_transport,
    )

    if result.error_class is not None:
        raw_detail = result.error_detail or "unknown endpoint error"
        body_text = result.body if isinstance(result.body, str) else repr(result.body)
        # The shared HTTP layer bounds provider messages before returning its
        # error detail. If that boundary cuts through an echoed credential,
        # exact-value redaction after the fact cannot recognize the remaining
        # prefix. The untruncated response body is still available here, so
        # redact it first whenever it contains either submitted credential.
        if any(value and value in body_text for value in (access_key, secret_key)):
            raw_detail = (
                f"HTTP {result.status_code}: "
                + _redact_workspace_credentials(
                    body_text,
                    access_key=access_key,
                    secret_key=secret_key,
                )[:400]
            )
        return WorkspaceResult(
            success=False,
            error=_redact_workspace_credentials(
                f"Workspace creation failed: {raw_detail}",
                access_key=access_key,
                secret_key=secret_key,
            ),
            status_code=result.status_code,
        )

    body_any: Any = result.body
    detail: dict[str, Any] = (
        cast("dict[str, Any]", body_any) if isinstance(body_any, dict) else {}
    )
    workspace_id_raw: Any = detail.get("id") or detail.get("workspace_id")
    workspace_id = (
        workspace_id_raw
        if isinstance(workspace_id_raw, str) and workspace_id_raw
        else None
    )
    if not workspace_id:
        return WorkspaceResult(
            success=False,
            error="TAO workspace create response did not include an id",
            status_code=result.status_code,
        )

    # Persist identity + bucket/endpoint + key refs on the singleton.
    _persist_workspace_identity(
        workspace_root,
        workspace_id=str(workspace_id),
        workspace_name=workspace_name,
        cloud_type=cloud_type,
        bucket=bucket,
        endpoint_url_internal=endpoint_url_internal,
        endpoint_url_external=endpoint_url_external,
    )

    logger.info(
        "Created TAO workspace name=%s id=%s cloud_type=%s",
        workspace_name,
        workspace_id,
        cloud_type,
    )

    return WorkspaceResult(
        success=True,
        workspace_id=str(workspace_id),
        workspace_detail=detail,
        already_provisioned=False,
        status_code=result.status_code,
    )


async def get_workspace(
    settings: Settings,
    *,
    workspace_id: str,
    _transport: httpx.AsyncBaseTransport | None = None,
) -> WorkspaceResult:
    """Fetch workspace detail via GET. Used by preflight + idempotent create."""
    missing = _require_tao_settings(settings)
    if missing:
        return WorkspaceResult(
            success=False,
            error=f"{missing} is not configured",
        )

    url = _workspaces_url(settings, workspace_id)
    result = await resilient_request(
        "GET",
        url,
        deadline_s=float(settings.HTTP_DEADLINE_INTERACTIVE_S),
        max_retries=1,
        headers=await tao_auth_headers(settings),
        _transport=_transport,
    )

    if result.error_class == "timeout":
        return WorkspaceResult(
            success=False,
            error="Connection to TAO timed out",
        )

    if result.error_class == "endpoint_error":
        if result.status_code == 404:
            return WorkspaceResult(
                success=False,
                error=f"Workspace {workspace_id} not found on TAO",
                status_code=result.status_code,
            )
        return WorkspaceResult(
            success=False,
            error=f"TAO workspace fetch failed: {result.error_detail}",
            status_code=result.status_code,
        )

    body_any: Any = result.body
    detail: dict[str, Any] = (
        cast("dict[str, Any]", body_any) if isinstance(body_any, dict) else {}
    )
    return WorkspaceResult(
        success=True,
        workspace_id=workspace_id,
        workspace_detail=detail,
        status_code=result.status_code,
    )


# ── Base-experiment discovery (FTMS 6.26.3) ─────────────────────────────────


async def find_base_experiment_by_arch(
    settings: Settings,
    *,
    network_arch: str,
    name_substring: str | None = None,
    _transport: httpx.AsyncBaseTransport | None = None,
) -> dict[str, Any] | None:
    """Query ``jobs:list_base_experiments`` for an already-indexed base model.

    FTMS 6.26.3 does not expose a client-driven ``:pull_from_ngc``
    endpoint — base experiments are indexed into the workspace by the
    TAO admin (either through the NGC catalog feed or an airgapped
    load). The Blueprint's job is to *find* the right UUID, not to pull.

    Returns the first matching entry (or ``None`` when no match). On
    a fresh workspace with no NGC catalog indexing, this returns
    ``None`` and the caller should surface a plain-language
    "ask the TAO admin to pull Cosmos Reason2 into this workspace"
    message rather than retry forever.
    """
    missing = _require_tao_settings(settings)
    if missing:
        return None

    url = (
        f"{tao_base_url(settings)}/orgs/{settings.TAO_ORG_NAME}"
        f"/jobs:list_base_experiments?network_arch={network_arch}&size=50"
    )
    result = await resilient_request(
        "GET",
        url,
        deadline_s=float(settings.HTTP_DEADLINE_INTERACTIVE_S),
        max_retries=1,
        headers=await tao_auth_headers(settings),
        _transport=_transport,
    )
    if result.error_class is not None:
        logger.warning(
            "list_base_experiments failed for network_arch=%s: %s",
            network_arch,
            result.error_detail,
        )
        return None

    body_any: Any = result.body
    body: dict[str, Any] = (
        cast("dict[str, Any]", body_any) if isinstance(body_any, dict) else {}
    )
    experiments_raw: Any = body.get("experiments") or []
    experiments: list[Any] = (
        cast("list[Any]", experiments_raw) if isinstance(experiments_raw, list) else []
    )
    for raw_exp in experiments:
        if not isinstance(raw_exp, dict):
            continue
        exp = cast("dict[str, Any]", raw_exp)
        if name_substring:
            name_raw: Any = exp.get("name") or exp.get("display_name") or ""
            name = name_raw.lower() if isinstance(name_raw, str) else ""
            if name_substring.lower() not in name:
                continue
        return exp
    return None
