# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NIM endpoint management and environment assessment routes.

Two router objects:
  - ``nim_router``: deployment-scoped (no project_id) — environment, test_connection,
    embedding_deployment_config.
  - ``nim_endpoints_router``: project-scoped — NimEndpoint CRUD.
"""

from __future__ import annotations

import base64
from typing import Any
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, HTTPException, Query

from vlm_feedback_loop.config import Settings
from vlm_feedback_loop.model_catalog_constants import (
    COSMOS_REASON2_8B_NIM_IMAGE,
)
from vlm_feedback_loop.routers.projects import get_current_settings
from vlm_feedback_loop.schemas.nim import (
    ConnectionTestRequest,
    ConnectionTestResponse,
    CredentialTestRequest,
    EmbeddingDeploymentConfigResponse,
    EmbeddingDeploymentConfigUpdate,
    EnvironmentResponse,
    NimEndpointCreate,
    NimEndpointListResponse,
    NimEndpointResponse,
    NimEndpointUpdate,
    SelfHostedEmbeddingConfigureRequest,
    SelfHostedTeacherConfigureRequest,
    SelfHostedTeacherConfigureResponse,
)
from vlm_feedback_loop.services import local_nim_service, nim_client
from vlm_feedback_loop.services.environment import (
    get_cached_environment,
    invalidate_machine_assessment_cache,
)
from vlm_feedback_loop.services.http_client import resilient_request
from vlm_feedback_loop.services.nim_client import (
    NIM_DEFAULT_HEADERS,
    build_auth_headers,
)
from vlm_feedback_loop.services.nim_endpoint_service import (
    NimEndpointConfigurationError,
    configure_self_hosted_teacher,
    create_nim_endpoint,
    get_nim_endpoint,
    list_nim_endpoints,
    normalize_self_hosted_base_url,
    update_nim_endpoint,
)
from vlm_feedback_loop.services.runtime_secrets import get_effective_secret

# ── Deployment-scoped router ────────────────────────────────────────────────

nim_router = APIRouter(tags=["nim"])


@nim_router.get("/environment", response_model=EnvironmentResponse)
async def get_environment(
    refresh_hardware: bool = Query(
        default=False,
        description=(
            "Re-probe Docker, NVIDIA Container Toolkit, and GPU inventory before "
            "returning the deployment-scoped assessment."
        ),
    ),
    settings: Settings = Depends(get_current_settings),
) -> EnvironmentResponse:
    """Return deployment recommendations over the cached machine assessment.

    Deployment-scoped (no project_id). No secrets in response. Stable hardware
    capabilities are cached for the backend process lifetime; credentials,
    embedding configuration, and active NIM residents are composed fresh.
    """
    if refresh_hardware:
        invalidate_machine_assessment_cache()
    result = await get_cached_environment(settings)
    return EnvironmentResponse(**result)


@nim_router.post("/nim/test_connection", response_model=ConnectionTestResponse)
async def test_connection(
    body: ConnectionTestRequest,
    settings: Settings = Depends(get_current_settings),
) -> ConnectionTestResponse:
    """Transient connection test proxy.

    ``credential_transient`` is held in request memory only for this probe,
    then discarded.  It is NEVER written to any durable store.
    """
    try:
        auth_headers = nim_client.build_auth_headers(
            auth_mode=body.auth_mode,
            credential=body.credential_transient,
        )
    except ValueError as exc:
        return ConnectionTestResponse(success=False, error=str(exc))

    probe_base_url = body.base_url
    if body.auth_mode == "none":
        try:
            probe_base_url = normalize_self_hosted_base_url(body.base_url)
        except NimEndpointConfigurationError as exc:
            return ConnectionTestResponse(success=False, error=exc.message)

    if body.probe_kind == "embeddings":
        try:
            await local_nim_service.probe_embedding_endpoint(
                probe_base_url,
                auth_headers,
                settings,
            )
        except NimEndpointConfigurationError as exc:
            return ConnectionTestResponse(success=False, error=exc.message)
        return ConnectionTestResponse(success=True)

    # Default: probe_kind == "models"
    result_models = await nim_client.list_models(
        base_url=probe_base_url,
        auth_headers=auth_headers,
        deadline_s=settings.HTTP_DEADLINE_INTERACTIVE_S,
        max_retries=1,
    )
    return ConnectionTestResponse(
        success=result_models.success,
        models=result_models.models,
        error=result_models.error,
    )


# ── Credential probes (shared shape) ────────────────────────────────────────
#
# The NGC and NVIDIA credential tests differ only in WHAT they probe (a
# registry token request with Basic auth vs. a gated chat completion with
# Bearer auth) and in the per-status error copy. The resolve/probe/classify
# skeleton is shared below so the two endpoints can't drift apart.


def _resolve_key_to_probe(
    credential_transient: str | None,
    secret_name: str,
    settings: Settings,
) -> str | None:
    """Pick the key a credential-test endpoint should probe.

    Two modes: a non-blank ``credential_transient`` wins; otherwise
    fall back to the currently-effective runtime secret. The latter is what
    the NIM setup gate uses when the SME's prior session left a key in
    ``.env`` and they didn't re-paste it; without it an already-bad
    persisted key would never get re-validated and would sail past the
    gate. Returns ``None`` when neither source yields a non-blank key.

    The credential is held in request memory only and discarded once the
    endpoint returns.  It is NEVER written to any durable store.
    """
    if credential_transient and credential_transient.strip():
        return credential_transient
    effective = get_effective_secret(secret_name, settings)
    if effective and effective.strip():
        return effective
    return None


async def _probe_and_classify(
    *,
    settings: Settings,
    method: str,
    url: str,
    headers: dict[str, str],
    host_label: str,
    auth_failure_errors: dict[int, str],
    json_body: dict[str, Any] | None = None,
    success_statuses: frozenset[int] = frozenset({200}),
) -> ConnectionTestResponse:
    """Fire one credential probe through the canonical client and classify it.

    Sends via ``resilient_request`` (Blueprint source header + shared
    deadline/retry machinery supplied by the caller's ``headers``), then
    maps the outcome: timeout / unreachable → friendly network error,
    a status in ``success_statuses`` (default: 200) → success, a status
    listed in ``auth_failure_errors`` → that endpoint's specific rejection
    copy, anything else → surfaced verbatim for debugging.
    """
    result = await resilient_request(
        method,
        url,
        deadline_s=float(settings.HTTP_DEADLINE_INTERACTIVE_S),
        max_retries=2,
        headers=headers,
        json_body=json_body,
    )
    if result.error_class == "timeout":
        return ConnectionTestResponse(
            success=False,
            error=f"Timed out reaching {host_label}. Check your network and try again.",
        )
    if result.status_code is None:
        return ConnectionTestResponse(
            success=False,
            error=f"Could not reach {host_label}: {result.error_detail}",
        )

    if result.status_code in success_statuses:
        return ConnectionTestResponse(success=True)
    if result.status_code in auth_failure_errors:
        return ConnectionTestResponse(
            success=False,
            error=auth_failure_errors[result.status_code],
        )
    return ConnectionTestResponse(
        success=False,
        error=f"Unexpected response from {host_label}: HTTP {result.status_code}",
    )


# Repository probed for pull scope by the NGC credential test — the
# Blueprint's own Teacher NIM repo, derived from the canonical image ref
# (registry prefix and tag stripped) so the probe target tracks the catalog.
_NGC_PROBE_REPOSITORY = COSMOS_REASON2_8B_NIM_IMAGE.removeprefix("nvcr.io/").rsplit(
    ":", 1
)[0]


@nim_router.post("/nim/test_ngc_credential", response_model=ConnectionTestResponse)
async def test_ngc_credential(
    body: CredentialTestRequest,
    settings: Settings = Depends(get_current_settings),
) -> ConnectionTestResponse:
    """Validate an NGC API key against the nvcr.io registry.

    The standard ``test_connection`` endpoint probes OpenAI-compatible
    NIM endpoints (build.nvidia.com / self-hosted) with Bearer auth; that
    doesn't work for NGC keys because the registry uses HTTP Basic with
    username ``$oauthtoken`` and the key as the password.  This endpoint
    is the parallel probe for NGC: it requests a pull-scope token from
    ``nvcr.io/proxy_auth`` and classifies the response.

    Status classification (empirically verified):
      - 200 → key valid and has the NGC Catalog / Private Registry scopes
      - 403 (``{"code":"DENIED"}``) → key is structurally valid but lacks
        the registry-pull scope (build.nvidia.com Personal API Key vs
        NGC key with Catalog + Private Registry services enabled at key
        creation time)
      - 401 → key is empty / malformed / unrecognised
      - anything else → surfaced verbatim for debugging

    ``credential_transient`` is held in request memory only and discarded
    once this function returns.  It is NEVER written to any durable store.
    """
    key_to_probe = _resolve_key_to_probe(
        body.credential_transient, "NGC_API_KEY", settings
    )
    if key_to_probe is None:
        return ConnectionTestResponse(
            success=False,
            error=(
                "No NGC API key is configured. Paste a key with the NGC "
                "Catalog and Private Registry scopes to continue."
            ),
        )

    # The registry speaks HTTP Basic (username ``$oauthtoken``), not the
    # NIM bearer scheme — build that one header here, but send it through
    # the canonical client so the probe carries the Blueprint source
    # header and the shared deadline/rate-limit machinery.
    query = urlencode({"scope": f"repository:{_NGC_PROBE_REPOSITORY}:pull"})
    basic = base64.b64encode(f"$oauthtoken:{key_to_probe}".encode()).decode()

    return await _probe_and_classify(
        settings=settings,
        method="GET",
        url=f"https://nvcr.io/proxy_auth?{query}",
        headers={**NIM_DEFAULT_HEADERS, "Authorization": f"Basic {basic}"},
        host_label="nvcr.io",
        auth_failure_errors={
            403: (
                "NGC key rejected by nvcr.io. The key needs the NGC Catalog "
                "and Private Registry scopes. Regenerate at "
                "https://org.ngc.nvidia.com/setup/api-key with both services "
                "enabled, then paste the new key."
            ),
            401: (
                "NGC key not recognised by nvcr.io. Double-check that you "
                "pasted the full key value."
            ),
        },
    )


@nim_router.post(
    "/nim/test_nvidia_credential",
    response_model=ConnectionTestResponse,
)
async def test_nvidia_credential(
    body: CredentialTestRequest,
    settings: Settings = Depends(get_current_settings),
) -> ConnectionTestResponse:
    """Validate an NVIDIA API key against ``build.nvidia.com``.

    Mirror of :func:`test_ngc_credential` for the NVIDIA / hosted-NIM key.

    **Why this exists separately from ``test_connection``**: the
    ``test_connection`` endpoint probes ``GET /v1/models``, which on
    ``https://integrate.api.nvidia.com/v1`` is **fully public** — it
    returns the catalog whether you pass a valid bearer, a garbage
    bearer, or no auth header at all. So ``test_connection`` cannot
    validate a key; it only confirms the host is reachable. This
    endpoint hits ``POST /v1/chat/completions`` — which IS gated
    (returns 403 "Authorization failed" on bad bearer) — with an
    intentionally invalid payload that clears auth but never reaches
    inference, so the probe is fast regardless of model queue load.

    Two modes:

      * ``credential_transient`` non-empty → probe that key.
      * ``credential_transient`` null/empty → probe the currently
        effective NVIDIA key resolved via :func:`get_effective_secret`.

    Credential is held in request memory only and discarded once this
    function returns.  NEVER written to any durable store and NEVER
    logged.
    """
    key_to_probe = _resolve_key_to_probe(
        body.credential_transient, "NVIDIA_API_KEY", settings
    )
    if key_to_probe is None:
        return ConnectionTestResponse(
            success=False,
            error=(
                "No NVIDIA API key is configured. Paste a key from "
                "https://build.nvidia.com/settings/api-keys to continue."
            ),
        )

    # Auth-gate probe with a deliberately invalid payload (empty
    # ``messages``) against the seeded default Teacher's route — the only
    # actually-gated path (GET /v1/models is public and can't validate a
    # key). The gateway authenticates the bearer BEFORE request
    # validation, so a valid key returns 400 in ~1 s and a bad key
    # 401/403 in ~0.2 s, without ever entering the model's inference
    # queue. The previous probe was a real 1-token completion and sat in
    # that queue — 20–52 s per probe measured under load (2026-07-20),
    # which is what the FTU setup screen's dwell time was made of.
    # Verified live: auth precedes validation on integrate.api.nvidia.com
    # (empty messages → 400 with a valid bearer, 403 garbage, 401 none).
    # If that ordering ever flips, a bad key would misclassify as valid
    # here and surface at first proposal instead — click-time probes and
    # downstream error surfaces remain the backstop.
    payload: dict[str, Any] = {
        "model": settings.DEFAULT_TEACHER_MODEL,
        "messages": [],
        "max_tokens": 1,
    }

    rejected = (
        "NVIDIA API key was rejected by build.nvidia.com. Generate "
        "a new key at https://build.nvidia.com/settings/api-keys "
        "and paste it here."
    )
    return await _probe_and_classify(
        settings=settings,
        method="POST",
        url="https://integrate.api.nvidia.com/v1/chat/completions",
        headers={
            **NIM_DEFAULT_HEADERS,
            **build_auth_headers("bearer", credential=key_to_probe),
        },
        host_label="build.nvidia.com",
        auth_failure_errors={401: rejected, 403: rejected},
        json_body=payload,
        # 400/422 mean the bearer cleared auth and the request died at
        # validation — exactly the signal this probe is engineered to hit.
        # A 429 is also an authenticated outcome: the hosted quota gate
        # accepted the bearer before rate-limiting the request. Invalid
        # bearers still return 401/403. Treating 429 as rejection traps users
        # in setup even after they paste a fresh, valid key.
        success_statuses=frozenset({200, 400, 422, 429}),
    )


@nim_router.patch(
    "/embedding_deployment_config",
    response_model=EmbeddingDeploymentConfigResponse,
)
def update_embedding_config(
    body: EmbeddingDeploymentConfigUpdate,
    settings: Settings = Depends(get_current_settings),
) -> EmbeddingDeploymentConfigResponse:
    """Update the deployment-scoped embedding-NIM configuration singleton."""
    config = local_nim_service.update_embedding_deployment_config(
        settings.WORKSPACE_ROOT,
        body.model_dump(exclude_unset=True),
    )
    if config is None:
        raise HTTPException(
            status_code=500,
            detail="EmbeddingDeploymentConfig singleton not found",
        )
    return EmbeddingDeploymentConfigResponse.model_validate(config)


@nim_router.post(
    "/embedding_deployment_config:configure_self_hosted",
    response_model=EmbeddingDeploymentConfigResponse,
)
async def configure_self_hosted_embedding_endpoint(
    body: SelfHostedEmbeddingConfigureRequest,
    settings: Settings = Depends(get_current_settings),
) -> EmbeddingDeploymentConfigResponse:
    """Durably apply a live-verified external embedding NIM."""
    try:
        config = await local_nim_service.configure_self_hosted_embedding(
            body.base_url,
            settings,
        )
    except NimEndpointConfigurationError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail={"code": exc.code, "message": exc.message},
        ) from exc
    return EmbeddingDeploymentConfigResponse.model_validate(config)


# ── Project-scoped router ──────────────────────────────────────────────────

nim_endpoints_router = APIRouter(
    prefix="/projects/{project_id}/nim_endpoints",
    tags=["nim_endpoints"],
)


@nim_endpoints_router.post(
    ":configure_self_hosted_teacher",
    response_model=SelfHostedTeacherConfigureResponse,
)
async def configure_self_hosted_teacher_endpoint(
    project_id: str,
    body: SelfHostedTeacherConfigureRequest,
    settings: Settings = Depends(get_current_settings),
) -> SelfHostedTeacherConfigureResponse:
    """Durably apply a verified self-hosted endpoint to one Teacher."""
    try:
        endpoint, model_config = await configure_self_hosted_teacher(
            project_id=project_id,
            model_config_id=body.model_config_id,
            base_url=body.base_url,
            workspace_root=settings.WORKSPACE_ROOT,
            settings=settings,
        )
    except NimEndpointConfigurationError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail={"code": exc.code, "message": exc.message},
        ) from exc
    return SelfHostedTeacherConfigureResponse(
        endpoint=NimEndpointResponse.model_validate(endpoint),
        model_config_id=model_config.model_config_id,
        model_name=model_config.model_name,
        structured_generation_support=model_config.structured_generation_support,
        thinking_toggle_support=model_config.thinking_toggle_support,
        visual_budget_support=model_config.visual_budget_support,
    )


@nim_endpoints_router.post("", status_code=201, response_model=NimEndpointResponse)
async def create_endpoint(
    project_id: str,
    body: NimEndpointCreate,
    settings: Settings = Depends(get_current_settings),
) -> NimEndpointResponse:
    """Create a NimEndpoint record with auto-probe."""
    endpoint = await create_nim_endpoint(
        project_id=project_id,
        data=body.model_dump(),
        workspace_root=settings.WORKSPACE_ROOT,
        settings=settings,
    )
    if endpoint is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return NimEndpointResponse.model_validate(endpoint)


@nim_endpoints_router.get("", response_model=NimEndpointListResponse)
def list_endpoints(
    project_id: str,
    settings: Settings = Depends(get_current_settings),
) -> NimEndpointListResponse:
    """List all NimEndpoint records for a project."""
    endpoints = list_nim_endpoints(
        project_id=project_id,
        workspace_root=settings.WORKSPACE_ROOT,
    )
    return NimEndpointListResponse(
        items=[NimEndpointResponse.model_validate(ep) for ep in endpoints],
    )


@nim_endpoints_router.get("/{endpoint_id}", response_model=NimEndpointResponse)
def get_endpoint_detail(
    project_id: str,
    endpoint_id: str,
    settings: Settings = Depends(get_current_settings),
) -> NimEndpointResponse:
    """Retrieve a single NimEndpoint by ID."""
    endpoint = get_nim_endpoint(
        project_id=project_id,
        endpoint_id=endpoint_id,
        workspace_root=settings.WORKSPACE_ROOT,
    )
    if endpoint is None:
        raise HTTPException(status_code=404, detail="Endpoint not found")
    return NimEndpointResponse.model_validate(endpoint)


@nim_endpoints_router.patch("/{endpoint_id}", response_model=NimEndpointResponse)
async def update_endpoint_detail(
    project_id: str,
    endpoint_id: str,
    body: NimEndpointUpdate,
    settings: Settings = Depends(get_current_settings),
) -> NimEndpointResponse:
    """Update a NimEndpoint record. Re-probes if base_url or auth_mode changed."""
    updates = body.model_dump(exclude_unset=True)
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")

    endpoint = await update_nim_endpoint(
        project_id=project_id,
        endpoint_id=endpoint_id,
        updates=updates,
        workspace_root=settings.WORKSPACE_ROOT,
        settings=settings,
    )
    if endpoint is None:
        raise HTTPException(status_code=404, detail="Endpoint not found")
    return NimEndpointResponse.model_validate(endpoint)
