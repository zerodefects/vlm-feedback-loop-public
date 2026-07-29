# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TAO FTMS connection probe and authentication.

Provides lightweight functions for testing TAO reachability and
performing the FTMS login exchange (NGC key → JWT).  Used by the
``tao_setup`` Action Request generator and the training preflight.
"""

from __future__ import annotations

from typing import Any, cast

from vlm_feedback_loop.config import Settings
from vlm_feedback_loop.services.http_client import resilient_request
from vlm_feedback_loop.services.nim_client import NIM_DEFAULT_HEADERS
from vlm_feedback_loop.services.tao_auth import tao_base_url, tao_preflight


async def probe_tao_connection(settings: Settings) -> dict[str, Any]:
    """Probe the TAO FTMS endpoint to confirm access.

    Sends ``GET {TAO_API_BASE_URL}/orgs/{TAO_ORG_NAME}/jobs?limit=1``
    using ``TAO_API_KEY`` as the Bearer token, then reads the server's
    OpenAPI document to confirm its v2 ``ExperimentJobReq`` accepts the
    required per-job ``timeout_minutes`` safety override.

    Returns::

        {
            "success": bool,
            "error": str | None,
            "status_code": int | None,
            "job_timeout_supported": bool,
            "job_timeout_error": str | None,
        }
    """
    headers, preflight_error = await tao_preflight(settings)
    if headers is None:
        return {
            "success": False,
            "error": preflight_error,
            "status_code": None,
            "job_timeout_supported": False,
            "job_timeout_error": (
                "TAO job-timeout compatibility cannot be verified until the "
                "endpoint is reachable."
            ),
        }

    url = f"{tao_base_url(settings)}/orgs/{settings.TAO_ORG_NAME}/jobs?limit=1"
    result = await resilient_request(
        "GET",
        url,
        deadline_s=float(settings.HTTP_DEADLINE_INTERACTIVE_S),
        max_retries=1,
        headers=headers,
    )

    if result.error_class == "timeout":
        return {
            "success": False,
            "error": "Connection to TAO timed out",
            "status_code": None,
            "job_timeout_supported": False,
            "job_timeout_error": (
                "TAO job-timeout compatibility cannot be verified until the "
                "endpoint is reachable."
            ),
        }

    if result.error_class == "endpoint_error":
        if result.status_code in (401, 403):
            return {
                "success": False,
                "error": "Could not connect to TAO. Verify the API URL, key, and organization name.",
                "status_code": result.status_code,
                "job_timeout_supported": False,
                "job_timeout_error": (
                    "TAO job-timeout compatibility cannot be verified until "
                    "the endpoint is reachable."
                ),
            }
        return {
            "success": False,
            "error": f"TAO endpoint error: {result.error_detail}",
            "status_code": result.status_code,
            "job_timeout_supported": False,
            "job_timeout_error": (
                "TAO job-timeout compatibility cannot be verified until the "
                "endpoint is reachable."
            ),
        }

    openapi_url = f"{tao_base_url(settings)}/openapi.json"
    openapi_result = await resilient_request(
        "GET",
        openapi_url,
        deadline_s=float(settings.HTTP_DEADLINE_INTERACTIVE_S),
        max_retries=1,
        headers=headers,
    )
    if openapi_result.error_class is not None:
        timeout_supported = False
        timeout_error = (
            "TAO job-timeout compatibility could not be verified: "
            f"{openapi_result.error_detail or 'OpenAPI request failed'}."
        )
    else:
        timeout_supported = _openapi_accepts_job_timeout(openapi_result.body)
        timeout_error = (
            None
            if timeout_supported
            else (
                "TAO does not declare `timeout_minutes` on its v2 "
                "ExperimentJobReq schema."
            )
        )

    return {
        "success": True,
        "error": None,
        "status_code": result.status_code,
        "job_timeout_supported": timeout_supported,
        "job_timeout_error": timeout_error,
    }


def _openapi_accepts_job_timeout(body: Any) -> bool:
    """Return whether the FTMS v2 experiment-job schema accepts a timeout.

    Training-suite actions are all submitted as ``kind=experiment``, so this
    is the exact non-mutating compatibility check needed before the Blueprint
    allows a suite to start.
    """
    if not isinstance(body, dict):
        return False
    body_dict = cast("dict[str, Any]", body)
    components = body_dict.get("components", {})
    if not isinstance(components, dict):
        return False
    component_dict = cast("dict[str, Any]", components)
    schemas = component_dict.get("schemas", {})
    if not isinstance(schemas, dict):
        return False
    schema_dict = cast("dict[str, Any]", schemas)
    experiment_schema = schema_dict.get("ExperimentJobReq", {})
    if not isinstance(experiment_schema, dict):
        return False
    experiment_dict = cast("dict[str, Any]", experiment_schema)
    properties = experiment_dict.get("properties", {})
    return isinstance(properties, dict) and "timeout_minutes" in properties


async def login_tao(
    tao_api_base_url: str,
    ngc_api_key: str,
    org_name: str,
    deadline_s: float = 30.0,
) -> dict[str, Any]:
    """Exchange an NGC Personal API Key for a FTMS JWT.

    Sends ``POST {tao_api_base_url}/login`` with the NGC key and org name.
    On a stock FTMS v2 deployment, the returned JWT is used as the
    Bearer token for subsequent Jobs API calls.

    The request body fields (``ngc_key``, ``ngc_org_name``) are validated
    against the FTMS 6.26.3 OpenAPI spec (``LoginReq`` schema).

    Returns::

        {"success": bool, "token": str | None, "error": str | None}
    """
    base = tao_api_base_url.rstrip("/")
    url = f"{base}/login"

    # Field names per FTMS v2 OpenAPI spec (LoginReq schema):
    #   ngc_key: str       — NGC Personal API Key
    #   ngc_org_name: str  — Organization name
    result = await resilient_request(
        "POST",
        url,
        deadline_s=deadline_s,
        max_retries=1,
        headers=NIM_DEFAULT_HEADERS,
        json_body={"ngc_key": ngc_api_key, "ngc_org_name": org_name},
    )

    if result.error_class is not None:
        return {
            "success": False,
            "token": None,
            "error": f"TAO login failed: {result.error_detail}",
        }

    body: Any = result.body
    if isinstance(body, dict):
        body_dict = cast("dict[str, Any]", body)
        token: Any = body_dict.get("token") or body_dict.get("access_token")
        if token:
            return {"success": True, "token": str(token), "error": None}

    return {
        "success": False,
        "token": None,
        "error": "TAO login: unexpected response format",
    }
