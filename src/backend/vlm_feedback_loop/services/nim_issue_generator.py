# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""nim_issue Action Request content generator.

Registered at import time.  Consumed by the labeling screen's
[Report NIM Issue] button.
"""

from __future__ import annotations

import logging
from typing import Any

from vlm_feedback_loop.services.action_requests import register_generator

logger = logging.getLogger("vlm_feedback_loop.services.nim_issue_generator")


def _generate_nim_issue(
    project_name: str,
    project_id: str,
    context: dict[str, Any],
) -> dict[str, Any]:
    """Generate a nim_issue Action Request.

    Pre-fills: NIM endpoint base URL, model name, sanitized error details,
    timestamp, suggested diagnostic.

    When the caller does not pass ``base_url`` / ``model_name`` in ``context``
    (the typical case — the labeling screen only knows a proposal failed, not
    which endpoint was hit), we fall back to the project's active Teacher and
    its endpoint so the report contains actionable information.
    """
    base_url = context.get("base_url")
    model_name = context.get("model_name")

    if not base_url or not model_name:
        fallback = _load_active_teacher_fallback(project_id)
        base_url = base_url or fallback.get("base_url") or "(not available)"
        model_name = model_name or fallback.get("model_name") or "(not available)"

    error = context.get("error", "(no error details)")
    error_timestamp = context.get("error_timestamp", "(not available)")

    # Path-join that doesn't double-up on `/v1`. The NimEndpoint
    # ``base_url`` already ends in ``/v1`` for OpenAI-compatible
    # endpoints (hosted: ``https://integrate.api.nvidia.com/v1``;
    # local NIM: ``http://localhost:8000/v1``) — appending another
    # ``/v1/models`` would yield a useless ``/v1/v1/models`` in the
    # rendered Action Request. Strip ONE trailing ``/v1`` before
    # appending the suffix.
    api_root = base_url.rstrip("/")
    if api_root.endswith("/v1"):
        api_root = api_root[: -len("/v1")]
    models_url = f"{api_root}/v1/models"
    health_url = f"{api_root}/v1/health/ready"

    rendered_text = (
        f"NIM Issue Report\n"
        f"\n"
        f"Project: {project_name}\n"
        f"\n"
        f"A NIM endpoint returned an error during labeling.\n"
        f"\n"
        f"Endpoint: {base_url}\n"
        f"Model: {model_name}\n"
        f"Error: {error}\n"
        f"Time: {error_timestamp}\n"
        f"\n"
        f"Diagnostic:\n"
        f"  GET {models_url}\n"
        f"  (should return the model list)\n"
        f"\n"
        f"Possible causes:\n"
        f"  - NIM container has stopped or crashed\n"
        f"  - GPU out of memory\n"
        f"  - Model not loaded\n"
        f"  - Network connectivity issue\n"
        f"\n"
        f"Resolution steps:\n"
        f"  1. Check container status: docker ps | grep nim\n"
        f"  2. Check container logs: docker logs <container>\n"
        f"  3. Verify endpoint health: GET {health_url}\n"
        f"  4. Restart container if needed\n"
    )

    technical_requirements = {
        "endpoint_base_url": base_url,
        "model_name": model_name,
        "error": error,
        "error_timestamp": error_timestamp,
        "diagnostic_endpoint": f"GET {models_url}",
    }

    return {
        "technical_requirements": technical_requirements,
        "current_environment": {},
        "rendered_text": rendered_text,
    }


def _load_active_teacher_fallback(project_id: str) -> dict[str, str]:
    """Return the current Teacher's endpoint and model name, or empty strings.

    Keeps the Action Request informative when the UI has no invocation context.
    All lookups are best-effort: any DB / lookup failure silently yields
    empty strings and the caller falls through to ``"(not available)"``.
    """
    try:
        # Imported lazily to avoid import cycles at module-load time.
        from sqlalchemy.orm import Session

        from vlm_feedback_loop.config import get_settings
        from vlm_feedback_loop.db.models import ModelConfig, NimEndpoint, Project
        from vlm_feedback_loop.services.project_service import get_project_engine

        settings = get_settings()
        engine = get_project_engine(project_id, settings.WORKSPACE_ROOT)
        if engine is None:
            return {}
        with Session(engine) as session:
            project = session.query(Project).filter_by(project_id=project_id).first()
            if project is None or not project.teacher_model_config_id:
                return {}
            mc = (
                session.query(ModelConfig)
                .filter_by(
                    project_id=project_id,
                    model_config_id=project.teacher_model_config_id,
                )
                .first()
            )
            if mc is None:
                return {}
            endpoint = (
                session.query(NimEndpoint)
                .filter_by(project_id=project_id, endpoint_id=mc.endpoint_id)
                .first()
            )
            return {
                "model_name": mc.model_name or "",
                "base_url": (endpoint.base_url if endpoint else "") or "",
            }
    except Exception:  # pragma: no cover — defensive, nim_issue must not error
        logger.exception("nim_issue fallback lookup failed")
        return {}


# Register at import time (side-effect import in main.py)
register_generator("nim_issue", _generate_nim_issue)
