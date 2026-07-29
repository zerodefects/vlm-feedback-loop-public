# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Action Request generation and type registry.

Provides the generic framework; type-specific generators are registered
by their owning modules via ``register_generator()``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from vlm_feedback_loop.db.base import utc_now
from vlm_feedback_loop.db.models.audit_event import AuditEvent
from vlm_feedback_loop.services.logging_config import redact, redact_value

VALID_REQUEST_TYPES = {
    "tao_setup",
    "nim_setup",
    "nim_issue",
    "missing_files",
    "student_nim_deploy",
    "tao_issue",
    "deployment_handoff",
}

# Registry populated via register_generator() by the owning modules.
_generators: dict[str, Callable[..., dict[str, Any]]] = {}


def register_generator(
    request_type: str, generator: Callable[..., dict[str, Any]]
) -> None:
    """Register a type-specific Action Request generator.

    The generator is called with ``(project_name=..., project_id=..., context={})``.
    It must return ``{"technical_requirements": {...}, "current_environment": {...}, "rendered_text": "..."}``.
    """
    _generators[request_type] = generator


def generate_action_request(
    request_type: str,
    project_name: str,
    project_id: str,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate an Action Request response.

    Raises ValueError for unknown request types.
    """
    if request_type not in VALID_REQUEST_TYPES:
        raise ValueError(f"Unknown request_type: {request_type}")

    try:
        generator = _generators[request_type]
    except KeyError:
        raise RuntimeError(
            f"No generator registered for {request_type!r} — the owning "
            "module must be imported at startup (see main.py)."
        ) from None
    result: dict[str, Any] = generator(
        project_name=project_name,
        project_id=project_id,
        context=context or {},
    )

    rendered_raw: Any = result.get("rendered_text", "")
    return {
        "request_type": request_type,
        "generated_at": utc_now(),
        "project_name": project_name,
        # The structured payloads can carry provider error text and env
        # echoes, so they get the same secret scrub as rendered_text.
        "technical_requirements": redact_value(
            result.get("technical_requirements", {})
        ),
        "current_environment": redact_value(result.get("current_environment", {})),
        "rendered_text": redact(rendered_raw if isinstance(rendered_raw, str) else ""),
    }


def log_copy(
    engine: Engine,
    project_id: str,
    request_type: str,
    rendered_text: str,
) -> str:
    """Persist an ``action_request_copied`` AuditEvent; return its id."""
    with Session(engine) as session:
        event = AuditEvent(
            project_id=project_id,
            event_type="action_request_copied",
            event_data={
                "request_type": request_type,
                "rendered_text": rendered_text,
            },
        )
        session.add(event)
        session.commit()
        session.refresh(event)
        return event.audit_event_id
