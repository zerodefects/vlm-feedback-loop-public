# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Centralized API error construction and the one service-error classifier.

Follows the RAG Blueprint pattern (``src/nvidia_rag/rag_server/server.py``):
errors log at construction, carry a specific HTTP status code, and inherit
from :class:`fastapi.HTTPException` so FastAPI's default exception handler
serializes them as ``{"detail": "..."}`` with the right status.

The helpers cover the status codes CLAUDE.md's "Error responses" section
mandates for NIM-driven workflows:

* :func:`not_found`            → 404 Not Found
* :func:`conflict`             → 409 Conflict
* :func:`validation_failed`    → 400 Bad Request

Upstream 503/504 (unreachable / timed-out NIM and TAO endpoints) are
produced by :func:`map_service_error` from the ``tao_unreachable:`` /
``tao_timeout:`` machine tokens.

Services signal request-path errors by returning a message string instead
of a result dict; :func:`map_service_error` turns those service-layer
strings into HTTP statuses. Routers should not re-implement that mapping —
seven per-router copies of it drifted apart (the same error produced 400 on
one endpoint and 422 on another) before being consolidated here. Simple
literal cases in routers (a 404 for a missing record, a 409 for a known
state conflict) may still raise ``HTTPException`` directly; the helpers are
for classifying service-layer error strings consistently.

A stable prefix (``"validation: "``, ``"conflict: "``, ``"not found: "``)
is the author's explicit classification and always wins. The substring
fallback exists for legacy unprefixed strings and has a known failure
mode: the message interpolates user-controlled content (enum vocabularies,
field names, submitted values), so a stray ``"insufficient"`` or
``"not found"`` inside that content would misclassify — prefix new service
error strings instead of relying on the fallback.
"""

from __future__ import annotations

import logging

from fastapi import HTTPException

logger = logging.getLogger("vlm_feedback_loop.errors")


class APIError(HTTPException):
    """HTTPException that logs on construction.

    Raising an ``APIError`` produces a JSON response ``{"detail": "..."}``
    with the carried status code — identical wire behaviour to raising
    ``HTTPException`` directly, with the added benefit of a structured log
    entry (component ``vlm_feedback_loop.errors``) for post-hoc triage.
    """

    def __init__(self, *, status_code: int, detail: str) -> None:
        logger.info(
            "api_error status=%d detail=%s",
            status_code,
            detail,
            extra={"status_code": status_code},
        )
        super().__init__(status_code=status_code, detail=detail)


def validation_failed(detail: str) -> APIError:
    """400 Bad Request — the request cannot be processed as given.

    Use for domain validation errors where the client could fix the input:
    invalid enum values, missing required references, out-of-range integers.
    Distinct from FastAPI's automatic 422 for Pydantic schema validation.
    """
    return APIError(status_code=400, detail=detail)


def conflict(detail: str) -> APIError:
    """409 Conflict — state transition violates an invariant.

    Use for stale proposals (superseded by a later retry), terminal-state
    transitions (cancelling a completed run), duplicate keys, and single-
    process project locks.
    """
    return APIError(status_code=409, detail=detail)


def not_found(detail: str) -> APIError:
    """404 Not Found — the referenced project or record does not exist."""
    return APIError(status_code=404, detail=detail)


# Case-sensitive machine tokens that services embed in error strings as
# stable codes. Checked after the explicit prefixes, before the substring
# fallback.
_TOKEN_RULES: tuple[tuple[str, int], ...] = (
    ("VISUAL_BUDGET_PARITY_VIOLATION", 400),
    ("INFERENCE_CONTRACT_MISMATCH", 409),
    ("student_nim_not_deployed", 409),
    ("tao_eval_already_in_suite", 409),
    ("student_nim_serving_run_missing", 409),
    ("tao_unreachable:", 503),
    ("tao_timeout:", 504),
    ("tao_error:", 502),
)


def map_service_error(result: str) -> APIError:
    """Classify a service error string into the one true HTTP status.

    Services prefix their error strings with a stable marker —
    ``"not found: "``, ``"conflict: "``, ``"validation: "`` — or embed one
    of the machine tokens above. The rules, in precedence order:

    * explicit prefix         → its status (the author's classification;
      checked first because the message body often interpolates
      user-controlled content that would trip the substring fallback)
    * machine token           → its listed status
    * ``not found``           → 404
    * ``conflict`` / ``terminal`` / ``superseded`` / ``insufficient`` → 409
    * anything else           → 400 (domain validation — the client could
      fix the input). 422 is reserved for FastAPI's automatic Pydantic
      schema validation, per the CLAUDE.md error contract.
    """
    lower = result.lower()
    if lower.startswith("validation:"):
        return validation_failed(result)
    if lower.startswith("conflict:"):
        return conflict(result)
    if lower.startswith("not found:"):
        return not_found(result)
    for token, status in _TOKEN_RULES:
        if token in result:
            return APIError(status_code=status, detail=result)
    if "not found" in lower:
        return not_found(result)
    if any(t in lower for t in ("conflict", "terminal", "superseded", "insufficient")):
        return conflict(result)
    return validation_failed(result)
