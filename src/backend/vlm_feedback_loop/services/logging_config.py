# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structured JSON logging with dual output and secret redaction.

Log format: one JSON object per line with 7 required fields.
Redaction: ``nvapi-…``, ``Bearer …``, ``hf_…``, JWT (``eyJ…``) and
``…access_key/…secret_key`` key/value patterns → ``[REDACTED]``, across the
message, args, ``details``, and the rendered exception traceback.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any, cast

# ── Redaction ────────────────────────────────────────────────────────────────

# Key names that hold S3-style credentials (TAO workspace MinIO/SeaweedFS
# keys, boto3 kwargs, ``TAO_WORKSPACE_S3_*`` env vars). The values are
# prefix-less arbitrary strings, so they can only be caught by key context.
_CREDENTIAL_KEY = r"[\w.-]*(?:access|secret)[_-]?key(?:[_-]?id)?"

REDACTION_PATTERNS = [
    # build.nvidia.com / NGC personal API keys.
    re.compile(r"nvapi-[A-Za-z0-9_-]+"),
    # Any Authorization: Bearer <token> value.
    re.compile(r"Bearer\s+[A-Za-z0-9_./+=-]+"),
    # HuggingFace access tokens (gated Cosmos base pulls).
    re.compile(r"hf_[A-Za-z0-9]{16,}"),
    # JWTs (TAO FTMS bearer tokens) — three base64url segments after the
    # distinctive ``eyJ`` header prefix. Matched even without a "Bearer "
    # prefix so a bare token in a URL/error body is caught too.
    re.compile(r"eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+"),
    # A credential key followed by ``=`` or ``:`` and its value — catches
    # dict reprs, JSON bodies, env-file lines, and presigned-URL params.
    # ``(?!<)`` spares documentation placeholders (``KEY=<paste here>``)
    # in rendered action-request text, which shares these patterns.
    re.compile(rf"(?i){_CREDENTIAL_KEY}[\"']?\s*[=:]\s*[\"']?(?!<)[^\"'\s,}}&]+"),
]
REDACTED = "[REDACTED]"

# Dict keys whose values are replaced wholesale by ``redact_value``: inside a
# dict a credential value is a bare string with no key context, so the
# key=value pattern above cannot see it (e.g. a logged boto3 config dict).
_CREDENTIAL_KEY_RE = re.compile(rf"(?i){_CREDENTIAL_KEY}")


def redact(text: str) -> str:
    """Replace secret patterns with [REDACTED]."""
    for pattern in REDACTION_PATTERNS:
        text = pattern.sub(REDACTED, text)
    return text


def redact_exact_secrets(text: str, values: Iterable[str | None]) -> str:
    """Remove known opaque secret values that pattern matching cannot infer."""
    private_values = {value for value in values if value}
    for value in sorted(private_values, key=len, reverse=True):
        text = text.replace(value, REDACTED)
    return text


def redact_value(value: Any) -> Any:
    """Recursively redact strings in dicts, lists, and plain values.

    Dict values under a credential key name are replaced wholesale — the
    value alone carries no redactable pattern.
    """
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        value_dict = cast("dict[Any, Any]", value)
        return {
            k: (
                REDACTED
                if isinstance(k, str) and _CREDENTIAL_KEY_RE.fullmatch(k)
                else redact_value(v)
            )
            for k, v in value_dict.items()
        }
    if isinstance(value, (list, tuple)):
        value_seq = cast("list[Any] | tuple[Any, ...]", value)
        return type(value_seq)(redact_value(v) for v in value_seq)
    return value


# ── Filter ───────────────────────────────────────────────────────────────────


class RedactionFilter(logging.Filter):
    """Redact secrets from log messages and extra fields."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact(str(record.msg))
        if record.args:
            # Use redact_value (recursive, type-preserving), NOT
            # redact(str(...)): stringifying every arg before the
            # formatter runs breaks every "%d"/"%f" format spec in the
            # codebase with
            # ``TypeError: %d format: a real number is required, not str``.
            # redact_value preserves int/float/bool/etc. while still
            # redacting strings inside dicts and lists.
            if isinstance(record.args, dict):
                record.args = redact_value(record.args)
            else:
                record.args = tuple(redact_value(a) for a in record.args)
        # ``details`` is attached as an ``extra={...}`` keyword by callers; it
        # is NOT a native LogRecord attribute, so we go through __dict__.
        details_attr: Any = record.__dict__.get("details")
        if details_attr is not None:
            record.__dict__["details"] = redact_value(details_attr)
        return True


# ── Formatter ────────────────────────────────────────────────────────────────


class StructuredJsonFormatter(logging.Formatter):
    """Emit each log record as a single JSON line with 7 required fields."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "level": record.levelname.lower(),
            "component": getattr(record, "component", record.name),
            "project_id": getattr(record, "project_id", None),
            "correlation_id": getattr(record, "correlation_id", None),
            "message": record.getMessage(),
            "details": getattr(record, "details", None),
        }
        # When the caller passes ``exc_info=...`` (e.g.
        # ``BackgroundTaskManager._on_task_done`` for failed tasks),
        # render the traceback into ``details`` so the structured log
        # surfaces the full call site. Without this, exceptions whose
        # ``__str__`` is empty (bare ``Exception()``,
        # ``httpx.ReadError("")``, SQLAlchemy autoflush wrapper, …)
        # are silent: the message is just ``"Background task
        # failed: <id> — "`` and the traceback is discarded by the
        # formatter — a cosmos-2b ``_poll_health`` failure spent hours
        # hidden behind exactly that empty message.
        if record.exc_info:
            details_raw: Any = entry["details"] or {}
            details: dict[str, Any] = (
                cast("dict[str, Any]", details_raw)
                if isinstance(details_raw, dict)
                else {"prior": details_raw}
            )
            # Redact the traceback text HERE: RedactionFilter runs before the
            # formatter and never touches exc_info, so a secret embedded in an
            # exception message / traceback (a TAO error echoing a Bearer JWT,
            # an httpx error carrying a token in the URL) would otherwise land
            # in the log unredacted.
            details["exc_info"] = redact(self.formatException(record.exc_info))
            entry["details"] = details
        return json.dumps(entry, default=str)


# ── Setup ────────────────────────────────────────────────────────────────────


def setup_logging(log_level: str = "info") -> None:
    """Configure root logger with structured JSON to stdout."""
    level = getattr(logging, log_level.upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(StructuredJsonFormatter())
    handler.addFilter(RedactionFilter())
    root.addHandler(handler)


def get_logger(
    component: str,
    project_id: str | None = None,
    correlation_id: str | None = None,
) -> logging.LoggerAdapter[logging.Logger]:
    """Return a LoggerAdapter with structured extra fields pre-bound."""
    base = logging.getLogger(f"vlm_feedback_loop.{component}")
    return logging.LoggerAdapter(
        base,
        extra={
            "component": component,
            "project_id": project_id,
            "correlation_id": correlation_id,
        },
    )
