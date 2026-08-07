# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for structured logging."""

from __future__ import annotations

import io
import json
import logging

import pytest

from vlm_feedback_loop.services.logging_config import (
    RedactionFilter,
    StructuredJsonFormatter,
    get_logger,
    redact,
)


@pytest.fixture(autouse=True)
def _reset_root_logger():
    """Ensure root logger is clean between tests."""
    root = logging.getLogger()
    old_handlers = root.handlers[:]
    old_level = root.level
    yield
    root.handlers = old_handlers
    root.level = old_level


def _capture_log_output(log_level: str = "info") -> tuple[io.StringIO, logging.Logger]:
    """Set up logging to capture to a StringIO buffer."""
    buf = io.StringIO()
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(getattr(logging, log_level.upper()))

    handler = logging.StreamHandler(buf)
    handler.setFormatter(StructuredJsonFormatter())
    handler.addFilter(RedactionFilter())
    root.addHandler(handler)

    return buf, root


class TestStructuredJsonFields:
    """Every log entry has the 7 required JSON fields."""

    def test_seven_fields_present(self):
        buf, _ = _capture_log_output()

        log = get_logger("test_comp", project_id="proj1", correlation_id="corr1")
        log.info("Test message")

        output = buf.getvalue().strip()
        entry = json.loads(output)

        assert "timestamp" in entry
        assert entry["level"] == "info"
        assert entry["component"] == "test_comp"
        assert entry["project_id"] == "proj1"
        assert entry["correlation_id"] == "corr1"
        assert entry["message"] == "Test message"
        assert "details" in entry  # may be None


class TestLogLevelConfigurable:
    """LOG_LEVEL defaults to info; debug produces more."""

    def test_info_filters_debug(self):
        buf, _ = _capture_log_output("info")
        log = get_logger("test")
        log.debug("debug msg")
        log.info("info msg")

        lines = [line for line in buf.getvalue().strip().split("\n") if line]
        assert len(lines) == 1
        assert "info msg" in lines[0]

    def test_debug_shows_debug(self):
        buf, _ = _capture_log_output("debug")
        log = get_logger("test")
        log.debug("debug msg")

        lines = [line for line in buf.getvalue().strip().split("\n") if line]
        assert len(lines) >= 1
        assert "debug msg" in lines[0]


class TestRedaction:
    """Secret patterns are redacted."""

    def test_nvapi_redacted(self):
        buf, _ = _capture_log_output()
        log = get_logger("test")
        log.info("Key is nvapi-abc123XYZ_secret")

        output = buf.getvalue()
        assert "nvapi-abc123XYZ_secret" not in output
        assert "[REDACTED]" in output

    def test_bearer_redacted(self):
        buf, _ = _capture_log_output()
        log = get_logger("test")
        log.info("Authorization: Bearer my_secret_token_here")

        output = buf.getvalue()
        assert "my_secret_token_here" not in output
        assert "[REDACTED]" in output

    def test_redact_function_directly(self):
        assert redact("key=nvapi-REAL_KEY") == "key=[REDACTED]"
        assert redact("Bearer tok123") == "[REDACTED]"

    def test_hf_and_jwt_tokens_redacted(self):
        # HuggingFace token and a TAO FTMS JWT (bare, no Bearer prefix).
        fake_hf_token = "hf_ABCdef0123456789ghijkl"  # gitleaks:allow
        assert "hf_" not in redact(f"token={fake_hf_token}").replace("[REDACTED]", "")
        jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.abc-DEF_123"
        assert jwt not in redact(f"Auth failed for {jwt}")
        assert "[REDACTED]" in redact(f"Auth failed for {jwt}")

    def test_s3_credentials_redacted_by_key_value_context(self):
        # TAO workspace S3 creds (MinIO/SeaweedFS) are prefix-less arbitrary
        # strings — they are caught by key=value context, not value shape.
        assert "seaweedfs123" not in redact("aws_secret_access_key='seaweedfs123'")
        assert "AKIAXYZ" not in redact('{"access_key": "AKIAXYZ"}')
        assert "sekrit" not in redact("TAO_WORKSPACE_S3_SECRET_KEY=sekrit")
        # A bare mention of the key name (no value) stays readable.
        assert redact("secret_key not configured") == "secret_key not configured"

    def test_s3_credentials_redacted_inside_logged_dict(self):
        # A boto3-style config dict interpolated via %s: the values are bare
        # strings inside the dict, so redaction must key off the dict keys.
        buf, root = _capture_log_output()
        root.info(
            "building s3 client: %s",
            {
                "aws_access_key_id": "seaweedfs",
                "aws_secret_access_key": "seaweedfs123",
                "endpoint_url": "http://127.0.0.1:8333",
            },
        )
        output = buf.getvalue()
        assert "seaweedfs" not in output
        assert "seaweedfs123" not in output
        assert "[REDACTED]" in output
        # Non-credential fields survive for debuggability.
        assert "http://127.0.0.1:8333" in output

    def test_secret_in_exception_traceback_redacted(self):
        # RedactionFilter never touches exc_info; the formatter must redact
        # the rendered traceback or a token in an exception message leaks.
        buf, _ = _capture_log_output()
        log = get_logger("test")
        try:
            raise RuntimeError("TAO rejected Bearer eyJx.eyJy.zzz-secret")
        except RuntimeError:
            log.error("job failed", exc_info=True)

        output = buf.getvalue()
        assert "zzz-secret" not in output
        assert "eyJx.eyJy.zzz-secret" not in output
        assert "[REDACTED]" in output

    def test_no_secrets_in_output(self):
        buf, _ = _capture_log_output()
        log = get_logger("test")
        log.info(
            "Config: api_key=nvapi-SECRET123 token=Bearer SECRET_TOKEN",  # gitleaks:allow
        )

        output = buf.getvalue()
        assert "SECRET123" not in output
        assert "SECRET_TOKEN" not in output


class TestRedactionFilterPreservesArgTypes:
    """Regression — RedactionFilter MUST NOT stringify int/float args.

    A filter that does ``tuple(redact(str(a)) for a in record.args)``
    converts every arg (including ints) to a string before the formatter
    runs ``msg % self.args``, so every ``%d`` / ``%f`` format spec in the
    codebase trips with
    ``TypeError: %d format: a real number is required, not str``.
    The most visible victim is ``clip_embedding_service``'s
    ``Embedding worker: completed %d/%d for project %s`` log call, which
    crashes on every embedding-worker completion. This test class pins
    the type-preserving behavior.
    """

    def test_percent_d_with_int_args_does_not_crash(self):
        """Smoke regression for the embedding-worker %d crash."""
        buf, _ = _capture_log_output()
        log = get_logger("test")

        # With a stringifying filter this raises
        # ``TypeError: %d format: a real number is required, not str``
        # because the int arg reaches ``%d`` as a string.
        log.info("Embedding worker: completed %d/%d for project %s", 9, 9, "p1")

        output = buf.getvalue().strip()
        entry = json.loads(output)
        assert entry["message"] == "Embedding worker: completed 9/9 for project p1"

    def test_percent_d_with_zero_args_does_not_crash(self):
        buf, _ = _capture_log_output()
        log = get_logger("test")
        log.info("count=%d", 0)

        entry = json.loads(buf.getvalue().strip())
        assert entry["message"] == "count=0"

    def test_percent_f_with_float_arg_preserves_format(self):
        buf, _ = _capture_log_output()
        log = get_logger("test")
        log.info("ratio=%.2f", 0.875)

        entry = json.loads(buf.getvalue().strip())
        assert entry["message"] == "ratio=0.88"

    def test_percent_s_with_string_arg_still_redacts(self):
        """The fix preserves the redaction smoke check that motivated the filter."""
        buf, _ = _capture_log_output()
        log = get_logger("test")
        log.info("auth=%s", "Bearer tok_xyz_secret")

        output = buf.getvalue()
        assert "tok_xyz_secret" not in output
        assert "[REDACTED]" in output

    def test_dict_args_preserve_int_types(self):
        """The dict path of record.args also must preserve types."""
        buf, _ = _capture_log_output()
        log = get_logger("test")
        log.info("processed=%(p)d total=%(t)d", {"p": 3, "t": 5})

        entry = json.loads(buf.getvalue().strip())
        assert entry["message"] == "processed=3 total=5"

    def test_mixed_args_int_string(self):
        buf, _ = _capture_log_output()
        log = get_logger("test")
        log.info("provider=%s, concurrency=%d, batch_size=%d", "hosted_nvclip", 1, 8)

        entry = json.loads(buf.getvalue().strip())
        assert entry["message"] == "provider=hosted_nvclip, concurrency=1, batch_size=8"


def test_credential_key_placeholder_docs_survive_redaction():
    """Rendered guidance text documents env keys as ``KEY=<placeholder>``;
    the scrub must redact real values without eating the documentation
    (a redacted placeholder deletes the key name the operator needs)."""
    from vlm_feedback_loop.services.logging_config import redact as redact_text

    doc = "TAO_WORKSPACE_S3_ACCESS_KEY=<S3 access key — persisted by `tao-bootstrap`>"
    assert redact_text(doc) == doc
    real = "TAO_WORKSPACE_S3_ACCESS_KEY=AKIA123456789EXAMPLE"
    assert "AKIA123456789EXAMPLE" not in redact_text(real)
