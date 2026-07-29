# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the nim_issue Action Request generator."""

from __future__ import annotations

import pytest

import vlm_feedback_loop.services.nim_issue_generator  # noqa: F401 — registers generator
from vlm_feedback_loop.services.action_requests import generate_action_request


@pytest.fixture(autouse=True)
def _hermetic_config(isolated_config):
    """The generator reaches ``get_settings()`` (project-DB lookup for
    endpoint context), which hard-exits when no config file exists — the
    state of a fresh CI runner."""


class TestNimIssueGenerator:
    """AC: nim_issue Action Request contains endpoint info, diagnostic, and no secrets."""

    def test_generates_with_endpoint_info(self):
        result = generate_action_request(
            "nim_issue",
            "Test Project",
            "proj-1",
            {
                "base_url": "https://nim.example.com/v1",
                "model_name": "nvidia/cosmos-reason2-8b",
                "error": "Connection refused",
                "error_timestamp": "2026-04-14T12:00:00Z",
            },
        )
        assert "nim.example.com" in result["rendered_text"]
        assert "nvidia/cosmos-reason2-8b" in result["rendered_text"]

    def test_contains_diagnostic_endpoint(self):
        result = generate_action_request(
            "nim_issue",
            "Test",
            "p1",
            {"base_url": "https://nim.example.com/v1"},
        )
        # Regression guard: naively rendering ``{base_url}/v1/models``
        # produces ``/v1/v1/models`` because base_url already includes
        # ``/v1`` — the NIM Issue Report then asks the operator to GET an
        # endpoint that doesn't exist. The generator must strip one ``/v1``
        # before re-appending the full ``/v1/models`` suffix.
        text = result["rendered_text"]
        assert "GET https://nim.example.com/v1/models" in text
        assert "/v1/v1/models" not in text, (
            "doubled /v1/ path regressed; "
            "see services/nim_issue_generator.py api_root logic"
        )

    def test_diagnostic_endpoint_works_when_base_url_has_no_v1_suffix(self):
        # Defensive: some self-hosted operators wire ``base_url``
        # without the ``/v1`` suffix. The strip-then-append logic
        # must still produce a valid ``/v1/models`` URL.
        result = generate_action_request(
            "nim_issue",
            "Test",
            "p1",
            {"base_url": "http://localhost:8000"},
        )
        text = result["rendered_text"]
        assert "GET http://localhost:8000/v1/models" in text
        assert "/v1/v1/models" not in text

    def test_contains_error_info(self):
        result = generate_action_request(
            "nim_issue",
            "Test",
            "p1",
            {"error": "502 Bad Gateway: upstream timeout"},
        )
        assert "502 Bad Gateway" in result["rendered_text"]

    def test_graceful_with_empty_context(self):
        result = generate_action_request("nim_issue", "Test", "p1", {})
        assert result["rendered_text"]  # non-empty
        assert "(not available)" in result["rendered_text"]
