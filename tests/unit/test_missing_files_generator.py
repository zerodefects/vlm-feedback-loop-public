# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the missing_files Action Request generator."""

from __future__ import annotations

import vlm_feedback_loop.services.missing_files_generator  # noqa: F401 — registers generator
from vlm_feedback_loop.services.action_requests import generate_action_request


class TestMissingFilesGenerator:
    """AC: missing_files Action Request contains paths, remap suggestion, and no secrets."""

    def test_generates_with_paths(self):
        result = generate_action_request(
            "missing_files",
            "Test Project",
            "proj-1",
            {"missing_paths": ["/data/images/img_001.jpg", "/data/images/img_002.jpg"]},
        )
        assert "img_001.jpg" in result["rendered_text"]
        assert "img_002.jpg" in result["rendered_text"]

    def test_contains_remap_suggestion(self):
        result = generate_action_request(
            "missing_files",
            "Test",
            "p1",
            {"missing_paths": ["/old/path/img.jpg"]},
        )
        assert "remap_paths" in result["rendered_text"]
        assert "dry_run" in result["rendered_text"]

    def test_caps_displayed_paths(self):
        paths = [f"/data/img_{i:03d}.jpg" for i in range(30)]
        result = generate_action_request(
            "missing_files",
            "Test",
            "p1",
            {"missing_paths": paths},
        )
        text = result["rendered_text"]
        assert "...and 10 more" in text

    def test_empty_paths_handled(self):
        result = generate_action_request("missing_files", "Test", "p1", {})
        assert result["rendered_text"]  # non-empty
        assert "(no paths provided)" in result["rendered_text"]
