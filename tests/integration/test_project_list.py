# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Live round-trip acceptance test for the Project List screen's API surface.

Coverage map for the screen's behaviors:
  - Card rendering (name, description, counts, dates), empty-state copy,
    null-description line omitted, pending_relabel not displayed,
    last-active-screen in localStorage, create-dialog cancel, lock-dialog
    copy: vitest (src/ui/src/pages/__tests__/ProjectListPage.test.tsx).
  - Empty name -> 422 naming the missing field, through the proxy:
    tests/integration/test_frontend_shell.py (TestDevProxy) and at unit
    level in tests/unit/test_projects.py.
  - Zero counts incl. pending_relabel, project-locked 409 handler: unit
    (tests/unit/test_projects.py).
  - This file: one live round trip through the Vite proxy — create (with
    and without description) -> list envelope -> card fields -> detail —
    pinning the API shape the cards render from against real servers.

Server fixtures (backend_server, frontend_server) are provided by
tests/integration/conftest.py.
"""

from __future__ import annotations

import re

import httpx

FRONTEND_URL = "http://127.0.0.1:5173"

ISO_8601_UTC_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


def _create_project(name: str, description: str | None = None) -> dict:
    """Create a project through the proxy and return the full response body."""
    body: dict = {"name": name}
    if description is not None:
        body["description"] = description
    resp = httpx.post(f"{FRONTEND_URL}/v1/projects", json=body, timeout=10)
    assert resp.status_code == 201, f"Create failed: {resp.text}"
    return resp.json()


class TestProjectListRoundTrip:
    """Create -> list -> detail through the Vite proxy returns the card data."""

    def test_create_list_detail_round_trip(self, frontend_server):  # noqa: ARG002
        """The list envelope carries everything a project card renders from."""
        with_desc = _create_project("Round Trip Test", "Test description")
        no_desc = _create_project("Round Trip No Desc")

        # Create response: 201 asserted in the helper; id, echoed inputs,
        # UTC ISO-8601 Z timestamps.
        assert "project_id" in with_desc
        assert with_desc["name"] == "Round Trip Test"
        assert no_desc["description"] is None
        for stamp in (with_desc["created_at"], with_desc["updated_at"]):
            assert ISO_8601_UTC_PATTERN.match(stamp), stamp

        # List envelope through the proxy: items + cursor + the
        # workspace-global archived flag the UI reads for "Show archived".
        resp = httpx.get(f"{FRONTEND_URL}/v1/projects", timeout=5)
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data["items"], list)
        assert "next_cursor" in data
        assert isinstance(data["has_archived"], bool)

        (item,) = [
            i for i in data["items"] if i["project_id"] == with_desc["project_id"]
        ]
        assert item["name"] == "Round Trip Test"
        assert item["description"] == "Test description"
        # Just-created project: every count at zero (pending_relabel is in
        # the payload even though the card does not display it).
        counts = item["counts"]
        for key in (
            "verified",
            "unlabeled",
            "auto_labeled",
            "omitted",
            "pending_relabel",
        ):
            assert counts[key] == 0, (key, counts)

        # Card click target: the detail endpoint serves the same project.
        detail = httpx.get(
            f"{FRONTEND_URL}/v1/projects/{with_desc['project_id']}", timeout=5
        )
        assert detail.status_code == 200
        assert detail.json()["project_id"] == with_desc["project_id"]
