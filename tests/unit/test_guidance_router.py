# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance tests for Guidance CRUD and SchemaCore type system.

Covers:
  - Guidance CRUD and versioning
  - SchemaCore boundary wiring (per-rule validation is pinned in
    test_schema_core.py)
Plus: draft validation, active_guidance_id validation, JSON Schema round-trip.
"""

from __future__ import annotations

import json
import re
from unittest.mock import patch

from starlette.testclient import TestClient

from conftest import create_project_via_api

UUID4_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
ISO8601_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


# ── Helpers ─────────────────────────────────────────────────────────────────


def _create_project(client: TestClient, name: str = "Test") -> str:
    """Create a project and return its project_id."""
    return create_project_via_api(client, name)["project_id"]


def _valid_schema() -> list[dict]:
    """Minimal valid schema: one Core enum field."""
    return [
        {
            "field_name": "damage_type",
            "type": "enum",
            "role": "core",
            "allowed_values": ["crack", "dent"],
            "display_order": 1,
        }
    ]


def _create_guidance(
    client: TestClient,
    project_id: str,
    description: str = "Classify damage.",
    schema: list[dict] | None = None,
    rules: str = "",
) -> dict:
    """Create a Guidance version and return the response dict."""
    resp = client.post(
        f"/v1/projects/{project_id}/guidance",
        json={
            "description": description,
            "schema": schema or _valid_schema(),
            "rules": rules,
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


# ══════════════════════════════════════════════════════════════════════════════
# Guidance CRUD
# ══════════════════════════════════════════════════════════════════════════════


class TestGuidanceCrudImmutability:
    """POST creates immutable version; subsequent calls create new versions."""

    def test_create_returns_201(self, test_app_client: TestClient):
        pid = _create_project(test_app_client)
        data = _create_guidance(test_app_client, pid)
        assert data["version_number"] == 1

    def test_second_create_returns_version_2(self, test_app_client: TestClient):
        pid = _create_project(test_app_client)
        _create_guidance(test_app_client, pid)
        data = _create_guidance(test_app_client, pid, description="Updated task.")
        assert data["version_number"] == 2

    def test_create_does_not_modify_previous(self, test_app_client: TestClient):
        pid = _create_project(test_app_client)
        v1 = _create_guidance(test_app_client, pid, description="Original")
        _create_guidance(test_app_client, pid, description="Updated")
        # Re-read v1 — must still be "Original"
        resp = test_app_client.get(f"/v1/projects/{pid}/guidance/{v1['guidance_id']}")
        assert resp.status_code == 200
        assert resp.json()["description"] == "Original"

    def test_create_joins_project_write_queue(self, test_app_client: TestClient):
        """Guidance saves must not race background ingest/embedding DB writes."""
        from vlm_feedback_loop.services import guidance_service
        from vlm_feedback_loop.services.project_db_locks import (
            get_project_write_lock,
        )

        pid = _create_project(test_app_client)
        original = guidance_service.create_guidance
        lock_seen = False

        def assert_locked(**kwargs):
            nonlocal lock_seen
            lock_seen = get_project_write_lock(pid).locked()
            return original(**kwargs)

        with patch.object(
            guidance_service, "create_guidance", side_effect=assert_locked
        ):
            data = _create_guidance(test_app_client, pid)

        assert data["version_number"] == 1
        assert lock_seen


class TestVersionNumber:
    """version_number is 1-based, monotonic, backend-assigned, immutable."""

    def test_monotonically_increasing(self, test_app_client: TestClient):
        pid = _create_project(test_app_client)
        versions = []
        for i in range(5):
            data = _create_guidance(test_app_client, pid, description=f"v{i + 1}")
            versions.append(data["version_number"])
        assert versions == [1, 2, 3, 4, 5]


class TestGetGuidance:
    """GET returns all required fields."""

    def test_returns_all_required_fields(self, test_app_client: TestClient):
        pid = _create_project(test_app_client)
        created = _create_guidance(test_app_client, pid, rules="Be careful.")

        resp = test_app_client.get(
            f"/v1/projects/{pid}/guidance/{created['guidance_id']}"
        )
        assert resp.status_code == 200
        data = resp.json()

        # Required fields on the guidance record
        assert UUID4_RE.match(data["guidance_id"])
        assert data["project_id"] == pid
        assert data["version_number"] == 1
        assert data["description"] == "Classify damage."
        assert data["rules"] == "Be careful."
        assert ISO8601_RE.match(data["created_at"])

        # Schema fields include field_id per field
        for field in data["schema_fields"]:
            assert UUID4_RE.match(field["field_id"])
            assert field["field_name"]
            assert field["type"]
            assert field["role"]

        # Derived artifacts
        assert isinstance(data["derived_json_schema"], dict)
        assert isinstance(data["generation_order"], list)
        assert isinstance(data["schema_hash"], str)

    def test_nonexistent_returns_404(self, test_app_client: TestClient):
        pid = _create_project(test_app_client)
        resp = test_app_client.get(
            f"/v1/projects/{pid}/guidance/00000000-0000-4000-a000-000000000000"
        )
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Guidance not found"


class TestListGuidances:
    """LIST returns newest-first with cursor pagination."""

    def test_newest_first(self, test_app_client: TestClient):
        pid = _create_project(test_app_client)
        _create_guidance(test_app_client, pid, description="v1")
        _create_guidance(test_app_client, pid, description="v2")
        _create_guidance(test_app_client, pid, description="v3")

        resp = test_app_client.get(f"/v1/projects/{pid}/guidance")
        assert resp.status_code == 200
        data = resp.json()
        versions = [item["version_number"] for item in data["items"]]
        assert versions == [3, 2, 1]

    def test_cursor_pagination(self, test_app_client: TestClient):
        pid = _create_project(test_app_client)
        for i in range(5):
            _create_guidance(test_app_client, pid, description=f"v{i + 1}")

        # Page 1: limit=2
        resp = test_app_client.get(f"/v1/projects/{pid}/guidance?limit=2")
        data = resp.json()
        assert len(data["items"]) == 2
        assert data["items"][0]["version_number"] == 5
        assert data["items"][1]["version_number"] == 4
        assert data["next_cursor"] is not None

        # Page 2: use cursor
        resp = test_app_client.get(
            f"/v1/projects/{pid}/guidance?limit=2&cursor={data['next_cursor']}"
        )
        data2 = resp.json()
        assert len(data2["items"]) == 2
        assert data2["items"][0]["version_number"] == 3
        assert data2["items"][1]["version_number"] == 2

    def test_empty_project_returns_empty_list(self, test_app_client: TestClient):
        pid = _create_project(test_app_client)
        resp = test_app_client.get(f"/v1/projects/{pid}/guidance")
        data = resp.json()
        assert data["items"] == []
        assert data["next_cursor"] is None


class TestActiveGuidanceId:
    """PATCH project with active_guidance_id changes active version."""

    def test_set_active_guidance_id(self, test_app_client: TestClient):
        pid = _create_project(test_app_client)
        g = _create_guidance(test_app_client, pid)
        gid = g["guidance_id"]

        resp = test_app_client.patch(
            f"/v1/projects/{pid}",
            json={"active_guidance_id": gid},
        )
        assert resp.status_code == 200
        assert resp.json()["active_guidance_id"] == gid

    def test_does_not_modify_guidance_record(self, test_app_client: TestClient):
        pid = _create_project(test_app_client)
        g = _create_guidance(test_app_client, pid)
        gid = g["guidance_id"]

        test_app_client.patch(f"/v1/projects/{pid}", json={"active_guidance_id": gid})

        # Read guidance — unchanged
        resp = test_app_client.get(f"/v1/projects/{pid}/guidance/{gid}")
        assert resp.json()["description"] == g["description"]

    def test_nonexistent_guidance_rejected(self, test_app_client: TestClient):
        pid = _create_project(test_app_client)
        resp = test_app_client.patch(
            f"/v1/projects/{pid}",
            json={"active_guidance_id": "00000000-0000-4000-a000-000000000000"},
        )
        assert resp.status_code == 400
        assert "does not exist" in resp.json()["detail"]


class TestGuidanceProjectScoping:
    """Guidance records are project-scoped."""

    def test_versions_from_project_a_not_visible_in_project_b(
        self, test_app_client: TestClient
    ):
        pid_a = _create_project(test_app_client, "Project A")
        pid_b = _create_project(test_app_client, "Project B")

        g = _create_guidance(test_app_client, pid_a)

        # Not visible in project B
        resp = test_app_client.get(f"/v1/projects/{pid_b}/guidance/{g['guidance_id']}")
        assert resp.status_code == 404

        # Not in project B's list
        resp = test_app_client.get(f"/v1/projects/{pid_b}/guidance")
        assert len(resp.json()["items"]) == 0


class TestGuidanceDescriptionRules:
    """Description optional; empty rules allowed."""

    def test_empty_description_accepted(self, test_app_client: TestClient):
        """An empty Task Description saves fine — the schema is the only structural requirement."""
        pid = _create_project(test_app_client)
        resp = test_app_client.post(
            f"/v1/projects/{pid}/guidance",
            json={"description": "", "schema": _valid_schema(), "rules": ""},
        )
        assert resp.status_code == 201, resp.text
        data = resp.json()
        assert data["description"] == ""
        # Stored, not just echoed: re-read the version
        resp = test_app_client.get(f"/v1/projects/{pid}/guidance/{data['guidance_id']}")
        assert resp.status_code == 200
        assert resp.json()["description"] == ""

    def test_omitted_description_defaults_to_empty(self, test_app_client: TestClient):
        """Description omitted from the request body defaults to empty and saves fine."""
        pid = _create_project(test_app_client)
        resp = test_app_client.post(
            f"/v1/projects/{pid}/guidance",
            json={"schema": _valid_schema(), "rules": ""},
        )
        assert resp.status_code == 201, resp.text
        assert resp.json()["description"] == ""

    def test_empty_rules_allowed(self, test_app_client: TestClient):
        pid = _create_project(test_app_client)
        data = _create_guidance(test_app_client, pid, rules="")
        assert data["rules"] == ""


# ══════════════════════════════════════════════════════════════════════════════
# SchemaCore type system
# ══════════════════════════════════════════════════════════════════════════════


class TestSchemaFieldTypes:
    """Only 5 types accepted; unknown type rejected."""

    def test_all_five_types_accepted(self, test_app_client: TestClient):
        pid = _create_project(test_app_client)
        schema = [
            {
                "field_name": "cat",
                "type": "enum",
                "role": "core",
                "allowed_values": ["a", "b"],
                "display_order": 1,
            },
            {
                "field_name": "tags",
                "type": "enum_set",
                "role": "core",
                "allowed_values": ["x", "y"],
                "display_order": 2,
            },
            {
                "field_name": "flag",
                "type": "boolean",
                "role": "core",
                "display_order": 3,
            },
            {
                "field_name": "score",
                "type": "integer",
                "role": "core",
                "display_order": 4,
            },
            {
                "field_name": "desc",
                "type": "string",
                "role": "core",
                "display_order": 5,
            },
        ]
        data = _create_guidance(test_app_client, pid, schema=schema)
        assert data["version_number"] == 1

    def test_unknown_type_rejected(self, test_app_client: TestClient):
        pid = _create_project(test_app_client)
        # Pydantic Literal rejects unknown types with 422 at the request
        # model — schema_core's UNSUPPORTED_TYPE path is unreachable through
        # HTTP; this pins the API contract.
        resp = test_app_client.post(
            f"/v1/projects/{pid}/guidance",
            json={
                "description": "Test.",
                "schema": [
                    {
                        "field_name": "x",
                        "type": "float",
                        "role": "core",
                        "display_order": 1,
                    }
                ],
                "rules": "",
            },
        )
        assert resp.status_code == 422
        assert "float" in resp.text, "422 detail must point at the offending type value"


class TestBooleanNoConstraints:
    """Boolean has no additional constraints."""

    def test_boolean_accepted(self, test_app_client: TestClient):
        pid = _create_project(test_app_client)
        data = _create_guidance(
            test_app_client,
            pid,
            schema=[
                {
                    "field_name": "flag",
                    "type": "boolean",
                    "role": "core",
                    "display_order": 1,
                }
            ],
        )
        assert data["version_number"] == 1


class TestRationaleNoteReserved:
    """rationale_note is an optional reserved field."""

    def test_disabled_by_default_when_missing(self, test_app_client: TestClient):
        pid = _create_project(test_app_client)
        data = _create_guidance(test_app_client, pid)
        names = [f["field_name"] for f in data["schema_fields"]]
        assert "rationale_note" not in names
        assert "rationale_note" not in data["derived_json_schema"]["properties"]


class TestFieldIdGeneration:
    """Creation rejects client identity; responses expose backend identity."""

    def test_client_provided_field_id_rejected(self, test_app_client: TestClient):
        pid = _create_project(test_app_client)
        resp = test_app_client.post(
            f"/v1/projects/{pid}/guidance",
            json={
                "description": "Test.",
                "schema": [
                    {
                        "field_name": "cat",
                        "type": "enum",
                        "role": "core",
                        "allowed_values": ["a", "b"],
                        "display_order": 1,
                        "field_id": "client-id-should-be-ignored",
                    },
                ],
                "rules": "",
            },
        )
        assert resp.status_code == 422
        assert any(
            "field_id" in error.get("loc", []) for error in resp.json()["detail"]
        )


class TestNoCoreFields:
    """At least one Core field required. Representative service-validation
    rejection: pins that a schema_core failure surfaces as 400 with the
    issue code in the detail string (str result -> map_service_error).
    Per-rule validation coverage lives in test_schema_core.py."""

    def test_only_aux_fields_rejected(self, test_app_client: TestClient):
        pid = _create_project(test_app_client)
        resp = test_app_client.post(
            f"/v1/projects/{pid}/guidance",
            json={
                "description": "Test.",
                "schema": [
                    {
                        "field_name": "obs",
                        "type": "string",
                        "role": "aux",
                        "display_order": 1,
                    },
                ],
                "rules": "",
            },
        )
        assert resp.status_code == 400
        assert "NO_CORE_FIELDS" in resp.text


class TestDisplayOrder:
    """display_order controls within-group ordering."""

    def test_generation_order_respects_display_order(self, test_app_client: TestClient):
        pid = _create_project(test_app_client)
        schema = [
            {
                "field_name": "z_core",
                "type": "boolean",
                "role": "core",
                "display_order": 10,
            },
            {
                "field_name": "a_core",
                "type": "boolean",
                "role": "core",
                "display_order": 5,
            },
            {
                "field_name": "z_aux",
                "type": "string",
                "role": "aux",
                "display_order": 3,
            },
            {
                "field_name": "a_aux",
                "type": "string",
                "role": "aux",
                "display_order": 1,
            },
        ]
        data = _create_guidance(test_app_client, pid, schema=schema)
        order = data["generation_order"]

        # Aux sorted by display_order
        aux_section = order[:2]
        assert aux_section == ["a_aux", "z_aux"]
        # Core sorted by display_order
        core_section = order[2:]
        assert core_section == ["a_core", "z_core"]


class TestRationaleNoteDisplayOrder:
    """rationale_note has lowest display_order (system-enforced)."""

    def test_rationale_note_always_first_in_generation_order(
        self, test_app_client: TestClient
    ):
        pid = _create_project(test_app_client)
        data = _create_guidance(
            test_app_client,
            pid,
            schema=[
                {
                    "field_name": "rationale_note",
                    "type": "string",
                    "role": "aux",
                    "display_order": 99,
                },
                *_valid_schema(),
            ],
        )
        assert data["generation_order"][0] == "rationale_note"

    def test_rationale_note_display_order_lower_than_all_others(
        self, test_app_client: TestClient
    ):
        pid = _create_project(test_app_client)
        data = _create_guidance(
            test_app_client,
            pid,
            schema=[
                {
                    "field_name": "rationale_note",
                    "type": "string",
                    "role": "aux",
                    "display_order": 99,
                },
                *_valid_schema(),
            ],
        )
        rn = [f for f in data["schema_fields"] if f["field_name"] == "rationale_note"][
            0
        ]
        others = [
            f for f in data["schema_fields"] if f["field_name"] != "rationale_note"
        ]
        for f in others:
            assert rn["display_order"] < f["display_order"]


# ══════════════════════════════════════════════════════════════════════════════
# Draft Validation
# ══════════════════════════════════════════════════════════════════════════════


class TestDraftValidation:
    def test_returns_issues_and_derived_schema(self, test_app_client: TestClient):
        pid = _create_project(test_app_client)
        resp = test_app_client.post(
            f"/v1/projects/{pid}/guidance:validate_draft",
            json={
                "description": "Test.",
                "schema": _valid_schema(),
                "rules": "",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["save_allowed"] is True
        assert isinstance(data["derived_json_schema"], dict)
        assert isinstance(data["schema_hash"], str)
        assert data["issues"] == []

    def test_save_not_allowed_when_errors(self, test_app_client: TestClient):
        """A draft with no Core fields is blocked by NO_CORE_FIELDS — the sole structural save requirement."""
        pid = _create_project(test_app_client)
        resp = test_app_client.post(
            f"/v1/projects/{pid}/guidance:validate_draft",
            json={
                "description": "Test.",
                "schema": [],
                "rules": "",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["save_allowed"] is False
        assert "NO_CORE_FIELDS" in [issue["code"] for issue in data["issues"]]

    def test_empty_description_draft_save_allowed(self, test_app_client: TestClient):
        """An empty description with one valid Core field validates clean — description is optional."""
        pid = _create_project(test_app_client)
        resp = test_app_client.post(
            f"/v1/projects/{pid}/guidance:validate_draft",
            json={
                "description": "",
                "schema": _valid_schema(),
                "rules": "",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["save_allowed"] is True
        assert data["issues"] == []

    def test_draft_and_save_use_same_derivation(self, test_app_client: TestClient):
        """The same validate_and_derive function serves both paths."""
        pid = _create_project(test_app_client)
        schema = _valid_schema()

        # Draft
        draft_resp = test_app_client.post(
            f"/v1/projects/{pid}/guidance:validate_draft",
            json={"description": "Test.", "schema": schema, "rules": ""},
        )
        draft = draft_resp.json()

        # Save
        save_resp = test_app_client.post(
            f"/v1/projects/{pid}/guidance",
            json={"description": "Test.", "schema": schema, "rules": ""},
        )
        saved = save_resp.json()

        # schema_hash must be identical (proves same derivation)
        assert draft["schema_hash"] == saved["schema_hash"]

    def test_schema_hash_deterministic(self, test_app_client: TestClient):
        pid = _create_project(test_app_client)
        schema = _valid_schema()

        r1 = test_app_client.post(
            f"/v1/projects/{pid}/guidance:validate_draft",
            json={"description": "Test.", "schema": schema, "rules": ""},
        )
        r2 = test_app_client.post(
            f"/v1/projects/{pid}/guidance:validate_draft",
            json={"description": "Test.", "schema": schema, "rules": ""},
        )
        assert r1.json()["schema_hash"] == r2.json()["schema_hash"]

    def test_schema_compile_failure_reported(self, test_app_client: TestClient):
        """SCHEMA_COMPILE_FAILURE detected on internally inconsistent schema."""
        pid = _create_project(test_app_client)
        with patch(
            "vlm_feedback_loop.services.schema_core._derive_json_schema",
            side_effect=ValueError("synthetic compile failure"),
        ):
            resp = test_app_client.post(
                f"/v1/projects/{pid}/guidance:validate_draft",
                json={"description": "Test.", "schema": _valid_schema(), "rules": ""},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["save_allowed"] is False
        codes = [i["code"] for i in data["issues"]]
        assert "SCHEMA_COMPILE_FAILURE" in codes

    def test_project_not_found_returns_404(self, test_app_client: TestClient):
        """Pins this router's project-not-found wiring only (the
        validate_draft handler's None sentinel → 404); the shared
        service-error→status mapping rule is pinned in
        test_error_mapping.py."""
        resp = test_app_client.post(
            "/v1/projects/00000000-0000-4000-a000-000000000000/guidance:validate_draft",
            json={"description": "Test.", "schema": _valid_schema(), "rules": ""},
        )
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Project not found"


# ══════════════════════════════════════════════════════════════════════════════
# JSON Schema Round-Trip (Byte Identity)
# ══════════════════════════════════════════════════════════════════════════════


class TestJsonSchemaRoundTrip:
    """Derived JSON Schema must survive SQLite round-trip with byte identity."""

    def test_derived_schema_round_trips_through_sqlite(
        self, test_app_client: TestClient
    ):
        pid = _create_project(test_app_client)
        schema = [
            {
                "field_name": "cat",
                "type": "enum",
                "role": "core",
                "allowed_values": ["crack", "dent", "scratch"],
                "display_order": 1,
            },
            {
                "field_name": "severity",
                "type": "integer",
                "role": "core",
                "minimum": 0,
                "maximum": 5,
                "display_order": 2,
            },
            {"field_name": "obs", "type": "string", "role": "aux", "display_order": 0},
        ]

        # Create
        created = _create_guidance(test_app_client, pid, schema=schema)

        # Read back
        resp = test_app_client.get(
            f"/v1/projects/{pid}/guidance/{created['guidance_id']}"
        )
        retrieved = resp.json()

        # Byte-identical comparison via canonical JSON serialization
        created_json = json.dumps(
            created["derived_json_schema"], sort_keys=True, separators=(",", ":")
        )
        retrieved_json = json.dumps(
            retrieved["derived_json_schema"], sort_keys=True, separators=(",", ":")
        )
        assert created_json == retrieved_json

    def test_properties_order_preserved(self, test_app_client: TestClient):
        """Property ordering (generation_order) must survive the round-trip."""
        pid = _create_project(test_app_client)
        schema = [
            {"field_name": "obs", "type": "string", "role": "aux", "display_order": 0},
            {
                "field_name": "cat",
                "type": "enum",
                "role": "core",
                "allowed_values": ["a", "b"],
                "display_order": 1,
            },
            {
                "field_name": "flag",
                "type": "boolean",
                "role": "core",
                "display_order": 2,
            },
        ]
        created = _create_guidance(test_app_client, pid, schema=schema)

        resp = test_app_client.get(
            f"/v1/projects/{pid}/guidance/{created['guidance_id']}"
        )
        retrieved = resp.json()

        # generation_order must match
        assert created["generation_order"] == retrieved["generation_order"]

        # x-generation-order in derived schema must match
        assert (
            created["derived_json_schema"]["x-generation-order"]
            == retrieved["derived_json_schema"]["x-generation-order"]
        )


# ══════════════════════════════════════════════════════════════════════════════
# ICL count
# ══════════════════════════════════════════════════════════════════════════════


class TestIclCount:
    """GET /v1/projects/{id}/guidance:icl_count returns the non-pool Edit count."""

    def test_returns_zero_when_no_labels(self, test_app_client: TestClient):
        pid = _create_project(test_app_client)
        _create_guidance(test_app_client, pid)

        resp = test_app_client.get(f"/v1/projects/{pid}/guidance:icl_count")
        assert resp.status_code == 200
        assert resp.json() == {"eligible_count": 0}

    def test_returns_404_for_missing_project(self, test_app_client: TestClient):
        resp = test_app_client.get("/v1/projects/nonexistent-pid/guidance:icl_count")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Project not found"
