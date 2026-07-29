# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the path remapping endpoint."""

from __future__ import annotations

from pathlib import Path

from starlette.testclient import TestClient

from conftest import create_project_via_api, make_api_client, make_test_image

# ── Helpers ─────────────────────────────────────────────────────────────────


def _ingest(client: TestClient, pid: str, examples: list[dict]) -> dict:
    # :ingest returns 202 (the pHash sweeper runs in the
    # background). Skeleton rows are created with phash=null and
    # remapping still works correctly against them.
    resp = client.post(
        f"/v1/projects/{pid}/examples:ingest",
        json={"examples": examples},
    )
    assert resp.status_code == 202, resp.text
    return resp.json()


def _remap(
    client: TestClient, pid: str, old_prefix: str, new_prefix: str, dry_run: bool = True
) -> dict:
    resp = client.post(
        f"/v1/projects/{pid}/examples:remap_paths",
        json={"old_prefix": old_prefix, "new_prefix": new_prefix, "dry_run": dry_run},
    )
    return resp.json(), resp.status_code


# ═══════════════════════════════════════════════════════════════════════════
# Storage path remapping
# ═══════════════════════════════════════════════════════════════════════════


class TestRemapDryRun:
    """Verify: dry_run=true returns preview without modifying records."""

    def test_dry_run_returns_preview(self, tmp_path: Path):
        old_dir = tmp_path / "old_nas" / "images"
        old_dir.mkdir(parents=True)
        img = make_test_image(old_dir / "img.jpg")

        client = make_api_client(tmp_path)
        project = create_project_via_api(client)
        pid = project["project_id"]

        _ingest(client, pid, [{"example_key": "k1", "storage_ref": str(img)}])

        body, status = _remap(
            client,
            pid,
            old_prefix=str(tmp_path / "old_nas"),
            new_prefix=str(tmp_path / "new_nas"),
            dry_run=True,
        )

        assert status == 200
        assert body["dry_run"] is True
        assert body["matched_count"] == 1
        assert len(body["sample_remappings"]) == 1
        assert body["sample_remappings"][0]["old_storage_ref"] == str(img)
        assert "new_nas" in body["sample_remappings"][0]["new_storage_ref"]
        assert body["validation"]["sample_checked"] == 1

    def test_dry_run_does_not_modify_records(self, tmp_path: Path):
        old_dir = tmp_path / "old_nas" / "images"
        img = make_test_image(old_dir / "img.jpg")

        client = make_api_client(tmp_path)
        project = create_project_via_api(client)
        pid = project["project_id"]

        _ingest(client, pid, [{"example_key": "k1", "storage_ref": str(img)}])

        _remap(
            client,
            pid,
            str(tmp_path / "old_nas"),
            str(tmp_path / "new_nas"),
            dry_run=True,
        )

        # Query the example — storage_ref should be unchanged
        resp = client.get(f"/v1/projects/{pid}/examples")
        items = resp.json()["items"]
        assert items[0]["example"]["storage_ref"] == str(img)


class TestRemapCommit:
    """Verify: commit replaces prefixes; non-matching unaffected."""

    def test_commit_replaces_prefix(self, tmp_path: Path):
        old_dir = tmp_path / "old_nas" / "images"
        new_dir = tmp_path / "new_nas" / "images"
        img = make_test_image(old_dir / "img.jpg")
        # Create file at new location too
        make_test_image(new_dir / "img.jpg")

        client = make_api_client(tmp_path)
        project = create_project_via_api(client)
        pid = project["project_id"]

        _ingest(client, pid, [{"example_key": "k1", "storage_ref": str(img)}])

        body, status = _remap(
            client,
            pid,
            old_prefix=str(tmp_path / "old_nas"),
            new_prefix=str(tmp_path / "new_nas"),
            dry_run=False,
        )

        assert status == 200
        assert body["dry_run"] is False
        assert body["remapped_count"] == 1

        # Verify storage_ref changed
        resp = client.get(f"/v1/projects/{pid}/examples")
        new_ref = resp.json()["items"][0]["example"]["storage_ref"]
        assert new_ref == str(new_dir / "img.jpg")

    def test_nonmatching_unaffected(self, tmp_path: Path):
        dir_a = tmp_path / "dir_a"
        dir_b = tmp_path / "dir_b"
        img_a = make_test_image(dir_a / "img.jpg")
        img_b = make_test_image(dir_b / "img.jpg")

        # Create file at new location for dir_a
        new_a = tmp_path / "new_a"
        make_test_image(new_a / "img.jpg")

        client = make_api_client(tmp_path)
        project = create_project_via_api(client)
        pid = project["project_id"]

        _ingest(
            client,
            pid,
            [
                {"example_key": "ka", "storage_ref": str(img_a)},
                {"example_key": "kb", "storage_ref": str(img_b)},
            ],
        )

        _remap(client, pid, str(dir_a), str(new_a), dry_run=False)

        resp = client.get(f"/v1/projects/{pid}/examples")
        items = {
            i["example"]["example_key"]: i["example"]["storage_ref"]
            for i in resp.json()["items"]
        }
        assert "new_a" in items["ka"]
        assert items["kb"] == str(img_b)  # unchanged

    def test_prefix_matches_byte_literally(self, tmp_path: Path):
        """The remap prefix must match byte-literally. SQL LIKE — which a
        column .startswith compiles to — treats '_' as a single-character
        wildcard and folds ASCII case, so remapping /data/my_data also
        matched /data/myXdata (wildcard) and /data/My_Data (case fold),
        splicing the wrong tail onto unrelated rows on a case-sensitive
        filesystem. Both bystander classes must be untouched."""
        dir_target = tmp_path / "my_data"
        dir_wildcard = tmp_path / "myXdata"
        dir_casefold = tmp_path / "My_Data"
        img_t = make_test_image(dir_target / "img.jpg")
        img_w = make_test_image(dir_wildcard / "img.jpg")
        img_c = make_test_image(dir_casefold / "img.jpg")
        new_dir = tmp_path / "relocated"
        make_test_image(new_dir / "img.jpg")

        client = make_api_client(tmp_path)
        project = create_project_via_api(client)
        pid = project["project_id"]

        _ingest(
            client,
            pid,
            [
                {"example_key": "kt", "storage_ref": str(img_t)},
                {"example_key": "kw", "storage_ref": str(img_w)},
                {"example_key": "kc", "storage_ref": str(img_c)},
            ],
        )

        body, status = _remap(client, pid, str(dir_target), str(new_dir), dry_run=False)
        assert status == 200
        assert body["remapped_count"] == 1

        resp = client.get(f"/v1/projects/{pid}/examples")
        items = {
            i["example"]["example_key"]: i["example"]["storage_ref"]
            for i in resp.json()["items"]
        }
        assert items["kt"] == str(new_dir / "img.jpg")
        assert items["kw"] == str(img_w)  # wildcard near-miss untouched
        assert items["kc"] == str(img_c)  # case-fold near-miss untouched


class TestRemapZeroResolveRejection:
    """Verify: commit rejected when zero sampled paths resolve."""

    def test_zero_resolve_returns_400(self, tmp_path: Path):
        old_dir = tmp_path / "old_nas" / "images"
        img = make_test_image(old_dir / "img.jpg")

        client = make_api_client(tmp_path)
        project = create_project_via_api(client)
        pid = project["project_id"]

        _ingest(client, pid, [{"example_key": "k1", "storage_ref": str(img)}])

        # No files exist at new_prefix — commit should fail
        body, status = _remap(
            client,
            pid,
            old_prefix=str(tmp_path / "old_nas"),
            new_prefix=str(tmp_path / "nonexistent_nas"),
            dry_run=False,
        )

        assert status == 400
        assert "resolve" in body["detail"].lower()


class TestRemapAuditEvent:
    """Verify: commit creates AuditEvent with storage_ref_remap."""

    def test_audit_event_created(self, tmp_path: Path):
        old_dir = tmp_path / "old_nas" / "images"
        new_dir = tmp_path / "new_nas" / "images"
        img = make_test_image(old_dir / "img.jpg")
        make_test_image(new_dir / "img.jpg")

        client = make_api_client(tmp_path)
        project = create_project_via_api(client)
        pid = project["project_id"]

        _ingest(client, pid, [{"example_key": "k1", "storage_ref": str(img)}])

        body, status = _remap(
            client,
            pid,
            old_prefix=str(tmp_path / "old_nas"),
            new_prefix=str(tmp_path / "new_nas"),
            dry_run=False,
        )

        assert status == 200
        assert body["audit_event_id"] is not None

        # Verify by querying the DB directly
        from sqlalchemy import select
        from sqlalchemy.orm import Session

        from vlm_feedback_loop.db.models.audit_event import AuditEvent
        from vlm_feedback_loop.services.project_service import get_project_engine

        engine = get_project_engine(pid, str(tmp_path / "workspace"))
        with Session(engine) as session:
            audit = session.execute(
                select(AuditEvent).where(
                    AuditEvent.project_id == pid,
                    AuditEvent.event_type == "storage_ref_remap",
                )
            ).scalar_one_or_none()

            assert audit is not None
            assert audit.event_data["old_prefix"] == str(tmp_path / "old_nas")
            assert audit.event_data["new_prefix"] == str(tmp_path / "new_nas")
            assert audit.event_data["remapped_count"] == 1


class TestRemapAbsolutePaths:
    """Verify: both prefixes must be non-empty absolute paths."""

    def test_relative_old_prefix_rejected(self, tmp_path: Path):
        client = make_api_client(tmp_path)
        project = create_project_via_api(client)
        pid = project["project_id"]

        body, status = _remap(client, pid, "relative/path", "/new/path", dry_run=True)
        assert status == 400

    def test_relative_new_prefix_rejected(self, tmp_path: Path):
        client = make_api_client(tmp_path)
        project = create_project_via_api(client)
        pid = project["project_id"]

        body, status = _remap(client, pid, "/old/path", "relative/path", dry_run=True)
        assert status == 400

    def test_empty_old_prefix_rejected(self, tmp_path: Path):
        client = make_api_client(tmp_path)
        project = create_project_via_api(client)
        pid = project["project_id"]

        body, status = _remap(client, pid, "", "/new/path", dry_run=True)
        assert status == 400

    def test_empty_new_prefix_rejected(self, tmp_path: Path):
        client = make_api_client(tmp_path)
        project = create_project_via_api(client)
        pid = project["project_id"]

        body, status = _remap(client, pid, "/old/path", "", dry_run=True)
        assert status == 400


class TestRemapProjectNotFound:
    """Verify: 404 for nonexistent project."""

    def test_remap_project_not_found(self, tmp_path: Path):
        client = make_api_client(tmp_path)
        resp = client.post(
            "/v1/projects/nonexistent/examples:remap_paths",
            json={"old_prefix": "/old", "new_prefix": "/new"},
        )
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Project not found"
