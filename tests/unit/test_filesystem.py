# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for filesystem browse and scan endpoints.

Covers: sorted entries, hidden excluded, image filter, 404/403,
IMAGE_ROOT scenarios (3), symlinks, deployment-scoped,
deterministic key, collision checking statuses (3), skipped files.
"""

from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath

import pytest
from starlette.testclient import TestClient

from conftest import make_api_client, make_settings

# ── Helpers ─────────────────────────────────────────────────────────────────


def _create_image_tree(base: Path) -> Path:
    """Create a realistic directory tree with images and non-images."""
    img_dir = base / "images"
    img_dir.mkdir(parents=True)

    # Supported image files
    (img_dir / "photo_001.jpg").write_bytes(b"\xff\xd8" + b"\x00" * 100)
    (img_dir / "photo_002.png").write_bytes(b"\x89PNG" + b"\x00" * 100)
    (img_dir / "photo_003.webp").write_bytes(b"RIFF" + b"\x00" * 100)
    (img_dir / "photo_004.bmp").write_bytes(b"BM" + b"\x00" * 100)
    (img_dir / "photo_005.tiff").write_bytes(b"II" + b"\x00" * 100)

    # Subdirectory with more images
    sub = img_dir / "batch_01"
    sub.mkdir()
    (sub / "img_a.jpg").write_bytes(b"\xff\xd8" + b"\x00" * 50)
    (sub / "img_b.png").write_bytes(b"\x89PNG" + b"\x00" * 50)

    # Non-image files
    (img_dir / "notes.txt").write_text("some notes")
    (img_dir / "data.csv").write_text("a,b,c")

    # Hidden files and directory
    (img_dir / ".hidden_file.jpg").write_bytes(b"\xff\xd8" + b"\x00" * 10)
    hidden_dir = img_dir / ".hidden_dir"
    hidden_dir.mkdir()
    (hidden_dir / "secret.jpg").write_bytes(b"\xff\xd8" + b"\x00" * 10)

    return img_dir


def _expected_key(relative_path: str) -> str:
    """Recompute the expected example key for a given relative path."""
    import re

    posix = PurePosixPath(relative_path)
    stem_path = str(posix.with_suffix(""))
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", stem_path).strip("_")
    hash12 = hashlib.sha256(str(posix).encode("utf-8")).hexdigest()[:12]
    return f"{slug}--{hash12}"


# ── Browse: sorted entries ──────────────────────────────────────────────────


class TestBrowseSortedEntries:
    """Verify: entries sorted directories-first alphabetical, then files alphabetical."""

    def test_sorted_dirs_first_then_files(self, tmp_path: Path):
        img_dir = _create_image_tree(tmp_path)
        settings = make_settings(tmp_path)

        from vlm_feedback_loop.services.filesystem_service import browse_directory

        result = browse_directory(str(img_dir), settings, image_formats_only=False)
        names = [e["name"] for e in result["entries"]]

        # Directories should come first
        dirs = [e["name"] for e in result["entries"] if e["type"] == "directory"]
        files = [e["name"] for e in result["entries"] if e["type"] == "file"]

        assert dirs == sorted(dirs), "Directories not alphabetically sorted"
        assert files == sorted(files), "Files not alphabetically sorted"

        # Verify all dirs come before all files in the flat list
        dir_indices = [names.index(d) for d in dirs]
        file_indices = [names.index(f) for f in files]
        if dir_indices and file_indices:
            assert max(dir_indices) < min(file_indices), (
                "Directories should come before files"
            )

    def test_via_endpoint(self, tmp_path: Path):
        img_dir = _create_image_tree(tmp_path)
        client = make_api_client(tmp_path)

        resp = client.get(
            "/v1/filesystem/browse",
            params={"path": str(img_dir), "image_formats_only": "false"},
        )
        assert resp.status_code == 200
        entries = resp.json()["entries"]

        types_order = [e["type"] for e in entries]
        dir_end = 0
        for i, t in enumerate(types_order):
            if t == "directory":
                dir_end = i
        for i, t in enumerate(types_order):
            if t == "file":
                assert i > dir_end or not any(
                    tt == "directory" for tt in types_order[i:]
                )


# ── Browse: hidden excluded ─────────────────────────────────────────────────


class TestBrowseHiddenExcluded:
    """Verify: hidden files and directories (starting with .) are excluded."""

    def test_hidden_files_excluded(self, tmp_path: Path):
        img_dir = _create_image_tree(tmp_path)
        settings = make_settings(tmp_path)

        from vlm_feedback_loop.services.filesystem_service import browse_directory

        result = browse_directory(str(img_dir), settings, image_formats_only=False)
        names = [e["name"] for e in result["entries"]]

        assert ".hidden_file.jpg" not in names
        assert ".hidden_dir" not in names


# ── Browse: image format filter ─────────────────────────────────────────────


class TestBrowseImageFilter:
    """Verify: image_formats_only=true filters to JPEG, PNG, WebP, BMP, TIFF."""

    def test_image_filter_excludes_non_images(self, tmp_path: Path):
        img_dir = _create_image_tree(tmp_path)
        settings = make_settings(tmp_path)

        from vlm_feedback_loop.services.filesystem_service import browse_directory

        result = browse_directory(str(img_dir), settings, image_formats_only=True)
        files = [e for e in result["entries"] if e["type"] == "file"]
        file_names = [e["name"] for e in files]

        assert "notes.txt" not in file_names
        assert "data.csv" not in file_names
        # Image files should be present
        assert "photo_001.jpg" in file_names
        assert "photo_002.png" in file_names
        assert "photo_003.webp" in file_names
        assert "photo_004.bmp" in file_names
        assert "photo_005.tiff" in file_names

    def test_image_filter_off_shows_all(self, tmp_path: Path):
        img_dir = _create_image_tree(tmp_path)
        settings = make_settings(tmp_path)

        from vlm_feedback_loop.services.filesystem_service import browse_directory

        result = browse_directory(str(img_dir), settings, image_formats_only=False)
        files = [e for e in result["entries"] if e["type"] == "file"]
        file_names = [e["name"] for e in files]

        assert "notes.txt" in file_names
        assert "data.csv" in file_names

    def test_show_files_false_omits_files(self, tmp_path: Path):
        img_dir = _create_image_tree(tmp_path)
        settings = make_settings(tmp_path)

        from vlm_feedback_loop.services.filesystem_service import browse_directory

        result = browse_directory(str(img_dir), settings, show_files=False)
        assert all(e["type"] == "directory" for e in result["entries"])


# ── Browse: 404 and 403 ────────────────────────────────────────────────────


class TestBrowse404And403:
    """Verify: path not found → 404, permission denied → 403."""

    def test_path_not_found(self, tmp_path: Path):
        client = make_api_client(tmp_path)
        resp = client.get(
            "/v1/filesystem/browse",
            params={"path": str(tmp_path / "nonexistent")},
        )
        assert resp.status_code == 404
        assert "Directory not found" in resp.json()["detail"]

    def test_not_a_directory(self, tmp_path: Path):
        f = tmp_path / "file.txt"
        f.write_text("data")
        client = make_api_client(tmp_path)
        resp = client.get("/v1/filesystem/browse", params={"path": str(f)})
        assert resp.status_code == 404
        assert "Not a directory" in resp.json()["detail"]

    def test_relative_path_rejected(self, tmp_path: Path):
        client = make_api_client(tmp_path)
        resp = client.get("/v1/filesystem/browse", params={"path": "relative/path"})
        assert resp.status_code == 404
        assert "absolute" in resp.json()["detail"].lower()


# ── Browse: IMAGE_ROOT 3 scenarios ─────────────────────────────────────────


class TestBrowseRootsSecurity:
    """Verify IMAGE_ROOT behavior across bind-address scenarios.

    Scenario 1: loopback + no root → unrestricted.
    Scenario 2: non-loopback + no root → 403 with configuration guidance.
    Scenario 3: a configured root exposes only its directory tree.
    """

    def test_loopback_without_root_unrestricted(self, tmp_path: Path):
        """Loopback bind + no root lets a caller browse any absolute path."""
        img_dir = tmp_path / "anywhere"
        img_dir.mkdir()
        (img_dir / "test.jpg").write_bytes(b"\xff\xd8" + b"\x00" * 10)

        client = make_api_client(
            tmp_path,
            BIND_HOST="127.0.0.1",
            IMAGE_ROOT=None,
        )
        resp = client.get("/v1/filesystem/browse", params={"path": str(img_dir)})
        assert resp.status_code == 200
        assert any(e["name"] == "test.jpg" for e in resp.json()["entries"])

    def test_nonloopback_without_root_disabled(self, tmp_path: Path):
        """Non-loopback bind + no root returns actionable configuration guidance."""
        img_dir = tmp_path / "data"
        img_dir.mkdir()

        client = make_api_client(
            tmp_path,
            BIND_HOST="0.0.0.0",
            IMAGE_ROOT=None,
        )
        resp = client.get("/v1/filesystem/browse", params={"path": str(img_dir)})
        assert resp.status_code == 403
        detail = resp.json()["detail"]
        assert "Filesystem browsing is disabled" in detail
        assert "IMAGE_ROOT" in detail

    def test_configured_root_allows_inside(self, tmp_path: Path):
        """A configured root allows paths inside it."""
        allowed = tmp_path / "allowed"
        allowed.mkdir()
        (allowed / "img.jpg").write_bytes(b"\xff\xd8" + b"\x00" * 10)

        client = make_api_client(
            tmp_path,
            BIND_HOST="0.0.0.0",
            IMAGE_ROOT=str(allowed),
        )
        resp = client.get("/v1/filesystem/browse", params={"path": str(allowed)})
        assert resp.status_code == 200
        assert any(e["name"] == "img.jpg" for e in resp.json()["entries"])

    def test_omitted_path_opens_configured_root(self, tmp_path: Path):
        """The picker can start without knowing a deployment-specific path."""
        image_root = tmp_path / "images"
        image_root.mkdir()
        (image_root / "img.jpg").write_bytes(b"\xff\xd8" + b"\x00" * 10)

        client = make_api_client(
            tmp_path,
            BIND_HOST="0.0.0.0",
            IMAGE_ROOT=str(image_root),
        )
        resp = client.get("/v1/filesystem/browse")

        assert resp.status_code == 200
        assert resp.json()["path"] == str(image_root)
        assert resp.json()["parent"] is None
        assert [entry["name"] for entry in resp.json()["entries"]] == ["img.jpg"]

    def test_configured_root_rejects_outside(self, tmp_path: Path):
        """A configured root rejects paths outside it."""
        allowed = tmp_path / "allowed"
        allowed.mkdir()
        forbidden = tmp_path / "forbidden"
        forbidden.mkdir()

        client = make_api_client(
            tmp_path,
            BIND_HOST="0.0.0.0",
            IMAGE_ROOT=str(allowed),
        )
        resp = client.get("/v1/filesystem/browse", params={"path": str(forbidden)})
        assert resp.status_code == 403
        assert "outside" in resp.json()["detail"].lower()


# ── Browse: symlink rejection ───────────────────────────────────────────────


class TestBrowseSymlinkRejection:
    """Verify: symlinks pointing outside IMAGE_ROOT are rejected."""

    def test_symlink_escaping_root_excluded(self, tmp_path: Path):
        allowed = tmp_path / "allowed"
        allowed.mkdir()

        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secret.jpg").write_bytes(b"\xff\xd8" + b"\x00" * 10)

        # Create a symlink inside allowed that points outside
        link = allowed / "escape_link"
        try:
            link.symlink_to(outside)
        except OSError:
            pytest.skip("Cannot create symlinks on this filesystem")

        settings = make_settings(
            tmp_path,
            BIND_HOST="0.0.0.0",
            IMAGE_ROOT=str(allowed),
        )

        from vlm_feedback_loop.services.filesystem_service import browse_directory

        result = browse_directory(str(allowed), settings)
        names = [e["name"] for e in result["entries"]]
        assert "escape_link" not in names

    def test_symlink_within_root_allowed(self, tmp_path: Path):
        allowed = tmp_path / "allowed"
        allowed.mkdir()
        sub = allowed / "sub"
        sub.mkdir()
        (sub / "img.jpg").write_bytes(b"\xff\xd8" + b"\x00" * 10)

        link = allowed / "link_to_sub"
        try:
            link.symlink_to(sub)
        except OSError:
            pytest.skip("Cannot create symlinks on this filesystem")

        settings = make_settings(
            tmp_path,
            BIND_HOST="0.0.0.0",
            IMAGE_ROOT=str(allowed),
        )

        from vlm_feedback_loop.services.filesystem_service import browse_directory

        result = browse_directory(str(allowed), settings)
        names = [e["name"] for e in result["entries"]]
        assert "link_to_sub" in names


# ── Browse: deployment-scoped ───────────────────────────────────────────────


class TestBrowseDeploymentScoped:
    """Verify: browse is deployment-scoped (no project_id in path)."""

    def test_no_project_id_in_path(self, tmp_path: Path):
        img_dir = tmp_path / "data"
        img_dir.mkdir()
        (img_dir / "test.jpg").write_bytes(b"\xff\xd8" + b"\x00" * 10)

        client = make_api_client(tmp_path)
        # The endpoint is /v1/filesystem/browse, NOT /v1/projects/{id}/filesystem/browse
        resp = client.get("/v1/filesystem/browse", params={"path": str(img_dir)})
        assert resp.status_code == 200
        assert any(e["name"] == "test.jpg" for e in resp.json()["entries"])


# ── Browse: parent field ────────────────────────────────────────────────────


class TestBrowseParent:
    """Verify: response includes parent directory path."""

    def test_parent_field(self, tmp_path: Path):
        sub = tmp_path / "data" / "images"
        sub.mkdir(parents=True)

        settings = make_settings(tmp_path)

        from vlm_feedback_loop.services.filesystem_service import browse_directory

        result = browse_directory(str(sub), settings)
        assert result["parent"] == str(sub.parent)

    def test_root_parent(self, tmp_path: Path):
        """Root directory reports parent=None."""
        settings = make_settings(tmp_path)

        from vlm_feedback_loop.services.filesystem_service import browse_directory

        # Path("/").parent is "/" itself; the service reports None instead
        # of a self-referencing parent.
        result = browse_directory("/", settings)
        assert result["parent"] is None

    def test_configured_image_root_parent(self, tmp_path: Path):
        """IMAGE_ROOT is the top of the picker even when it has an OS parent."""
        image_root = tmp_path / "images"
        image_root.mkdir()
        settings = make_settings(tmp_path, IMAGE_ROOT=str(image_root))

        from vlm_feedback_loop.services.filesystem_service import browse_directory

        result = browse_directory(None, settings)
        assert result["path"] == str(image_root)
        assert result["parent"] is None


# ── Browse: file size_bytes ─────────────────────────────────────────────────


class TestBrowseFileSize:
    """Verify: file entries include size_bytes."""

    def test_file_has_size(self, tmp_path: Path):
        d = tmp_path / "data"
        d.mkdir()
        content = b"\xff\xd8" + b"\x00" * 200
        (d / "img.jpg").write_bytes(content)

        settings = make_settings(tmp_path)

        from vlm_feedback_loop.services.filesystem_service import browse_directory

        result = browse_directory(str(d), settings)
        files = [e for e in result["entries"] if e["type"] == "file"]
        assert len(files) == 1
        assert files[0]["size_bytes"] == len(content)


class TestBundledSampleDiscovery:
    """The browser identifies the shipped sample only when it is in scope."""

    def test_configured_sample_root_is_advertised(self, tmp_path: Path):
        sample = tmp_path / "example-images"
        sample.mkdir()
        (sample / "LICENSE.DATA").write_text("sample license")
        settings = make_settings(tmp_path, IMAGE_ROOT=str(sample))

        from vlm_feedback_loop.services.filesystem_service import browse_directory

        result = browse_directory(None, settings)
        assert result["bundled_sample_path"] == str(sample)

    def test_custom_root_does_not_offer_repository_sample_outside_boundary(
        self, tmp_path: Path
    ):
        custom = tmp_path / "customer-images"
        custom.mkdir()
        settings = make_settings(tmp_path, IMAGE_ROOT=str(custom))

        from vlm_feedback_loop.services.filesystem_service import browse_directory

        result = browse_directory(None, settings)
        assert result["bundled_sample_path"] is None


# ── Scan: deterministic key generation ──────────────────────────────────────


class TestScanDeterministicKey:
    """Verify: suggested_example_key is deterministic slug + 12-hex SHA-256."""

    def test_key_format_and_determinism(self, tmp_path: Path):
        scan_root = tmp_path / "images"
        scan_root.mkdir()
        (scan_root / "img_001.jpg").write_bytes(b"\xff\xd8" + b"\x00" * 10)

        settings = make_settings(tmp_path, IMAGE_ROOT=str(scan_root))

        from vlm_feedback_loop.services.filesystem_service import scan_directory

        result = scan_directory(str(scan_root), settings)
        assert result["total_images"] == 1

        img = result["images"][0]
        expected = _expected_key("img_001.jpg")
        assert img["suggested_example_key"] == expected

        # Run again — must be identical
        result2 = scan_directory(str(scan_root), settings)
        assert result2["images"][0]["suggested_example_key"] == expected

    def test_key_includes_subdir_in_slug(self, tmp_path: Path):
        scan_root = tmp_path / "images"
        sub = scan_root / "batch_01" / "sub"
        sub.mkdir(parents=True)
        (sub / "img_001.jpg").write_bytes(b"\xff\xd8" + b"\x00" * 10)

        settings = make_settings(tmp_path, IMAGE_ROOT=str(scan_root))

        from vlm_feedback_loop.services.filesystem_service import scan_directory

        result = scan_directory(str(scan_root), settings)
        img = result["images"][0]
        expected = _expected_key("batch_01/sub/img_001.jpg")
        assert img["suggested_example_key"] == expected
        # Slug should include directory components
        assert "batch_01" in img["suggested_example_key"]

    def test_different_extensions_produce_different_keys(self, tmp_path: Path):
        """Extension participates in the hash so foo.jpg ≠ foo.png."""
        scan_root = tmp_path / "images"
        scan_root.mkdir()
        (scan_root / "foo.jpg").write_bytes(b"\xff\xd8" + b"\x00" * 10)
        (scan_root / "foo.png").write_bytes(b"\x89PNG" + b"\x00" * 10)

        settings = make_settings(tmp_path)

        from vlm_feedback_loop.services.filesystem_service import scan_directory

        result = scan_directory(str(scan_root), settings)
        keys = [img["suggested_example_key"] for img in result["images"]]
        assert len(set(keys)) == 2, (
            "Same stem + different ext must produce different keys"
        )

    def test_same_file_has_same_key_when_scanned_from_parent_or_subdirectory(
        self, tmp_path: Path
    ):
        image_root = tmp_path / "images"
        class_dir = image_root / "class-a"
        class_dir.mkdir(parents=True)
        image = class_dir / "sample.jpg"
        image.write_bytes(b"\xff\xd8" + b"\x00" * 10)
        settings = make_settings(tmp_path, IMAGE_ROOT=str(image_root))

        from vlm_feedback_loop.services.filesystem_service import scan_directory

        parent_scan = scan_directory(str(image_root), settings)
        class_scan = scan_directory(str(class_dir), settings)

        assert parent_scan["images"][0]["storage_ref"] == str(image)
        assert class_scan["images"][0]["storage_ref"] == str(image)
        assert (
            parent_scan["images"][0]["suggested_example_key"]
            == class_scan["images"][0]["suggested_example_key"]
        )

    def test_unconfigured_loopback_scan_is_independent_of_selected_root(
        self, tmp_path: Path
    ):
        parent = tmp_path / "datasets"
        class_dir = parent / "class-a"
        class_dir.mkdir(parents=True)
        image = class_dir / "sample.jpg"
        image.write_bytes(b"\xff\xd8" + b"\x00" * 10)
        settings = make_settings(tmp_path)

        from vlm_feedback_loop.services.filesystem_service import scan_directory

        parent_scan = scan_directory(str(parent), settings)
        class_scan = scan_directory(str(class_dir), settings)

        assert (
            parent_scan["images"][0]["suggested_example_key"]
            == class_scan["images"][0]["suggested_example_key"]
        )


# ── Scan: collision checking 3 statuses ─────────────────────────────────────


class TestScanCollisionChecking:
    """Verify: collision checking returns available / already_exists_same_path /
    collision_different_path with existing_storage_ref.
    """

    def _create_project_with_example(
        self, client: TestClient, img_path: Path, example_key: str
    ) -> str:
        """Create a project and insert an example record directly."""
        resp = client.post("/v1/projects", json={"name": "Collision Test"})
        assert resp.status_code == 201
        project_id = resp.json()["project_id"]
        project_dir = Path(resp.json()["project_dir"])

        from vlm_feedback_loop.db.base import utc_now
        from vlm_feedback_loop.db.engine import open_project_db
        from vlm_feedback_loop.db.models.example import Example

        engine = open_project_db(project_dir)
        from sqlalchemy.orm import Session

        with Session(engine) as session:
            ex = Example(
                example_key=example_key,
                project_id=project_id,
                storage_ref=str(img_path),
                ingested_at=utc_now(),
                source_metadata={},
                state="Unlabeled",
            )
            session.add(ex)
            session.commit()

        # Register engine so the scan service can find it
        from vlm_feedback_loop.services.project_service import set_project_engine

        set_project_engine(project_id, engine)

        return project_id

    def test_available_when_no_project(self, tmp_path: Path):
        scan_root = tmp_path / "images"
        scan_root.mkdir()
        (scan_root / "img.jpg").write_bytes(b"\xff\xd8" + b"\x00" * 10)

        settings = make_settings(tmp_path)

        from vlm_feedback_loop.services.filesystem_service import scan_directory

        result = scan_directory(str(scan_root), settings, project_id=None)
        assert result["images"][0]["key_status"] == "available"
        assert result["total_collisions"] == 0

    def test_available_when_key_not_in_project(self, tmp_path: Path):
        scan_root = tmp_path / "images"
        scan_root.mkdir()
        img_path = scan_root / "new_img.jpg"
        img_path.write_bytes(b"\xff\xd8" + b"\x00" * 10)

        client = make_api_client(tmp_path)
        # Create project with no examples
        resp = client.post("/v1/projects", json={"name": "Empty"})
        project_id = resp.json()["project_id"]

        settings = make_settings(tmp_path)
        from vlm_feedback_loop.services.filesystem_service import scan_directory

        result = scan_directory(
            str(scan_root),
            settings,
            project_id=project_id,
            workspace_root=settings.WORKSPACE_ROOT,
        )
        assert result["images"][0]["key_status"] == "available"

    def test_already_exists_same_path(self, tmp_path: Path):
        scan_root = tmp_path / "images"
        scan_root.mkdir()
        img_path = scan_root / "img.jpg"
        img_path.write_bytes(b"\xff\xd8" + b"\x00" * 10)

        expected_key = _expected_key("img.jpg")

        client = make_api_client(tmp_path)
        project_id = self._create_project_with_example(client, img_path, expected_key)

        settings = make_settings(tmp_path, IMAGE_ROOT=str(scan_root))
        from vlm_feedback_loop.services.filesystem_service import scan_directory

        result = scan_directory(
            str(scan_root),
            settings,
            project_id=project_id,
            workspace_root=settings.WORKSPACE_ROOT,
        )
        assert result["images"][0]["key_status"] == "already_exists_same_path"
        assert result["total_collisions"] == 0

    def test_existing_path_wins_when_legacy_key_differs(self, tmp_path: Path):
        scan_root = tmp_path / "images"
        scan_root.mkdir()
        img_path = scan_root / "img.jpg"
        img_path.write_bytes(b"\xff\xd8" + b"\x00" * 10)
        client = make_api_client(tmp_path)
        project_id = self._create_project_with_example(
            client, img_path, "legacy-scan-root-key"
        )
        settings = make_settings(tmp_path, IMAGE_ROOT=str(scan_root))

        from vlm_feedback_loop.services.filesystem_service import scan_directory

        result = scan_directory(
            str(scan_root),
            settings,
            project_id=project_id,
            workspace_root=settings.WORKSPACE_ROOT,
        )

        assert result["images"][0]["key_status"] == "already_exists_same_path"
        assert result["images"][0]["suggested_example_key"] == "legacy-scan-root-key"
        assert result["total_collisions"] == 0

    def test_collision_different_path(self, tmp_path: Path):
        scan_root = tmp_path / "images"
        scan_root.mkdir()
        img_path = scan_root / "img.jpg"
        img_path.write_bytes(b"\xff\xd8" + b"\x00" * 10)

        expected_key = _expected_key("img.jpg")
        different_path = tmp_path / "other" / "img.jpg"
        different_path.parent.mkdir(parents=True)
        different_path.write_bytes(b"\xff\xd8" + b"\x00" * 10)

        client = make_api_client(tmp_path)
        project_id = self._create_project_with_example(
            client, different_path, expected_key
        )

        settings = make_settings(tmp_path, IMAGE_ROOT=str(scan_root))
        from vlm_feedback_loop.services.filesystem_service import scan_directory

        result = scan_directory(
            str(scan_root),
            settings,
            project_id=project_id,
            workspace_root=settings.WORKSPACE_ROOT,
        )
        img = result["images"][0]
        assert img["key_status"] == "collision_different_path"
        assert img["existing_storage_ref"] == str(different_path)
        assert result["total_collisions"] == 1


# ── Scan: skipped files ─────────────────────────────────────────────────────


class TestScanSkipped:
    """Verify: non-image files reported in skipped[] with reasons."""

    def test_unsupported_formats_skipped(self, tmp_path: Path):
        scan_root = tmp_path / "images"
        scan_root.mkdir()
        (scan_root / "photo.jpg").write_bytes(b"\xff\xd8" + b"\x00" * 10)
        (scan_root / "notes.txt").write_text("text")
        (scan_root / "data.csv").write_text("a,b")
        (scan_root / "archive.zip").write_bytes(b"PK" + b"\x00" * 10)

        settings = make_settings(tmp_path)

        from vlm_feedback_loop.services.filesystem_service import scan_directory

        result = scan_directory(str(scan_root), settings)
        assert result["total_images"] == 1
        assert result["total_skipped"] == 3

        skipped_paths = [s["path"] for s in result["skipped"]]
        assert str(scan_root / "notes.txt") in skipped_paths
        assert str(scan_root / "data.csv") in skipped_paths
        assert str(scan_root / "archive.zip") in skipped_paths

        # All reasons should be unsupported_format
        for s in result["skipped"]:
            assert s["reason"] == "unsupported_format"


# ── Scan: recursive ─────────────────────────────────────────────────────────


class TestScanRecursive:
    """Verify: recursive scan discovers files in subdirectories."""

    def test_recursive_discovers_subdirs(self, tmp_path: Path):
        scan_root = tmp_path / "images"
        sub = scan_root / "batch_01"
        sub.mkdir(parents=True)
        (scan_root / "top.jpg").write_bytes(b"\xff\xd8" + b"\x00" * 10)
        (sub / "nested.jpg").write_bytes(b"\xff\xd8" + b"\x00" * 10)

        settings = make_settings(tmp_path)

        from vlm_feedback_loop.services.filesystem_service import scan_directory

        result = scan_directory(str(scan_root), settings, recursive=True)
        assert result["total_images"] == 2

        refs = [img["storage_ref"] for img in result["images"]]
        assert str(scan_root / "top.jpg") in refs
        assert str(sub / "nested.jpg") in refs

    def test_non_recursive_top_level_only(self, tmp_path: Path):
        scan_root = tmp_path / "images"
        sub = scan_root / "batch_01"
        sub.mkdir(parents=True)
        (scan_root / "top.jpg").write_bytes(b"\xff\xd8" + b"\x00" * 10)
        (sub / "nested.jpg").write_bytes(b"\xff\xd8" + b"\x00" * 10)

        settings = make_settings(tmp_path)

        from vlm_feedback_loop.services.filesystem_service import scan_directory

        result = scan_directory(str(scan_root), settings, recursive=False)
        assert result["total_images"] == 1
        assert "top.jpg" in result["images"][0]["storage_ref"]


# ── Scan: security ──────────────────────────────────────────────────────────


class TestScanSecurity:
    """Verify: scan respects the same security rules as browse."""

    def test_scan_nonloopback_empty_roots_disabled(self, tmp_path: Path):
        scan_root = tmp_path / "images"
        scan_root.mkdir()

        client = make_api_client(
            tmp_path,
            BIND_HOST="0.0.0.0",
            IMAGE_ROOT=None,
        )
        resp = client.post("/v1/filesystem/scan", json={"path": str(scan_root)})
        assert resp.status_code == 403
        assert "IMAGE_ROOT" in resp.json()["detail"]

    def test_scan_outside_roots_rejected(self, tmp_path: Path):
        allowed = tmp_path / "allowed"
        allowed.mkdir()
        forbidden = tmp_path / "forbidden"
        forbidden.mkdir()

        client = make_api_client(
            tmp_path,
            BIND_HOST="0.0.0.0",
            IMAGE_ROOT=str(allowed),
        )
        resp = client.post("/v1/filesystem/scan", json={"path": str(forbidden)})
        assert resp.status_code == 403
        assert "outside IMAGE_ROOT" in resp.json()["detail"]

    def test_scan_not_found(self, tmp_path: Path):
        client = make_api_client(tmp_path)
        resp = client.post(
            "/v1/filesystem/scan",
            json={"path": str(tmp_path / "nonexistent")},
        )
        assert resp.status_code == 404
        assert "Directory not found" in resp.json()["detail"]


# ── Scan: hidden files excluded ─────────────────────────────────────────────


class TestScanHiddenExcluded:
    """Verify: hidden files excluded from scan results."""

    def test_hidden_files_not_in_scan(self, tmp_path: Path):
        scan_root = tmp_path / "images"
        scan_root.mkdir()
        (scan_root / "visible.jpg").write_bytes(b"\xff\xd8" + b"\x00" * 10)
        (scan_root / ".hidden.jpg").write_bytes(b"\xff\xd8" + b"\x00" * 10)

        settings = make_settings(tmp_path)

        from vlm_feedback_loop.services.filesystem_service import scan_directory

        result = scan_directory(str(scan_root), settings)
        assert result["total_images"] == 1
        assert "visible" in result["images"][0]["storage_ref"]


# ── Endpoint integration: browse + scan via HTTP ────────────────────────────


class TestEndpointIntegration:
    """Full round-trip endpoint tests for browse and scan."""

    def test_browse_endpoint_response_shape(self, tmp_path: Path):
        d = tmp_path / "data"
        d.mkdir()
        sub = d / "subdir"
        sub.mkdir()
        (d / "img.jpg").write_bytes(b"\xff\xd8" + b"\x00" * 10)

        client = make_api_client(tmp_path)
        resp = client.get("/v1/filesystem/browse", params={"path": str(d)})
        assert resp.status_code == 200
        body = resp.json()

        assert body["path"] == str(d)
        assert body["parent"] == str(d.parent)
        assert isinstance(body["entries"], list)

        # Check entry shape
        for entry in body["entries"]:
            assert "name" in entry
            assert "type" in entry
            assert entry["type"] in ("directory", "file")
            assert "path" in entry
            if entry["type"] == "file":
                assert "size_bytes" in entry

    def test_scan_endpoint_response_shape(self, tmp_path: Path):
        d = tmp_path / "data"
        d.mkdir()
        (d / "img.jpg").write_bytes(b"\xff\xd8" + b"\x00" * 10)
        (d / "readme.txt").write_text("hello")

        client = make_api_client(tmp_path)
        resp = client.post("/v1/filesystem/scan", json={"path": str(d)})
        assert resp.status_code == 200
        body = resp.json()

        assert body["path"] == str(d)
        assert body["total_images"] == 1
        assert body["total_skipped"] == 1
        assert body["total_collisions"] == 0
        assert isinstance(body["images"], list)
        assert isinstance(body["skipped"], list)

        img = body["images"][0]
        assert "storage_ref" in img
        assert "suggested_example_key" in img
        assert "size_bytes" in img
        assert img["key_status"] == "available"

        skip = body["skipped"][0]
        assert "path" in skip
        assert skip["reason"] == "unsupported_format"

    def test_scan_with_project_id_collision_check(self, tmp_path: Path):
        """Full round-trip: create project, insert example, scan with collision."""
        d = tmp_path / "data"
        d.mkdir()
        img = d / "test_img.jpg"
        img.write_bytes(b"\xff\xd8" + b"\x00" * 20)

        client = make_api_client(tmp_path)

        # Create project
        resp = client.post("/v1/projects", json={"name": "Scan Test"})
        assert resp.status_code == 201
        project_id = resp.json()["project_id"]
        project_dir = Path(resp.json()["project_dir"])

        # Insert an example with the same key but different path
        expected_key = _expected_key(str(img.resolve()))

        from sqlalchemy.orm import Session

        from vlm_feedback_loop.db.base import utc_now
        from vlm_feedback_loop.db.engine import open_project_db
        from vlm_feedback_loop.db.models.example import Example
        from vlm_feedback_loop.services.project_service import set_project_engine

        engine = open_project_db(project_dir)
        set_project_engine(project_id, engine)

        with Session(engine) as session:
            ex = Example(
                example_key=expected_key,
                project_id=project_id,
                storage_ref="/some/other/path.jpg",
                ingested_at=utc_now(),
                source_metadata={},
                state="Unlabeled",
            )
            session.add(ex)
            session.commit()

        # Scan with project_id
        resp = client.post(
            "/v1/filesystem/scan",
            json={"path": str(d), "project_id": project_id},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["total_collisions"] == 1
        assert body["images"][0]["key_status"] == "collision_different_path"
        assert body["images"][0]["existing_storage_ref"] == "/some/other/path.jpg"


# ── Scan: symlink security ──────────────────────────────────────────────────


class TestScanSymlinks:
    """Verify: scan excludes symlinks escaping roots."""

    def test_symlink_files_escaping_root_excluded(self, tmp_path: Path):
        allowed = tmp_path / "allowed"
        allowed.mkdir()

        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secret.jpg").write_bytes(b"\xff\xd8" + b"\x00" * 10)

        # Symlink file inside allowed pointing outside
        link = allowed / "linked_secret.jpg"
        try:
            link.symlink_to(outside / "secret.jpg")
        except OSError:
            pytest.skip("Cannot create symlinks on this filesystem")

        # Also add a real image
        (allowed / "real.jpg").write_bytes(b"\xff\xd8" + b"\x00" * 10)

        settings = make_settings(
            tmp_path,
            BIND_HOST="0.0.0.0",
            IMAGE_ROOT=str(allowed),
        )

        from vlm_feedback_loop.services.filesystem_service import scan_directory

        result = scan_directory(str(allowed), settings)
        refs = [img["storage_ref"] for img in result["images"]]
        assert str(allowed / "real.jpg") in refs
        # The symlink should not be in results
        assert not any("secret" in r for r in refs)
