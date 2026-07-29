# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for image ingestion, pHash, image serving, and example query."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from PIL import Image
from starlette.testclient import TestClient

from conftest import (
    create_project_via_api,
    make_api_client,
    make_settings,
    make_test_image,
)

# ── Helpers ─────────────────────────────────────────────────────────────────
# Per-format image builders share conftest's make_test_image; each format
# keeps its own fill color so cross-format pairs stay visually distinct
# (the pHash tests rely on that).


def _make_png(path: Path, width: int = 100, height: int = 100) -> Path:
    return make_test_image(
        path, fmt="PNG", width=width, height=height, color=(32, 128, 64)
    )


def _make_bmp(path: Path, width: int = 100, height: int = 100) -> Path:
    return make_test_image(
        path, fmt="BMP", width=width, height=height, color=(64, 32, 128)
    )


def _make_webp(path: Path, width: int = 100, height: int = 100) -> Path:
    return make_test_image(
        path, fmt="WEBP", width=width, height=height, color=(200, 100, 50)
    )


def _make_tiff(path: Path, width: int = 100, height: int = 100) -> Path:
    return make_test_image(
        path, fmt="TIFF", width=width, height=height, color=(50, 200, 100)
    )


def _make_tiff_multipage(path: Path) -> Path:
    """Create a multi-page TIFF (2 pages)."""
    pages = [
        Image.new("RGB", (50, 50), color=(255, 0, 0)),
        Image.new("RGB", (50, 50), color=(0, 255, 0)),
    ]
    pages[0].save(str(path), format="TIFF", save_all=True, append_images=pages[1:])
    return path


def _ingest(client: TestClient, project_id: str, examples: list[dict]) -> dict:
    """Helper: call the ingest endpoint."""
    resp = client.post(
        f"/v1/projects/{project_id}/examples:ingest",
        json={"examples": examples},
    )
    return resp.json()


PHASH_RE = re.compile(r"^[0-9a-f]{16}$")


# ═══════════════════════════════════════════════════════════════════════════
# Image ingestion — detailed behavior
# ═══════════════════════════════════════════════════════════════════════════


class TestIdempotentReIngest:
    """Verify: same key + same path → status='exists', no duplicate."""

    def test_idempotent_reingest(self, tmp_path: Path):
        client = make_api_client(tmp_path)
        project = create_project_via_api(client)
        pid = project["project_id"]

        img = make_test_image(tmp_path / "img.jpg")
        item = {"example_key": "img1", "storage_ref": str(img)}

        r1 = _ingest(client, pid, [item])
        assert r1["results"][0]["status"] == "created"

        r2 = _ingest(client, pid, [item])
        assert r2["results"][0]["status"] == "exists"
        assert r2["results"][0]["example"]["example_key"] == "img1"


class TestKeyCollision:
    """Verify: same key + different path → error with error_code."""

    def test_key_collision_different_path(self, tmp_path: Path):
        client = make_api_client(tmp_path)
        project = create_project_via_api(client)
        pid = project["project_id"]

        img1 = make_test_image(tmp_path / "img_a.jpg")
        img2 = make_test_image(tmp_path / "img_b.jpg")

        _ingest(client, pid, [{"example_key": "k1", "storage_ref": str(img1)}])
        r2 = _ingest(client, pid, [{"example_key": "k1", "storage_ref": str(img2)}])

        result = r2["results"][0]
        assert result["status"] == "error"
        assert result["error_code"] == "example_key_collision"
        assert str(img1) in result["error"]
        assert str(img2) in result["error"]


class TestServerSetIngestedAt:
    """Verify: ingested_at is server-set, client value ignored."""

    def test_ingested_at_server_set(self, tmp_path: Path):
        client = make_api_client(tmp_path)
        project = create_project_via_api(client)
        pid = project["project_id"]

        img = make_test_image(tmp_path / "img.jpg")
        r = _ingest(
            client,
            pid,
            [
                {
                    "example_key": "k1",
                    "storage_ref": str(img),
                    "source_metadata": {"ingested_at": "1999-01-01T00:00:00Z"},
                }
            ],
        )
        ts = r["results"][0]["example"]["ingested_at"]
        assert ts != "1999-01-01T00:00:00Z"
        assert ts.endswith("Z")


class TestPHashAfterSweeper:
    """pHash is null at endpoint return and populated by the sweeper.

    pHash runs in a background sweeper so the endpoint can return
    202 in ~1s; this test asserts the two-stage guarantee — phash=null
    in the response, then populated after the sweeper runs.
    """

    def test_phash_present_after_sweep(self, tmp_path: Path):
        import asyncio

        from vlm_feedback_loop.services.ingest_sweeper_service import _ingest_worker

        client = make_api_client(tmp_path)
        project = create_project_via_api(client)
        pid = project["project_id"]

        img = make_test_image(tmp_path / "img.jpg")
        r = _ingest(client, pid, [{"example_key": "k1", "storage_ref": str(img)}])

        # phash is null at endpoint return.
        assert r["results"][0]["example"]["phash"] is None

        # Run the sweeper synchronously to assert the eventual state.
        # In production the sweeper runs as a background task that the
        # endpoint kicked off via trigger_ingest_processing — calling
        # _ingest_worker directly here is equivalent and deterministic.
        settings = make_settings(tmp_path / "workspace")
        asyncio.run(_ingest_worker(pid, settings.WORKSPACE_ROOT))

        # Re-fetch the example and assert pHash is now populated.
        from sqlalchemy import select
        from sqlalchemy.orm import Session

        from vlm_feedback_loop.db.models.example import Example
        from vlm_feedback_loop.services.project_service import get_project_engine

        engine = get_project_engine(pid, settings.WORKSPACE_ROOT)
        assert engine is not None
        with Session(engine) as session:
            ex = session.execute(
                select(Example).where(Example.example_key == "k1")
            ).scalar_one()
            phash = ex.phash
        assert phash is not None
        assert PHASH_RE.match(phash), f"pHash {phash!r} not 16 hex chars"


class TestClipTriggered:
    """Verify: CLIP computation triggered after successful ingest."""

    def test_clip_trigger_called(self, tmp_path: Path, monkeypatch):
        from unittest.mock import MagicMock

        mock_trigger = MagicMock()
        # Monkeypatch at the source module so the lazy import picks it up
        monkeypatch.setattr(
            "vlm_feedback_loop.services.clip_embedding_service.trigger_embedding_computation",
            mock_trigger,
        )

        client = make_api_client(tmp_path)
        project = create_project_via_api(client)
        pid = project["project_id"]

        img = make_test_image(tmp_path / "img.jpg")
        _ingest(client, pid, [{"example_key": "k1", "storage_ref": str(img)}])

        assert mock_trigger.call_count >= 1


class TestStateDefault:
    """Verify: default state is 'Unlabeled' after ingestion."""

    def test_state_unlabeled(self, tmp_path: Path):
        client = make_api_client(tmp_path)
        project = create_project_via_api(client)
        pid = project["project_id"]

        img = make_test_image(tmp_path / "img.jpg")
        r = _ingest(client, pid, [{"example_key": "k1", "storage_ref": str(img)}])
        assert r["results"][0]["example"]["state"] == "Unlabeled"


# ═══════════════════════════════════════════════════════════════════════════
# pHash computation
# ═══════════════════════════════════════════════════════════════════════════


class TestPHashNullAtEndpointReturn:
    """:ingest returns 202 with phash=null.

    Skeleton rows are created with phash=None and returned with 202;
    the background sweeper populates pHash asynchronously. This test
    pins the contract at the API boundary.
    """

    def test_endpoint_returns_202_with_null_phash(self, tmp_path: Path):
        client = make_api_client(tmp_path)
        project = create_project_via_api(client)
        pid = project["project_id"]

        img = make_test_image(tmp_path / "img.jpg")
        resp = client.post(
            f"/v1/projects/{pid}/examples:ingest",
            json={"examples": [{"example_key": "k1", "storage_ref": str(img)}]},
        )

        # Ingest contract: 202 Accepted, phash is null in the response body.
        assert resp.status_code == 202, resp.text
        body = resp.json()
        assert body["results"][0]["status"] == "created"
        assert body["results"][0]["example"]["phash"] is None


class TestSkeletonIngestValidation:
    """The non-blocking ingest path validates structure without decoding pixels."""

    def test_skeleton_ingest_does_not_expand_full_resolution_pixels(
        self, tmp_path: Path, monkeypatch
    ):
        from vlm_feedback_loop.services import example_service

        client = make_api_client(tmp_path)
        project = create_project_via_api(client)
        pid = project["project_id"]
        img = make_test_image(tmp_path / "large.jpg", width=1600, height=1200)

        def fail_full_decode(*_args, **_kwargs):
            raise AssertionError("skeleton ingest performed a full pixel decode")

        # Patch after creating the fixture: Pillow's writer legitimately loads
        # pixels while saving. The ingest validator must use verify(), leaving
        # the first full decode to the background pHash sweep.
        monkeypatch.setattr(Image.Image, "load", fail_full_decode)

        settings = make_settings(tmp_path / "workspace")
        results = example_service.ingest_examples(
            project_id=pid,
            workspace_root=settings.WORKSPACE_ROOT,
            items=[{"example_key": "large", "storage_ref": str(img)}],
            settings=settings,
        )

        assert results[0]["status"] == "created"
        assert results[0]["example"]["phash"] is None


class TestPHashHexEncoded:
    """Verify: pHash stored as hex-encoded string (16 chars = 64 bits)."""

    def test_phash_format(self, tmp_path: Path):
        from vlm_feedback_loop.services.phash import compute_phash_from_path

        img_path = make_test_image(tmp_path / "img.jpg")
        h = compute_phash_from_path(str(img_path))

        assert h is not None
        assert len(h) == 16
        assert PHASH_RE.match(h)


class TestPHashFailureDoesNotFailIngest:
    """Verify: pHash failure for one image doesn't fail ingestion."""

    def test_corrupt_phash_still_ingests(self, tmp_path: Path):
        client = make_api_client(tmp_path)
        project = create_project_via_api(client)
        pid = project["project_id"]

        # A 1x1 JPEG is an edge case for the DCT pHash (resized up to the
        # 32x32 DCT input); ingestion must still succeed.
        img = make_test_image(tmp_path / "tiny.jpg", width=1, height=1)
        r = _ingest(client, pid, [{"example_key": "k1", "storage_ref": str(img)}])

        # Should still succeed even for tiny images
        assert r["results"][0]["status"] == "created"


class TestPHashDeterministic:
    """Verify: same image produces same pHash across calls."""

    def test_determinism(self, tmp_path: Path):
        from vlm_feedback_loop.services.phash import compute_phash_from_path

        img_path = make_test_image(tmp_path / "img.jpg", width=200, height=200)
        h1 = compute_phash_from_path(str(img_path))
        h2 = compute_phash_from_path(str(img_path))
        assert h1 == h2


class TestPHashAlwaysComputed:
    """Verify: pHash always populated by sweeper regardless of EMBEDDINGS_AUTO_COMPUTE."""

    def test_phash_with_embeddings_off(self, tmp_path: Path):
        import asyncio

        from vlm_feedback_loop.services.ingest_sweeper_service import _ingest_worker

        client = make_api_client(tmp_path, EMBEDDINGS_AUTO_COMPUTE=False)
        project = create_project_via_api(client)
        pid = project["project_id"]

        img = make_test_image(tmp_path / "img.jpg")
        r = _ingest(client, pid, [{"example_key": "k1", "storage_ref": str(img)}])

        # phash is null in response — sweeper hasn't run yet.
        assert r["results"][0]["example"]["phash"] is None

        # Run the sweeper synchronously. EMBEDDINGS_AUTO_COMPUTE=False
        # disables CLIP, but the pHash sweeper is independent — it must
        # still populate phash.
        settings = make_settings(tmp_path / "workspace", EMBEDDINGS_AUTO_COMPUTE=False)
        asyncio.run(_ingest_worker(pid, settings.WORKSPACE_ROOT))

        from sqlalchemy import select
        from sqlalchemy.orm import Session

        from vlm_feedback_loop.db.models.example import Example
        from vlm_feedback_loop.services.project_service import get_project_engine

        engine = get_project_engine(pid, settings.WORKSPACE_ROOT)
        assert engine is not None
        with Session(engine) as session:
            ex = session.execute(
                select(Example).where(Example.example_key == "k1")
            ).scalar_one()
            phash = ex.phash
        assert phash is not None


class TestPHashAlgorithmCheck:
    """Verify: project.phash_algorithm is 'dct_phash_64' (checked on project open)."""

    def test_project_has_phash_algorithm(self, tmp_path: Path):
        client = make_api_client(tmp_path)
        project = create_project_via_api(client)
        assert project.get("phash_algorithm") == "dct_phash_64"


# ═══════════════════════════════════════════════════════════════════════════
# Ingestion and querying
# ═══════════════════════════════════════════════════════════════════════════


class TestBatchIdempotency:
    """Verify: batch supports mixed new/existing items."""

    def test_batch_mixed(self, tmp_path: Path):
        client = make_api_client(tmp_path)
        project = create_project_via_api(client)
        pid = project["project_id"]

        img1 = make_test_image(tmp_path / "a.jpg")
        img2 = make_test_image(tmp_path / "b.jpg")
        img3 = make_test_image(tmp_path / "c.jpg")

        # First ingest img1
        _ingest(client, pid, [{"example_key": "k1", "storage_ref": str(img1)}])

        # Batch: existing + new + new
        r = _ingest(
            client,
            pid,
            [
                {"example_key": "k1", "storage_ref": str(img1)},  # exists
                {"example_key": "k2", "storage_ref": str(img2)},  # new
                {"example_key": "k3", "storage_ref": str(img3)},  # new
            ],
        )

        statuses = [item["status"] for item in r["results"]]
        assert statuses == ["exists", "created", "created"]


class TestPartialSuccess:
    """Verify: one failure doesn't block other items in the batch."""

    def test_partial_success(self, tmp_path: Path):
        client = make_api_client(tmp_path)
        project = create_project_via_api(client)
        pid = project["project_id"]

        good_img = make_test_image(tmp_path / "good.jpg")
        bad_path = str(tmp_path / "nonexistent.jpg")

        r = _ingest(
            client,
            pid,
            [
                {"example_key": "good", "storage_ref": str(good_img)},
                {"example_key": "bad", "storage_ref": bad_path},
            ],
        )

        assert r["results"][0]["status"] == "created"
        assert r["results"][1]["status"] == "error"
        assert "not found" in r["results"][1]["error"].lower()


class TestQueryFiltersPaginationOrdering:
    """Verify: query supports state filter, cursor pagination, ordering."""

    def test_state_filter(self, tmp_path: Path):
        client = make_api_client(tmp_path)
        project = create_project_via_api(client)
        pid = project["project_id"]

        # Ingest 3 images (all Unlabeled)
        for i in range(3):
            img = make_test_image(tmp_path / f"img_{i}.jpg")
            _ingest(client, pid, [{"example_key": f"k{i}", "storage_ref": str(img)}])

        # Query all
        resp = client.get(f"/v1/projects/{pid}/examples")
        assert resp.status_code == 200
        assert len(resp.json()["items"]) == 3

        # Query by state
        resp = client.get(f"/v1/projects/{pid}/examples", params={"state": "Unlabeled"})
        assert resp.status_code == 200
        assert len(resp.json()["items"]) == 3

        resp = client.get(f"/v1/projects/{pid}/examples", params={"state": "Verified"})
        assert resp.status_code == 200
        assert len(resp.json()["items"]) == 0

    def test_cursor_pagination(self, tmp_path: Path):
        client = make_api_client(tmp_path)
        project = create_project_via_api(client)
        pid = project["project_id"]

        # Ingest 5 images
        for i in range(5):
            img = make_test_image(tmp_path / f"img_{i}.jpg")
            _ingest(
                client, pid, [{"example_key": f"k{i:02d}", "storage_ref": str(img)}]
            )

        # Page 1: limit=2
        resp = client.get(f"/v1/projects/{pid}/examples", params={"limit": 2})
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["items"]) == 2
        assert body["next_cursor"] is not None

        # Page 2
        resp2 = client.get(
            f"/v1/projects/{pid}/examples",
            params={"limit": 2, "cursor": body["next_cursor"]},
        )
        assert resp2.status_code == 200
        body2 = resp2.json()
        assert len(body2["items"]) == 2

        # Ensure no overlap between pages
        keys_p1 = {i["example"]["example_key"] for i in body["items"]}
        keys_p2 = {i["example"]["example_key"] for i in body2["items"]}
        assert keys_p1.isdisjoint(keys_p2)

    def test_include_verified_label(self, tmp_path: Path):
        """When include=verified_label but no labels exist, items have null verified_label."""
        client = make_api_client(tmp_path)
        project = create_project_via_api(client)
        pid = project["project_id"]

        img = make_test_image(tmp_path / "img.jpg")
        _ingest(client, pid, [{"example_key": "k1", "storage_ref": str(img)}])

        resp = client.get(
            f"/v1/projects/{pid}/examples",
            params={"include": "verified_label"},
        )
        assert resp.status_code == 200
        assert resp.json()["items"][0]["verified_label"] is None

    def test_pool_membership_filter_returns_only_pool_members(self, tmp_path: Path):
        """pool_membership=test_pool returns only examples whose verified
        label carries a test_pool assignment — non-members are filtered out."""
        from sqlalchemy.orm import Session

        from vlm_feedback_loop.db.base import utc_now
        from vlm_feedback_loop.db.models.label import Label
        from vlm_feedback_loop.services.project_service import get_project_engine

        client = make_api_client(tmp_path)
        project = create_project_via_api(client)
        pid = project["project_id"]

        img1 = make_test_image(tmp_path / "img1.jpg")
        img2 = _make_png(tmp_path / "img2.png")
        _ingest(
            client,
            pid,
            [
                {"example_key": "in_pool", "storage_ref": str(img1)},
                {"example_key": "not_in_pool", "storage_ref": str(img2)},
            ],
        )

        # Seed one verified label in the Test Pool and one outside it.
        engine = get_project_engine(pid, str(tmp_path / "workspace"))
        now = utc_now()
        with Session(engine) as s:
            for key, pool in (("in_pool", "test_pool"), ("not_in_pool", None)):
                s.add(
                    Label(
                        project_id=pid,
                        example_key=key,
                        label_status="verified",
                        guidance_id="g-1",
                        inference_invocation_id="inv-1",
                        label_json={"field": "v"},
                        labeled_at=now,
                        verified_outcome="Accept",
                        verified_at=now,
                        pool_assignment=pool,
                    )
                )
            s.commit()

        resp = client.get(
            f"/v1/projects/{pid}/examples",
            params={"pool_membership": "test_pool"},
        )
        assert resp.status_code == 200
        keys = [i["example"]["example_key"] for i in resp.json()["items"]]
        assert keys == ["in_pool"]


# ═══════════════════════════════════════════════════════════════════════════
# Image handling
# ═══════════════════════════════════════════════════════════════════════════


class TestFileExistence:
    """Verify: ingest validates file existence; missing → error."""

    def test_missing_file(self, tmp_path: Path):
        client = make_api_client(tmp_path)
        project = create_project_via_api(client)
        pid = project["project_id"]

        r = _ingest(
            client,
            pid,
            [{"example_key": "k1", "storage_ref": str(tmp_path / "nope.jpg")}],
        )
        assert r["results"][0]["status"] == "error"
        assert "not found" in r["results"][0]["error"].lower()


class TestFormatValidation:
    """Verify: accepted formats + rejected GIF/multi-TIFF."""

    @pytest.mark.parametrize(
        "make_image,filename",
        [
            pytest.param(make_test_image, "img.jpg", id="jpeg"),
            pytest.param(_make_png, "img.png", id="png"),
            pytest.param(_make_bmp, "img.bmp", id="bmp"),
            pytest.param(_make_webp, "img.webp", id="webp"),
            pytest.param(_make_tiff, "img.tiff", id="tiff-single-page"),
        ],
    )
    def test_supported_format_accepted(self, tmp_path: Path, make_image, filename):
        client = make_api_client(tmp_path)
        pid = create_project_via_api(client)["project_id"]

        img = make_image(tmp_path / filename)
        r = _ingest(client, pid, [{"example_key": "k1", "storage_ref": str(img)}])
        assert r["results"][0]["status"] == "created"

    def test_gif_rejected_as_unsupported_format(self, tmp_path: Path):
        """GIF is not in the accepted-format allowlist, so ingestion rejects it."""
        client = make_api_client(tmp_path)
        project = create_project_via_api(client)
        pid = project["project_id"]

        img = make_test_image(tmp_path / "img.gif", fmt="GIF")
        r = _ingest(client, pid, [{"example_key": "k1", "storage_ref": str(img)}])
        assert r["results"][0]["status"] == "error"
        assert "unsupported" in r["results"][0]["error"].lower()

    def test_multipage_tiff_rejected(self, tmp_path: Path):
        client = make_api_client(tmp_path)
        project = create_project_via_api(client)
        pid = project["project_id"]

        img = _make_tiff_multipage(tmp_path / "multi.tiff")
        r = _ingest(client, pid, [{"example_key": "k1", "storage_ref": str(img)}])
        assert r["results"][0]["status"] == "error"
        assert "multi" in r["results"][0]["error"].lower()

    def test_unsupported_format_rejected(self, tmp_path: Path):
        """A text file renamed to .jpg should be rejected by Pillow."""
        client = make_api_client(tmp_path)
        project = create_project_via_api(client)
        pid = project["project_id"]

        fake = tmp_path / "fake.jpg"
        fake.write_text("not an image")
        r = _ingest(client, pid, [{"example_key": "k1", "storage_ref": str(fake)}])
        assert r["results"][0]["status"] == "error"


class TestSizeWarnings:
    """Verify: large images produce warnings but still ingest."""

    def test_large_dimension_warning(self, tmp_path: Path):
        client = make_api_client(tmp_path)
        project = create_project_via_api(client)
        pid = project["project_id"]

        # Create an image with longest edge > 8192 px
        img = _make_png(tmp_path / "big.png", width=8200, height=100)
        r = _ingest(client, pid, [{"example_key": "k1", "storage_ref": str(img)}])

        result = r["results"][0]
        assert result["status"] == "created"
        assert any("8192" in w or "exceeds" in w.lower() for w in result["warnings"])


class TestStreamFromStorageRef:
    """Verify: image serving streams correct bytes and Content-Type."""

    def test_serve_jpeg(self, tmp_path: Path):
        client = make_api_client(tmp_path)
        project = create_project_via_api(client)
        pid = project["project_id"]

        img_path = make_test_image(tmp_path / "img.jpg")
        _ingest(client, pid, [{"example_key": "k1", "storage_ref": str(img_path)}])

        resp = client.get(f"/v1/projects/{pid}/examples/k1/image")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/jpeg"
        assert len(resp.content) > 0

        # Verify the bytes match the original file
        assert resp.content == img_path.read_bytes()

    def test_serve_png(self, tmp_path: Path):
        client = make_api_client(tmp_path)
        project = create_project_via_api(client)
        pid = project["project_id"]

        img_path = _make_png(tmp_path / "img.png")
        _ingest(client, pid, [{"example_key": "k1", "storage_ref": str(img_path)}])

        resp = client.get(f"/v1/projects/{pid}/examples/k1/image")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"

    def test_security_no_arbitrary_paths(self, tmp_path: Path):
        """The image endpoint only serves from persisted storage_ref,
        not arbitrary paths from the request."""
        client = make_api_client(tmp_path)
        project = create_project_via_api(client)
        pid = project["project_id"]

        # No example "evil" exists → 404, not a path traversal
        resp = client.get(f"/v1/projects/{pid}/examples/../../etc/passwd/image")
        assert resp.status_code in (404, 422)
        # The error response must not leak file contents.
        assert "root:" not in resp.text


class TestMissingExample404:
    """Verify: GET image for nonexistent example returns 404."""

    def test_nonexistent_example(self, tmp_path: Path):
        client = make_api_client(tmp_path)
        project = create_project_via_api(client)
        pid = project["project_id"]

        resp = client.get(f"/v1/projects/{pid}/examples/nonexistent/image")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Example not found"

    def test_missing_file_at_storage_ref(self, tmp_path: Path):
        """Example exists but file was deleted → appropriate error."""
        client = make_api_client(tmp_path)
        project = create_project_via_api(client)
        pid = project["project_id"]

        img_path = make_test_image(tmp_path / "img.jpg")
        _ingest(client, pid, [{"example_key": "k1", "storage_ref": str(img_path)}])

        # Delete the file
        img_path.unlink()

        resp = client.get(f"/v1/projects/{pid}/examples/k1/image")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()


# ═══════════════════════════════════════════════════════════════════════════
# pHash correctness and determinism
# ═══════════════════════════════════════════════════════════════════════════


class TestPHashCorrectness:
    """Verify the DCT pHash implementation produces correct, stable results."""

    def test_different_images_different_hashes(self, tmp_path: Path):
        from vlm_feedback_loop.services.phash import compute_phash_from_path

        img1 = make_test_image(tmp_path / "red.jpg", width=200, height=200)
        img2 = _make_png(tmp_path / "green.png", width=200, height=200)

        h1 = compute_phash_from_path(str(img1))
        h2 = compute_phash_from_path(str(img2))

        assert h1 is not None
        assert h2 is not None
        assert h1 != h2

    def test_visually_distinct_images_different_hashes(self, tmp_path: Path):
        """Visually distinct images should produce different pHash values."""
        from vlm_feedback_loop.services.phash import compute_phash

        # A gradient vs a checkerboard — visually very different
        img1 = Image.new("L", (200, 200))
        for y in range(200):
            for x in range(200):
                img1.putpixel((x, y), int(255 * x / 199))
        img1 = img1.convert("RGB")

        img2 = Image.new("L", (200, 200))
        for y in range(200):
            for x in range(200):
                img2.putpixel((x, y), 255 if (x // 25 + y // 25) % 2 == 0 else 0)
        img2 = img2.convert("RGB")

        h1 = compute_phash(img1)
        h2 = compute_phash(img2)

        assert h1 != h2, "Visually distinct images should produce different hashes"

    def test_solid_colour_hash_stable(self, tmp_path: Path):
        """A solid-colour image should always produce the same hash."""
        from vlm_feedback_loop.services.phash import compute_phash

        img = Image.new("RGB", (100, 100), color=(0, 0, 0))
        h1 = compute_phash(img)
        h2 = compute_phash(img)
        assert h1 == h2
        assert PHASH_RE.match(h1)

    def test_low_frequency_optimization_preserves_committed_hash_algorithm(self):
        """Skipping unused DCT coefficients must not change persisted hashes."""
        from vlm_feedback_loop.services.phash import compute_phash

        fixtures = {
            "gradient": (
                lambda x, y: (x * 7 + y * 3) % 256,
                "94294a5569baeb56",
            ),
            "checker": (
                lambda x, y: 255 if (x // 4 + y // 4) % 2 else 0,
                "9ea87fa87ea8aa20",
            ),
            "impulse": (
                lambda x, y: 255 if (x, y) == (7, 19) else 0,
                "e11e1ee1e11ee1e1",
            ),
        }

        for pixel_fn, expected in fixtures.values():
            img = Image.new("L", (32, 32))
            img.putdata([pixel_fn(x, y) for y in range(32) for x in range(32)])
            assert compute_phash(img) == expected


# ═══════════════════════════════════════════════════════════════════════════
# Project not found
# ═══════════════════════════════════════════════════════════════════════════


class TestProjectNotFound:
    """Verify: all endpoints return 404 for non-existent project."""

    def test_ingest_project_not_found(self, tmp_path: Path):
        client = make_api_client(tmp_path)
        resp = client.post(
            "/v1/projects/nonexistent/examples:ingest",
            json={"examples": []},
        )
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Project not found"

    def test_image_project_not_found(self, tmp_path: Path):
        client = make_api_client(tmp_path)
        resp = client.get("/v1/projects/nonexistent/examples/k1/image")
        assert resp.status_code == 404
        # The image route resolves the example directly; a missing project
        # surfaces as the example lookup failing.
        assert "not found" in resp.json()["detail"]

    def test_query_project_not_found(self, tmp_path: Path):
        client = make_api_client(tmp_path)
        resp = client.get("/v1/projects/nonexistent/examples")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Project not found"


# ═══════════════════════════════════════════════════════════════════════════
# Image-root enforcement + input validation
# ═══════════════════════════════════════════════════════════════════════════


class TestIngestImageRoot:
    """Ingest must enforce IMAGE_ROOT at the door: a row whose storage_ref
    escapes the root would let a later Teacher/embedding
    call base64-ship an arbitrary host file to a NIM endpoint, so the
    out-of-root reference must never be persisted."""

    def test_nonloopback_ingest_requires_image_root(self, tmp_path: Path):
        """Direct ingest cannot bypass a disabled network filesystem boundary."""
        image = make_test_image(tmp_path / "images" / "secret.jpg")
        client = make_api_client(
            tmp_path,
            BIND_HOST="0.0.0.0",
            IMAGE_ROOT=None,
        )
        pid = create_project_via_api(client)["project_id"]

        result = _ingest(
            client,
            pid,
            [{"example_key": "k1", "storage_ref": str(image)}],
        )

        assert result["results"][0]["status"] == "error"
        assert result["results"][0]["error_code"] == "path_not_allowed"
        assert "Configure IMAGE_ROOT" in result["results"][0]["error"]

    def test_ingest_rejects_storage_ref_outside_roots(self, tmp_path: Path):
        allowed = tmp_path / "allowed"
        allowed.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        out_img = make_test_image(outside / "secret.jpg")

        client = make_api_client(tmp_path, IMAGE_ROOT=str(allowed))
        pid = create_project_via_api(client)["project_id"]
        r = _ingest(client, pid, [{"example_key": "k1", "storage_ref": str(out_img)}])
        assert r["results"][0]["status"] == "error"
        assert r["results"][0]["error_code"] == "path_not_allowed"

        # No row was persisted, so serving 404s (not a broken 200).
        resp = client.get(f"/v1/projects/{pid}/examples/k1/image")
        assert resp.status_code == 404

    def test_ingest_allows_storage_ref_inside_roots(self, tmp_path: Path):
        allowed = tmp_path / "allowed"
        allowed.mkdir()
        in_img = make_test_image(allowed / "ok.jpg")

        client = make_api_client(tmp_path, IMAGE_ROOT=str(allowed))
        pid = create_project_via_api(client)["project_id"]
        r = _ingest(client, pid, [{"example_key": "k1", "storage_ref": str(in_img)}])
        assert r["results"][0]["status"] == "created"


class TestServeImageRoot:
    """Serving must enforce IMAGE_ROOT, not just browse.

    Defense in depth for rows ingested before the root boundary was configured:
    ingest is the primary gate (see TestIngestImageRoot), but the serve
    path re-checks so a pre-existing out-of-root row cannot be read either.
    """

    def test_serve_403_for_preexisting_out_of_root_row(self, tmp_path: Path):
        from sqlalchemy.orm import Session

        from vlm_feedback_loop.db.base import utc_now
        from vlm_feedback_loop.db.models.example import Example
        from vlm_feedback_loop.services.project_service import get_project_engine

        allowed = tmp_path / "allowed"
        allowed.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        out_img = make_test_image(outside / "secret.jpg")

        client = make_api_client(tmp_path, IMAGE_ROOT=str(allowed))
        pid = create_project_via_api(client)["project_id"]
        # Seed an out-of-root row directly, simulating a row ingested before
        # the root was tightened (ingest itself now rejects this).
        engine = get_project_engine(pid, str(tmp_path / "workspace"))
        assert engine is not None
        with Session(engine) as session:
            session.add(
                Example(
                    example_key="k1",
                    project_id=pid,
                    storage_ref=str(out_img),
                    ingested_at=utc_now(),
                    source_metadata={},
                    state="Unlabeled",
                    phash="a" * 16,
                )
            )
            session.commit()

        resp = client.get(f"/v1/projects/{pid}/examples/k1/image")
        assert resp.status_code == 403
        assert "outside IMAGE_ROOT" in resp.json()["detail"]

    def test_serve_200_when_storage_ref_inside_roots(self, tmp_path: Path):
        allowed = tmp_path / "allowed"
        allowed.mkdir()
        in_img = make_test_image(allowed / "ok.jpg")

        client = make_api_client(tmp_path, IMAGE_ROOT=str(allowed))
        pid = create_project_via_api(client)["project_id"]
        _ingest(client, pid, [{"example_key": "k1", "storage_ref": str(in_img)}])

        resp = client.get(f"/v1/projects/{pid}/examples/k1/image")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/jpeg"
        assert resp.content == in_img.read_bytes()

    def test_remap_403_when_new_prefix_outside_roots(self, tmp_path: Path):
        allowed = tmp_path / "allowed"
        allowed.mkdir()
        in_img = make_test_image(allowed / "ok.jpg")

        client = make_api_client(tmp_path, IMAGE_ROOT=str(allowed))
        pid = create_project_via_api(client)["project_id"]
        _ingest(client, pid, [{"example_key": "k1", "storage_ref": str(in_img)}])

        resp = client.post(
            f"/v1/projects/{pid}/examples:remap_paths",
            json={
                "old_prefix": str(allowed),
                "new_prefix": "/etc",
                "dry_run": False,
            },
        )
        assert resp.status_code == 403

        # The invariant the 403 protects: the remap was NOT applied.
        listed = client.get(f"/v1/projects/{pid}/examples")
        assert listed.json()["items"][0]["example"]["storage_ref"] == str(in_img)


class TestQueryCursorValidation:
    def test_malformed_cursor_returns_400_not_500(self, tmp_path: Path):
        client = make_api_client(tmp_path)
        pid = create_project_via_api(client)["project_id"]
        resp = client.get(f"/v1/projects/{pid}/examples?cursor=not-valid-base64!!!")
        assert resp.status_code == 400
        detail = resp.json()["detail"]
        assert "cursor" in detail
        assert "not-valid-base64!!!" in detail


class TestIngestIdentityValidation:
    """Ingest rejects unaddressable keys and unreachable states."""

    def test_key_with_slash_rejected(self, tmp_path: Path):
        client = make_api_client(tmp_path)
        pid = create_project_via_api(client)["project_id"]
        img = make_test_image(tmp_path / "img.jpg")
        r = _ingest(client, pid, [{"example_key": "a/b", "storage_ref": str(img)}])
        assert r["results"][0]["status"] == "error"
        assert "path separators" in r["results"][0]["error"]

    def test_unknown_state_rejected(self, tmp_path: Path):
        client = make_api_client(tmp_path)
        pid = create_project_via_api(client)["project_id"]
        img = make_test_image(tmp_path / "img.jpg")
        r = _ingest(
            client,
            pid,
            [{"example_key": "k1", "storage_ref": str(img), "state": "Bogus"}],
        )
        assert r["results"][0]["status"] == "error"
        assert "invalid state" in r["results"][0]["error"]
