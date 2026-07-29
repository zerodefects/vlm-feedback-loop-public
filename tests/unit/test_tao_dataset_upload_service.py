# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the TAO workspace dataset upload service.

Covers:
- small archive uses put_object
- large archive uses multipart upload
- idempotent via head_object + SHA-256 match
- dataset_upload_ref + dataset_upload_uri persisted on DatasetExport
- build_tao_spec_reference shape parametrised (s3:// vs http://)

The fake S3 client (``support.FakeS3Client``) is in-memory, so no
boto3 / moto dependency is required at test collection time.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from support import FakeS3Client
from vlm_feedback_loop.db.deployment_models import TAODeploymentConfig
from vlm_feedback_loop.db.models.dataset_export import DatasetExport
from vlm_feedback_loop.services.tao_dataset_upload_service import (
    MULTIPART_THRESHOLD_BYTES,
    SHA256_METADATA_KEY,
    build_s3_key,
    build_tao_spec_reference,
    upload_dataset_archive,
)

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_deployment_config(*, bucket: str = "vlm-bucket") -> TAODeploymentConfig:
    """Build an in-memory TAODeploymentConfig (no DB session needed)."""
    return TAODeploymentConfig(
        tao_deployment_config_id="cfg-1",
        tao_workspace_id="ws-1",
        tao_workspace_name="ws-name",
        tao_workspace_cloud_type="seaweedfs",
        tao_workspace_bucket=bucket,
        tao_workspace_s3_endpoint_url_internal="http://seaweedfs-s3:8333",
        tao_workspace_s3_endpoint_url_external="http://127.0.0.1:8333",
        tao_workspace_s3_access_key_ref="TAO_WORKSPACE_S3_ACCESS_KEY",
        tao_workspace_s3_secret_key_ref="TAO_WORKSPACE_S3_SECRET_KEY",
        bootstrap_status="bootstrapped",
    )


def _write_archive(path: Path, size_bytes: int, *, pattern: bytes = b"x") -> None:
    """Write an archive file of the given size using repeating bytes."""
    with open(path, "wb") as fh:
        written = 0
        chunk = pattern * 4096
        while written < size_bytes:
            to_write = min(len(chunk), size_bytes - written)
            fh.write(chunk[:to_write])
            written += to_write


def _persist_dataset_export(project_engine, archive_path: Path) -> DatasetExport:
    from vlm_feedback_loop.db.base import generate_uuid4

    export = DatasetExport(
        dataset_export_id=generate_uuid4(),
        project_id="proj-1",
        dataset_intent="training",
        export_field_mode="all",
        guidance_id="guidance-1",
        label_tier_filter="verified_only",
        selection_definition_snapshot={},
        artifact_refs={"archive_path": str(archive_path)},
        manifest_ref=str(archive_path.parent / "manifest.json"),
        example_count=3,
    )
    with Session(project_engine) as session:
        session.add(export)
        session.commit()
        session.refresh(export)
        export_id = export.dataset_export_id
    # Return a detached instance callers can pass to the service.
    with Session(project_engine) as session:
        return session.query(DatasetExport).filter_by(dataset_export_id=export_id).one()


# ── U1: small archive → put_object ──────────────────────────────────────────


class TestSmallArchiveUsesPutObject:
    @pytest.mark.asyncio
    async def test_small_archive_single_put(self, tmp_path: Path, project_engine):
        archive = tmp_path / "tiny.tar.gz"
        _write_archive(archive, size_bytes=1024)  # 1 KiB

        export = _persist_dataset_export(project_engine, archive)
        s3 = FakeS3Client()
        deployment = _make_deployment_config()

        with Session(project_engine) as session:
            # Attach the detached export to this session
            export = session.merge(export)
            result = await upload_dataset_archive(
                session,
                dataset_export=export,
                archive_path=archive,
                deployment_config=deployment,
                s3_client=s3,
            )
            session.commit()

        assert result.success is True
        assert result.already_uploaded is False
        # put_object called exactly once; no multipart calls
        methods = [m for m, _ in s3.calls]
        assert methods.count("put_object") == 1
        assert all(
            m
            not in (
                "create_multipart_upload",
                "upload_part",
                "complete_multipart_upload",
            )
            for m in methods
        )
        # SHA-256 present in metadata
        put_call = next(kw for m, kw in s3.calls if m == "put_object")
        assert SHA256_METADATA_KEY in (put_call["Metadata"] or {})


# ── U2: large archive → multipart ────────────────────────────────────────────


class TestLargeArchiveUsesMultipart:
    @pytest.mark.asyncio
    async def test_multipart_when_above_threshold(self, tmp_path: Path, project_engine):
        archive = tmp_path / "big.tar.gz"
        # Use a small threshold to keep the test fast.
        size = 1_000_000
        _write_archive(archive, size_bytes=size)

        export = _persist_dataset_export(project_engine, archive)
        s3 = FakeS3Client()
        deployment = _make_deployment_config()

        with Session(project_engine) as session:
            export = session.merge(export)
            result = await upload_dataset_archive(
                session,
                dataset_export=export,
                archive_path=archive,
                deployment_config=deployment,
                s3_client=s3,
                multipart_threshold_bytes=256 * 1024,  # force multipart
                multipart_part_size_bytes=256 * 1024,  # force multiple parts
            )
            session.commit()

        assert result.success is True
        methods = [m for m, _ in s3.calls]
        assert methods.count("create_multipart_upload") == 1
        assert methods.count("upload_part") >= 2
        assert methods.count("complete_multipart_upload") == 1
        assert "put_object" not in methods

    def test_default_threshold_is_8_mib(self):
        assert MULTIPART_THRESHOLD_BYTES == 8 * 1024 * 1024


# ── Idempotent re-upload via head_object + SHA-256 ──────────────────────────


class TestIdempotentReupload:
    @pytest.mark.asyncio
    async def test_head_object_match_skips_put(self, tmp_path: Path, project_engine):
        archive = tmp_path / "same.tar.gz"
        _write_archive(archive, size_bytes=2048)

        export = _persist_dataset_export(project_engine, archive)
        s3 = FakeS3Client()
        deployment = _make_deployment_config()

        # First upload.
        with Session(project_engine) as session:
            export = session.merge(export)
            first = await upload_dataset_archive(
                session,
                dataset_export=export,
                archive_path=archive,
                deployment_config=deployment,
                s3_client=s3,
            )
            session.commit()
        assert first.success is True
        assert first.already_uploaded is False

        first_call_count = len(s3.calls)

        # Second upload of the SAME archive should short-circuit after HEAD.
        with Session(project_engine) as session:
            export = session.merge(export)
            second = await upload_dataset_archive(
                session,
                dataset_export=export,
                archive_path=archive,
                deployment_config=deployment,
                s3_client=s3,
            )
            session.commit()

        assert second.success is True
        assert second.already_uploaded is True

        new_calls = s3.calls[first_call_count:]
        new_methods = [m for m, _ in new_calls]
        # Second invocation: HEAD only, no PUT/multipart
        assert new_methods == ["head_object"]


# ── U4: DatasetExport upload fields persisted ────────────────────────────────


class TestDatasetUploadPersistence:
    @pytest.mark.asyncio
    async def test_persists_upload_ref_and_uri(self, tmp_path: Path, project_engine):
        archive = tmp_path / "persist.tar.gz"
        _write_archive(archive, size_bytes=512)

        export = _persist_dataset_export(project_engine, archive)
        s3 = FakeS3Client()
        deployment = _make_deployment_config(bucket="vlm-persist-bucket")

        with Session(project_engine) as session:
            export = session.merge(export)
            result = await upload_dataset_archive(
                session,
                dataset_export=export,
                archive_path=archive,
                deployment_config=deployment,
                s3_client=s3,
            )
            session.commit()
            export_id = export.dataset_export_id

        assert result.success is True
        assert result.upload_uri == f"s3://vlm-persist-bucket/{result.key}"

        # Read back from DB and verify fields are persisted.
        with Session(project_engine) as session:
            reloaded = (
                session.query(DatasetExport)
                .filter_by(dataset_export_id=export_id)
                .one()
            )
            assert reloaded.dataset_upload_ref == result.key
            assert reloaded.dataset_upload_uri == result.upload_uri


# ── U5: build_tao_spec_reference shape (parametrised) ────────────────────────


class TestBuildTaoSpecReferenceShapes:
    def test_reference_carries_full_key(self):
        # Default cloud_type on the test fixture is "seaweedfs"; TAO
        # FTMS 6.26.3 dispatches on URL scheme, so seaweedfs workspaces
        # MUST get seaweedfs:// (an s3:// reference surfaces as
        # KeyError: 'aws' from TAO's cloud handler).
        deployment = _make_deployment_config()
        key = build_s3_key(
            project_id="p1",
            dataset_export_id="d1",
            archive_name="a.tar.gz",
        )
        uri = build_tao_spec_reference(deployment, bucket="vlm-bucket", key=key)
        assert uri.startswith("seaweedfs://vlm-bucket/")
        assert uri.endswith(key)

    @pytest.mark.parametrize(
        "cloud_type,expected_prefix",
        [
            ("seaweedfs", "seaweedfs://"),
            ("aws", "s3://"),
            ("azure", "azure://"),
            ("", "s3://"),  # legacy / unset → s3:// fallback
        ],
    )
    def test_scheme_picked_from_cloud_type(self, cloud_type: str, expected_prefix: str):
        """Spec reference URL scheme MUST follow workspace cloud_type so
        TAO's container-side cloud handler dispatches correctly. On FTMS
        6.26.3, a seaweedfs workspace with an s3:// reference fails with
        KeyError: 'aws'."""
        deployment = _make_deployment_config()
        deployment.tao_workspace_cloud_type = cloud_type
        uri = build_tao_spec_reference(deployment, bucket="b", key="p/d/a.tar.gz")
        assert uri.startswith(expected_prefix)


# ── Guardrails ───────────────────────────────────────────────────────────────


class TestGuardrails:
    @pytest.mark.asyncio
    async def test_missing_bucket_returns_error(self, tmp_path: Path, project_engine):
        archive = tmp_path / "nobucket.tar.gz"
        _write_archive(archive, size_bytes=128)
        export = _persist_dataset_export(project_engine, archive)

        deployment = _make_deployment_config()
        deployment.tao_workspace_bucket = None

        s3 = FakeS3Client()
        with Session(project_engine) as session:
            export = session.merge(export)
            result = await upload_dataset_archive(
                session,
                dataset_export=export,
                archive_path=archive,
                deployment_config=deployment,
                s3_client=s3,
            )

        assert result.success is False
        assert "tao_workspace_bucket" in (result.error or "")
        assert s3.calls == []

    @pytest.mark.asyncio
    async def test_missing_archive_returns_error(self, tmp_path: Path, project_engine):
        nonexistent = tmp_path / "does-not-exist.tar.gz"
        archive_exists = tmp_path / "placeholder.tar.gz"
        _write_archive(archive_exists, size_bytes=32)
        export = _persist_dataset_export(project_engine, archive_exists)

        deployment = _make_deployment_config()
        s3 = FakeS3Client()

        with Session(project_engine) as session:
            export = session.merge(export)
            result = await upload_dataset_archive(
                session,
                dataset_export=export,
                archive_path=nonexistent,
                deployment_config=deployment,
                s3_client=s3,
            )

        assert result.success is False
        assert "Archive not found" in (result.error or "")
        # No S3 calls issued when the file doesn't exist.
        assert s3.calls == []

    @pytest.mark.asyncio
    async def test_put_failure_surfaces_error_and_preserves_row(
        self, tmp_path: Path, project_engine
    ):
        archive = tmp_path / "fails.tar.gz"
        _write_archive(archive, size_bytes=256)
        export = _persist_dataset_export(project_engine, archive)

        deployment = _make_deployment_config()
        s3 = FakeS3Client(raise_on_put=True)

        with Session(project_engine) as session:
            export = session.merge(export)
            result = await upload_dataset_archive(
                session,
                dataset_export=export,
                archive_path=archive,
                deployment_config=deployment,
                s3_client=s3,
            )
            session.commit()
            export_id = export.dataset_export_id

        assert result.success is False
        assert "S3 upload failed" in (result.error or "")

        # Row remains with upload fields NULL — re-retriable.
        with Session(project_engine) as session:
            reloaded = (
                session.query(DatasetExport)
                .filter_by(dataset_export_id=export_id)
                .one()
            )
            assert reloaded.dataset_upload_ref is None
            assert reloaded.dataset_upload_uri is None


# ── Sidecar annotations.json upload ───────────────────────


class TestSidecarAnnotationsUpload:
    """``upload_dataset_archive(annotations_path=…)`` uploads a SECOND
    object next to the tarball so cosmos-rl's
    ``custom.train_dataset.annotation_path`` (or evaluate's
    ``dataset.annotation_path``) is a distinct JSON URL.

    cosmos-rl's ``tao_sft_example.py`` opens
    ``annotation_path`` via ``json.load(open(annotation_path))``. If
    ``annotation_path`` shares the URL with the tarball, TAO's URL-to-
    path resolution maps both to the extracted directory and the
    cosmos-rl worker crashes with ``FileNotFoundError`` on the
    annotations file.
    """

    @pytest.mark.asyncio
    async def test_sidecar_uploaded_as_separate_object(
        self, tmp_path: Path, project_engine
    ):
        archive = tmp_path / "tiny.tar.gz"
        _write_archive(archive, size_bytes=1024)
        # The sidecar is the ground-truth annotations.json that TAO
        # opens directly. Use the canonical name pattern emitted by
        # dataset_export_service: ``{export_id}_annotations.json``.
        export = _persist_dataset_export(project_engine, archive)
        ann_name = f"{export.dataset_export_id}_annotations.json"
        ann_path = tmp_path / ann_name
        ann_path.write_text('[{"id":"x","conversations":[]}]', encoding="utf-8")

        s3 = FakeS3Client()
        deployment = _make_deployment_config()

        with Session(project_engine) as session:
            export = session.merge(export)
            result = await upload_dataset_archive(
                session,
                dataset_export=export,
                archive_path=archive,
                deployment_config=deployment,
                s3_client=s3,
                annotations_path=ann_path,
            )
            session.commit()

        assert result.success is True
        # Two distinct put_object calls: tarball + sidecar.
        put_keys = [kw["Key"] for m, kw in s3.calls if m == "put_object"]
        assert len(put_keys) == 2, put_keys
        # One key ends in .tar.gz (the archive); the other ends in
        # _annotations.json (the sidecar).
        assert any(k.endswith(".tar.gz") for k in put_keys)
        assert any(k.endswith("_annotations.json") for k in put_keys)
        # The two URLs that flow into cosmos-rl's spec are distinct —
        # this is the contract that prevents ``FileNotFoundError`` on
        # ``annotation_path``.
        assert result.spec_reference is not None
        assert result.annotation_spec_reference is not None
        assert result.spec_reference != result.annotation_spec_reference
        # Sidecar URL points at the .json sibling, not the .tar.gz.
        assert result.annotation_spec_reference.endswith("_annotations.json")
        assert result.annotation_upload_uri is not None
        assert result.annotation_upload_uri.endswith(ann_name)

    @pytest.mark.asyncio
    async def test_no_sidecar_no_annotation_spec_reference(
        self, tmp_path: Path, project_engine
    ):
        """When ``annotations_path`` is omitted, the result MUST NOT
        carry sidecar fields — preserves backward compat for callers
        that don't yet emit a sidecar."""
        archive = tmp_path / "tiny.tar.gz"
        _write_archive(archive, size_bytes=1024)
        export = _persist_dataset_export(project_engine, archive)
        s3 = FakeS3Client()
        deployment = _make_deployment_config()

        with Session(project_engine) as session:
            export = session.merge(export)
            result = await upload_dataset_archive(
                session,
                dataset_export=export,
                archive_path=archive,
                deployment_config=deployment,
                s3_client=s3,
            )
            session.commit()

        assert result.success is True
        # Single put_object call (the tarball only).
        put_calls = [m for m, _ in s3.calls if m == "put_object"]
        assert len(put_calls) == 1
        assert result.annotation_spec_reference is None
        assert result.annotation_upload_uri is None
        assert result.annotation_key is None
