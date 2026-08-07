# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the TAO workspace dataset upload service.

Covers:
- single and multipart uploads for the archive/sidecar pair
- pair-wide SHA-256 idempotency and lineage persistence
- frozen checksum and type-sensitive JSON equivalence
- project-export confinement and pathname-replacement resistance
- distinct TAO media and annotation references

The fake S3 client (``support.FakeS3Client``) is in-memory, so no
boto3 / moto dependency is required at test collection time.
"""

from __future__ import annotations

import io
import json
import os
import random
import tarfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from sqlalchemy.orm import Session

from conftest import make_settings
from support import FakeS3Client
from vlm_feedback_loop.db.deployment_models import TAODeploymentConfig
from vlm_feedback_loop.db.models.dataset_export import DatasetExport
from vlm_feedback_loop.services import tao_dataset_upload_service
from vlm_feedback_loop.services.hashing import sha256_file
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


def _artifact_paths(tmp_path: Path, stem: str) -> tuple[Path, Path, Any]:
    workspace = tmp_path / "workspace"
    export_root = workspace / "projects" / "proj-1" / "exports"
    export_root.mkdir(parents=True, exist_ok=True)
    return (
        export_root / f"{stem}.tar.gz",
        export_root / f"{stem}_annotations.json",
        make_settings(workspace),
    )


def _write_annotation_artifacts(
    archive_path: Path,
    sidecar_path: Path,
    *,
    archive_annotations: object,
    sidecar_annotations: object | None = None,
    filler: bytes | None = None,
) -> None:
    archive_json = json.dumps(
        archive_annotations,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    with tarfile.open(archive_path, "w:gz") as tf:
        member = tarfile.TarInfo("annotations.json")
        member.size = len(archive_json)
        tf.addfile(member, io.BytesIO(archive_json))
        if filler is not None:
            filler_member = tarfile.TarInfo("images/filler.bin")
            filler_member.size = len(filler)
            tf.addfile(filler_member, io.BytesIO(filler))

    sidecar_path.write_text(
        json.dumps(
            archive_annotations if sidecar_annotations is None else sidecar_annotations,
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )


def _persist_dataset_export(
    project_engine,
    archive_path: Path,
    *,
    annotations_path: Path | None = None,
    checksum_sha256: object | None = None,
) -> DatasetExport:
    from vlm_feedback_loop.db.base import generate_uuid4

    export = DatasetExport(
        dataset_export_id=generate_uuid4(),
        project_id="proj-1",
        dataset_intent="training",
        export_field_mode="all",
        guidance_id="guidance-1",
        label_tier_filter="verified_only",
        selection_definition_snapshot={},
        artifact_refs={
            "archive_path": str(archive_path),
            **(
                {"annotations_path": str(annotations_path)}
                if annotations_path is not None
                else {}
            ),
            **(
                {"checksum_sha256": checksum_sha256}
                if checksum_sha256 is not None
                else {}
            ),
        },
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


def _persist_frozen_export(
    project_engine,
    archive_path: Path,
    sidecar_path: Path,
    *,
    annotations: object | None = None,
    sidecar_annotations: object | None = None,
    filler: bytes | None = None,
) -> DatasetExport:
    archive_annotations = (
        [{"id": "sample-1", "conversations": []}]
        if annotations is None
        else annotations
    )
    _write_annotation_artifacts(
        archive_path,
        sidecar_path,
        archive_annotations=archive_annotations,
        sidecar_annotations=sidecar_annotations,
        filler=filler,
    )
    return _persist_dataset_export(
        project_engine,
        archive_path,
        annotations_path=sidecar_path,
        checksum_sha256=sha256_file(archive_path),
    )


async def _upload(
    project_engine,
    *,
    export: DatasetExport,
    s3_client: FakeS3Client,
    settings: Any,
    deployment_config: TAODeploymentConfig | None = None,
    **kwargs: Any,
):
    with Session(project_engine) as session:
        attached = session.merge(export)
        result = await upload_dataset_archive(
            session,
            dataset_export=attached,
            deployment_config=deployment_config or _make_deployment_config(),
            s3_client=s3_client,
            settings=settings,
            **kwargs,
        )
        session.commit()
        return result


# ── U1: small archive → put_object ──────────────────────────────────────────


class TestSmallArchiveUsesPutObject:
    @pytest.mark.asyncio
    async def test_small_archive_single_put(self, tmp_path: Path, project_engine):
        archive, sidecar, settings = _artifact_paths(tmp_path, "tiny")
        export = _persist_frozen_export(project_engine, archive, sidecar)
        s3 = FakeS3Client()
        result = await _upload(
            project_engine,
            export=export,
            s3_client=s3,
            settings=settings,
        )

        assert result.success is True
        assert result.already_uploaded is False
        # Both frozen representations are uploaded before lineage is written.
        methods = [m for m, _ in s3.calls]
        assert methods.count("put_object") == 2
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
        archive_put = next(
            kw
            for method, kw in s3.calls
            if method == "put_object" and kw["Key"].endswith(".tar.gz")
        )
        assert archive_put["Metadata"] == {SHA256_METADATA_KEY: sha256_file(archive)}


# ── U2: large archive → multipart ────────────────────────────────────────────


class TestLargeArchiveUsesMultipart:
    @pytest.mark.asyncio
    async def test_multipart_when_above_threshold(self, tmp_path: Path, project_engine):
        archive, sidecar, settings = _artifact_paths(tmp_path, "big")
        export = _persist_frozen_export(
            project_engine,
            archive,
            sidecar,
            filler=random.Random(0).randbytes(4096),
        )
        threshold = sidecar.stat().st_size + 1
        assert archive.stat().st_size > threshold
        s3 = FakeS3Client()
        result = await _upload(
            project_engine,
            export=export,
            s3_client=s3,
            settings=settings,
            multipart_threshold_bytes=threshold,
            multipart_part_size_bytes=max(1, threshold // 2),
        )

        assert result.success is True
        methods = [m for m, _ in s3.calls]
        assert methods.count("create_multipart_upload") == 1
        assert methods.count("upload_part") >= 2
        assert methods.count("complete_multipart_upload") == 1
        assert methods.count("put_object") == 1

    def test_default_threshold_is_8_mib(self):
        assert MULTIPART_THRESHOLD_BYTES == 8 * 1024 * 1024


# ── Idempotent re-upload via head_object + SHA-256 ──────────────────────────


class TestIdempotentReupload:
    @pytest.mark.asyncio
    async def test_head_object_match_skips_only_complete_pair(
        self, tmp_path: Path, project_engine
    ):
        archive, sidecar, settings = _artifact_paths(tmp_path, "same")
        export = _persist_frozen_export(project_engine, archive, sidecar)
        s3 = FakeS3Client()

        first = await _upload(
            project_engine,
            export=export,
            s3_client=s3,
            settings=settings,
        )
        assert first.success is True
        assert first.already_uploaded is False

        first_call_count = len(s3.calls)

        second = await _upload(
            project_engine,
            export=export,
            s3_client=s3,
            settings=settings,
        )

        assert second.success is True
        assert second.already_uploaded is True

        new_calls = s3.calls[first_call_count:]
        new_methods = [m for m, _ in new_calls]
        assert new_methods == ["head_object", "head_object"]

        assert second.bucket is not None
        assert second.annotation_key is not None
        del s3.objects[(second.bucket, second.annotation_key)]
        partial_call_count = len(s3.calls)
        partial = await _upload(
            project_engine,
            export=export,
            s3_client=s3,
            settings=settings,
        )
        assert partial.success is True
        assert partial.already_uploaded is False
        assert [method for method, _ in s3.calls[partial_call_count:]] == [
            "head_object",
            "head_object",
            "put_object",
        ]


# ── U4: DatasetExport upload fields persisted ────────────────────────────────


class TestDatasetUploadPersistence:
    @pytest.mark.asyncio
    async def test_persists_upload_ref_and_uri(self, tmp_path: Path, project_engine):
        archive, sidecar, settings = _artifact_paths(tmp_path, "persist")
        export = _persist_frozen_export(project_engine, archive, sidecar)
        s3 = FakeS3Client()
        deployment = _make_deployment_config(bucket="vlm-persist-bucket")
        result = await _upload(
            project_engine,
            export=export,
            s3_client=s3,
            settings=settings,
            deployment_config=deployment,
        )
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
        archive, sidecar, settings = _artifact_paths(tmp_path, "nobucket")
        export = _persist_frozen_export(project_engine, archive, sidecar)

        deployment = _make_deployment_config()
        deployment.tao_workspace_bucket = None

        s3 = FakeS3Client()
        result = await _upload(
            project_engine,
            export=export,
            s3_client=s3,
            settings=settings,
            deployment_config=deployment,
        )

        assert result.success is False
        assert "tao_workspace_bucket" in (result.error or "")
        assert s3.calls == []

    @pytest.mark.asyncio
    async def test_missing_archive_returns_error(self, tmp_path: Path, project_engine):
        archive, sidecar, settings = _artifact_paths(tmp_path, "missing")
        export = _persist_frozen_export(project_engine, archive, sidecar)
        archive.unlink()
        s3 = FakeS3Client()
        result = await _upload(
            project_engine,
            export=export,
            s3_client=s3,
            settings=settings,
        )

        assert result.success is False
        assert "integrity check failed" in (result.error or "").lower()
        # No S3 calls issued when the file doesn't exist.
        assert s3.calls == []

    @pytest.mark.asyncio
    async def test_put_failure_surfaces_error_and_preserves_row(
        self, tmp_path: Path, project_engine
    ):
        archive, sidecar, settings = _artifact_paths(tmp_path, "fails")
        export = _persist_frozen_export(project_engine, archive, sidecar)

        deployment = _make_deployment_config()
        s3 = FakeS3Client(raise_on_put=True)

        result = await _upload(
            project_engine,
            export=export,
            s3_client=s3,
            settings=settings,
            deployment_config=deployment,
        )
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

    @pytest.mark.asyncio
    async def test_sidecar_failure_writes_no_lineage(
        self, tmp_path: Path, project_engine
    ):
        archive, sidecar, settings = _artifact_paths(tmp_path, "sidecar-fails")
        export = _persist_frozen_export(project_engine, archive, sidecar)
        s3 = FakeS3Client()
        original_put = s3.put_object
        put_count = 0

        def fail_second_put(**kwargs):
            nonlocal put_count
            put_count += 1
            if put_count == 2:
                raise RuntimeError("sidecar transfer failed")
            return original_put(**kwargs)

        with patch.object(s3, "put_object", side_effect=fail_second_put):
            result = await _upload(
                project_engine,
                export=export,
                s3_client=s3,
                settings=settings,
            )

        assert result.success is False
        assert "sidecar" in (result.error or "").lower()
        with Session(project_engine) as session:
            reloaded = session.get(DatasetExport, export.dataset_export_id)
            assert reloaded is not None
            assert reloaded.dataset_upload_ref is None
            assert reloaded.dataset_upload_uri is None


# ── Sidecar annotations.json upload ───────────────────────


class TestSidecarAnnotationsUpload:
    """The frozen sidecar uploads as a SECOND object so cosmos-rl's
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
        archive, ann_path, settings = _artifact_paths(tmp_path, "separate")
        export = _persist_frozen_export(
            project_engine,
            archive,
            ann_path,
            annotations=[{"id": "x", "conversations": []}],
        )
        s3 = FakeS3Client()
        result = await _upload(
            project_engine,
            export=export,
            s3_client=s3,
            settings=settings,
        )

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
        assert result.annotation_upload_uri.endswith(ann_path.name)

    @pytest.mark.asyncio
    async def test_missing_sidecar_reference_fails_before_s3(
        self, tmp_path: Path, project_engine
    ):
        archive, sidecar, settings = _artifact_paths(tmp_path, "incomplete")
        _write_annotation_artifacts(
            archive,
            sidecar,
            archive_annotations=[],
        )
        export = _persist_dataset_export(
            project_engine,
            archive,
            checksum_sha256=sha256_file(archive),
        )
        s3 = FakeS3Client()
        result = await _upload(
            project_engine,
            export=export,
            s3_client=s3,
            settings=settings,
        )

        assert result.success is False
        assert "sidecar path" in (result.error or "").lower()
        assert s3.calls == []

    @pytest.mark.asyncio
    async def test_colliding_archive_and_sidecar_keys_fail_before_s3(
        self, tmp_path: Path, project_engine
    ):
        archive, _, settings = _artifact_paths(tmp_path, "placeholder")
        export_root = archive.parent
        archive = export_root / "archive" / "artifact"
        sidecar = export_root / "sidecar" / "artifact"
        archive.parent.mkdir()
        sidecar.parent.mkdir()
        export = _persist_frozen_export(project_engine, archive, sidecar)
        s3 = FakeS3Client()

        result = await _upload(
            project_engine,
            export=export,
            s3_client=s3,
            settings=settings,
        )

        assert result.success is False
        assert "same workspace object" in (result.error or "")
        assert s3.calls == []


class TestFrozenArtifactIntegrity:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("recorded_checksum", [None, "", "not-a-sha256", 123])
    async def test_missing_or_malformed_checksum_fails_before_s3(
        self,
        tmp_path: Path,
        project_engine,
        recorded_checksum: object | None,
    ):
        archive, sidecar, settings = _artifact_paths(tmp_path, "bad-checksum")
        _write_annotation_artifacts(
            archive,
            sidecar,
            archive_annotations=[],
        )
        export = _persist_dataset_export(
            project_engine,
            archive,
            annotations_path=sidecar,
            checksum_sha256=recorded_checksum,
        )
        s3 = FakeS3Client()

        result = await _upload(
            project_engine,
            export=export,
            s3_client=s3,
            settings=settings,
        )

        assert result.success is False
        assert "checksum" in (result.error or "").lower()
        assert s3.calls == []

    @pytest.mark.asyncio
    async def test_checksum_mismatch_fails_before_s3(self, tmp_path, project_engine):
        archive, sidecar, settings = _artifact_paths(tmp_path, "changed")
        export = _persist_frozen_export(project_engine, archive, sidecar)
        archive.write_bytes(archive.read_bytes() + b"changed")
        s3 = FakeS3Client()

        result = await _upload(
            project_engine,
            export=export,
            s3_client=s3,
            settings=settings,
        )

        assert result.success is False
        assert "checksum" in (result.error or "").lower()
        assert s3.calls == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize("outside_ref", ["archive", "sidecar"])
    async def test_artifact_outside_project_exports_fails_before_s3(
        self,
        tmp_path: Path,
        project_engine,
        outside_ref: str,
    ):
        archive, sidecar, settings = _artifact_paths(tmp_path, "inside")
        export = _persist_frozen_export(project_engine, archive, sidecar)
        outside_archive = tmp_path / "outside.tar.gz"
        outside_sidecar = tmp_path / "outside_annotations.json"
        _write_annotation_artifacts(
            outside_archive,
            outside_sidecar,
            archive_annotations=[{"id": "outside"}],
        )
        refs = dict(export.artifact_refs or {})
        if outside_ref == "archive":
            refs["archive_path"] = str(outside_archive)
            refs["checksum_sha256"] = sha256_file(outside_archive)
        else:
            refs["annotations_path"] = str(outside_sidecar)
        export.artifact_refs = refs
        s3 = FakeS3Client()

        result = await _upload(
            project_engine,
            export=export,
            s3_client=s3,
            settings=settings,
        )

        assert result.success is False
        assert "authorized root" in (result.error or "").lower()
        assert s3.calls == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "archive_annotations,sidecar_annotations",
        [
            (
                [{"id": "sample", "prompt": "classify"}],
                [{"id": "sample", "prompt": "describe"}],
            ),
            (
                [{"id": "sample", "label": {"valid": True}}],
                [{"id": "sample", "label": {"valid": False}}],
            ),
            ([{"value": True}], [{"value": 1}]),
            ([{"value": 1}], [{"value": 1.0}]),
            ([{"items": ["first", "second"]}], [{"items": ["second", "first"]}]),
        ],
        ids=["prompt", "label", "bool-int", "int-float", "list-order"],
    )
    async def test_semantic_difference_fails_before_s3(
        self,
        tmp_path: Path,
        project_engine,
        archive_annotations: object,
        sidecar_annotations: object,
    ):
        archive, sidecar, settings = _artifact_paths(tmp_path, "divergent")
        export = _persist_frozen_export(
            project_engine,
            archive,
            sidecar,
            annotations=archive_annotations,
            sidecar_annotations=sidecar_annotations,
        )
        s3 = FakeS3Client()

        result = await _upload(
            project_engine,
            export=export,
            s3_client=s3,
            settings=settings,
        )

        assert result.success is False
        assert "annotations" in (result.error or "").lower()
        assert s3.calls == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "corruption",
        [
            "not-an-archive",
            "missing-archive-member",
            "archive-member-is-directory",
            "malformed-archive-json",
            "malformed-sidecar-json",
        ],
    )
    async def test_unreadable_representation_fails_before_s3(
        self,
        tmp_path: Path,
        project_engine,
        corruption: str,
    ):
        archive, sidecar, settings = _artifact_paths(tmp_path, corruption)
        annotations = [{"id": "sample"}]
        _write_annotation_artifacts(
            archive,
            sidecar,
            archive_annotations=annotations,
        )
        if corruption == "not-an-archive":
            archive.write_bytes(b"not a tar archive")
        elif corruption in {
            "missing-archive-member",
            "archive-member-is-directory",
            "malformed-archive-json",
        }:
            member_name = (
                "different.json"
                if corruption == "missing-archive-member"
                else "annotations.json"
            )
            with tarfile.open(archive, "w:gz") as tf:
                member = tarfile.TarInfo(member_name)
                if corruption == "archive-member-is-directory":
                    member.type = tarfile.DIRTYPE
                    tf.addfile(member)
                else:
                    payload = (
                        b"{not-json"
                        if corruption == "malformed-archive-json"
                        else json.dumps(annotations).encode()
                    )
                    member.size = len(payload)
                    tf.addfile(member, io.BytesIO(payload))
        else:
            sidecar.write_text("{not-json", encoding="utf-8")
        export = _persist_dataset_export(
            project_engine,
            archive,
            annotations_path=sidecar,
            checksum_sha256=sha256_file(archive),
        )
        s3 = FakeS3Client()

        result = await _upload(
            project_engine,
            export=export,
            s3_client=s3,
            settings=settings,
        )

        assert result.success is False
        assert "annotations" in (result.error or "").lower()
        assert s3.calls == []

    @pytest.mark.asyncio
    async def test_formatting_key_order_and_unicode_escaping_are_equivalent(
        self, tmp_path: Path, project_engine
    ):
        archive, sidecar, settings = _artifact_paths(tmp_path, "equivalent")
        export = _persist_frozen_export(
            project_engine,
            archive,
            sidecar,
            annotations=[
                {
                    "id": "café",
                    "conversations": [],
                    "metadata": {"first": 1, "second": 2},
                }
            ],
            sidecar_annotations=[
                {
                    "metadata": {"second": 2, "first": 1},
                    "conversations": [],
                    "id": "café",
                }
            ],
        )
        s3 = FakeS3Client()

        result = await _upload(
            project_engine,
            export=export,
            s3_client=s3,
            settings=settings,
        )

        assert result.success is True
        assert [method for method, _ in s3.calls].count("put_object") == 2

    @pytest.mark.asyncio
    async def test_path_replacement_after_validation_cannot_change_uploaded_bytes(
        self, tmp_path: Path, project_engine
    ):
        archive, sidecar, settings = _artifact_paths(tmp_path, "original")
        export = _persist_frozen_export(project_engine, archive, sidecar)
        original_archive = archive.read_bytes()
        original_sidecar = sidecar.read_bytes()
        replacement_archive = archive.with_name("replacement.tar.gz")
        replacement_sidecar = sidecar.with_name("replacement_annotations.json")
        _write_annotation_artifacts(
            replacement_archive,
            replacement_sidecar,
            archive_annotations=[{"id": "replacement"}],
        )
        original_validate = tao_dataset_upload_service._validate_frozen_export_artifacts

        def validate_then_replace(**kwargs):
            digests = original_validate(**kwargs)
            os.replace(replacement_archive, archive)
            os.replace(replacement_sidecar, sidecar)
            return digests

        s3 = FakeS3Client()
        with patch.object(
            tao_dataset_upload_service,
            "_validate_frozen_export_artifacts",
            side_effect=validate_then_replace,
        ):
            result = await _upload(
                project_engine,
                export=export,
                s3_client=s3,
                settings=settings,
            )

        assert result.success is True
        assert result.bucket is not None
        assert result.key is not None
        assert result.annotation_key is not None
        assert s3.objects[(result.bucket, result.key)]["Body"] == original_archive
        assert (
            s3.objects[(result.bucket, result.annotation_key)]["Body"]
            == original_sidecar
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("mutated_artifact", ["archive", "sidecar"])
    async def test_same_inode_overwrite_after_validation_is_not_uploaded(
        self,
        tmp_path: Path,
        project_engine,
        mutated_artifact: str,
    ):
        archive, sidecar, settings = _artifact_paths(tmp_path, "same-inode")
        export = _persist_frozen_export(project_engine, archive, sidecar)
        target = archive if mutated_artifact == "archive" else sidecar
        original_inode = target.stat().st_ino
        original_validate = tao_dataset_upload_service._validate_frozen_export_artifacts

        def validate_then_overwrite(**kwargs):
            digests = original_validate(**kwargs)
            target.write_bytes(target.read_bytes() + b"\nchanged after validation")
            assert target.stat().st_ino == original_inode
            return digests

        s3 = FakeS3Client()
        with patch.object(
            tao_dataset_upload_service,
            "_validate_frozen_export_artifacts",
            side_effect=validate_then_overwrite,
        ):
            result = await _upload(
                project_engine,
                export=export,
                s3_client=s3,
                settings=settings,
            )

        assert result.success is False
        assert "changed after validation" in (result.error or "")
        uploaded_keys = [
            kwargs["Key"] for method, kwargs in s3.calls if method == "put_object"
        ]
        if mutated_artifact == "archive":
            assert uploaded_keys == []
        else:
            assert len(uploaded_keys) == 1
            assert uploaded_keys[0].endswith(".tar.gz")
        with Session(project_engine) as session:
            reloaded = session.get(DatasetExport, export.dataset_export_id)
            assert reloaded is not None
            assert reloaded.dataset_upload_ref is None
            assert reloaded.dataset_upload_uri is None

    @pytest.mark.asyncio
    async def test_mutation_during_archive_parse_cannot_repair_a_mismatched_pair(
        self, tmp_path: Path, project_engine
    ):
        """Archive semantics and hash come from one sequential byte stream."""
        archive, sidecar, settings = _artifact_paths(tmp_path, "stream-bound")
        export = _persist_frozen_export(project_engine, archive, sidecar)
        original_archive = archive.read_bytes()
        original_sidecar = sidecar.read_bytes()
        s3 = FakeS3Client()
        first = await _upload(
            project_engine,
            export=export,
            s3_client=s3,
            settings=settings,
        )
        assert first.success is True

        replacement_archive = archive.with_name("stream-bound-new.tar.gz")
        replacement_sidecar = sidecar.with_name("stream-bound-new.json")
        _write_annotation_artifacts(
            replacement_archive,
            replacement_sidecar,
            archive_annotations=[{"id": "replacement"}],
        )
        replacement_archive_bytes = replacement_archive.read_bytes()
        replacement_sidecar_bytes = replacement_sidecar.read_bytes()
        original_read = tao_dataset_upload_service._DigestingReader.read
        mutated = False

        def read_then_mutate(reader, size=-1):
            nonlocal mutated
            data = original_read(reader, size)
            if not mutated and data:
                mutated = True
                archive.write_bytes(replacement_archive_bytes)
                sidecar.write_bytes(replacement_sidecar_bytes)
            return data

        s3.calls.clear()
        with patch.object(
            tao_dataset_upload_service._DigestingReader,
            "read",
            read_then_mutate,
        ):
            second = await _upload(
                project_engine,
                export=export,
                s3_client=s3,
                settings=settings,
            )

        assert second.success is False
        assert "integrity check failed" in (second.error or "").lower()
        assert s3.calls == []
        assert first.bucket is not None
        assert first.key is not None
        assert first.annotation_key is not None
        assert s3.objects[(first.bucket, first.key)]["Body"] == original_archive
        assert (
            s3.objects[(first.bucket, first.annotation_key)]["Body"] == original_sidecar
        )

    @pytest.mark.asyncio
    async def test_same_inode_truncate_aborts_multipart_before_completion(
        self, tmp_path: Path, project_engine
    ):
        archive, sidecar, settings = _artifact_paths(tmp_path, "truncate")
        export = _persist_frozen_export(
            project_engine,
            archive,
            sidecar,
            filler=random.Random(1).randbytes(4096),
        )
        original_inode = archive.stat().st_ino
        original_validate = tao_dataset_upload_service._validate_frozen_export_artifacts

        def validate_then_truncate(**kwargs):
            digests = original_validate(**kwargs)
            archive.write_bytes(b"truncated after validation")
            assert archive.stat().st_ino == original_inode
            return digests

        s3 = FakeS3Client()
        with patch.object(
            tao_dataset_upload_service,
            "_validate_frozen_export_artifacts",
            side_effect=validate_then_truncate,
        ):
            result = await _upload(
                project_engine,
                export=export,
                s3_client=s3,
                settings=settings,
                multipart_threshold_bytes=128,
                multipart_part_size_bytes=64,
            )

        assert result.success is False
        assert "multipart upload" in (result.error or "")
        methods = [method for method, _ in s3.calls]
        assert methods.count("create_multipart_upload") == 1
        assert methods.count("upload_part") == 1
        assert methods.count("abort_multipart_upload") == 1
        assert "complete_multipart_upload" not in methods
        assert s3.objects == {}

    @pytest.mark.asyncio
    async def test_noncompleted_export_fails_before_s3(self, tmp_path, project_engine):
        archive, sidecar, settings = _artifact_paths(tmp_path, "running")
        export = _persist_frozen_export(project_engine, archive, sidecar)
        export.status = "running"
        s3 = FakeS3Client()

        result = await _upload(
            project_engine,
            export=export,
            s3_client=s3,
            settings=settings,
        )

        assert result.success is False
        assert "not completed" in (result.error or "").lower()
        assert s3.calls == []
