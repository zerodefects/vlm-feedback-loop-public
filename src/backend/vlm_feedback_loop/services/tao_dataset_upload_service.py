# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-training dataset upload to the TAO workspace S3 bucket.

When a training suite is created the Blueprint streams the .tar.gz
archive produced by dataset export into the workspace's S3
bucket. TAO's own execution containers (not the Blueprint host) read
the archive from S3 at training time; the Blueprint never uploads files
directly through TAO.

Key layout::

    s3://{bucket}/vlm-feedback-loop/projects/{project_id}/
        dataset_exports/{dataset_export_id}/{archive_name}

Contract:

- Uses a small ``S3ClientProtocol`` so tests can inject a fake client
  without depending on ``boto3`` / ``moto``.
- Threshold for multipart: 8 MiB (configurable).
- Before any S3 call, the archive must match its recorded checksum and its
  root ``annotations.json`` must equal the separate sidecar as a complete,
  type-sensitive JSON value.
- Both files are opened beneath the owning project's ``exports/`` directory;
  validation and upload consume those same descriptors, and the exact bytes
  sent are rehashed to detect in-place mutation after validation.
- Idempotency: ``head_object`` + SHA-256 comparison short-circuits a
  re-upload only when both representations are already staged.
- Persistence: after a successful upload, the caller-provided session
  is used to finalise ``DatasetExport.dataset_upload_ref`` and
  ``DatasetExport.dataset_upload_uri``. The upload itself happens
  outside the caller's long DB transaction to respect the short-
  transaction discipline — the caller commits the row first,
  issues the upload, then finalises the upload fields in a tiny second
  transaction.
- ``build_tao_spec_reference(...)`` is the SOLE point that decides the
  URI shape TAO job specs reference (``s3://...`` vs an HTTP URL via
  the workspace's internal endpoint). Changing the shape touches no
  other code.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Protocol, cast, runtime_checkable

from sqlalchemy.orm import Session

from vlm_feedback_loop.config import Settings
from vlm_feedback_loop.db.deployment_models import TAODeploymentConfig
from vlm_feedback_loop.db.models.dataset_export import DatasetExport
from vlm_feedback_loop.services.authorized_file import (
    OpenedRegularFile,
    open_regular_file_beneath,
)
from vlm_feedback_loop.services.project_service import project_dir_path

logger = logging.getLogger("vlm_feedback_loop.services.tao_dataset_upload")

# 8 MiB — below this size we use a single PUT, above we use multipart.
MULTIPART_THRESHOLD_BYTES = 8 * 1024 * 1024
# 8 MiB part size for multipart uploads (SeaweedFS + AWS S3 both accept this).
MULTIPART_PART_SIZE_BYTES = 8 * 1024 * 1024

# Metadata key that carries the content SHA-256 on the uploaded object so
# idempotent re-upload can skip work (HEAD + compare).
SHA256_METADATA_KEY = "dataset-export-sha256"


class OpenedFileChangedError(RuntimeError):
    """The bytes read for upload no longer match the validated file."""


# ── S3 client protocol ───────────────────────────────────────────────────────


@runtime_checkable
class S3ClientProtocol(Protocol):
    """Narrow subset of the S3 API this service uses.

    Tests ship a hand-rolled fake; runtime uses ``boto3`` wrapped via
    ``asyncio.to_thread``. Keeping the interface narrow avoids coupling
    to boto3 and makes the TAO S3 endpoint swappable. Parameter names
    (``Bucket``, ``Key``, …) mirror boto3's PascalCase kwargs.
    """

    def head_object(self, *, Bucket: str, Key: str) -> dict[str, Any]: ...

    def put_object(
        self,
        *,
        Bucket: str,
        Key: str,
        Body: bytes,
        Metadata: dict[str, str] | None = None,
    ) -> dict[str, Any]: ...

    def create_multipart_upload(
        self,
        *,
        Bucket: str,
        Key: str,
        Metadata: dict[str, str] | None = None,
    ) -> dict[str, Any]: ...

    def upload_part(
        self,
        *,
        Bucket: str,
        Key: str,
        PartNumber: int,
        UploadId: str,
        Body: bytes,
    ) -> dict[str, Any]: ...

    def complete_multipart_upload(
        self,
        *,
        Bucket: str,
        Key: str,
        UploadId: str,
        MultipartUpload: dict[str, Any],
    ) -> dict[str, Any]: ...

    def abort_multipart_upload(
        self,
        *,
        Bucket: str,
        Key: str,
        UploadId: str,
    ) -> dict[str, Any]: ...


# ── Factory for the real boto3 client ────────────────────────────────────────


def build_s3_client(
    deployment_config: TAODeploymentConfig,
    *,
    settings: Settings | None = None,
    region_name: str | None = None,
) -> S3ClientProtocol:
    """Build a real boto3 S3 client using the workspace deployment config.

    The endpoint URL is the **external** view — what the Blueprint host
    uses to reach the bucket. TAO containers use the internal URL (never
    passed to boto3 on the Blueprint side).

    Secrets (access key / secret key) are not stored on the config — they
    come from environment variables. ``settings`` supplies them;
    when omitted, the process-wide singleton from
    ``vlm_feedback_loop.config.get_settings()`` is used. ``region_name``
    is forwarded to the boto3 constructor when set (the upload path
    omits it; the polling artifact-download path pins ``us-east-1``).

    Both workspace-S3 consumers (uploads here, ``tao_polling_service``
    artifact downloads) construct through this factory and share the same
    SigV4 + bounded-retry configuration; only the polling path's
    not-configured precheck (an actionable ``RuntimeError``, versus this
    factory's ``ValueError``) stays at its call site.
    """
    # Lazy import so tests that never touch the real client avoid
    # importing boto3 at collection time.
    from vlm_feedback_loop.services._boto3_shim import BotoConfig, boto3

    endpoint = deployment_config.tao_workspace_s3_endpoint_url_external
    if not endpoint:
        raise ValueError(
            "TAODeploymentConfig.tao_workspace_s3_endpoint_url_external "
            "is not configured"
        )

    if settings is None:
        from vlm_feedback_loop.config import get_settings

        settings = get_settings()
    access_key = settings.TAO_WORKSPACE_S3_ACCESS_KEY
    secret_key = settings.TAO_WORKSPACE_S3_SECRET_KEY

    if not access_key or not secret_key:
        raise ValueError(
            "TAO_WORKSPACE_S3_ACCESS_KEY / TAO_WORKSPACE_S3_SECRET_KEY "
            "must be configured for workspace uploads"
        )

    kwargs: dict[str, Any] = {
        "endpoint_url": endpoint,
        "aws_access_key_id": access_key,
        "aws_secret_access_key": secret_key,
        "config": BotoConfig(
            signature_version="s3v4",
            retries={"max_attempts": 3, "mode": "standard"},
        ),
    }
    if region_name is not None:
        kwargs["region_name"] = region_name
    return boto3.client("s3", **kwargs)


# ── Public key + reference helpers ───────────────────────────────────────────


def build_s3_key(*, project_id: str, dataset_export_id: str, archive_name: str) -> str:
    """Canonical S3 key layout."""
    return (
        f"vlm-feedback-loop/projects/{project_id}/dataset_exports/"
        f"{dataset_export_id}/{archive_name}"
    )


def build_tao_spec_reference(
    deployment_config: TAODeploymentConfig,
    *,
    bucket: str,
    key: str,
) -> str:
    """Build the URI that TAO job specs should reference for this object.

    This is the SOLE point that decides what URL scheme TAO specs read
    from. The scheme depends on the workspace's ``cloud_type``
    (behavior verified against FTMS 6.26.3):

    * ``cloud_type="seaweedfs"`` → ``seaweedfs://{bucket}/{key}``. TAO's
      cloud handler dispatches on the URL scheme; using ``s3://`` against
      a seaweedfs workspace surfaces as ``KeyError: 'aws'`` from
      ``download_files_from_cloud`` (cloud_data has no ``aws`` entry).
    * ``cloud_type="aws"`` → ``s3://{bucket}/{key}``. Standard S3.
    * ``cloud_type="azure"`` → ``azure://{bucket}/{key}``.
    """
    cloud_type = (deployment_config.tao_workspace_cloud_type or "").strip().lower()
    if cloud_type == "seaweedfs":
        return f"seaweedfs://{bucket}/{key}"
    if cloud_type == "azure":
        return f"azure://{bucket}/{key}"
    # Default to s3:// (covers cloud_type="aws" + any unset/legacy state).
    return f"s3://{bucket}/{key}"


# ── Upload result ────────────────────────────────────────────────────────────


@dataclass
class UploadResult:
    """Outcome of an upload_dataset_archive call.

    ``spec_reference`` is the URL of the .tar.gz archive (the ``media``
    side of the cosmos-rl spec). ``annotation_spec_reference`` is the
    URL of the sidecar annotations.json (the ``annotation`` side).
    Cosmos-RL's CustomDataset opens annotation_path as a JSON file, so
    media_path and annotation_path MUST be distinct URLs.
    """

    success: bool
    dataset_export_id: str
    bucket: str | None = None
    key: str | None = None
    upload_uri: str | None = None
    spec_reference: str | None = None
    annotation_key: str | None = None
    annotation_upload_uri: str | None = None
    annotation_spec_reference: str | None = None
    sha256: str | None = None
    already_uploaded: bool = False
    error: str | None = None


# ── Upload orchestration ─────────────────────────────────────────────────────


def do_single_put(
    s3_client: S3ClientProtocol,
    *,
    bucket: str,
    key: str,
    opened_file: OpenedRegularFile,
    sha256: str,
) -> None:
    with opened_file.open_binary() as fh:
        fh.seek(0)
        body = fh.read()
        actual_sha256 = hashlib.sha256(body).hexdigest()
        if actual_sha256 != sha256:
            raise OpenedFileChangedError(
                "File changed after validation and before upload"
            )
        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=body,
            Metadata={SHA256_METADATA_KEY: sha256},
        )


def do_multipart_put(
    s3_client: S3ClientProtocol,
    *,
    bucket: str,
    key: str,
    opened_file: OpenedRegularFile,
    sha256: str,
    part_size_bytes: int = MULTIPART_PART_SIZE_BYTES,
) -> None:
    create = s3_client.create_multipart_upload(
        Bucket=bucket,
        Key=key,
        Metadata={SHA256_METADATA_KEY: sha256},
    )
    upload_id = create["UploadId"]

    parts: list[dict[str, Any]] = []
    try:
        uploaded_digest = hashlib.sha256()
        with opened_file.open_binary() as fh:
            fh.seek(0)
            part_number = 1
            while True:
                chunk = fh.read(part_size_bytes)
                if not chunk:
                    break
                uploaded_digest.update(chunk)
                resp = s3_client.upload_part(
                    Bucket=bucket,
                    Key=key,
                    PartNumber=part_number,
                    UploadId=upload_id,
                    Body=chunk,
                )
                parts.append({"ETag": resp.get("ETag", ""), "PartNumber": part_number})
                part_number += 1

        if uploaded_digest.hexdigest() != sha256:
            raise OpenedFileChangedError(
                "File changed after validation and during multipart upload"
            )

        s3_client.complete_multipart_upload(
            Bucket=bucket,
            Key=key,
            UploadId=upload_id,
            MultipartUpload={"Parts": parts},
        )
    except Exception:
        # Best-effort abort so SeaweedFS / S3 does not accumulate orphan parts.
        try:
            s3_client.abort_multipart_upload(Bucket=bucket, Key=key, UploadId=upload_id)
        except Exception:
            logger.exception("Failed to abort multipart upload %s", upload_id)
        raise


def already_uploaded(
    s3_client: S3ClientProtocol,
    *,
    bucket: str,
    key: str,
    sha256: str,
) -> bool:
    """Return True when head_object shows a matching SHA-256 in metadata."""
    try:
        resp = s3_client.head_object(Bucket=bucket, Key=key)
    except Exception:
        return False
    meta_raw: Any = resp.get("Metadata") or {}
    meta: dict[str, Any] = (
        cast("dict[str, Any]", meta_raw) if isinstance(meta_raw, dict) else {}
    )
    return meta.get(SHA256_METADATA_KEY) == sha256


def _run_upload_sync(
    s3_client: S3ClientProtocol,
    *,
    bucket: str,
    key: str,
    opened_file: OpenedRegularFile,
    sha256: str,
    multipart_threshold_bytes: int,
    multipart_part_size_bytes: int,
) -> bool:
    """Synchronous upload path — wrapped in asyncio.to_thread by the caller.

    Returns whether the matching object was already present.
    """
    if already_uploaded(s3_client, bucket=bucket, key=key, sha256=sha256):
        logger.info(
            "Dataset archive already uploaded: s3://%s/%s (sha256=%s)",
            bucket,
            key,
            sha256[:12],
        )
        return True

    size_bytes = opened_file.stat_result.st_size
    if size_bytes > multipart_threshold_bytes:
        do_multipart_put(
            s3_client,
            bucket=bucket,
            key=key,
            opened_file=opened_file,
            sha256=sha256,
            part_size_bytes=multipart_part_size_bytes,
        )
    else:
        do_single_put(
            s3_client,
            bucket=bucket,
            key=key,
            opened_file=opened_file,
            sha256=sha256,
        )
    return False


def _json_values_equal(left: Any, right: Any) -> bool:
    """Compare parsed JSON without conflating booleans and numbers."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        left_dict = cast("dict[str, Any]", left)
        right_dict = cast("dict[str, Any]", right)
        return left_dict.keys() == right_dict.keys() and all(
            _json_values_equal(left_dict[key], right_dict[key]) for key in left_dict
        )
    if isinstance(left, list):
        left_list = cast("list[Any]", left)
        right_list = cast("list[Any]", right)
        return len(left_list) == len(right_list) and all(
            _json_values_equal(left_item, right_item)
            for left_item, right_item in zip(left_list, right_list, strict=True)
        )
    return bool(left == right)


def _artifact_path(dataset_export: DatasetExport, key: str, label: str) -> Path:
    refs = dataset_export.artifact_refs
    raw_path: Any = refs.get(key) if isinstance(refs, dict) else None
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(
            f"DatasetExport has no {label}; re-export the dataset before TAO upload"
        )
    return Path(raw_path)


class _DigestingReader:
    """Hash each source byte as a streaming parser consumes it."""

    def __init__(self, source: BinaryIO) -> None:
        self._source = source
        self._digest = hashlib.sha256()

    def read(self, size: int = -1) -> bytes:
        data = self._source.read(size)
        self._digest.update(data)
        return data

    def hexdigest(self) -> str:
        return self._digest.hexdigest()


def _validate_frozen_export_artifacts(
    *,
    dataset_export: DatasetExport,
    archive: OpenedRegularFile,
    annotations: OpenedRegularFile,
) -> tuple[str, str]:
    """Validate both frozen representations through their authorized inodes."""
    artifact_refs = dataset_export.artifact_refs
    recorded_checksum_raw: Any = (
        artifact_refs.get("checksum_sha256")
        if isinstance(artifact_refs, dict)
        else None
    )
    if (
        not isinstance(recorded_checksum_raw, str)
        or len(recorded_checksum_raw) != 64
        or any(char not in "0123456789abcdefABCDEF" for char in recorded_checksum_raw)
    ):
        raise ValueError(
            "DatasetExport has no valid recorded archive checksum; re-export "
            "the dataset before TAO upload"
        )

    with archive.open_binary() as archive_stream:
        archive_stream.seek(0)
        digesting_stream = _DigestingReader(archive_stream)
        missing_annotations = object()
        archive_annotations: Any = missing_annotations
        try:
            with tarfile.open(
                fileobj=cast("BinaryIO", digesting_stream),
                mode="r|gz",
            ) as tf:
                for member in tf:
                    if member.name != "annotations.json":
                        continue
                    if archive_annotations is not missing_annotations:
                        raise ValueError(
                            "Archive contains duplicate root annotations.json members"
                        )
                    if not member.isfile():
                        raise ValueError(
                            "Archive annotations.json is not a regular file"
                        )
                    extracted = tf.extractfile(member)
                    if extracted is None:
                        raise ValueError("Archive annotations.json cannot be read")
                    archive_annotations = json.loads(extracted.read())
            # Tar readers may stop at the end marker before the compressed
            # source reaches EOF. Drain the remaining bytes so the digest is
            # the complete archive hash, not merely the parsed prefix.
            while digesting_stream.read(1024 * 1024):
                pass
        except (EOFError, OSError, tarfile.TarError, UnicodeError) as exc:
            raise ValueError(f"Archive annotations.json is unreadable: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"Archive annotations.json is malformed: {exc}") from exc
        if archive_annotations is missing_annotations:
            raise ValueError("Archive does not contain annotations.json")
        actual_checksum = digesting_stream.hexdigest()
        if actual_checksum.lower() != recorded_checksum_raw.lower():
            raise ValueError(
                "Archive checksum does not match the frozen DatasetExport; "
                "re-export the dataset before TAO upload"
            )

    with annotations.open_binary() as annotations_stream:
        annotations_stream.seek(0)
        try:
            annotations_bytes = annotations_stream.read()
        except (OSError, UnicodeError) as exc:
            raise ValueError(f"Annotations sidecar is unreadable: {exc}") from exc
        annotations_checksum = hashlib.sha256(annotations_bytes).hexdigest()
        try:
            sidecar_annotations = json.loads(annotations_bytes)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Annotations sidecar is malformed: {exc}") from exc

    if not _json_values_equal(archive_annotations, sidecar_annotations):
        raise ValueError(
            "Annotations sidecar does not match the frozen archive annotations"
        )
    return actual_checksum, annotations_checksum


async def upload_dataset_archive(
    session: Session,
    *,
    dataset_export: DatasetExport,
    deployment_config: TAODeploymentConfig,
    s3_client: S3ClientProtocol,
    settings: Settings,
    multipart_threshold_bytes: int = MULTIPART_THRESHOLD_BYTES,
    multipart_part_size_bytes: int = MULTIPART_PART_SIZE_BYTES,
) -> UploadResult:
    """Upload a dataset archive to the workspace bucket and persist lineage.

    Both paths come from the completed DatasetExport record. Before any S3
    operation, the archive must match its recorded SHA-256 and its root
    ``annotations.json`` must be type-sensitively equal to the required
    standalone sidecar. Validation and upload reuse the same two descriptors
    opened beneath the canonical project ``exports/`` directory.

    The caller owns the ``Session``. This function:
    - Reads fields off the provided ``dataset_export`` instance (no
      additional DB reads required).
    - Performs the upload outside any long transaction.
    - Uses the session to persist ``dataset_upload_ref`` +
      ``dataset_upload_uri`` in a short write transaction. The caller
      commits.
    """
    project_id = dataset_export.project_id
    dataset_export_id = dataset_export.dataset_export_id

    bucket = deployment_config.tao_workspace_bucket
    if not bucket:
        return UploadResult(
            success=False,
            dataset_export_id=dataset_export_id,
            error=(
                "TAODeploymentConfig.tao_workspace_bucket is not configured "
                "— run `vlm-feedback-loop tao-bootstrap` first."
            ),
        )
    if dataset_export.status != "completed":
        return UploadResult(
            success=False,
            dataset_export_id=dataset_export_id,
            bucket=bucket,
            error=(
                "Dataset export integrity check failed: DatasetExport is not "
                "completed; wait for completion or re-export before TAO upload"
            ),
        )

    try:
        archive_path = _artifact_path(
            dataset_export,
            "archive_path",
            "archive path",
        )
        annotations_path = _artifact_path(
            dataset_export,
            "annotations_path",
            "annotations sidecar path",
        )
    except ValueError as exc:
        return UploadResult(
            success=False,
            dataset_export_id=dataset_export_id,
            bucket=bucket,
            error=f"Dataset export integrity check failed: {exc}",
        )

    archive_name = archive_path.name
    key = build_s3_key(
        project_id=project_id,
        dataset_export_id=dataset_export_id,
        archive_name=archive_name,
    )
    spec_reference = build_tao_spec_reference(deployment_config, bucket=bucket, key=key)

    annotation_key = build_s3_key(
        project_id=project_id,
        dataset_export_id=dataset_export_id,
        archive_name=annotations_path.name,
    )
    annotation_spec_reference = build_tao_spec_reference(
        deployment_config,
        bucket=bucket,
        key=annotation_key,
    )
    if annotation_key == key:
        return UploadResult(
            success=False,
            dataset_export_id=dataset_export_id,
            bucket=bucket,
            key=key,
            spec_reference=spec_reference,
            annotation_key=annotation_key,
            annotation_spec_reference=annotation_spec_reference,
            error=(
                "Dataset export integrity check failed: archive and annotations "
                "sidecar map to the same workspace object; re-export the dataset"
            ),
        )
    export_root = project_dir_path(settings.WORKSPACE_ROOT, project_id) / "exports"
    try:
        with (
            open_regular_file_beneath(archive_path, export_root) as archive,
            open_regular_file_beneath(annotations_path, export_root) as annotations,
        ):
            try:
                archive_sha256, annotations_sha256 = await asyncio.to_thread(
                    _validate_frozen_export_artifacts,
                    dataset_export=dataset_export,
                    archive=archive,
                    annotations=annotations,
                )
            except (OSError, ValueError) as exc:
                return UploadResult(
                    success=False,
                    dataset_export_id=dataset_export_id,
                    bucket=bucket,
                    key=key,
                    spec_reference=spec_reference,
                    annotation_key=annotation_key,
                    annotation_spec_reference=annotation_spec_reference,
                    error=f"Dataset export integrity check failed: {exc}",
                )
            try:
                archive_already_uploaded = await asyncio.to_thread(
                    _run_upload_sync,
                    s3_client,
                    bucket=bucket,
                    key=key,
                    opened_file=archive,
                    sha256=archive_sha256,
                    multipart_threshold_bytes=multipart_threshold_bytes,
                    multipart_part_size_bytes=multipart_part_size_bytes,
                )
            except OpenedFileChangedError as exc:
                return UploadResult(
                    success=False,
                    dataset_export_id=dataset_export_id,
                    bucket=bucket,
                    key=key,
                    spec_reference=spec_reference,
                    annotation_key=annotation_key,
                    annotation_spec_reference=annotation_spec_reference,
                    error=f"Dataset export integrity check failed: {exc}",
                )
            except Exception as exc:
                logger.exception(
                    "Dataset archive upload failed for %s (bucket=%s key=%s)",
                    dataset_export_id,
                    bucket,
                    key,
                )
                return UploadResult(
                    success=False,
                    dataset_export_id=dataset_export_id,
                    bucket=bucket,
                    key=key,
                    spec_reference=spec_reference,
                    annotation_key=annotation_key,
                    annotation_spec_reference=annotation_spec_reference,
                    error=f"Archive S3 upload failed: {exc}",
                )
            try:
                sidecar_already_uploaded = await asyncio.to_thread(
                    _run_upload_sync,
                    s3_client,
                    bucket=bucket,
                    key=annotation_key,
                    opened_file=annotations,
                    sha256=annotations_sha256,
                    multipart_threshold_bytes=multipart_threshold_bytes,
                    multipart_part_size_bytes=multipart_part_size_bytes,
                )
            except OpenedFileChangedError as exc:
                return UploadResult(
                    success=False,
                    dataset_export_id=dataset_export_id,
                    bucket=bucket,
                    key=key,
                    spec_reference=spec_reference,
                    annotation_key=annotation_key,
                    annotation_spec_reference=annotation_spec_reference,
                    error=f"Dataset export integrity check failed: {exc}",
                )
            except Exception as exc:
                logger.exception(
                    "Annotations sidecar upload failed for %s (bucket=%s key=%s)",
                    dataset_export_id,
                    bucket,
                    annotation_key,
                )
                return UploadResult(
                    success=False,
                    dataset_export_id=dataset_export_id,
                    bucket=bucket,
                    key=key,
                    spec_reference=spec_reference,
                    annotation_key=annotation_key,
                    annotation_spec_reference=annotation_spec_reference,
                    error=f"Annotations sidecar S3 upload failed: {exc}",
                )
    except (FileNotFoundError, PermissionError) as exc:
        return UploadResult(
            success=False,
            dataset_export_id=dataset_export_id,
            bucket=bucket,
            key=key,
            spec_reference=spec_reference,
            annotation_key=annotation_key,
            annotation_spec_reference=annotation_spec_reference,
            error=f"Dataset export integrity check failed: {exc}",
        )

    upload_uri = f"s3://{bucket}/{key}"
    annotation_upload_uri = f"s3://{bucket}/{annotation_key}"

    # Persist lineage only after both representations are staged.
    dataset_export.dataset_upload_ref = key
    dataset_export.dataset_upload_uri = upload_uri
    session.add(dataset_export)

    return UploadResult(
        success=True,
        dataset_export_id=dataset_export_id,
        bucket=bucket,
        key=key,
        upload_uri=upload_uri,
        spec_reference=spec_reference,
        annotation_key=annotation_key,
        annotation_upload_uri=annotation_upload_uri,
        annotation_spec_reference=annotation_spec_reference,
        sha256=archive_sha256,
        already_uploaded=archive_already_uploaded and sidecar_already_uploaded,
    )
