# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Image transport service.

Reads images from ``storage_ref``, normalises formats, encodes as base64
data URLs, and returns a batched preparation result that preserves input
order.  All images — query and ICL examples — are transmitted inline as
base64 data URLs.

The sole transport mode is ``base64_inline``.

There is deliberately no NVCF Asset API upload mode for large images: the
hosted OpenAI-compatible endpoint at ``integrate.api.nvidia.com`` does not
accept asset references (it returns ``"Only base64 data URLs are supported
for now."``), and NVIDIA's NVCF deprecation retired the ``/v2/nvcf/assets``
endpoints outright with the guidance "send payloads directly during
invocation" — which is exactly what inline base64 does.
"""

from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from PIL import Image, ImageOps

from vlm_feedback_loop.services.authorized_file import open_authorized_image

if TYPE_CHECKING:
    from vlm_feedback_loop.config import Settings

logger = logging.getLogger("vlm_feedback_loop.services.image_transport")


# Sentinel: prepare_images resolves the transport downscale limit from
# Settings unless the caller passes an explicit value (tests do). An enum
# member rather than object() so pyright narrows the ``is`` check on it.
class _Sentinel(Enum):
    FROM_SETTINGS = 0


_FROM_SETTINGS = _Sentinel.FROM_SETTINGS

# Formats kept as-is (no re-encoding).  Everything else transcodes to PNG.
_PASSTHROUGH_FORMATS = {"JPEG", "PNG"}

# ── Supported image formats — single source of truth ────────────────────────
# The Blueprint supports JPEG, PNG, WebP, BMP, and single-page TIFF.
# Filesystem browsing, ingestion validation, and the image-serving endpoint all
# derive from the constants below.

# PIL format name → MIME type mapping
_FORMAT_TO_MIME: dict[str, str] = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "BMP": "image/bmp",
    "WEBP": "image/webp",
    "TIFF": "image/tiff",
}

# PIL format names accepted at ingestion (``example_service``).
ACCEPTED_PIL_FORMATS: frozenset[str] = frozenset(_FORMAT_TO_MIME)

# File extension → MIME type for the same format set (image serving).
EXT_TO_MIME: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
}

# File extensions recognised as images (filesystem browse, ingestion).
SUPPORTED_IMAGE_EXTENSIONS: frozenset[str] = frozenset(EXT_TO_MIME)


# ── Result types ────────────────────────────────────────────────────────────


@dataclass
class PreparedImage:
    """A single image prepared for NIM dispatch."""

    content_part: dict[str, Any]
    transport_mode: str  # base64_inline
    format_transmitted: str  # MIME type
    error: str | None = None


@dataclass
class BatchPrepareResult:
    """Result of preparing a batch of images for one model invocation."""

    images: list[PreparedImage]
    success: bool  # False if any required image failed


# ── Read and normalise ──────────────────────────────────────────────────────


def read_and_normalize(
    storage_ref: str,
    max_longest_edge: int | None = None,
    *,
    settings: Settings | None = None,
) -> tuple[bytes, str]:
    """Read an image from ``storage_ref`` and normalise to a NIM-safe format.

    JPEG stays JPEG (no lossy re-encoding).  PNG stays PNG.  All other
    supported formats (BMP, WebP, single-page TIFF) are transcoded to PNG.

    When ``max_longest_edge`` is set and the image's longest edge exceeds
    it, the image is downscaled (aspect-preserving, LANCZOS) before
    encoding — JPEG re-encodes as JPEG (quality 90), everything else as
    PNG.  EXIF orientation is applied before resizing so phone photos keep
    their upright orientation once the EXIF block is dropped by
    re-encoding.  Images at or under the limit pass through untouched.

    Returns:
        ``(image_bytes, mime_type)``

    Raises:
        FileNotFoundError: file does not exist or is not a regular file.
        PermissionError: file is outside the current image-access policy.
        ValueError: unsupported or corrupt image format.
    """
    if settings is None:
        from vlm_feedback_loop.config import get_settings

        settings = get_settings()

    try:
        opened_file = open_authorized_image(storage_ref, settings)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Image not found: {storage_ref}") from exc

    with (
        opened_file as opened,
        opened.open_binary() as stream,
    ):
        try:
            img = Image.open(stream)
        except Exception as exc:
            raise ValueError(f"Cannot open image {storage_ref}: {exc}") from exc

        with img:
            pil_format = img.format
            if pil_format is None:
                raise ValueError(f"Unknown image format: {storage_ref}")

            # Reject animated GIFs and multi-page TIFFs.
            if pil_format == "GIF" and getattr(img, "is_animated", False):
                raise ValueError(f"Animated GIFs are not supported: {storage_ref}")
            if pil_format == "TIFF":
                try:
                    img.seek(1)
                    raise ValueError(
                        f"Multi-page TIFFs are not supported: {storage_ref}"
                    )
                except EOFError:
                    img.seek(0)

            # None means no downscale (unset, disabled, or already small).
            limit: int | None = None
            if (
                max_longest_edge is not None
                and max_longest_edge > 0
                and max(img.size) > max_longest_edge
            ):
                limit = max_longest_edge

            if pil_format in _PASSTHROUGH_FORMATS and limit is None:
                # Read from the same authorized inode Pillow inspected.
                stream.seek(0)
                raw = stream.read()
                mime = _FORMAT_TO_MIME.get(pil_format, "image/png")
                return (raw, mime)

            output_img = img
            try:
                if limit is not None:
                    # Bake in EXIF orientation before re-encoding drops it.
                    output_img = ImageOps.exif_transpose(img)
                    original_size = output_img.size
                    output_img.thumbnail((limit, limit), Image.Resampling.LANCZOS)
                    logger.debug(
                        "transport downscale %s: %sx%s -> %sx%s (limit %s)",
                        storage_ref,
                        original_size[0],
                        original_size[1],
                        output_img.size[0],
                        output_img.size[1],
                        limit,
                    )
                    if pil_format == "JPEG":
                        jpeg_img = (
                            output_img
                            if output_img.mode == "RGB"
                            else output_img.convert("RGB")
                        )
                        try:
                            buf = io.BytesIO()
                            jpeg_img.save(buf, format="JPEG", quality=90)
                            return (buf.getvalue(), "image/jpeg")
                        finally:
                            if jpeg_img is not output_img:
                                jpeg_img.close()

                # Transcode every other supported format to PNG.
                png_img = (
                    output_img
                    if output_img.mode in ("RGB", "RGBA")
                    else output_img.convert("RGB")
                )
                try:
                    buf = io.BytesIO()
                    png_img.save(buf, format="PNG")
                    return (buf.getvalue(), "image/png")
                finally:
                    if png_img is not output_img:
                        png_img.close()
            finally:
                if output_img is not img:
                    output_img.close()


def to_base64_data_url(image_bytes: bytes, mime_type: str) -> str:
    """Encode image bytes as a base64 data URL.

    Public: also used by ``clip_embedding_service`` to feed the
    embeddings API, so the transport and embedding paths encode
    identically.
    """
    b64 = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{b64}"


def _prepare_single(
    storage_ref: str,
    max_longest_edge: int | None,
    settings: Settings,
) -> PreparedImage:
    """Read, normalise, and base64-encode one image (CPU-bound, sync).

    Kept as a plain sync function so ``prepare_images`` can hand the whole
    decode → downscale → re-encode → base64 chain to a worker thread via
    ``run_in_thread`` — none of it may run on the event loop.
    """
    try:
        image_bytes, mime_type = read_and_normalize(
            storage_ref,
            max_longest_edge=max_longest_edge,
            settings=settings,
        )
    except (FileNotFoundError, PermissionError, ValueError) as exc:
        return PreparedImage(
            content_part={},
            transport_mode="base64_inline",
            format_transmitted="",
            error=str(exc),
        )

    data_url = to_base64_data_url(image_bytes, mime_type)
    return PreparedImage(
        content_part={
            "type": "image_url",
            "image_url": {"url": data_url},
        },
        transport_mode="base64_inline",
        format_transmitted=mime_type,
    )


# ── Batched preparation ─────────────────────────────────────────────────────


async def prepare_images(
    storage_refs: list[str],
    max_longest_edge: int | None | _Sentinel = _FROM_SETTINGS,
    *,
    settings: Settings | None = None,
) -> BatchPrepareResult:
    """Prepare a batch of images for one model invocation.

    Each image is read from disk, normalised to a NIM-safe format, and
    encoded as an inline base64 data URL.  This applies uniformly to all
    endpoint modes (hosted, self-hosted, local): the hosted
    OpenAI-compatible endpoint accepts only base64 data URLs, and
    self-hosted/local NIMs accept inline base64 as well.

    Args:
        storage_refs: Filesystem paths to images (from Example.storage_ref).
        max_longest_edge: Transport downscale limit in pixels.  Defaults to
            ``Settings.IMAGE_TRANSPORT_MAX_LONGEST_EDGE``; pass ``None``
            explicitly to force send-as-is regardless of settings.

    Returns:
        ``BatchPrepareResult`` with images in input order and a success
        flag.  A read/normalise failure on any image sets ``success=False``
        — the caller should not dispatch the model request when a required
        image failed.

    Image prep is CPU-bound (PIL decode, LANCZOS downscale, re-encode,
    base64), so each image is normalised in a worker thread via
    ``run_in_thread`` — never on the event loop.  The images are prepared
    sequentially so the returned list preserves input order; the
    ``await`` between images also yields the loop so SSE streams, the
    health probe, and concurrent requests are not starved during a
    multi-image ICL invocation.
    """
    if settings is None:
        from vlm_feedback_loop.config import get_settings

        settings = get_settings()
    if isinstance(max_longest_edge, _Sentinel):
        max_longest_edge = settings.IMAGE_TRANSPORT_MAX_LONGEST_EDGE

    # Lazy import keeps image_transport free of a hard services.background
    # dependency at module load (mirrors the config import above).
    from vlm_feedback_loop.services.background import run_in_thread

    prepared: list[PreparedImage] = []
    for ref in storage_refs:
        prepared.append(
            await run_in_thread(_prepare_single, ref, max_longest_edge, settings)
        )

    any_failure = any(img.error is not None for img in prepared)
    return BatchPrepareResult(images=prepared, success=not any_failure)
