# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for image transport service (services/image_transport.py).

Test images are generated programmatically via Pillow — no committed binaries.

Image transport is inline-base64-only: every image is read,
format-normalised, and encoded as a base64 data URL regardless of size or
endpoint mode. There is no NVCF Asset API large-image path — hosted
endpoints only accept inline base64 (see the image_transport module
docstring).
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

import pytest
from PIL import Image

from vlm_feedback_loop.services.image_transport import (
    prepare_images,
    read_and_normalize,
    to_base64_data_url,
)

# prepare_images resolves its transport cap from Settings by default, and
# the loader hard-exits on hosts without a canonical config.yaml (fresh CI
# runners) — every test here needs the hermetic config. The fixture
# supplies the shipped defaults, under which all pre-existing fixtures
# (<=400 px) pass through byte-identical.
pytestmark = pytest.mark.usefixtures("isolated_config")

# ── Fixtures: generate test images ──────────────────────────────────────────


@pytest.fixture()
def jpeg_path(tmp_path: Path) -> Path:
    """1×1 JPEG image."""
    p = tmp_path / "test.jpg"
    img = Image.new("RGB", (1, 1), color=(128, 64, 32))
    img.save(p, format="JPEG")
    return p


@pytest.fixture()
def png_path(tmp_path: Path) -> Path:
    """1×1 PNG image."""
    p = tmp_path / "test.png"
    img = Image.new("RGB", (1, 1), color=(10, 20, 30))
    img.save(p, format="PNG")
    return p


@pytest.fixture()
def bmp_path(tmp_path: Path) -> Path:
    """1×1 BMP image (needs transcoding to PNG)."""
    p = tmp_path / "test.bmp"
    img = Image.new("RGB", (1, 1), color=(50, 100, 150))
    img.save(p, format="BMP")
    return p


@pytest.fixture()
def webp_path(tmp_path: Path) -> Path:
    """1×1 WebP image (needs transcoding to PNG)."""
    p = tmp_path / "test.webp"
    img = Image.new("RGB", (1, 1), color=(200, 100, 50))
    img.save(p, format="WEBP")
    return p


@pytest.fixture()
def tiff_path(tmp_path: Path) -> Path:
    """1×1 single-page TIFF image (needs transcoding to PNG)."""
    p = tmp_path / "test.tiff"
    img = Image.new("RGB", (1, 1), color=(80, 160, 240))
    img.save(p, format="TIFF")
    return p


@pytest.fixture()
def large_png_path(tmp_path: Path) -> Path:
    """A large PNG (>180 KB encoded) — proves transport does not branch on size.

    Uses os.urandom for pixel data — incompressible noise guarantees a large
    PNG even with maximum deflate compression.
    """
    import os

    p = tmp_path / "large.png"
    width, height = 400, 400
    raw = os.urandom(width * height * 3)  # 480,000 bytes of random RGB
    img = Image.frombytes("RGB", (width, height), raw)
    img.save(p, format="PNG")
    return p


# ── read_and_normalize ──────────────────────────────────────────────────────


class TestReadAndNormalize:
    def test_reads_jpeg_keeps_format(self, jpeg_path):
        data, mime = read_and_normalize(str(jpeg_path))
        assert mime == "image/jpeg"
        assert len(data) > 0
        # Verify it's valid JPEG (starts with FFD8)
        assert data[:2] == b"\xff\xd8"

    def test_reads_png_keeps_format(self, png_path):
        data, mime = read_and_normalize(str(png_path))
        assert mime == "image/png"
        assert data[:4] == b"\x89PNG"

    def test_transcodes_bmp_to_png(self, bmp_path):
        data, mime = read_and_normalize(str(bmp_path))
        assert mime == "image/png"
        assert data[:4] == b"\x89PNG"

    def test_transcodes_webp_to_png(self, webp_path):
        data, mime = read_and_normalize(str(webp_path))
        assert mime == "image/png"
        assert data[:4] == b"\x89PNG"

    def test_transcodes_tiff_to_png(self, tiff_path):
        data, mime = read_and_normalize(str(tiff_path))
        assert mime == "image/png"
        assert data[:4] == b"\x89PNG"

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError, match="Image not found"):
            read_and_normalize("/nonexistent/path/image.jpg")

    def test_unreadable_file_raises(self, tmp_path):
        bad = tmp_path / "corrupt.jpg"
        bad.write_bytes(b"not an image at all")
        with pytest.raises(ValueError, match="Cannot open image"):
            read_and_normalize(str(bad))


# ── to_base64_data_url ─────────────────────────────────────────────────────


class TestBase64DataUrl:
    def test_produces_valid_data_url(self):
        url = to_base64_data_url(b"\x89PNG\r\n\x1a\n", "image/png")
        assert url.startswith("data:image/png;base64,")
        # Verify round-trip
        b64_part = url.split(",", 1)[1]
        decoded = base64.b64decode(b64_part)
        assert decoded == b"\x89PNG\r\n\x1a\n"

    def test_jpeg_mime_type(self):
        url = to_base64_data_url(b"\xff\xd8\xff", "image/jpeg")
        assert url.startswith("data:image/jpeg;base64,")


# ── prepare_images ──────────────────────────────────────────────────────────


class TestPrepareImages:
    @pytest.mark.asyncio
    async def test_inline_base64(self, png_path):
        result = await prepare_images([str(png_path)])
        assert result.success is True
        assert len(result.images) == 1
        img = result.images[0]
        assert img.transport_mode == "base64_inline"
        assert img.format_transmitted == "image/png"
        assert img.content_part["type"] == "image_url"
        assert img.content_part["image_url"]["url"].startswith("data:image/png;base64,")

    @pytest.mark.asyncio
    async def test_large_image_still_inline(self, large_png_path):
        """Transport does not branch on size — large images are inline too.

        There is no NVCF Asset API large-image path; hosted endpoints only
        accept inline base64.
        """
        data, _ = read_and_normalize(str(large_png_path))
        assert (
            len(data) > 184320
        )  # comfortably above 180 KB — any size-based branch would trigger

        result = await prepare_images([str(large_png_path)])
        assert result.success is True
        assert result.images[0].transport_mode == "base64_inline"

    @pytest.mark.asyncio
    async def test_preserves_input_order(self, jpeg_path, png_path, bmp_path):
        refs = [str(jpeg_path), str(png_path), str(bmp_path)]
        result = await prepare_images(refs)
        assert len(result.images) == 3
        assert result.images[0].format_transmitted == "image/jpeg"
        assert result.images[1].format_transmitted == "image/png"
        assert result.images[2].format_transmitted == "image/png"  # BMP → PNG

    @pytest.mark.asyncio
    async def test_storage_ref_never_in_content_part(self, jpeg_path):
        result = await prepare_images([str(jpeg_path)])
        content_str = str(result.images[0].content_part)
        assert str(jpeg_path) not in content_str
        assert "/tmp" not in content_str or "base64" in content_str

    @pytest.mark.asyncio
    async def test_missing_file_marks_failure(self):
        result = await prepare_images(["/nonexistent/img.jpg"])
        assert result.success is False
        assert result.images[0].error is not None
        assert "not found" in result.images[0].error.lower()

    @pytest.mark.asyncio
    async def test_cpu_work_runs_off_event_loop(self, monkeypatch, png_path):
        """The CPU-bound decode/encode must run in a worker thread, not the loop.

        Regression guard for the event-loop freeze: ``prepare_images`` hands
        ``read_and_normalize`` to ``run_in_thread`` (asyncio.to_thread), so the
        heavy PIL work executes off the single event loop.  We assert it runs on
        a *different* thread than the one running the coroutine.
        """
        import threading

        from vlm_feedback_loop.services import image_transport

        loop_thread = threading.get_ident()
        seen: list[int] = []
        real = image_transport.read_and_normalize

        def spy(storage_ref, max_longest_edge=None):
            seen.append(threading.get_ident())
            return real(storage_ref, max_longest_edge=max_longest_edge)

        monkeypatch.setattr(image_transport, "read_and_normalize", spy)

        result = await prepare_images([str(png_path)])

        assert result.success is True
        assert seen, "read_and_normalize was never invoked"
        assert all(tid != loop_thread for tid in seen), (
            "image prep ran on the event-loop thread — it must run via run_in_thread"
        )


# ── Transport downscale (IMAGE_TRANSPORT_MAX_LONGEST_EDGE) ──────────────────


class TestTransportDownscale:
    """Client-side max-longest-edge resize before base64 encoding.

    Oversized source images blow hosted-provider gateway limits on
    multi-image ICL requests (full-res phone photos → 504 at the provider
    edge); the transport cap resizes only what is transmitted — stored
    originals are untouched.
    """

    @pytest.fixture()
    def big_jpeg_path(self, tmp_path: Path) -> Path:
        """4000x3000 JPEG — above any sensible cap."""
        p = tmp_path / "big.jpg"
        Image.new("RGB", (4000, 3000), color=(90, 120, 150)).save(p, format="JPEG")
        return p

    @pytest.fixture()
    def big_png_path(self, tmp_path: Path) -> Path:
        """3000x4000 portrait PNG with alpha."""
        p = tmp_path / "big.png"
        Image.new("RGBA", (3000, 4000), color=(90, 120, 150, 200)).save(p, format="PNG")
        return p

    def test_downscales_jpeg_over_limit_stays_jpeg(self, big_jpeg_path):
        data, mime = read_and_normalize(str(big_jpeg_path), max_longest_edge=2090)
        assert mime == "image/jpeg"
        out = Image.open(io.BytesIO(data))
        assert max(out.size) == 2090
        assert out.size == (2090, 1568)  # aspect preserved (4:3)

    def test_downscales_png_over_limit_stays_png_with_alpha(self, big_png_path):
        data, mime = read_and_normalize(str(big_png_path), max_longest_edge=2090)
        assert mime == "image/png"
        out = Image.open(io.BytesIO(data))
        assert max(out.size) == 2090
        assert out.mode == "RGBA"  # alpha survives the resize

    def test_under_limit_passes_through_byte_identical(self, big_jpeg_path):
        raw = Path(big_jpeg_path).read_bytes()
        data, mime = read_and_normalize(str(big_jpeg_path), max_longest_edge=4096)
        assert data == raw  # no re-encode when no resize is needed
        assert mime == "image/jpeg"

    def test_none_disables_downscale(self, big_jpeg_path):
        raw = Path(big_jpeg_path).read_bytes()
        data, _ = read_and_normalize(str(big_jpeg_path), max_longest_edge=None)
        assert data == raw

    def test_zero_disables_downscale(self, big_jpeg_path):
        raw = Path(big_jpeg_path).read_bytes()
        data, _ = read_and_normalize(str(big_jpeg_path), max_longest_edge=0)
        assert data == raw

    def test_exif_orientation_baked_in_before_reencode(self, tmp_path):
        # A 4000x3000 JPEG tagged orientation=6 (rotate 90 CW) renders as
        # 3000x4000 portrait; after downscale the EXIF block is gone, so
        # the pixels themselves must carry the rotation.
        p = tmp_path / "rotated.jpg"
        img = Image.new("RGB", (4000, 3000), color=(10, 200, 10))
        exif = img.getexif()
        exif[0x0112] = 6  # Orientation: Rotate 90 CW
        img.save(p, format="JPEG", exif=exif)

        data, _ = read_and_normalize(str(p), max_longest_edge=2000)
        out = Image.open(io.BytesIO(data))
        assert out.size == (1500, 2000)  # portrait after baked-in rotation

    @pytest.mark.asyncio
    async def test_prepare_images_explicit_limit(self, big_jpeg_path, jpeg_path):
        result = await prepare_images(
            [str(big_jpeg_path), str(jpeg_path)], max_longest_edge=1000
        )
        assert result.success is True
        for prepared in result.images:
            url = prepared.content_part["image_url"]["url"]
            payload = base64.b64decode(url.split(",", 1)[1])
            img = Image.open(io.BytesIO(payload))
            assert max(img.size) <= 1000

    @pytest.mark.asyncio
    async def test_prepare_images_defaults_from_settings(
        self, big_jpeg_path, monkeypatch
    ):
        from vlm_feedback_loop import config as config_module

        settings = config_module.get_settings().model_copy(
            update={"IMAGE_TRANSPORT_MAX_LONGEST_EDGE": 1200}
        )
        monkeypatch.setattr(config_module, "get_settings", lambda: settings)

        result = await prepare_images([str(big_jpeg_path)])
        url = result.images[0].content_part["image_url"]["url"]
        img = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
        assert max(img.size) == 1200

    def test_settings_normalize_zero_to_disabled(self):
        from vlm_feedback_loop.config import Settings

        s = Settings(WORKSPACE_ROOT="/tmp/ws", IMAGE_TRANSPORT_MAX_LONGEST_EDGE=0)
        assert s.IMAGE_TRANSPORT_MAX_LONGEST_EDGE is None
