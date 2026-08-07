"""Behavior checks for compression-insensitive architecture image drift."""

from __future__ import annotations

import importlib.util
import io
from pathlib import Path
from types import ModuleType

from PIL import Image


def _load_renderer() -> ModuleType:
    script = Path(__file__).parents[2] / "scripts" / "render_architecture_diagram.py"
    spec = importlib.util.spec_from_file_location("render_architecture_diagram", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _png_bytes(color: tuple[int, int, int], *, compress_level: int) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (8, 5), color).save(
        output, format="PNG", compress_level=compress_level
    )
    return output.getvalue()


def test_pixel_check_ignores_png_compression_metadata(tmp_path: Path) -> None:
    """Equivalent rendered pixels pass even when their PNG bytes differ."""

    renderer = _load_renderer()
    committed = tmp_path / "architecture.png"
    committed.write_bytes(_png_bytes((17, 17, 17), compress_level=0))
    rendered = _png_bytes((17, 17, 17), compress_level=9)

    assert committed.read_bytes() != rendered
    assert renderer._same_pixels(rendered, committed) is True


def test_pixel_check_rejects_a_visible_change(tmp_path: Path) -> None:
    """Any changed diagram pixel remains a release-blocking drift."""

    renderer = _load_renderer()
    committed = tmp_path / "architecture.png"
    committed.write_bytes(_png_bytes((17, 17, 17), compress_level=9))
    rendered = _png_bytes((118, 185, 0), compress_level=9)

    assert renderer._same_pixels(rendered, committed) is False
