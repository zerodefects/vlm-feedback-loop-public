# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DCT-based 64-bit perceptual hash.

Pure-Python implementation using only Pillow and the ``math`` standard library.
No numpy, scipy, or imagehash dependency.

Algorithm (``dct_phash_64``):
  1. Convert image to grayscale, resize to 32×32.
  2. Compute 2D DCT of the 32×32 pixel matrix.
  3. Compute the top-left 8×8 submatrix (low frequencies).
  4. Compute the median of those 64 values.
  5. Each bit is 1 if value > median, 0 otherwise.
  6. Return a 16-character hex string (64 bits).

The DCT basis matrix is precomputed once at module load. Only the 8×8 block
consumed by the hash is evaluated; computing the unused high-frequency
coefficients would add Python/GIL work without changing the result.
"""

from __future__ import annotations

import contextlib
import logging
import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from PIL.Image import Image

logger = logging.getLogger("vlm_feedback_loop.phash")

# ── Constants ───────────────────────────────────────────────────────────────

_N = 32  # Resize dimension for DCT
_LOW = 8  # Low-frequency submatrix size (8×8 = 64 bits)

# ── Precomputed DCT-II basis matrix (32×32) ────────────────────────────────


def _build_dct_matrix(n: int) -> list[list[float]]:
    """Build the *n*×*n* DCT-II basis matrix.

    ``DCT[k][i] = scale(k) * cos(π * (2i + 1) * k / (2N))``

    where ``scale(0) = sqrt(1/N)`` and ``scale(k>0) = sqrt(2/N)``.
    """
    matrix: list[list[float]] = []
    scale_0 = math.sqrt(1.0 / n)
    scale_k = math.sqrt(2.0 / n)
    for k in range(n):
        row: list[float] = []
        s = scale_0 if k == 0 else scale_k
        for i in range(n):
            row.append(s * math.cos(math.pi * (2 * i + 1) * k / (2 * n)))
        matrix.append(row)
    return matrix


_DCT: list[list[float]] = _build_dct_matrix(_N)


# ── Low-frequency DCT (pure Python, exact persisted-hash compatibility) ────


def _low_frequency_dct(pixels: list[list[float]]) -> list[list[float]]:
    """Return only the 8×8 DCT block used by the 64-bit hash.

    The prior implementation computed both full 32×32 matrix products and
    discarded 960 of the 1,024 coefficients. Computing the required rows and
    columns only reduces the Python/GIL-bound multiply-add count by ~6.4×.
    Loop order and the precomputed basis values are unchanged, so the retained
    coefficients — and every persisted ``dct_phash_64`` value — stay bit-for-
    bit compatible.
    """
    # First product: only the first _LOW rows of DCT @ pixels are needed.
    tmp: list[list[float]] = [[0.0] * _N for _ in range(_LOW)]
    for r in range(_LOW):
        basis_row = _DCT[r]
        for c in range(_N):
            value = 0.0
            for k in range(_N):
                value += basis_row[k] * pixels[k][c]
            tmp[r][c] = value

    # Second product: retain only the first _LOW columns. DCT^T[k][c]
    # equals DCT[c][k], avoiding a full transposed matrix allocation.
    result: list[list[float]] = [[0.0] * _LOW for _ in range(_LOW)]
    for r in range(_LOW):
        tmp_row = tmp[r]
        for c in range(_LOW):
            basis_row = _DCT[c]
            value = 0.0
            for k in range(_N):
                value += tmp_row[k] * basis_row[k]
            result[r][c] = value
    return result


# ── Public API ──────────────────────────────────────────────────────────────


def compute_phash(img: Image) -> str:
    """Compute a DCT-based 64-bit perceptual hash for *img*.

    Returns a 16-character lowercase hex string.
    """
    # 1. Grayscale + resize to 32×32. draft() first lets the JPEG decoder
    #    emit a reduced-resolution frame (DCT scaling to 1/2, 1/4, 1/8)
    #    instead of the full image — we only need a 32×32 downsample, so on
    #    a multi-megapixel photo this cuts the decode ~5–8× (a 1.6 MB JPEG
    #    drops from ~190 ms to ~30 ms) with no material effect on the hash.
    #    No-op for formats without draft support (e.g. PNG) and for images
    #    already loaded.
    with contextlib.suppress(Exception):  # draft is best-effort
        img.draft("L", (_N, _N))
    converted: Any = img.convert("L")
    gray: Any = converted.resize((_N, _N), resample=3)  # 3 = LANCZOS

    # 2. Extract pixels as 32×32 float matrix
    pixels_flat: list[Any] = list(gray.getdata())
    pixels: list[list[float]] = [
        [float(pixels_flat[r * _N + c]) for c in range(_N)] for r in range(_N)
    ]

    # 3. 2D DCT: retain only the low-frequency block the hash consumes.
    dct_result = _low_frequency_dct(pixels)

    # 4. Extract top-left 8×8 low-frequency values
    low_freq: list[float] = []
    for r in range(_LOW):
        for c in range(_LOW):
            low_freq.append(dct_result[r][c])

    # 5. Median threshold
    sorted_vals = sorted(low_freq)
    mid = len(sorted_vals) // 2
    if len(sorted_vals) % 2 == 0:
        median = (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0
    else:
        median = sorted_vals[mid]

    # 6. Build 64-bit hash
    hash_int = 0
    for val in low_freq:
        hash_int = (hash_int << 1) | (1 if val > median else 0)

    return f"{hash_int:016x}"


def compute_phash_from_path(storage_ref: str) -> str | None:
    """Compute pHash from a filesystem path.

    Returns ``None`` on any failure — pHash failure MUST NOT fail ingestion.
    """
    try:
        from PIL import Image as PILImage

        with PILImage.open(storage_ref) as img:
            return compute_phash(img)
    except Exception:
        logger.warning("pHash computation failed for %s", storage_ref, exc_info=True)
        return None
