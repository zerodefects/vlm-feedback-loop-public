# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU memory floor for Student NIM deploys — the ONE resolution policy.

Pure module, zero I/O. Shared by the Student NIM lifecycle preflight
and the deployment-handoff generator so the two can never disagree — a
diverged copy that returns 0 for a base it doesn't recognize ships
handoffs with no memory floor at all.

Policy: Cosmos Reason2 8B BF16 → 56 GB; FP8 / W4A16 → 48 GB.
Cosmos Reason2 2B BF16 → 36 GB; FP8 / W4A16 → 24 GB. Anything else
falls back to the base model's published
``local_deploy_metadata.nim_gpu_memory_minimum_gb`` (the BF16 default),
then conservatively to the 8B BF16 floor.
"""

from __future__ import annotations

from typing import Any

from vlm_feedback_loop.config import Settings

# Quantization identifiers that imply the reduced-precision floor.
_QUANTIZED_METHODS = frozenset({"fp8", "fp8_dynamic", "w4a16", "w8a8", "w8a16"})


def resolve_gpu_memory_floor_gb(
    *,
    base_model_name: str,
    quantization_method: str | None,
    base_local_deploy_metadata: dict[str, Any] | None,
    settings: Settings,
) -> int:
    """Resolve the GPU memory floor from base model + precision."""
    base = (base_model_name or "").lower()
    quant = (quantization_method or "").lower()
    is_quantized = quant in _QUANTIZED_METHODS

    if "8b" in base:
        return (
            settings.NIM_GPU_MEMORY_8B_FP8_GB
            if is_quantized
            else settings.NIM_GPU_MEMORY_8B_BF16_GB
        )
    if "2b" in base:
        return (
            settings.NIM_GPU_MEMORY_2B_FP8_GB
            if is_quantized
            else settings.NIM_GPU_MEMORY_2B_BF16_GB
        )

    # Unknown base — fall back to the published minimum on the catalog row.
    metadata = base_local_deploy_metadata or {}
    fallback = metadata.get("nim_gpu_memory_minimum_gb")
    if isinstance(fallback, int):
        return fallback
    return settings.NIM_GPU_MEMORY_8B_BF16_GB  # conservative
