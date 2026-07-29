# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Training preset resolver.

Pure module. Zero I/O. Maps a user-facing training intensity preset and a
Cosmos Reason2 base-model name to a deterministic hyperparameter patch that
is then merged into ``tao_create_job_request.specs`` by the training-suite
service. Same ``(preset, model)`` pair → same patch, always.

The patch targets these Cosmos-RL SFT schema fields:
``train.epoch``, ``train.resume``, ``train.ckpt.*`` (checkpoint config
including ``enable_checkpoint``, ``save_freq_in_epoch``, ``max_keep``, and
``export_safetensors``).
"""

from __future__ import annotations

from typing import Any

from vlm_feedback_loop.model_catalog_constants import (
    COSMOS3_NANO_REASONER,
    COSMOS3_SUPER_REASONER,
    COSMOS_REASON2_2B,
    COSMOS_REASON2_8B,
)

# Canonical set of user-facing presets.
TRAINING_PRESETS: frozenset[str] = frozenset(
    {"quick", "standard", "high_quality", "max_quality"}
)


# Default epoch counts per preset.
_DEFAULT_EPOCHS: dict[str, int] = {
    "quick": 1,
    "standard": 3,
    "high_quality": 9,
    "max_quality": 18,
}


# Optional model-aware epoch overrides. Bigger models need fewer epochs to avoid
# overfit; 2B benefits from more epochs.
_EPOCH_TABLE_BY_MODEL: dict[tuple[str, str], int] = {
    # Cosmos Reason2 2B — scaled up for smaller model.
    (COSMOS_REASON2_2B, "quick"): 1,
    (COSMOS_REASON2_2B, "standard"): 3,
    (COSMOS_REASON2_2B, "high_quality"): 12,
    (COSMOS_REASON2_2B, "max_quality"): 24,
    # Cosmos Reason2 8B — matches spec defaults (and NVIDIA Cookbook recipe).
    (COSMOS_REASON2_8B, "quick"): 1,
    (COSMOS_REASON2_8B, "standard"): 3,
    (COSMOS_REASON2_8B, "high_quality"): 9,
    (COSMOS_REASON2_8B, "max_quality"): 18,
    # Cosmos 3 Nano-Reasoner (~8B-class qwen3_vl) — mirrors the 8B schedule.
    (COSMOS3_NANO_REASONER, "quick"): 1,
    (COSMOS3_NANO_REASONER, "standard"): 3,
    (COSMOS3_NANO_REASONER, "high_quality"): 9,
    (COSMOS3_NANO_REASONER, "max_quality"): 18,
    # Cosmos 3 Super-Reasoner (~30B-class) — fewer epochs; large models
    # overfit faster and each epoch is far costlier on the 8-GPU shard.
    (COSMOS3_SUPER_REASONER, "quick"): 1,
    (COSMOS3_SUPER_REASONER, "standard"): 2,
    (COSMOS3_SUPER_REASONER, "high_quality"): 6,
    (COSMOS3_SUPER_REASONER, "max_quality"): 12,
}


def resolve_epochs(preset: str, model_name: str) -> int:
    """Return the epoch count for a (preset, model_name) pair.

    Falls back to the spec default for unknown models.
    """
    if preset not in TRAINING_PRESETS:
        raise ValueError(f"Unknown training preset: {preset!r}")
    normalized = model_name.strip().lower()
    return _EPOCH_TABLE_BY_MODEL.get((normalized, preset), _DEFAULT_EPOCHS[preset])


def resolve_training_preset(preset: str, model_name: str) -> dict[str, Any]:
    """Resolve a preset + model name to a deterministic hyperparameter patch.

    The patch is the exact ``job_config.hyperparameters`` value persisted
    on every training TAOJob, and is the same patch merged into
    ``tao_create_job_request.specs``.

    All presets share per-epoch checkpointing with ``export_safetensors
    =true`` for ``best_model`` selection plus NIM-compatible
    serialization. ``max_keep=1`` because every Blueprint training flow
    runs with ``resume=False`` and only ever pulls down the latest epoch:
    cosmos-rl writes one safetensors directory per saved epoch under
    ``results/<job>/<timestamp>/safetensors/epoch_<N>/`` and uploads each
    to the workspace S3 in a separate transfer at the end of the run.
    With ``max_keep=8`` and a ``high_quality`` 9-epoch 8B run, that's
    ~117.8 GB upload across 8 retained epochs; the Blueprint's checkpoint
    selector (``_select_hf_checkpoint_keys``) only ever pulls the latest
    epoch (~14.7 GB), so the other 7 retained epochs are dead weight
    consuming TAO worker time + workspace S3 storage. Setting
    ``max_keep=1`` cuts the upload by ~8× without losing any
    Blueprint-visible state. Empirical evidence: an 8B high_quality
    train spent 50 min on safetensors upload alone while the rest of the
    chain blocked on TAO's single-cluster gate. (A resume-supporting flow
    would need a higher ``max_keep`` to give ``train.resume`` fallback
    restart points.)

    Raises:
        ValueError: if ``preset`` is not one of ``TRAINING_PRESETS``.
    """
    epochs = resolve_epochs(preset, model_name)
    return {
        "train": {
            "epoch": epochs,
            "resume": False,
            "ckpt": {
                "enable_checkpoint": True,
                "save_freq_in_epoch": 1,
                # Only the latest epoch is ever consumed. See docstring.
                "max_keep": 1,
                "export_safetensors": True,
            },
        },
    }


__all__ = [
    "TRAINING_PRESETS",
    "resolve_epochs",
    "resolve_training_preset",
]
