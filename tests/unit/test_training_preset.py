# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the training_preset resolver: four presets resolve
to deterministic patches, same input → same output, and the patches target
the correct Cosmos-RL SFT spec fields.
"""

from __future__ import annotations

import pytest

from vlm_feedback_loop.model_catalog_constants import (
    COSMOS3_NANO_REASONER,
    COSMOS3_SUPER_REASONER,
    COSMOS_REASON2_2B,
    COSMOS_REASON2_8B,
)
from vlm_feedback_loop.services.training_preset import (
    TRAINING_PRESETS,
    resolve_epochs,
    resolve_training_preset,
)


class TestPresetConstants:
    def test_exactly_four_presets(self):
        assert (
            frozenset({"quick", "standard", "high_quality", "max_quality"})
            == TRAINING_PRESETS
        )


class TestResolveEpochs:
    @pytest.mark.parametrize(
        "preset,expected",
        [("quick", 1), ("standard", 3), ("high_quality", 9), ("max_quality", 18)],
    )
    def test_cosmos_reason_2_8b_matches_spec_defaults(self, preset, expected):
        assert resolve_epochs(preset, COSMOS_REASON2_8B) == expected

    @pytest.mark.parametrize(
        "preset,expected",
        # Model-aware table: 2B scales up for high/max quality.
        [("quick", 1), ("standard", 3), ("high_quality", 12), ("max_quality", 24)],
    )
    def test_cosmos_reason_2_2b_scaled_for_smaller_model(self, preset, expected):
        assert resolve_epochs(preset, COSMOS_REASON2_2B) == expected

    @pytest.mark.parametrize(
        "preset,expected",
        # Cosmos 3 Nano-Reasoner (~8B-class) mirrors the 8B schedule.
        [("quick", 1), ("standard", 3), ("high_quality", 9), ("max_quality", 18)],
    )
    def test_cosmos3_nano_reasoner_mirrors_8b(self, preset, expected):
        assert resolve_epochs(preset, COSMOS3_NANO_REASONER) == expected

    @pytest.mark.parametrize(
        "preset,expected",
        # Cosmos 3 Super-Reasoner (~30B-class) — fewer epochs (overfit + cost).
        [("quick", 1), ("standard", 2), ("high_quality", 6), ("max_quality", 12)],
    )
    def test_cosmos3_super_reasoner_fewer_epochs(self, preset, expected):
        assert resolve_epochs(preset, COSMOS3_SUPER_REASONER) == expected

    @pytest.mark.parametrize(
        "preset,expected",
        [("quick", 1), ("standard", 3), ("high_quality", 9), ("max_quality", 18)],
    )
    def test_unknown_model_falls_back_to_spec_defaults(self, preset, expected):
        assert resolve_epochs(preset, "some-unknown-model") == expected

    def test_case_insensitive_model_name(self):
        # Model lookup normalizes whitespace + case.
        assert resolve_epochs("standard", "COSMOS-REASON-2 8B") == 3
        assert resolve_epochs("standard", "  nvidia/cosmos-reason2-8b  ") == 3

    def test_invalid_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown training preset"):
            resolve_epochs("ludicrous", COSMOS_REASON2_8B)


class TestResolveTrainingPreset:
    def test_patch_targets_cosmos_rl_sft_train_fields(self):
        # Exact patch shape.
        patch = resolve_training_preset("standard", COSMOS_REASON2_8B)
        assert set(patch.keys()) == {"train"}
        train = patch["train"]
        assert train["epoch"] == 3
        assert train["resume"] is False
        ckpt = train["ckpt"]
        assert ckpt["enable_checkpoint"] is True
        assert ckpt["save_freq_in_epoch"] == 1
        # max_keep is conditional on resume. Every Blueprint
        # training flow runs with
        # resume=False today, so max_keep MUST be 1 (only the latest
        # epoch is needed by the local checkpoint selector; retaining
        # 8 epochs uploads ~8× more data with no Blueprint-visible
        # benefit).
        assert ckpt["max_keep"] == 1
        assert ckpt["export_safetensors"] is True

    def test_max_keep_reflects_resume_default_false(self):
        """When resume=False (the default for every
        Blueprint flow) the patch MUST emit max_keep=1. Evidence: on
        an 8B high_quality train, max_keep=8 pushed the safetensors
        upload to ~117.8 GB / ~50 min, blocking every other chain on
        TAO's single-cluster gate.
        """
        for preset in ("quick", "standard", "high_quality", "max_quality"):
            patch = resolve_training_preset(preset, COSMOS_REASON2_2B)
            assert patch["train"]["resume"] is False
            assert patch["train"]["ckpt"]["max_keep"] == 1, (
                f"expected max_keep=1 with resume=False for preset={preset}"
            )

    def test_determinism(self):
        # Same input → same output, byte-equivalent.
        a = resolve_training_preset("standard", COSMOS_REASON2_8B)
        b = resolve_training_preset("standard", COSMOS_REASON2_8B)
        assert a == b
        # Distinct dict instances (no shared mutable state).
        a["train"]["epoch"] = 99
        assert b["train"]["epoch"] == 3

    def test_invalid_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown training preset"):
            resolve_training_preset("max_depth", COSMOS_REASON2_8B)
