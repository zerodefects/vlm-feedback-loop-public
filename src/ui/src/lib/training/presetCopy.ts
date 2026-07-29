// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * SME-facing copy for the four training presets on the Student
 * Training screen.
 *
 * The resolved hyperparameter PATCHES are server-provided
 * (``resolved_presets`` from the training-preset resolver) — the
 * backend resolver (services/training_preset.py) is the one
 * source of truth. A former frontend mirror of that resolver lived here
 * and drifted (wrong ``max_keep``, wrong cosmos3-super epochs); do not
 * reintroduce one.
 */

import type { TrainingPreset } from "@/types/training";

export const PRESET_HELP: Readonly<Record<TrainingPreset, string>> = {
  quick: "Fast run for a quick signal.",
  standard: "Recommended default.",
  high_quality: "More training for better accuracy.",
  max_quality: "Large-epoch training for best quality (slowest).",
};

export const PRESET_LABEL: Readonly<Record<TrainingPreset, string>> = {
  quick: "Quick",
  standard: "Standard",
  high_quality: "High Quality",
  max_quality: "Max Quality",
};
