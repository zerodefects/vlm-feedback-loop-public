// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { TrainingPreflightCheckName } from "@/types/training";

const DATA_READINESS_CHECKS: ReadonlySet<TrainingPreflightCheckName> = new Set([
  "verified_train_examples",
  "min_test_pool_size",
]);

const CONFIGURATION_CHECKS: ReadonlySet<TrainingPreflightCheckName> = new Set([
  "student_base_role",
  "training_mode_compatible",
  "quantization_compatible",
]);

export function isTrainingDataReadinessCheck(
  checkName: TrainingPreflightCheckName,
): boolean {
  return DATA_READINESS_CHECKS.has(checkName);
}

export function isTrainingConfigurationCheck(
  checkName: TrainingPreflightCheckName,
): boolean {
  return CONFIGURATION_CHECKS.has(checkName);
}
