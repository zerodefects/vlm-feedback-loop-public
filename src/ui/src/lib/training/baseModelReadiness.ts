// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ModelConfigResponse } from "@/types/nim";

/**
 * Whether a Student base is already ready and can skip first-use provisioning.
 *
 * The backend remains authoritative: these fields are the provisioning state
 * returned by the ModelConfig API, and suite launch re-validates the
 * referenced experiment before materializing TAO chains.
 */
export function hasReadyTaoBaseExperiment(model: ModelConfigResponse): boolean {
  return (
    typeof model.tao_base_experiment_id === "string" &&
    model.tao_base_experiment_id.length > 0 &&
    model.tao_base_experiment_pull_status === "pull_complete"
  );
}
