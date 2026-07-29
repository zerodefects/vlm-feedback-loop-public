// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Shared helpers for local-NIM deployment surfaces
 * (``NIMSetupGatePage``, ``LocalDeployBanner``).
 */

import type { LocalNimDeploymentResponse } from "@/types/nim";

/**
 * Polling cadence for the local-NIM deployment query while a deploy is
 * still in ``starting``. Shared by every surface that polls
 * ``GET /v1/projects/{id}/local_nim/deployments`` so they stay in
 * lockstep and don't race for cache invalidations.
 */
export const LOCAL_NIM_POLL_INTERVAL_MS = 10_000;

/**
 * Group deployments by role, keeping the most recent per role. A new
 * successful deploy supersedes an older failed one even though both
 * records persist in the DB.
 */
export function latestPerRole(
  items: LocalNimDeploymentResponse[],
): LocalNimDeploymentResponse[] {
  const byRole = new Map<string, LocalNimDeploymentResponse>();
  for (const d of items) {
    const existing = byRole.get(d.role);
    if (
      existing === undefined ||
      new Date(d.created_at).getTime() > new Date(existing.created_at).getTime()
    ) {
      byRole.set(d.role, d);
    }
  }
  return Array.from(byRole.values());
}
