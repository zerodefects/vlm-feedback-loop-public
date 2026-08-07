// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Shared React Query ``refetchInterval`` factory for run-status polling.
 *
 * Every "walk away and come back" screen (Batch Run Status, Training Job
 * Monitor, inline evaluation, local-NIM deploy) polls its run endpoint
 * while the run is in flight and stops (or relaxes) once it settles.
 * This factory keeps that scaffolding in one place; pages supply their
 * own settled-predicate and interval since those genuinely differ.
 */

import { TERMINAL_TAO_STATUSES, type TAOJob } from "@/types/training";

interface RunStatusPollingOptions<TData> {
  /**
   * True once the observed data has reached a settled state (terminal
   * status, paused, all chain jobs terminal, …). Receives the query's
   * current cached data, which is ``undefined`` before the first fetch.
   */
  isSettled: (data: TData | undefined) => boolean;
  /** Poll interval in milliseconds while not settled. */
  activeMs: number;
  /**
   * Optional relaxed interval once settled. Omit to stop polling
   * entirely (the common case for terminal run statuses).
   */
  settledMs?: number;
}

/**
 * Build a ``refetchInterval`` callback: polls every ``activeMs`` until
 * ``isSettled(data)`` is true, then stops — or drops to ``settledMs``
 * when provided. React Query re-evaluates the callback reactively, so
 * status transitions are picked up without manual state.
 */
export function runStatusRefetchInterval<TData>({
  isSettled,
  activeMs,
  settledMs,
}: RunStatusPollingOptions<TData>): (query: {
  state: { data: TData | undefined };
}) => number | false {
  return (query) => (isSettled(query.state.data) ? (settledMs ?? false) : activeMs);
}

/**
 * TAO success is only settled for the monitor after the Blueprint finishes
 * consuming the remote result. Large checkpoint downloads can continue long
 * after TAO itself reports success, so stopping on ``status`` alone leaves the
 * screen frozen in a misleading intermediate state.
 */
export function isTAOJobPollingSettled(job: TAOJob | undefined): boolean {
  if (!job || !TERMINAL_TAO_STATUSES.has(job.status)) return false;
  if (job.status !== "succeeded") return true;
  return (
    job.outputs_fetch_status === "completed" || job.outputs_fetch_status === "failed"
  );
}
