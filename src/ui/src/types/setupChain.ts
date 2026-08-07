// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Shared state passed through ``location.state`` between the three FTU
 * setup screens (NIMNvidiaKeyPage → NIMNgcKeyPage → NIMSetupGatePage).
 *
 * ``activePath`` is the load-bearing field for the path-aware setup
 * flow: NIMNvidiaKeyPage computes it from the EnvironmentResponse and
 * the SME's choice; NIMNgcKeyPage forwards it unchanged (only
 * hosted-path flows reach that screen — local/hybrid flows collect NGC
 * inline and skip straight to the gate); NIMSetupGatePage drives the
 * summary list and the queue of local-NIM models to dispatch.
 */

import type { RecommendedMode } from "@/types/nim";

/**
 * Which post-setup model stack the SME has committed to.
 *
 * - ``"hosted"`` — the configured default hosted Teacher + hosted embedding,
 *   or hybrid where the embedding is local but the Teacher is hosted.
 *   No local Teacher container is queued.
 * - ``"local"`` — recommended local Teacher (Omni on supported ≥80 GB /
 *   CR3-Nano at ≥56 GB / Cosmos Reason2 2B at 36–55 GB) + local embedding.
 *   No hosted endpoint
 *   is required to start labeling.
 * - ``"hybrid"`` — Hosted bridge + preferred local Teacher. Labeling starts
 *   immediately on the hosted default; the local Teacher deploys in the
 *   background and becomes active automatically after backend verification.
 */
export type ActivePath = "hosted" | "local" | "hybrid";

/**
 * Acknowledgment payload forwarded through ``location.state`` when the
 * setup chain completes without the SME making a single choice (every
 * screen auto-skipped). ConfirmDefaultsPage's auto-skip path builds it;
 * the landing screen (Ingest or Labeling) renders it as the
 * non-blocking ``SetupAutoSkipBanner`` so the auto-selected
 * configuration — including hosted-NIM data egress — is surfaced at
 * least once. Router state is deliberate: it does not persist, so the
 * banner is one-shot.
 */
export interface SetupAutoSkipState {
  teacherMode: RecommendedMode;
  embeddingMode: RecommendedMode;
  /** Project's default Teacher model name; null when unresolvable. */
  teacherName: string | null;
}

export interface SetupChainState {
  activePath: ActivePath;
  /**
   * True iff the SME reached this screen without making an explicit
   * choice — every upstream screen auto-skipped. Propagated forward so
   * the confirmation gate can decide whether to auto-skip its own
   * confirmation.
   */
  cameFromAutoSkip: boolean;
  /**
   * Names of local NIM models the FTUE has queued for background
   * deployment at gate-confirm time. NIMSetupGatePage iterates this list,
   * issuing one ``POST :deploy`` per entry, and forwards it verbatim to
   * ``:mark_setup_completed`` for the AuditEvent payload. The hosted
   * path can still carry the embedding NIM: local embeddings are the
   * default whenever the host GPU fits them, so the NGC key screen
   * queues that deploy for hosted-Teacher chains (the small-GPU host
   * class — GPU below every Teacher floor but at/above the embedding
   * floor).
   */
  localDeployQueued: string[];
}

/**
 * Default state for direct-URL navigation (an SME deep-links to
 * ``/setup/ngc`` without going through the setup-choice screen first).
 * Hosted-only is the safe assumption — the NGC key screen won't strip a
 * queue that was never built, and the confirmation gate's summary
 * collapses cleanly.
 */
export const DEFAULT_SETUP_CHAIN_STATE: SetupChainState = {
  activePath: "hosted",
  cameFromAutoSkip: false,
  localDeployQueued: [],
};

/**
 * Distinguish an in-progress onboarding transition from a copied/deep URL.
 * Completed projects must not reconstruct a fake setup summary from
 * ``DEFAULT_SETUP_CHAIN_STATE`` when the ephemeral chain state is absent.
 */
export function isSetupChainState(value: unknown): value is SetupChainState {
  if (typeof value !== "object" || value === null) return false;
  const candidate = value as Record<string, unknown>;
  return (
    (candidate.activePath === "hosted" ||
      candidate.activePath === "local" ||
      candidate.activePath === "hybrid") &&
    typeof candidate.cameFromAutoSkip === "boolean" &&
    Array.isArray(candidate.localDeployQueued) &&
    candidate.localDeployQueued.every((name) => typeof name === "string")
  );
}
