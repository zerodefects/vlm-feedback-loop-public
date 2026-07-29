// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Generic tone-driven status pill used by any screen that needs to
 * render a "state chip" next to a title (Batch Run Status, Training Job
 * Monitor, etc.).
 *
 * Domain-specific badges (e.g. TAOJob statusDisplay, batch run status
 * → label mapping) resolve their canonical state into a ``Tone`` + a
 * display ``label`` and pass it here.  All the chrome — radius,
 * padding, typography, optional spinner — lives in exactly one place.
 *
 * Do not fork per-screen variants of this pill: hand-rolled copies of
 * the same tone-to-color mapping reliably diverge by a single Tailwind
 * utility or test-id, causing visual drift between screens.
 */

import type { ReactNode } from "react";

import { MiniSpinner } from "@/components/common/MiniSpinner";
import { TONE_STYLES, type Tone } from "@/lib/tone";

export interface StatusPillProps {
  tone: Tone;
  label: ReactNode;
  /** When true, a small spinner appears before the label (in-progress states). */
  spinner?: boolean;
  "data-testid"?: string;
  /** Optional DOM attribute passthrough for tests that want to assert the raw status. */
  "data-status"?: string;
  "data-halted"?: "true" | "false";
}

export function StatusPill({
  tone,
  label,
  spinner = false,
  "data-testid": testid = "status-pill",
  "data-status": dataStatus,
  "data-halted": dataHalted,
}: StatusPillProps) {
  const style = TONE_STYLES[tone];
  return (
    <span
      className="inline-flex w-fit items-center gap-1.5 rounded-full px-3 py-1 text-xs font-medium"
      style={{
        backgroundColor: style.background,
        color: style.color,
      }}
      data-testid={testid}
      data-status={dataStatus}
      data-halted={dataHalted}
    >
      {spinner && <MiniSpinner />}
      {label}
    </span>
  );
}
