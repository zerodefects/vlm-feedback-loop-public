// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Standard glass-card section shell used by configuration-summary /
 * dashboard screens (Scale-Up Hub, Batch Pre-Run, Batch Run Status,
 * Student Training, Training Job Monitor).
 *
 * Consolidates the per-page glass-card shells otherwise duplicated
 * across those pages so tuning of card padding / inner rhythm happens
 * in exactly one place.
 *
 * Two density variants:
 *   * ``density="standard"`` (default) — ``p-5 gap-3``. Matches the
 *     spacing used by Scale-Up Hub and Student Training, where each
 *     section contains multiple controls with helper copy.
 *   * ``density="dense"`` — ``p-4 gap-2``. Matches Batch Pre-Run's
 *     ``Configuration`` card, where the card is
 *     primarily a stack of short ``ConfigRow`` lines and a tighter rhythm
 *     reads better.
 */

import type { ReactNode } from "react";

type SectionCardDensity = "standard" | "dense";

export interface SectionCardProps {
  children: ReactNode;
  /** Spacing density. Defaults to ``"standard"``. */
  density?: SectionCardDensity;
  "data-testid"?: string;
}

// ``flex flex-col gap-*`` rather than ``space-y-*``. Tailwind 4's
// ``space-y`` utilities rely on ``margin-block-end`` via a ``:where()``
// selector clamped to 0 specificity; preflight's ``* { margin: 0 }``
// wins the cascade (identical 0,0,0,0) and the sibling margin collapses
// to zero. ``gap`` is applied on the parent flex container and is
// immune to that cascade fight.
const DENSITY_CLASSES: Readonly<Record<SectionCardDensity, string>> = {
  standard: "p-5 flex flex-col gap-3",
  dense: "p-4 flex flex-col gap-2",
};

export function SectionCard({
  children,
  density = "standard",
  "data-testid": testid,
}: SectionCardProps) {
  const shellClass = `glass-card ${DENSITY_CLASSES[density]}`;
  return (
    <div className={shellClass} data-testid={testid}>
      {children}
    </div>
  );
}
