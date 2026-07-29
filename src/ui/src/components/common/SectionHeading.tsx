// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Uppercase eyebrow-style section heading matching the Retail Agentic Commerce
 * and Retail Catalog Enrichment blueprints (e.g. CONFIGURATION / Advanced options,
 * AGENT PERFORMANCE, REVENUE OVER TIME).
 *
 * Pattern: muted uppercase tracked caps, used as the first child inside a glass
 * card to signal "what this card contains." Used on configuration-summary pages
 * (ConfirmDefaults, BatchPreRun, ScaleUpHub,
 * StudentTraining, TrainingJobMonitor).
 *
 * Case + tracking values come from the shared `.section-eyebrow` CSS utility
 * (see `index.css`) so every eyebrow site in the app shares one definition.
 * Color and font-weight stay inline because they're contextual: this component
 * uses muted+600 (slightly bolder than the basic class) to match the Retail
 * dashboard heading weight.
 */

import { Text } from "@kui/react";
import type { ReactNode } from "react";

export interface SectionHeadingProps {
  children: ReactNode;
  /** Optional right-aligned slot (e.g. a status chip). */
  trailing?: ReactNode;
  "data-testid"?: string;
}

export function SectionHeading({
  children,
  trailing,
  "data-testid": testid,
}: SectionHeadingProps) {
  return (
    <div className="flex items-baseline justify-between gap-3">
      <Text
        kind="label/regular/xs"
        className="section-eyebrow"
        style={{
          color: "var(--text-muted)",
          fontWeight: 600,
        }}
        data-testid={testid ?? "section-heading"}
      >
        {children}
      </Text>
      {trailing}
    </div>
  );
}
