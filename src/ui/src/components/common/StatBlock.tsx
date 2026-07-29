// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * KPI stat treatment shared by dashboard-style cards: a muted uppercase
 * micro-label over a bold value (Retail Blueprint metrics pattern).
 *
 * ``tone="green"`` renders the value in NVIDIA green — reserve it for the
 * headline number of a card so at-a-glance scanning has one anchor.
 */

import { Text } from "@kui/react";
import type { ReactNode } from "react";

export interface StatBlockProps {
  label: string;
  value: ReactNode;
  tone?: "green" | "neutral";
  "data-testid"?: string;
}

export function StatBlock({
  label,
  value,
  tone = "neutral",
  "data-testid": testid,
}: StatBlockProps) {
  return (
    <div className="flex flex-col gap-0.5" data-testid={testid}>
      <Text
        kind="label/regular/xs"
        className="block section-eyebrow"
        style={{ color: "var(--text-faint)" }}
      >
        {label}
      </Text>
      <Text
        kind="label/bold/lg"
        className="block"
        style={{
          color: tone === "green" ? "var(--accent-green)" : "var(--text-primary)",
        }}
      >
        {value}
      </Text>
    </div>
  );
}
