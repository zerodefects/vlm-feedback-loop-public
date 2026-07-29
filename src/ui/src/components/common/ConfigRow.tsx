// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Configuration row with label + value — shared across the Batch Pre-Run
 * and other configuration-summary cards.
 *
 * Label renders muted; value renders at primary emphasis so glass-card
 * backgrounds don't swallow the value text (inline-styled dark-on-dark
 * values are unreadable on glass panels).
 */

import { Text } from "@kui/react";
import type { ReactNode } from "react";

type TextSize = "xs" | "sm";

export interface ConfigRowProps {
  label: string;
  value: ReactNode;
  /**
   * KUI `<Text>` size variant for both the label and value. Defaults to
   * `"sm"` for Blueprint-aligned configuration-summary density shared
   * with Batch Pre-Run. `"xs"` remains available for compact contexts.
   */
  size?: TextSize;
}

export function ConfigRow({ label, value, size = "sm" }: ConfigRowProps) {
  const labelKind = size === "sm" ? "body/regular/sm" : "label/regular/xs";
  const valueKind = size === "sm" ? "body/regular/sm" : "label/regular/xs";
  return (
    <div className="flex items-center justify-between gap-4">
      <Text kind={labelKind} style={{ color: "var(--text-muted)" }}>
        {label}
      </Text>
      <Text kind={valueKind} style={{ color: "var(--text-primary)" }}>
        {value}
      </Text>
    </div>
  );
}
