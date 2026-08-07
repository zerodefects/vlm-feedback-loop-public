// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Shared segmented pill control.
 *
 * Multiple options laid out inside a single rounded-full track. The active
 * option gets the NVIDIA-green tint + border + text; inactive options are
 * muted. Matches the Retail Blueprint segmented-pill rhythm (references
 * INDEX.md: "active = green tint + green text") and centralizes the pill
 * styling used by the Labeling top bar.
 */

import { Button, Text } from "@kui/react";

interface SegmentedOption {
  key: string;
  label: string;
}

interface SegmentedControlProps {
  testId: string;
  label?: string;
  ariaLabel?: string;
  options: SegmentedOption[];
  value: string;
  disabled?: boolean;
  onChange: (key: string) => void;
}

export function SegmentedControl({
  testId,
  label,
  ariaLabel,
  options,
  value,
  disabled = false,
  onChange,
}: SegmentedControlProps) {
  const accessibleLabel = ariaLabel ?? label;

  return (
    <div
      className="flex items-center gap-1.5"
      role={accessibleLabel ? "group" : undefined}
      aria-label={accessibleLabel}
      data-testid={testId}
    >
      {label && (
        <Text kind="label/regular/sm" style={{ color: "var(--text-muted)" }}>
          {label}:
        </Text>
      )}
      <div
        className="flex rounded-full border"
        style={{
          borderColor: "var(--glass-border)",
          background: "var(--block-bg)",
        }}
      >
        {options.map((opt) => {
          const active = opt.key === value;
          return (
            <Button
              key={opt.key}
              kind="tertiary"
              className="px-3 py-1 text-xs font-medium transition-all"
              style={{
                borderRadius: 999,
                background: active ? "var(--accent-green-bg)" : "transparent",
                color: active ? "var(--accent-green)" : "var(--text-muted)",
                border: active
                  ? "1px solid var(--accent-green-border)"
                  : "1px solid transparent",
              }}
              onClick={() => onChange(opt.key)}
              disabled={disabled}
              aria-pressed={active}
              data-testid={`${testId}-${opt.key}`}
            >
              {opt.label}
            </Button>
          );
        })}
      </div>
    </div>
  );
}
