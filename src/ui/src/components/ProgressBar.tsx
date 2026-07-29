// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Shared progress-bar primitive used by Ingestion, Batch Labeling,
 * and Training Job Monitor. A single primitive keeps
 * track color, height, and fill transition from drifting between per-site
 * inline implementations.
 *
 * This component standardizes:
 *   - Track pill outlined by `--glass-border-subtle` so the bar stays
 *     visible at 0% against the card background (the empty state).
 *   - Fill variant mirrors the status-pill tone taxonomy so a single status
 *     can drive both the pill and the bar: `default` (success/running — NVIDIA
 *     green), `paused` (warning — amber), `error` (failed — red), `neutral`
 *     (canceled — muted).
 *   - Two heights: `sm` (h-2) and `md` (h-3).
 */

import type { CSSProperties } from "react";

export type ProgressBarVariant = "default" | "paused" | "error" | "neutral";
type ProgressBarSize = "sm" | "md";

interface ProgressBarProps {
  /** 0–100. Values outside this range are clamped. */
  percent: number;
  variant?: ProgressBarVariant;
  size?: ProgressBarSize;
  className?: string;
  /** Test id applied to the fill element (for state assertions). */
  fillTestId?: string;
  /** Accessible label for the progress bar role. */
  ariaLabel?: string;
}

const HEIGHT_CLASS: Record<ProgressBarSize, string> = {
  sm: "h-2",
  md: "h-3",
};

const FILL_COLOR: Record<ProgressBarVariant, string> = {
  default: "var(--accent-green, #76b900)",
  paused: "var(--warning-amber, #f59e0b)",
  error: "var(--error-red, #e52020)",
  neutral: "var(--text-muted, rgba(255, 255, 255, 0.62))",
};

const TRACK_STYLE: CSSProperties = {
  backgroundColor: "var(--block-bg-elevated)",
  border: "1px solid var(--glass-border-subtle)",
};

export function ProgressBar({
  percent,
  variant = "default",
  size = "sm",
  className,
  fillTestId,
  ariaLabel,
}: ProgressBarProps) {
  const clamped = Math.max(0, Math.min(100, percent));
  const heightCls = HEIGHT_CLASS[size];
  return (
    <div
      className={`${heightCls} rounded-full overflow-hidden${className ? ` ${className}` : ""}`}
      style={TRACK_STYLE}
      role="progressbar"
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={Math.round(clamped)}
      aria-label={ariaLabel}
    >
      <div
        className="h-full rounded-full transition-all"
        style={{ width: `${clamped}%`, backgroundColor: FILL_COLOR[variant] }}
        data-testid={fillTestId}
      />
    </div>
  );
}
