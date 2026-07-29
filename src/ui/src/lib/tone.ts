// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Shared tone system used by status pills, info banners, and any other
 * surface that needs semantic color coding.  Keeping the mapping in one
 * place prevents the Training status badge, Batch Run status badge, and
 * various glass-info banners from drifting apart.
 */

export type Tone = "neutral" | "info" | "success" | "warning" | "error" | "subdued";

export interface ToneStyle {
  background: string;
  color: string;
}

export const TONE_STYLES: Readonly<Record<Tone, ToneStyle>> = {
  neutral: {
    background: "rgba(255, 255, 255, 0.08)",
    color: "rgba(255, 255, 255, 0.72)",
  },
  info: {
    background: "rgba(96, 165, 250, 0.14)",
    color: "#60a5fa",
  },
  success: {
    background: "rgba(118, 185, 0, 0.14)",
    color: "var(--accent-green, #76b900)",
  },
  warning: {
    background: "rgba(245, 158, 11, 0.14)",
    color: "var(--warning-amber, #f59e0b)",
  },
  error: {
    background: "var(--error-red-bg, rgba(229, 32, 32, 0.15))",
    color: "var(--error-red-text, #ff6b6b)",
  },
  subdued: {
    background: "rgba(255, 255, 255, 0.04)",
    color: "rgba(255, 255, 255, 0.35)",
  },
};
