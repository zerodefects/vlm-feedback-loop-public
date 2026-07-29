// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Environment-assessment display helpers for the NIM Connection
 * screen.  Pure functions — no React, no fetch.
 */

import type { GpuInfo } from "@/types/nim";

// Matches memory tokens commonly embedded in nvidia-smi GPU names (e.g.
// "NVIDIA A100-SXM4-80GB", "NVIDIA H100 80GB HBM3").  Used by the detected
// line to avoid printing "A100-SXM4-80GB 80 GB".
const EMBEDDED_MEMORY_TOKEN_RE =
  /[\s-]*\b(?:16|24|32|40|48|80|94|141|192)\s*gb\b(?:\s*hbm[0-9e]*)?/i;

function stripEmbeddedMemory(name: string): string {
  const cleaned = name
    .replace(EMBEDDED_MEMORY_TOKEN_RE, "")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/-+$/, "")
    .trim();
  return cleaned || name.trim();
}

/**
 * Format the "Detected: …" line shown at the top of the NIM Connection
 * recommendation screen.  Collapses identical GPUs to `<name> <memory> GB × N`
 * and always appends the Docker state.
 *
 * Returns an empty string when there is nothing to show (no GPUs detected
 * AND Docker unavailable — e.g. a laptop with no CUDA runtime).
 */
export function formatDetectedLine(gpus: GpuInfo[], dockerAvailable: boolean): string {
  const groups = new Map<string, { name: string; memory: number; count: number }>();
  for (const g of gpus) {
    const displayName = stripEmbeddedMemory(g.name);
    const key = `${displayName}|${g.memory_total_gb}`;
    const existing = groups.get(key);
    if (existing) {
      existing.count += 1;
    } else {
      groups.set(key, {
        name: displayName,
        memory: g.memory_total_gb,
        count: 1,
      });
    }
  }

  const gpuTokens = Array.from(groups.values()).map((g) => {
    const base = `${g.name} ${g.memory} GB`;
    return g.count > 1 ? `${base} × ${g.count}` : base;
  });

  const dockerToken = dockerAvailable ? "Docker ready" : "Docker not found";

  if (gpuTokens.length === 0 && !dockerAvailable) {
    return "";
  }

  const parts = gpuTokens.length > 0 ? [...gpuTokens, dockerToken] : [dockerToken];
  return parts.join(" · ");
}
