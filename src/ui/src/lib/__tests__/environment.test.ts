// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, it, expect } from "vitest";
import { formatDetectedLine } from "@/lib/environment";
import type { GpuInfo } from "@/types/nim";

function gpu(name: string, memory: number): GpuInfo {
  return { name, memory_total_gb: memory };
}

describe("formatDetectedLine", () => {
  it("collapses 8 identical A100s to a single group with × 8", () => {
    const gpus: GpuInfo[] = Array.from({ length: 8 }, () =>
      gpu("NVIDIA A100-SXM4-80GB", 80),
    );
    expect(formatDetectedLine(gpus, true)).toBe(
      "NVIDIA A100-SXM4 80 GB × 8 · Docker ready",
    );
  });

  it("strips embedded memory tokens (H100 80GB HBM3)", () => {
    expect(formatDetectedLine([gpu("NVIDIA H100 80GB HBM3", 80)], true)).toBe(
      "NVIDIA H100 80 GB · Docker ready",
    );
  });

  it("leaves names without embedded memory alone", () => {
    expect(formatDetectedLine([gpu("NVIDIA A10G", 24)], true)).toBe(
      "NVIDIA A10G 24 GB · Docker ready",
    );
  });

  it("groups distinct GPUs separately, joined with commas", () => {
    const gpus = [
      gpu("NVIDIA H100", 80),
      gpu("NVIDIA H100", 80),
      gpu("NVIDIA A10G", 24),
    ];
    expect(formatDetectedLine(gpus, true)).toBe(
      "NVIDIA H100 80 GB × 2 · NVIDIA A10G 24 GB · Docker ready",
    );
  });

  it("emits 'Docker not found' when Docker is unavailable", () => {
    expect(formatDetectedLine([gpu("NVIDIA A100-SXM4-80GB", 80)], false)).toBe(
      "NVIDIA A100-SXM4 80 GB · Docker not found",
    );
  });

  it("returns an empty string when no GPUs and no Docker", () => {
    expect(formatDetectedLine([], false)).toBe("");
  });

  it("shows only Docker state when no GPUs but Docker is present", () => {
    expect(formatDetectedLine([], true)).toBe("Docker ready");
  });
});
