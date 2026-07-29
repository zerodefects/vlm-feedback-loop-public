// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";

import { formatBytes } from "../format-bytes";

describe("formatBytes", () => {
  it("ladders B → KB → MB → GB with the (4.2 MB) decimal convention", () => {
    expect(formatBytes(0)).toBe("0 B");
    expect(formatBytes(512)).toBe("512 B");
    expect(formatBytes(4_300)).toBe("4.2 KB");
    expect(formatBytes(4_404_019)).toBe("4.2 MB");
    expect(formatBytes(2_147_483_648)).toBe("2.00 GB");
  });

  it("returns null for missing or negative sizes so callers can render conditionally", () => {
    expect(formatBytes(null)).toBeNull();
    expect(formatBytes(undefined)).toBeNull();
    expect(formatBytes(-1)).toBeNull();
  });
});
