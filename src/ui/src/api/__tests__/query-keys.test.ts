// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Tests for the React Query key factories that carry real logic.
 *
 * Most factories are literal arrays and are not tested (a test would just
 * echo them). These three encode behavior a cache collision or miss would
 * break: argument normalization, order-insensitive hashing, and the
 * deliberate separation of two same-endpoint caches.
 */

import { describe, it, expect } from "vitest";

import { evaluationKeys, projectKeys, trainingKeys } from "@/api/query-keys";

describe("projectKeys.list", () => {
  it("collapses an omitted includeArchived to the same key as false", () => {
    // Callers that omit the flag and callers that pass false must share
    // one cache entry, or the Project List refetches on every mount.
    expect(projectKeys.list()).toEqual(projectKeys.list(false));
  });

  it("keys the archived-inclusive list separately", () => {
    expect(projectKeys.list(true)).not.toEqual(projectKeys.list(false));
  });
});

describe("trainingKeys.presets", () => {
  it("produces the same key regardless of model-config-id order", () => {
    // Preset resolution is one backend computation per id *set*; selection order in
    // the UI must not fragment the cache.
    expect(trainingKeys.presets("p1", ["mc-b", "mc-a"])).toEqual(
      trainingKeys.presets("p1", ["mc-a", "mc-b"]),
    );
  });

  it("produces distinct keys for distinct id sets", () => {
    expect(trainingKeys.presets("p1", ["mc-a"])).not.toEqual(
      trainingKeys.presets("p1", ["mc-a", "mc-b"]),
    );
  });

  it("does not mutate the caller's id array", () => {
    const ids = ["mc-b", "mc-a"];
    trainingKeys.presets("p1", ids);
    expect(ids).toEqual(["mc-b", "mc-a"]);
  });
});

describe("trainingKeys.preflight", () => {
  it("is stable across model ordering and varies with data policy", () => {
    expect(trainingKeys.preflight("p1", ["mc-b", "mc-a"], true)).toEqual(
      trainingKeys.preflight("p1", ["mc-a", "mc-b"], true),
    );
    expect(trainingKeys.preflight("p1", ["mc-a"], true)).not.toEqual(
      trainingKeys.preflight("p1", ["mc-a"], false),
    );
  });
});

describe("evaluationKeys", () => {
  it("keeps the Compare page's completed-runs cache separate from the strip's list cache", () => {
    // The two queries hit the same endpoint with different params; sharing
    // a key would let their queryFns race for one cache entry.
    expect(evaluationKeys.completedList("p1")).not.toEqual(evaluationKeys.list("p1"));
  });
});
