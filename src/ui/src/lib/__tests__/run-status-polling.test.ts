// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";

import { runStatusRefetchInterval } from "../run-status-polling";

interface FakeRun {
  status: string;
}

const TERMINAL = new Set(["completed", "failed"]);

function query(data: FakeRun | undefined) {
  return { state: { data } };
}

describe("runStatusRefetchInterval", () => {
  const interval = runStatusRefetchInterval({
    isSettled: (run: FakeRun | undefined) => !!run && TERMINAL.has(run.status),
    activeMs: 3000,
  });

  it("polls at the active interval while the run is in flight", () => {
    expect(interval(query({ status: "running" }))).toBe(3000);
  });

  it("polls at the active interval before the first fetch resolves", () => {
    expect(interval(query(undefined))).toBe(3000);
  });

  it("stops polling once the run settles", () => {
    expect(interval(query({ status: "completed" }))).toBe(false);
    expect(interval(query({ status: "failed" }))).toBe(false);
  });

  it("relaxes to settledMs instead of stopping when provided", () => {
    const fastSlow = runStatusRefetchInterval({
      isSettled: (run: FakeRun | undefined) => run?.status !== "running",
      activeMs: 750,
      settledMs: 5000,
    });
    expect(fastSlow(query({ status: "running" }))).toBe(750);
    expect(fastSlow(query({ status: "completed" }))).toBe(5000);
  });
});
