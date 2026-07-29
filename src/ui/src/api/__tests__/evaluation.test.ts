// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Tests for the evaluation API client wrappers.
 *
 * Pins the wire contract (URL, method, body, query-string building,
 * response unwrapping) of each thin fetch wrapper — a path typo or
 * query-param rename survives compilation and only breaks at runtime.
 */

import { describe, it, expect } from "vitest";
import { setupFetchMock } from "@/test/fetch-mock";

import {
  createEvaluationRun,
  getEvaluationRun,
  listEvaluationRuns,
  cancelEvaluationRun,
  fetchTriggerStatus,
  dismissTrigger,
  fetchScaleUpGate,
} from "@/api/evaluation";

const { fetchMock, lastCall } = setupFetchMock();

describe("createEvaluationRun", () => {
  it("POSTs /v1/projects/{id}/evaluation_runs with the serialized options", async () => {
    await createEvaluationRun("proj-1", {
      icl_mode: "with_icl",
      structured_generation_mode: null,
    });
    const { url, init } = lastCall();
    expect(url).toBe("/v1/projects/proj-1/evaluation_runs");
    expect(init?.method).toBe("POST");
    expect(JSON.parse(init?.body as string)).toEqual({
      icl_mode: "with_icl",
      structured_generation_mode: null,
    });
  });
});

describe("getEvaluationRun", () => {
  it("GETs /v1/projects/{id}/evaluation_runs/{runId} and returns the JSON body", async () => {
    fetchMock.mockResolvedValueOnce({
      ok: true,
      text: () => Promise.resolve(""),
      json: () => Promise.resolve({ run_id: "run-1", status: "succeeded" }),
    });
    const result = await getEvaluationRun("proj-1", "run-1");
    const { url, init } = lastCall();
    expect(url).toBe("/v1/projects/proj-1/evaluation_runs/run-1");
    expect(init?.method).toBeUndefined();
    expect(result).toEqual({ run_id: "run-1", status: "succeeded" });
  });
});

describe("listEvaluationRuns", () => {
  it("GETs /v1/projects/{id}/evaluation_runs with no query string when params are omitted", async () => {
    await listEvaluationRuns("proj-1");
    expect(lastCall().url).toBe("/v1/projects/proj-1/evaluation_runs");
  });

  it("encodes status, basis, limit, and cursor as query params when provided", async () => {
    await listEvaluationRuns("proj-1", {
      status: "running",
      basis: "gate",
      limit: 10,
      cursor: "c1",
    });
    expect(lastCall().url).toBe(
      "/v1/projects/proj-1/evaluation_runs?status=running&basis=gate&limit=10&cursor=c1",
    );
  });

  it("omits absent params instead of encoding empty values", async () => {
    await listEvaluationRuns("proj-1", { limit: 5 });
    expect(lastCall().url).toBe("/v1/projects/proj-1/evaluation_runs?limit=5");
  });
});

describe("cancelEvaluationRun", () => {
  it("POSTs /v1/projects/{id}/evaluation_runs/{runId}:cancel with an empty JSON body", async () => {
    await cancelEvaluationRun("proj-1", "run-1");
    const { url, init } = lastCall();
    expect(url).toBe("/v1/projects/proj-1/evaluation_runs/run-1:cancel");
    expect(init?.method).toBe("POST");
    expect(JSON.parse(init?.body as string)).toEqual({});
  });
});

describe("fetchTriggerStatus", () => {
  it("GETs /v1/projects/{id}/evaluation_trigger_status", async () => {
    await fetchTriggerStatus("proj-1");
    const { url, init } = lastCall();
    expect(url).toBe("/v1/projects/proj-1/evaluation_trigger_status");
    expect(init?.method).toBeUndefined();
  });
});

describe("dismissTrigger", () => {
  it("POSTs /v1/projects/{id}/evaluation_trigger_status:dismiss with the trigger type", async () => {
    await dismissTrigger("proj-1", "verified_delta");
    const { url, init } = lastCall();
    expect(url).toBe("/v1/projects/proj-1/evaluation_trigger_status:dismiss");
    expect(init?.method).toBe("POST");
    expect(JSON.parse(init?.body as string)).toEqual({
      trigger_type: "verified_delta",
    });
  });
});

describe("fetchScaleUpGate", () => {
  it("GETs /v1/projects/{id}/scaleup_gate", async () => {
    await fetchScaleUpGate("proj-1");
    const { url, init } = lastCall();
    expect(url).toBe("/v1/projects/proj-1/scaleup_gate");
    expect(init?.method).toBeUndefined();
  });
});
