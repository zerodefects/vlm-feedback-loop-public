// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Tests for the batch labeling / dataset export API client wrappers.
 *
 * Pins the wire contract (URL, method, body, response unwrapping) of
 * each thin fetch wrapper — a path typo or `:verb` rename survives
 * compilation and only breaks at runtime against the real backend.
 */

import { describe, it, expect } from "vitest";
import { setupFetchMock } from "@/test/fetch-mock";

import {
  createBatchLabelRun,
  getBatchLabelRun,
  resumeBatchLabelRun,
  cancelBatchLabelRun,
  getSchemaInvalidManifest,
  createDatasetExport,
  datasetExportArchiveUrl,
} from "@/api/batch";

const { fetchMock, lastCall } = setupFetchMock();

describe("createBatchLabelRun", () => {
  it("POSTs /v1/projects/{id}/batch_label_runs with the serialized options", async () => {
    await createBatchLabelRun("proj-1", {
      include_auto_labeled: true,
      run_limit: 50,
      structured_generation_mode: "json_schema",
      ingested_after: "2026-01-01T00:00:00Z",
      ingested_before: null,
    });
    const { url, init } = lastCall();
    expect(url).toBe("/v1/projects/proj-1/batch_label_runs");
    expect(init?.method).toBe("POST");
    expect(JSON.parse(init?.body as string)).toEqual({
      include_auto_labeled: true,
      run_limit: 50,
      structured_generation_mode: "json_schema",
      ingested_after: "2026-01-01T00:00:00Z",
      ingested_before: null,
    });
  });
});

describe("getBatchLabelRun", () => {
  it("GETs /v1/projects/{id}/batch_label_runs/{runId} and returns the JSON body", async () => {
    fetchMock.mockResolvedValueOnce({
      ok: true,
      text: () => Promise.resolve(""),
      json: () => Promise.resolve({ run_id: "run-1", status: "running" }),
    });
    const result = await getBatchLabelRun("proj-1", "run-1");
    const { url, init } = lastCall();
    expect(url).toBe("/v1/projects/proj-1/batch_label_runs/run-1");
    expect(init?.method).toBeUndefined();
    expect(result).toEqual({ run_id: "run-1", status: "running" });
  });
});

describe.each([
  ["resumeBatchLabelRun", resumeBatchLabelRun, ":resume"],
  ["cancelBatchLabelRun", cancelBatchLabelRun, ":cancel"],
] as const)("%s", (_name, fn, verb) => {
  it(`POSTs /v1/projects/{id}/batch_label_runs/{runId}${verb} with an empty JSON body`, async () => {
    await fn("proj-1", "run-1");
    const { url, init } = lastCall();
    expect(url).toBe(`/v1/projects/proj-1/batch_label_runs/run-1${verb}`);
    expect(init?.method).toBe("POST");
    expect(JSON.parse(init?.body as string)).toEqual({});
  });
});

describe("getSchemaInvalidManifest", () => {
  it("GETs /v1/projects/{id}/batch_label_runs/{runId}/schema_invalid_manifest", async () => {
    await getSchemaInvalidManifest("proj-1", "run-1");
    const { url, init } = lastCall();
    expect(url).toBe(
      "/v1/projects/proj-1/batch_label_runs/run-1/schema_invalid_manifest",
    );
    expect(init?.method).toBeUndefined();
  });
});

describe("createDatasetExport", () => {
  it("POSTs /v1/projects/{id}/dataset_exports with the serialized export options", async () => {
    await createDatasetExport("proj-1", {
      dataset_intent: "student_training",
      label_tier_filter: "verified_only",
      export_field_mode: "core_only",
      batch_label_run_id: "run-1",
    });
    const { url, init } = lastCall();
    expect(url).toBe("/v1/projects/proj-1/dataset_exports");
    expect(init?.method).toBe("POST");
    expect(JSON.parse(init?.body as string)).toEqual({
      dataset_intent: "student_training",
      label_tier_filter: "verified_only",
      export_field_mode: "core_only",
      batch_label_run_id: "run-1",
    });
  });
});

describe("datasetExportArchiveUrl", () => {
  it("builds the same-origin archive download path and escapes IDs", () => {
    expect(datasetExportArchiveUrl("project/one", "export two")).toBe(
      "/v1/projects/project%2Fone/dataset_exports/export%20two/archive",
    );
  });
});
