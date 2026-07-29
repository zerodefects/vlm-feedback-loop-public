// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Tests for the Interactive Labeling API client wrappers.
 *
 * Pins the wire contract (URL, method, body, response unwrapping) of
 * each thin fetch wrapper, plus the `imageUrl` helper whose string is
 * consumed directly by `<img src>` — a typo there breaks every image
 * on the review screen with no compile-time signal.
 */

import { describe, it, expect } from "vitest";
import { setupFetchMock } from "@/test/fetch-mock";

import {
  fetchNextReviewItem,
  createProposal,
  saveLabel,
  skipExample,
  restoreOmitted,
  regenerateRationale,
  imageUrl,
} from "@/api/labeling";
import type { LabelSaveRequest } from "@/types/labeling";

const { fetchMock, lastCall } = setupFetchMock();

describe("fetchNextReviewItem", () => {
  it("GETs /v1/projects/{id}/review_selector/next and returns the JSON body", async () => {
    fetchMock.mockResolvedValueOnce({
      ok: true,
      text: () => Promise.resolve(""),
      json: () => Promise.resolve({ example_key: "ex-1", pool_exhausted: false }),
    });
    const result = await fetchNextReviewItem("proj-1");
    const { url, init } = lastCall();
    expect(url).toBe("/v1/projects/proj-1/review_selector/next");
    expect(init?.method).toBeUndefined();
    expect(result).toEqual({ example_key: "ex-1", pool_exhausted: false });
  });
});

describe("createProposal", () => {
  it("POSTs /v1/projects/{id}/proposals with the serialized request", async () => {
    await createProposal("proj-1", {
      example_key: "ex-1",
      thinking_mode_override: "on",
    });
    const { url, init } = lastCall();
    expect(url).toBe("/v1/projects/proj-1/proposals");
    expect(init?.method).toBe("POST");
    expect(JSON.parse(init?.body as string)).toEqual({
      example_key: "ex-1",
      thinking_mode_override: "on",
    });
  });
});

describe("saveLabel", () => {
  it("POSTs /v1/projects/{id}/labels with the serialized label", async () => {
    const body: LabelSaveRequest = {
      example_key: "ex-1",
      inference_invocation_id: "inv-1",
      label_json: { category: "rock" },
      rationale_source: "teacher_original",
    };
    await saveLabel("proj-1", body);
    const { url, init } = lastCall();
    expect(url).toBe("/v1/projects/proj-1/labels");
    expect(init?.method).toBe("POST");
    expect(JSON.parse(init?.body as string)).toEqual(body);
  });
});

describe("skipExample", () => {
  it("POSTs /v1/projects/{id}/examples/{key}:skip with no body", async () => {
    await skipExample("proj-1", "ex-1");
    const { url, init } = lastCall();
    expect(url).toBe("/v1/projects/proj-1/examples/ex-1:skip");
    expect(init?.method).toBe("POST");
    expect(init?.body).toBeUndefined();
  });
});

describe("restoreOmitted", () => {
  it("POSTs /v1/projects/{id}/examples:restore_omitted with no body", async () => {
    await restoreOmitted("proj-1");
    const { url, init } = lastCall();
    expect(url).toBe("/v1/projects/proj-1/examples:restore_omitted");
    expect(init?.method).toBe("POST");
    expect(init?.body).toBeUndefined();
  });
});

describe("regenerateRationale", () => {
  it("POSTs rationale regeneration without label values", async () => {
    await regenerateRationale("proj-1", "ex-1", {
      teacher_model_config_id: "mc-1",
    });
    const { url, init } = lastCall();
    expect(url).toBe("/v1/projects/proj-1/examples/ex-1:regenerate_rationale");
    expect(init?.method).toBe("POST");
    expect(JSON.parse(init?.body as string)).toEqual({
      teacher_model_config_id: "mc-1",
    });
  });
});

describe("imageUrl", () => {
  it("builds the /v1 image serving path used as <img src>", () => {
    expect(imageUrl("proj-1", "ex-1")).toBe("/v1/projects/proj-1/examples/ex-1/image");
  });
});
