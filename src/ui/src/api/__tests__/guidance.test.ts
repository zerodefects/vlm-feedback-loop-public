// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Tests for the Guidance API client wrappers.
 *
 * Pins the wire contract (URL, method, body, query-string building,
 * response unwrapping) of each thin fetch wrapper. The preview/execute
 * pair shares one endpoint and differs only in the injected `dry_run`
 * flag — swapping those silently corrupts Guidance history, so the
 * flag injection is pinned explicitly.
 */

import { describe, it, expect } from "vitest";
import { setupFetchMock } from "@/test/fetch-mock";

import {
  createGuidance,
  validateDraft,
  fetchGuidance,
  listGuidances,
  editGuidancePreview,
  editGuidanceExecute,
  fetchIclCount,
  fetchReminderStatus,
  dismissReminder,
} from "@/api/guidance";
import type { SchemaFieldInput } from "@/types/guidance";

const { fetchMock, lastCall } = setupFetchMock();

const schemaField: SchemaFieldInput = {
  field_name: "category",
  type: "string",
  role: "core",
  allowed_values: ["rock", "paper"],
  display_order: 0,
};

const draftBody = { description: "desc", schema: [schemaField], rules: "rules" };

describe("createGuidance", () => {
  it("POSTs /v1/projects/{id}/guidance with the serialized draft", async () => {
    await createGuidance("proj-1", draftBody);
    const { url, init } = lastCall();
    expect(url).toBe("/v1/projects/proj-1/guidance");
    expect(init?.method).toBe("POST");
    expect(JSON.parse(init?.body as string)).toEqual(draftBody);
  });
});

describe("validateDraft", () => {
  it("POSTs /v1/projects/{id}/guidance:validate_draft with the serialized draft", async () => {
    await validateDraft("proj-1", draftBody);
    const { url, init } = lastCall();
    expect(url).toBe("/v1/projects/proj-1/guidance:validate_draft");
    expect(init?.method).toBe("POST");
    expect(JSON.parse(init?.body as string)).toEqual(draftBody);
  });
});

describe("fetchGuidance", () => {
  it("GETs /v1/projects/{id}/guidance/{guidanceId} and returns the JSON body", async () => {
    fetchMock.mockResolvedValueOnce({
      ok: true,
      text: () => Promise.resolve(""),
      json: () => Promise.resolve({ guidance_id: "g-1", version: 3 }),
    });
    const result = await fetchGuidance("proj-1", "g-1");
    const { url, init } = lastCall();
    expect(url).toBe("/v1/projects/proj-1/guidance/g-1");
    expect(init?.method).toBeUndefined();
    expect(result).toEqual({ guidance_id: "g-1", version: 3 });
  });
});

describe("listGuidances", () => {
  it("GETs /v1/projects/{id}/guidance with no query string when params are omitted", async () => {
    await listGuidances("proj-1");
    expect(lastCall().url).toBe("/v1/projects/proj-1/guidance");
  });

  it("encodes cursor and limit as query params when provided", async () => {
    await listGuidances("proj-1", "c1", 10);
    expect(lastCall().url).toBe("/v1/projects/proj-1/guidance?cursor=c1&limit=10");
  });
});

describe.each([
  ["editGuidancePreview", editGuidancePreview, true],
  ["editGuidanceExecute", editGuidanceExecute, false],
] as const)("%s", (_name, fn, dryRun) => {
  it(`POSTs /v1/projects/{id}/guidance:edit with dry_run=${dryRun} injected into the body`, async () => {
    await fn("proj-1", {
      ...draftBody,
      schema_change_context_example_key: "ex-1",
    });
    const { url, init } = lastCall();
    expect(url).toBe("/v1/projects/proj-1/guidance:edit");
    expect(init?.method).toBe("POST");
    expect(JSON.parse(init?.body as string)).toEqual({
      ...draftBody,
      schema_change_context_example_key: "ex-1",
      dry_run: dryRun,
    });
  });
});

describe.each([
  ["fetchIclCount", fetchIclCount, ":icl_count"],
  ["fetchReminderStatus", fetchReminderStatus, ":reminder_status"],
] as const)("%s", (_name, fn, verb) => {
  it(`GETs /v1/projects/{id}/guidance${verb}`, async () => {
    await fn("proj-1");
    const { url, init } = lastCall();
    expect(url).toBe(`/v1/projects/proj-1/guidance${verb}`);
    expect(init?.method).toBeUndefined();
  });
});

describe("dismissReminder", () => {
  it("POSTs /v1/projects/{id}/guidance:dismiss_reminder with no body", async () => {
    await dismissReminder("proj-1");
    const { url, init } = lastCall();
    expect(url).toBe("/v1/projects/proj-1/guidance:dismiss_reminder");
    expect(init?.method).toBe("POST");
    expect(init?.body).toBeUndefined();
  });
});
