// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Tests for the filesystem API client wrappers.
 *
 * Pins the wire contract (URL, method, query string, body) of the thin
 * apiFetch shells, including query-param encoding of filesystem paths and
 * the undefined→null normalization of scan's optional project_id.
 */

import { describe, it, expect } from "vitest";
import { setupFetchMock } from "@/test/fetch-mock";

import { browseFilesystem, scanDirectory, ingestExamples } from "@/api/filesystem";

const { lastCall } = setupFetchMock();

describe("browseFilesystem", () => {
  it("omits path so the backend selects the deployment image root", async () => {
    await browseFilesystem();
    const { url } = lastCall();
    expect(url).toBe("/v1/filesystem/browse?show_files=true&image_formats_only=true");
  });

  it("GETs /v1/filesystem/browse with the path URL-encoded and default flags true", async () => {
    await browseFilesystem("/data/images");
    const { url, init } = lastCall();
    expect(url).toBe(
      "/v1/filesystem/browse?path=%2Fdata%2Fimages&show_files=true&image_formats_only=true",
    );
    expect(init?.method).toBeUndefined();
  });

  it("encodes flag overrides as false rather than omitting them", async () => {
    await browseFilesystem("/data", false, false);
    const { url } = lastCall();
    expect(url).toContain("show_files=false");
    expect(url).toContain("image_formats_only=false");
  });
});

describe("scanDirectory", () => {
  it("POSTs /v1/filesystem/scan with path, recursive, and a null project_id by default", async () => {
    await scanDirectory("/data/images");
    const { url, init } = lastCall();
    expect(url).toBe("/v1/filesystem/scan");
    expect(init?.method).toBe("POST");
    // Omitted projectId is sent as an explicit null, not dropped.
    expect(JSON.parse(init?.body as string)).toEqual({
      path: "/data/images",
      recursive: true,
      project_id: null,
    });
  });

  it("passes recursive=false and a concrete project_id through the body", async () => {
    await scanDirectory("/data/images", false, "proj-1");
    const body = JSON.parse(lastCall().init?.body as string);
    expect(body).toEqual({
      path: "/data/images",
      recursive: false,
      project_id: "proj-1",
    });
  });
});

describe("ingestExamples", () => {
  it("POSTs /v1/projects/{id}/examples:ingest with the examples payload", async () => {
    const body = {
      examples: [{ example_key: "img-1.png", storage_ref: "/data/img-1.png" }],
    };
    await ingestExamples("proj-1", body);
    const { url, init } = lastCall();
    expect(url).toBe("/v1/projects/proj-1/examples:ingest");
    expect(init?.method).toBe("POST");
    expect(JSON.parse(init?.body as string)).toEqual(body);
  });
});
