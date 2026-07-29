// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Tests for the project API client wrappers.
 *
 * These functions are thin shells over apiFetch — they look trivial,
 * but a typo in the URL or method name is exactly the kind of bug that
 * survives compilation and only shows up at runtime. The wire contract
 * (path, method, body shape, query string) is tested directly here so
 * a refactor of the client.ts wrapper or a path rename surfaces
 * immediately.
 */

import { describe, it, expect } from "vitest";
import { setupFetchMock } from "@/test/fetch-mock";

import {
  fetchProject,
  createProject,
  fetchProjectList,
  archiveProject,
  unarchiveProject,
} from "@/api/projects";

// ── Fetch mock ──────────────────────────────────────────────────────────────

const { fetchMock, lastCall } = setupFetchMock();

describe("fetchProject", () => {
  it("GETs /v1/projects/{id}", async () => {
    await fetchProject("proj-1");
    const { url, init } = lastCall();
    expect(url).toBe("/v1/projects/proj-1");
    // GET is the default method; init may be undefined or have no method.
    expect(init?.method).toBeUndefined();
  });
});

describe("createProject", () => {
  it("POSTs /v1/projects with serialized body", async () => {
    await createProject({ name: "Test Project", description: "desc" });
    const { url, init } = lastCall();
    expect(url).toBe("/v1/projects");
    expect(init?.method).toBe("POST");
    const body = JSON.parse(init?.body as string);
    expect(body).toEqual({ name: "Test Project", description: "desc" });
  });
});

describe("fetchProjectList", () => {
  // Helper for the multi-page response shape. setupFetchMock's default
  // resolves to `{}` which has no `next_cursor` and no `items` — which
  // terminates the auto-pagination loop after one fetch with [].
  function mockPagesOnce(
    pages: Array<{
      items: unknown[];
      next_cursor: string | null;
      has_archived?: boolean;
    }>,
  ) {
    fetchMock.mockReset();
    for (const page of pages) {
      fetchMock.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(""),
        json: () => Promise.resolve(page),
      });
    }
  }

  it("requests /v1/projects with the max page size (200)", async () => {
    await fetchProjectList();
    const { url } = lastCall();
    expect(url).toBe("/v1/projects?limit=200");
  });

  it("appends include_archived only when truthy", async () => {
    await fetchProjectList(true);
    expect(lastCall().url).toContain("include_archived=true");
    expect(lastCall().url).toContain("limit=200");

    fetchMock.mockClear();
    fetchMock.mockResolvedValue({
      ok: true,
      text: () => Promise.resolve(""),
      json: () => Promise.resolve({}),
    });
    await fetchProjectList(false);
    // Falsy path: param is omitted, not encoded as "false".
    expect(lastCall().url).not.toContain("include_archived");
  });

  it("walks next_cursor across multiple pages and returns the merged list", async () => {
    mockPagesOnce([
      { items: [{ project_id: "p1" }, { project_id: "p2" }], next_cursor: "c1" },
      { items: [{ project_id: "p3" }], next_cursor: "c2" },
      { items: [{ project_id: "p4" }], next_cursor: null, has_archived: true },
    ]);

    const result = await fetchProjectList();
    expect(result.items.map((p: { project_id: string }) => p.project_id)).toEqual([
      "p1",
      "p2",
      "p3",
      "p4",
    ]);
    expect(result.next_cursor).toBeNull();
    // Workspace-global flag carries through the aggregated result.
    expect(result.has_archived).toBe(true);

    expect(fetchMock).toHaveBeenCalledTimes(3);
    // First page: no cursor. Subsequent pages: cursor from prior response.
    expect(fetchMock.mock.calls[0][0]).toBe("/v1/projects?limit=200");
    expect(fetchMock.mock.calls[1][0]).toContain("cursor=c1");
    expect(fetchMock.mock.calls[2][0]).toContain("cursor=c2");
  });
});

describe("archiveProject", () => {
  it("POSTs /v1/projects/{id}:archive with no body content", async () => {
    await archiveProject("proj-1");
    const { url, init } = lastCall();
    expect(url).toBe("/v1/projects/proj-1:archive");
    expect(init?.method).toBe("POST");
  });
});

describe("unarchiveProject", () => {
  it("POSTs /v1/projects/{id}:unarchive", async () => {
    await unarchiveProject("proj-9");
    const { url, init } = lastCall();
    expect(url).toBe("/v1/projects/proj-9:unarchive");
    expect(init?.method).toBe("POST");
  });
});
