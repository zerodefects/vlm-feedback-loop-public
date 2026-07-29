// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Tests for client.ts error-body parsing.
 *
 * parseApiErrorDetail / parseApiErrorReasons sit between raw HTTP error
 * bodies and every user-facing error message. Backend handlers emit
 * `{"detail": ...}` JSON, but nginx/proxy failures deliver HTML or plain
 * text, and FastAPI 422s carry a non-string `detail` array — if parsing
 * throws or returns the wrong shape on any of those, error surfaces
 * across the app crash or go blank instead of showing a fallback.
 */

import { describe, it, expect } from "vitest";

import { ApiError, parseApiErrorDetail, parseApiErrorReasons } from "@/api/client";

describe("parseApiErrorDetail", () => {
  it("extracts the detail string from a structured backend error body", () => {
    const err = new ApiError(404, JSON.stringify({ detail: "Project not found" }));
    expect(parseApiErrorDetail(err)).toBe("Project not found");
  });

  it("returns null for a plain-text proxy body instead of throwing", () => {
    // nginx 502/504 pages are HTML, not JSON — the JSON.parse failure must
    // be swallowed so callers can fall back to their own message.
    const err = new ApiError(502, "<html><body>502 Bad Gateway</body></html>");
    expect(parseApiErrorDetail(err)).toBeNull();
  });

  it("returns null when detail is not a string (FastAPI 422 array shape)", () => {
    const err = new ApiError(
      422,
      JSON.stringify({ detail: [{ loc: ["body", "name"], msg: "field required" }] }),
    );
    expect(parseApiErrorDetail(err)).toBeNull();
  });

  it("returns null for an empty-string detail so callers use their fallback", () => {
    const err = new ApiError(500, JSON.stringify({ detail: "" }));
    expect(parseApiErrorDetail(err)).toBeNull();
  });

  it("returns null for JSON bodies without a detail key", () => {
    const err = new ApiError(500, JSON.stringify({ error: "boom" }));
    expect(parseApiErrorDetail(err)).toBeNull();
  });

  it("returns null for non-ApiError values", () => {
    expect(parseApiErrorDetail(new Error('{"detail": "not an ApiError"}'))).toBeNull();
    expect(parseApiErrorDetail(undefined)).toBeNull();
  });
});

describe("parseApiErrorReasons", () => {
  it("extracts the reasons list from the busy-gate 409 body", () => {
    const err = new ApiError(
      409,
      JSON.stringify({
        detail: "Project is busy",
        reasons: ["evaluation running", "batch run active"],
      }),
    );
    expect(parseApiErrorReasons(err)).toEqual([
      "evaluation running",
      "batch run active",
    ]);
  });

  it("returns [] for a plain-text proxy body instead of throwing", () => {
    const err = new ApiError(504, "upstream request timed out");
    expect(parseApiErrorReasons(err)).toEqual([]);
  });

  it("returns [] when reasons is not an array", () => {
    const err = new ApiError(409, JSON.stringify({ reasons: "evaluation running" }));
    expect(parseApiErrorReasons(err)).toEqual([]);
  });

  it("returns [] when any element of reasons is not a string", () => {
    // A malformed list is dropped wholesale, never partially rendered.
    const err = new ApiError(
      409,
      JSON.stringify({ reasons: ["evaluation running", 42] }),
    );
    expect(parseApiErrorReasons(err)).toEqual([]);
  });

  it("returns [] when reasons is absent", () => {
    const err = new ApiError(409, JSON.stringify({ detail: "Conflict" }));
    expect(parseApiErrorReasons(err)).toEqual([]);
  });

  it("returns [] for non-ApiError values", () => {
    expect(parseApiErrorReasons(new Error("boom"))).toEqual([]);
  });
});
