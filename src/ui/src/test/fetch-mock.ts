// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Shared global-fetch stub for API-wrapper tests (api/__tests__/*).
 *
 * `setupFetchMock()` registers the beforeEach/afterEach lifecycle for the
 * calling test file: a fresh JSON-friendly default response per test and
 * `vi.unstubAllGlobals()` on teardown. The returned `fetchMock` reference is
 * stable across tests (it is `mockReset()` between them), so per-test
 * overrides via `fetchMock.mockResolvedValueOnce(...)` work as usual.
 */

import { afterEach, beforeEach, expect, vi } from "vitest";

export interface FetchMockHarness {
  /** The stubbed global fetch. Stable identity; reset before each test. */
  fetchMock: ReturnType<typeof vi.fn>;
  /**
   * URL and RequestInit of the sole recorded fetch call. Asserts exactly one
   * call was made — API-wrapper tests exercise one request per test.
   */
  lastCall(): { url: string; init: RequestInit | undefined };
}

export function setupFetchMock(): FetchMockHarness {
  const fetchMock = vi.fn();

  beforeEach(() => {
    fetchMock.mockReset();
    fetchMock.mockResolvedValue({
      ok: true,
      text: () => Promise.resolve(""),
      json: () => Promise.resolve({}),
    });
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  function lastCall() {
    expect(fetchMock).toHaveBeenCalledTimes(1);
    return {
      url: fetchMock.mock.calls[0][0] as string,
      init: fetchMock.mock.calls[0][1] as RequestInit | undefined,
    };
  }

  return { fetchMock, lastCall };
}
