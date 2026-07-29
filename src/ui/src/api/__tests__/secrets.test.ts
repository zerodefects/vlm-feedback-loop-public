// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Tests for the secrets API client wrapper.
 *
 * Pins the wire contract (URL, method, body) of the ``:set`` custom-action
 * endpoint — the colon-suffixed path is easy to break in a refactor and the
 * bug only surfaces at runtime.
 */

import { describe, it, expect } from "vitest";
import { setupFetchMock } from "@/test/fetch-mock";

import { setSecret } from "@/api/secrets";

const { lastCall } = setupFetchMock();

describe("setSecret", () => {
  it("POSTs /v1/secrets:set with name/value/persist in the body", async () => {
    await setSecret({ name: "NVIDIA_API_KEY", value: "nvapi-xyz", persist: true });
    const { url, init } = lastCall();
    expect(url).toBe("/v1/secrets:set");
    expect(init?.method).toBe("POST");
    expect(JSON.parse(init?.body as string)).toEqual({
      name: "NVIDIA_API_KEY",
      value: "nvapi-xyz",
      persist: true,
    });
  });
});
