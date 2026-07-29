// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Tests for the model-config API client wrappers.
 *
 * Pins the wire contract (URL, method, query string, body) of the thin
 * apiFetch shells — a path typo or a renamed query param survives
 * compilation and would otherwise only surface at runtime.
 */

import { describe, it, expect } from "vitest";
import { setupFetchMock } from "@/test/fetch-mock";

import { fetchModelConfigs, updateProject } from "@/api/model-configs";

const { lastCall } = setupFetchMock();

describe("fetchModelConfigs", () => {
  it("GETs /v1/projects/{id}/model_configs with no query string by default", async () => {
    await fetchModelConfigs("proj-1");
    const { url, init } = lastCall();
    expect(url).toBe("/v1/projects/proj-1/model_configs");
    expect(init?.method).toBeUndefined();
  });

  it("appends eligible_role only when a role is given", async () => {
    await fetchModelConfigs("proj-1", "teacher");
    expect(lastCall().url).toBe(
      "/v1/projects/proj-1/model_configs?eligible_role=teacher",
    );
  });
});

describe("updateProject", () => {
  it("PATCHes /v1/projects/{id} with the serialized body", async () => {
    await updateProject("proj-1", { teacher_model_config_id: "mc-2" });
    const { url, init } = lastCall();
    expect(url).toBe("/v1/projects/proj-1");
    expect(init?.method).toBe("PATCH");
    expect(JSON.parse(init?.body as string)).toEqual({
      teacher_model_config_id: "mc-2",
    });
  });
});
