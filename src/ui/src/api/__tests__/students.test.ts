// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Tests for the StudentModel API client wrappers.
 *
 * Pins the wire contract (URL, method, query string, body) of the thin
 * apiFetch shells. The colon-suffixed action paths matter especially:
 * ``:deployment_handoff`` is the only gated handoff path, so a typo that
 * silently fell back to a different endpoint would skip the readiness gates.
 */

import { describe, it, expect } from "vitest";
import { setupFetchMock } from "@/test/fetch-mock";

import {
  deploymentBundleUrl,
  listStudentModels,
  deployNim,
  rerescoreStudentModel,
  requestDeploymentHandoff,
} from "@/api/students";
import type { DeployNimRequest } from "@/types/training";

const { lastCall } = setupFetchMock();

describe("listStudentModels", () => {
  it("GETs /v1/projects/{id}/student_models with no query string by default", async () => {
    await listStudentModels("proj-1");
    const { url, init } = lastCall();
    expect(url).toBe("/v1/projects/proj-1/student_models");
    expect(init?.method).toBeUndefined();
  });

  it("encodes limit and cursor as query params when given", async () => {
    await listStudentModels("proj-1", { limit: 5, cursor: "c-2" });
    expect(lastCall().url).toBe(
      "/v1/projects/proj-1/student_models?limit=5&cursor=c-2",
    );
  });
});

describe("deployNim", () => {
  it("POSTs /v1/projects/{id}/student_models/{id}:deploy_nim with the request body", async () => {
    const body: DeployNimRequest = {
      nim_endpoint_url: "http://gpu-box:8801/v1",
      auth_mode: "none",
      benchmark_kv_cache_reuse: "disabled",
    };
    await deployNim("proj-1", "sm-1", body);
    const { url, init } = lastCall();
    expect(url).toBe("/v1/projects/proj-1/student_models/sm-1:deploy_nim");
    expect(init?.method).toBe("POST");
    expect(JSON.parse(init?.body as string)).toEqual(body);
  });
});

describe("rerescoreStudentModel", () => {
  it("POSTs /v1/projects/{id}/student_models/{id}:rerescore with an empty body", async () => {
    await rerescoreStudentModel("proj-1", "sm-1");
    const { url, init } = lastCall();
    expect(url).toBe("/v1/projects/proj-1/student_models/sm-1:rerescore");
    expect(init?.method).toBe("POST");
    expect(JSON.parse(init?.body as string)).toEqual({});
  });
});

describe("requestDeploymentHandoff", () => {
  it("POSTs the gated :deployment_handoff action with an empty body", async () => {
    await requestDeploymentHandoff("proj-1", "sm-1");
    const { url, init } = lastCall();
    expect(url).toBe("/v1/projects/proj-1/student_models/sm-1:deployment_handoff");
    expect(init?.method).toBe("POST");
    expect(JSON.parse(init?.body as string)).toEqual({});
  });
});

describe("deploymentBundleUrl", () => {
  it("returns the same-origin streaming download path with encoded IDs", () => {
    expect(deploymentBundleUrl("proj one", "student/two")).toBe(
      "/v1/projects/proj%20one/student_models/student%2Ftwo/deployment_bundle",
    );
  });
});
