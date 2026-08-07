// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Tests for the NIM API client wrappers.
 *
 * Validates the wire contract (URL, method, body) for each exported
 * function. These thin wrappers are easy to break with a path rename
 * or method typo and the bug only surfaces at runtime; testing them
 * directly lets a refactor surface the regression in CI.
 */

import { describe, it, expect } from "vitest";
import { setupFetchMock } from "@/test/fetch-mock";
import { ApiError } from "@/api/client";

import {
  fetchEnvironment,
  fetchNimEndpoints,
  configureSelfHostedEmbedding,
  configureSelfHostedTeacher,
  testConnection,
  runPreflight,
  deployLocalNim,
  generateActionRequest,
  logActionRequestCopy,
  parseLocalNimGpuConflict,
  stopLocalNim,
} from "@/api/nim";
import type {
  ActionRequestGenerateRequest,
  ConnectionTestRequest,
  LocalNimDeployRequest,
  LocalNimPreflightRequest,
} from "@/types/nim";

const { lastCall } = setupFetchMock();

describe("fetchEnvironment", () => {
  it("GETs /v1/environment", async () => {
    await fetchEnvironment();
    expect(lastCall().url).toBe("/v1/environment");
  });

  it("requests an explicit machine re-probe when asked", async () => {
    await fetchEnvironment(true);
    expect(lastCall().url).toBe("/v1/environment?refresh_hardware=true");
  });
});

describe("fetchNimEndpoints", () => {
  it("GETs the project-scoped endpoint catalog", async () => {
    await fetchNimEndpoints("proj-1");
    expect(lastCall().url).toBe("/v1/projects/proj-1/nim_endpoints");
  });
});

describe("testConnection", () => {
  it("POSTs /v1/nim/test_connection with the request body", async () => {
    const body: ConnectionTestRequest = {
      base_url: "https://x",
      credential_transient: "k",
    };
    await testConnection(body);
    const { url, init } = lastCall();
    expect(url).toBe("/v1/nim/test_connection");
    expect(init?.method).toBe("POST");
    expect(JSON.parse(init?.body as string)).toEqual(body);
  });
});

describe("configureSelfHostedTeacher", () => {
  it("POSTs the verified endpoint and exact catalog model", async () => {
    const body = {
      base_url: "http://nim.internal:8000/v1",
      model_config_id: "mc-1",
    };
    await configureSelfHostedTeacher("proj-1", body);
    const { url, init } = lastCall();
    expect(url).toBe("/v1/projects/proj-1/nim_endpoints:configure_self_hosted_teacher");
    expect(init?.method).toBe("POST");
    expect(JSON.parse(init?.body as string)).toEqual(body);
  });
});

describe("configureSelfHostedEmbedding", () => {
  it("POSTs the live-verified deployment-scoped endpoint", async () => {
    const body = { base_url: "http://embedding.internal:8000/v1" };
    await configureSelfHostedEmbedding(body);
    const { url, init } = lastCall();
    expect(url).toBe("/v1/embedding_deployment_config:configure_self_hosted");
    expect(init?.method).toBe("POST");
    expect(JSON.parse(init?.body as string)).toEqual(body);
  });
});

describe("runPreflight", () => {
  it("POSTs /v1/projects/{id}/local_nim/preflight with body", async () => {
    const body: LocalNimPreflightRequest = { role: "teacher", model_config_id: "mc-1" };
    await runPreflight("proj-1", body);
    const { url, init } = lastCall();
    expect(url).toBe("/v1/projects/proj-1/local_nim/preflight");
    expect(init?.method).toBe("POST");
    expect(JSON.parse(init?.body as string)).toEqual(body);
  });
});

describe("deployLocalNim", () => {
  it("POSTs /v1/projects/{id}/local_nim/deploy with body", async () => {
    const body: LocalNimDeployRequest = { role: "teacher", model_config_id: "mc-1" };
    await deployLocalNim("proj-1", body);
    const { url, init } = lastCall();
    expect(url).toBe("/v1/projects/proj-1/local_nim/deploy");
    expect(init?.method).toBe("POST");
    expect(JSON.parse(init?.body as string)).toEqual(body);
  });
});

describe("stopLocalNim", () => {
  it("POSTs the project-owned deployment stop action", async () => {
    await stopLocalNim("proj-1", "dep-1");
    const { url, init } = lastCall();
    expect(url).toBe("/v1/projects/proj-1/local_nim/deployments/dep-1:stop");
    expect(init?.method).toBe("POST");
  });
});

describe("parseLocalNimGpuConflict", () => {
  it("parses FastAPI's nested resident replacement detail", () => {
    const error = new ApiError(
      409,
      JSON.stringify({
        detail: {
          code: "gpu_occupied",
          message: "occupied",
          can_replace: true,
          matches_requested_model: false,
          resident: {
            project_id: "old-project",
            project_name: "Old project",
            local_nim_deployment_id: "dep-1",
            role: "teacher",
            model_name: "nvidia/cosmos3-nano-reasoner",
            nim_container_image: "image",
            gpu_assignment: "device=0",
            status: "running",
          },
        },
      }),
    );

    expect(parseLocalNimGpuConflict(error)).toEqual(
      expect.objectContaining({
        code: "gpu_occupied",
        can_replace: true,
        resident: expect.objectContaining({ project_name: "Old project" }),
      }),
    );
  });
});

describe("generateActionRequest", () => {
  it("POSTs /v1/projects/{id}/action_requests:generate with body", async () => {
    const body: ActionRequestGenerateRequest = { request_type: "tao_setup" };
    await generateActionRequest("proj-1", body);
    const { url, init } = lastCall();
    expect(url).toBe("/v1/projects/proj-1/action_requests:generate");
    expect(init?.method).toBe("POST");
    expect(JSON.parse(init?.body as string)).toEqual(body);
  });
});

describe("logActionRequestCopy", () => {
  it("POSTs /v1/projects/{id}/action_requests:log_copy with body", async () => {
    const body = { request_type: "tao_setup", rendered_text: "..." };
    await logActionRequestCopy("proj-1", body);
    const { url, init } = lastCall();
    expect(url).toBe("/v1/projects/proj-1/action_requests:log_copy");
    expect(init?.method).toBe("POST");
    expect(JSON.parse(init?.body as string)).toEqual(body);
  });
});
