// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Tests for the training API client wrappers.
 *
 * Covers each exported function's wire contract: URL, method, body,
 * and (for getTAOJob) the optional refresh query param.
 */

import { describe, it, expect } from "vitest";
import { setupFetchMock } from "@/test/fetch-mock";

import {
  createTrainingSuite,
  cancelTrainingSuite,
  getTAOBaseProvisioning,
  getTrainingSuite,
  listTrainingSuites,
  resolveTrainingPresets,
  getTAOJob,
  cancelTAOJob,
  listStudentBaseModelConfigs,
  runTrainingPreflight,
  startTAOBaseProvisioning,
} from "@/api/training";
import type { TrainingSuiteCreateRequest } from "@/types/training";

const { lastCall } = setupFetchMock();

describe("TAO base provisioning", () => {
  it("POSTs the selected Student base ids", async () => {
    await startTAOBaseProvisioning("proj-1", ["mc-2b", "mc-8b"]);
    const { url, init } = lastCall();
    expect(url).toBe("/v1/projects/proj-1/tao_base_experiment_provisioning");
    expect(init?.method).toBe("POST");
    expect(JSON.parse(init?.body as string)).toEqual({
      student_base_model_config_ids: ["mc-2b", "mc-8b"],
    });
  });

  it("GETs one tracked provisioning run", async () => {
    await getTAOBaseProvisioning("proj-1", "provision-1");
    expect(lastCall().url).toBe(
      "/v1/projects/proj-1/tao_base_experiment_provisioning/provision-1",
    );
  });
});

describe("createTrainingSuite", () => {
  it("POSTs /v1/projects/{id}/training_suites with body", async () => {
    const body: TrainingSuiteCreateRequest = {
      student_base_model_config_ids: ["mc-student-1"],
      training_preset: "standard",
      include_auto_labeled: false,
      export_field_mode: "all",
      quantization_schemes: [],
      idempotency_key: "idem-1",
    };
    await createTrainingSuite("proj-1", body);
    const { url, init } = lastCall();
    expect(url).toBe("/v1/projects/proj-1/training_suites");
    expect(init?.method).toBe("POST");
    expect(JSON.parse(init?.body as string)).toEqual(body);
  });
});

describe("runTrainingPreflight", () => {
  it("POSTs selected bases and the Auto-Labeled inclusion policy", async () => {
    await runTrainingPreflight("proj-1", ["mc-2b"], false);
    const { url, init } = lastCall();
    expect(url).toBe("/v1/projects/proj-1/training_preflight");
    expect(init?.method).toBe("POST");
    expect(JSON.parse(init?.body as string)).toEqual({
      student_base_model_config_ids: ["mc-2b"],
      include_auto_labeled: false,
    });
  });
});

describe("getTrainingSuite", () => {
  it("GETs /v1/projects/{id}/training_suites/{id}", async () => {
    await getTrainingSuite("proj-1", "suite-1");
    expect(lastCall().url).toBe("/v1/projects/proj-1/training_suites/suite-1");
  });
});

describe("listTrainingSuites", () => {
  it("GETs the newest training suites with the requested limit", async () => {
    await listTrainingSuites("proj-1", 25);
    expect(lastCall().url).toBe("/v1/projects/proj-1/training_suites?limit=25");
  });
});

describe("cancelTrainingSuite", () => {
  it("POSTs one suite-level cancellation request", async () => {
    await cancelTrainingSuite("proj-1", "suite-1");
    const { url, init } = lastCall();
    expect(url).toBe("/v1/projects/proj-1/training_suites/suite-1:cancel");
    expect(init?.method).toBe("POST");
    expect(JSON.parse(init?.body as string)).toEqual({});
  });
});

describe("resolveTrainingPresets", () => {
  it("POSTs the student_base_model_config_ids list", async () => {
    await resolveTrainingPresets("proj-1", ["mc-2b", "mc-8b"]);
    const { url, init } = lastCall();
    expect(url).toBe("/v1/projects/proj-1/training_presets:resolve");
    expect(init?.method).toBe("POST");
    expect(JSON.parse(init?.body as string)).toEqual({
      student_base_model_config_ids: ["mc-2b", "mc-8b"],
    });
  });
});

describe("getTAOJob", () => {
  it("GETs /v1/projects/{id}/tao_jobs/{id} when refresh is omitted", async () => {
    await getTAOJob("proj-1", "tao-1");
    expect(lastCall().url).toBe("/v1/projects/proj-1/tao_jobs/tao-1");
  });

  it("appends ?refresh=true when refresh is true", async () => {
    await getTAOJob("proj-1", "tao-1", true);
    expect(lastCall().url).toBe("/v1/projects/proj-1/tao_jobs/tao-1?refresh=true");
  });

  it("omits ?refresh=true when refresh is false", async () => {
    await getTAOJob("proj-1", "tao-1", false);
    // Falsy refresh → no query suffix (the refresh-disabled default).
    expect(lastCall().url).toBe("/v1/projects/proj-1/tao_jobs/tao-1");
  });
});

describe("cancelTAOJob", () => {
  it("POSTs /v1/projects/{id}/tao_jobs/{id}:cancel with empty body", async () => {
    await cancelTAOJob("proj-1", "tao-1");
    const { url, init } = lastCall();
    expect(url).toBe("/v1/projects/proj-1/tao_jobs/tao-1:cancel");
    expect(init?.method).toBe("POST");
    expect(JSON.parse(init?.body as string)).toEqual({});
  });
});

describe("listStudentBaseModelConfigs", () => {
  it("delegates to fetchModelConfigs with eligible_role=student_base", async () => {
    await listStudentBaseModelConfigs("proj-1");
    const { url } = lastCall();
    // Implementation calls fetchModelConfigs(projectId, "student_base"),
    // which queries with the role as a filter.
    expect(url).toBe("/v1/projects/proj-1/model_configs?eligible_role=student_base");
  });
});
