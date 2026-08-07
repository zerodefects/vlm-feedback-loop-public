// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Student Training safety tests.
 *
 * The screen must start with a small validation suite, render
 * backend-authoritative export counts, fail closed on preflight, and require an
 * exact workload confirmation before creating remote TAO jobs.
 */

import { beforeEach, describe, expect, it, vi } from "vitest";
import { installEventSourceMock } from "@/test/event-source-mock";
import { makeProjectResponse } from "@/test/fixtures";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter, Outlet, Route, Routes } from "react-router-dom";

import { ApiError } from "@/api/client";
import type { QuantizationScheme } from "@/types/training";
import { StudentTrainingPage } from "../StudentTrainingPage";

vi.mock("@/api/training", () => ({
  createTrainingSuite: vi.fn(),
  listStudentBaseModelConfigs: vi.fn(),
  resolveTrainingPresets: vi.fn(),
  runTrainingPreflight: vi.fn(),
}));

vi.mock("@/api/nim", () => ({
  generateActionRequest: vi.fn().mockResolvedValue({
    request_type: "tao_setup",
    rendered_text: "TAO Setup Request\n...",
    generated_at: "2026-07-29T00:00:00Z",
    project_name: "Test",
    technical_requirements: {},
    current_environment: {},
  }),
  logActionRequestCopy: vi.fn().mockResolvedValue({ audit_event_id: "ae-1" }),
}));

vi.mock("@/pages/setup-context", () => ({
  useSetupContext: vi.fn(),
}));

installEventSourceMock();

import {
  createTrainingSuite,
  listStudentBaseModelConfigs,
  resolveTrainingPresets,
  runTrainingPreflight,
} from "@/api/training";
import { useSetupContext } from "@/pages/setup-context";

const mockCreateSuite = createTrainingSuite as ReturnType<typeof vi.fn>;
const mockListBases = listStudentBaseModelConfigs as ReturnType<typeof vi.fn>;
const mockResolvePresets = resolveTrainingPresets as ReturnType<typeof vi.fn>;
const mockRunPreflight = runTrainingPreflight as ReturnType<typeof vi.fn>;
const mockSetup = useSetupContext as ReturnType<typeof vi.fn>;

const PROJECT = makeProjectResponse({
  project_id: "pid-1",
  project_dir: "/tmp/p",
  teacher_model_config_id: "mc-t",
  embedding_provider: "hosted_nvclip",
  counts: {
    verified: 120,
    unlabeled: 60,
    auto_labeled: 341,
    omitted: 5,
    pending_relabel: 0,
  },
});

const BASES = {
  items: [
    {
      model_config_id: "mc-8b",
      project_id: "pid-1",
      endpoint_id: "ep",
      model_name: "nvidia/cosmos-reason2-8b",
      context_window_tokens: 256000,
      eligible_roles: ["student_base"],
      supports_image_input: true,
      structured_generation_support: "supported",
      thinking_toggle_mode: "qwen_enable_thinking",
      thinking_toggle_support: "supported",
      visual_budget_mode: "mm_processor_size",
      visual_budget_support: "supported",
      model_quantization: null,
      nim_model_profile: null,
      nim_profile_metadata: null,
      local_deploy_metadata: null,
      tao_base_experiment_id: "base-8b",
      tao_base_experiment_pull_status: "pull_complete",
      created_at: "2026-07-29T00:00:00Z",
    },
    {
      model_config_id: "mc-2b",
      project_id: "pid-1",
      endpoint_id: "ep",
      model_name: "nvidia/cosmos-reason2-2b",
      context_window_tokens: 256000,
      eligible_roles: ["student_base"],
      supports_image_input: true,
      structured_generation_support: "supported",
      thinking_toggle_mode: "qwen_enable_thinking",
      thinking_toggle_support: "supported",
      visual_budget_mode: "mm_processor_size",
      visual_budget_support: "supported",
      model_quantization: null,
      nim_model_profile: null,
      nim_profile_metadata: null,
      local_deploy_metadata: null,
      tao_base_experiment_id: "base-2b",
      tao_base_experiment_pull_status: "pull_complete",
      created_at: "2026-07-29T00:00:00Z",
    },
  ],
  next_cursor: null,
};

function patch(epoch: number) {
  return {
    train: {
      epoch,
      resume: false,
      ckpt: {
        enable_checkpoint: true,
        save_freq_in_epoch: 1,
        max_keep: 1,
        export_safetensors: true,
      },
    },
  };
}

const RESOLVED_PRESETS = {
  resolved_presets: {
    "mc-8b": {
      quick: patch(1),
      standard: patch(3),
      high_quality: patch(9),
      max_quality: patch(18),
    },
    "mc-2b": {
      quick: patch(1),
      standard: patch(3),
      high_quality: patch(12),
      max_quality: patch(24),
    },
  },
};

function preflight(includeAutoLabeled = true) {
  return {
    status: "passed" as const,
    checks: [
      {
        check_name: "tao_reachable" as const,
        passed: true,
        message: "TAO endpoint reachable.",
        model_config_id: null,
        provisioning_required: false,
        remediation: null,
      },
      {
        check_name: "verified_train_examples" as const,
        passed: true,
        message: "72 Verified training examples available (Test Pool excluded).",
        model_config_id: null,
        provisioning_required: false,
        remediation: null,
      },
      {
        check_name: "min_test_pool_size" as const,
        passed: true,
        message: "Test Pool has 48 held-out evaluation examples (need 48).",
        model_config_id: null,
        provisioning_required: false,
        remediation: null,
      },
    ],
    data_summary: {
      verified_training_count: 72,
      test_pool_count: 48,
      required_test_pool_count: 48,
      auto_labeled_eligible_count: 341,
      auto_labeled_included_count: includeAutoLabeled ? 341 : 0,
      excluded_test_pool_count: 48,
      excluded_auto_labeled_count: includeAutoLabeled ? 0 : 341,
      usable_training_count: 72 + (includeAutoLabeled ? 341 : 0),
    },
    resolved_presets: RESOLVED_PRESETS.resolved_presets,
  };
}

let queryClient: QueryClient;

beforeEach(() => {
  vi.clearAllMocks();
  queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  mockSetup.mockReturnValue({
    projectId: "pid-1",
    project: PROJECT,
    environment: {},
  });
  mockListBases.mockResolvedValue(BASES);
  mockResolvePresets.mockResolvedValue(RESOLVED_PRESETS);
  mockRunPreflight.mockImplementation(
    (
      _projectId: string,
      _ids: string[],
      includeAutoLabeled: boolean,
      _enableLora: boolean,
      _quantizationSchemes: QuantizationScheme[],
    ) => Promise.resolve(preflight(includeAutoLabeled)),
  );
  mockCreateSuite.mockResolvedValue({
    training_suite_id: "ts-1",
    project_id: "pid-1",
    idempotency_key: "idem",
    guidance_id: "g-1",
    training_preset: "quick",
    export_field_mode: "all",
    include_auto_labeled: true,
    quantization_schemes: ["FP8_DYNAMIC"],
    training_dataset_export_id: "de-train",
    evaluation_dataset_export_id: "de-eval",
    selected_student_base_model_config_ids: ["mc-2b"],
    chain_ids_ordered: ["chain-2b"],
    chains: [],
    provisioning_run_id: null,
    provisioning_model_names: [],
    setup_error_ref: null,
    status: "initialized",
    created_at: "2026-07-29T00:00:00Z",
    started_at: null,
    completed_at: null,
  });
});

function renderPage() {
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={["/projects/pid-1/training"]}>
        <Routes>
          <Route path="/projects/:projectId" element={<Outlet />}>
            <Route path="training" element={<StudentTrainingPage />} />
            <Route
              path="training/:trainingSuiteId"
              element={<div data-testid="monitor-page" />}
            />
            <Route path="scale-up" element={<div data-testid="scale-up-page" />} />
          </Route>
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

describe("StudentTrainingPage", () => {
  it("defaults to one 2B base, Quick preset, and baseline + FP8", async () => {
    renderPage();
    await screen.findByText("Ready to create the training jobs");

    expect(
      (screen.getByTestId("base-model-checkbox-mc-8b") as HTMLInputElement).checked,
    ).toBe(false);
    expect(
      (screen.getByTestId("base-model-checkbox-mc-2b") as HTMLInputElement).checked,
    ).toBe(true);
    expect(
      (screen.getByTestId("training-preset-select") as HTMLSelectElement).value,
    ).toBe("quick");
    expect(screen.getByRole("combobox", { name: "Training intensity" })).toBe(
      screen.getByTestId("training-preset-select"),
    );
    expect(
      (screen.getByTestId("quant-checkbox-FP8_DYNAMIC") as HTMLInputElement).checked,
    ).toBe(true);
    expect(
      (screen.getByTestId("quant-checkbox-W4A16") as HTMLInputElement).checked,
    ).toBe(false);
    expect(screen.getByTestId("training-start")).toHaveTextContent("Review 4 jobs");
    expect(screen.getByTestId("training-method-lora")).toHaveTextContent("LoRA");
    expect(screen.queryByTestId("training-method-full-weight")).not.toBeInTheDocument();
  });

  it("makes the broad multi-model comparison an explicit advanced intent", async () => {
    renderPage();
    await screen.findByText("Ready to create the training jobs");
    fireEvent.click(screen.getByTestId("training-intent-compare"));

    await waitFor(() => {
      expect(
        (screen.getByTestId("base-model-checkbox-mc-8b") as HTMLInputElement).checked,
      ).toBe(true);
      expect(
        (screen.getByTestId("base-model-checkbox-mc-2b") as HTMLInputElement).checked,
      ).toBe(true);
    });
    expect(
      (screen.getByTestId("training-preset-select") as HTMLSelectElement).value,
    ).toBe("standard");
    expect(
      (screen.getByTestId("quant-checkbox-W4A16") as HTMLInputElement).checked,
    ).toBe(true);
    expect(screen.getByTestId("training-start")).toHaveTextContent("Review 12 jobs");
  });

  it("confirms selected bases in the same order sent to the backend", async () => {
    renderPage();
    await screen.findByText("Ready to create the training jobs");
    fireEvent.click(screen.getByTestId("training-intent-compare"));

    for (const modelId of ["mc-8b", "mc-2b"]) {
      fireEvent.click(screen.getByTestId(`base-model-checkbox-${modelId}`));
    }
    fireEvent.click(screen.getByTestId("base-model-checkbox-mc-2b"));
    fireEvent.click(screen.getByTestId("base-model-checkbox-mc-8b"));

    await waitFor(() => expect(screen.getByTestId("training-start")).toBeEnabled());
    fireEvent.click(screen.getByTestId("training-start"));
    expect(
      await screen.findByTestId("training-confirmation-summary"),
    ).toHaveTextContent("Cosmos Reason2 2B, Cosmos Reason2 8B");

    fireEvent.click(screen.getByTestId("training-confirm-start"));
    await waitFor(() =>
      expect(mockCreateSuite).toHaveBeenCalledWith(
        "pid-1",
        expect.objectContaining({
          student_base_model_config_ids: ["mc-2b", "mc-8b"],
        }),
      ),
    );
  });

  it("renders backend-authoritative train, Test Pool, exclusion, and usable counts", async () => {
    renderPage();
    await screen.findByText("Ready to create the training jobs");

    expect(screen.getByTestId("training-data-verified-count")).toHaveTextContent("72");
    expect(screen.getByTestId("training-data-test-pool-count")).toHaveTextContent("48");
    expect(screen.getByTestId("training-data-auto-labeled-count")).toHaveTextContent(
      "341",
    );
    expect(screen.getByTestId("training-data-excluded-count")).toHaveTextContent("48");
    expect(screen.getByTestId("training-data-total")).toHaveTextContent("413");

    fireEvent.click(screen.getByTestId("training-data-auto-labeled-checkbox"));
    await waitFor(() =>
      expect(screen.getByTestId("training-data-total")).toHaveTextContent("72"),
    );
    expect(screen.getByTestId("training-data-excluded-count")).toHaveTextContent("389");
  });

  it("fails closed when every Verified example is held out in the Test Pool", async () => {
    mockRunPreflight.mockResolvedValueOnce({
      status: "failed",
      checks: [
        {
          check_name: "verified_train_examples",
          passed: false,
          message: "No Verified training examples yet. Continue labeling.",
          model_config_id: null,
          provisioning_required: false,
          remediation: null,
        },
      ],
      data_summary: {
        verified_training_count: 0,
        test_pool_count: 12,
        required_test_pool_count: 12,
        auto_labeled_eligible_count: 0,
        auto_labeled_included_count: 0,
        excluded_test_pool_count: 12,
        excluded_auto_labeled_count: 0,
        usable_training_count: 0,
      },
      resolved_presets: RESOLVED_PRESETS.resolved_presets,
    });

    renderPage();
    await screen.findByText("Training setup is incomplete");
    expect(screen.getByTestId("training-data-verified-count")).toHaveTextContent("0");
    expect(screen.getByTestId("training-data-test-pool-count")).toHaveTextContent("12");
    expect(screen.getByTestId("training-start")).toBeDisabled();
  });

  it("treats an undersized Test Pool as a data blocker, not TAO setup", async () => {
    mockRunPreflight.mockResolvedValueOnce({
      ...preflight(),
      status: "failed",
      checks: [
        preflight().checks[0],
        preflight().checks[1],
        {
          check_name: "min_test_pool_size",
          passed: false,
          message:
            "Test Pool has 12 of 60 required held-out evaluation examples. Continue labeling to grow the pool.",
          model_config_id: null,
          provisioning_required: false,
          remediation: null,
        },
      ],
      data_summary: {
        ...preflight().data_summary,
        test_pool_count: 12,
        required_test_pool_count: 60,
        excluded_test_pool_count: 12,
      },
    });

    renderPage();
    expect(await screen.findByTestId("test-pool-minimum-warning")).toHaveTextContent(
      "12 of 60 required",
    );
    expect(screen.getByTestId("training-start")).toBeDisabled();
    expect(screen.queryByTestId("training-request-tao-setup")).not.toBeInTheDocument();
  });

  it("surfaces TAO setup as a structured Action Request", async () => {
    mockRunPreflight.mockResolvedValueOnce({
      ...preflight(),
      status: "failed",
      checks: [
        {
          check_name: "tao_reachable",
          passed: false,
          message: "TAO endpoint is not configured.",
          model_config_id: null,
          provisioning_required: false,
          remediation: "Ask infrastructure to configure TAO.",
        },
        preflight().checks[1],
      ],
    });

    renderPage();
    await screen.findByText("Training setup is incomplete");
    fireEvent.click(screen.getByTestId("training-request-tao-setup"));
    expect(await screen.findByText(/TAO Setup Request/)).toBeInTheDocument();
    expect(screen.getByTestId("training-start")).toBeDisabled();
  });

  it("blocks Go when a gated Student base needs provisioning without HF_TOKEN", async () => {
    mockRunPreflight.mockResolvedValueOnce({
      ...preflight(),
      status: "failed",
      checks: [
        {
          check_name: "tao_base_experiment_ready",
          passed: true,
          message:
            "Base experiment is not registered. Start Training will provision it automatically.",
          model_config_id: "mc-2b",
          provisioning_required: true,
          remediation: null,
        },
        {
          check_name: "hf_token_configured",
          passed: false,
          message:
            "HF_TOKEN is required to provision the selected gated Cosmos Student base.",
          model_config_id: null,
          provisioning_required: false,
          remediation:
            "Set HF_TOKEN in ~/.vlm_feedback_loop/.env, restart the Blueprint backend, then rerun readiness.",
        },
        preflight().checks[1],
      ],
    });

    renderPage();
    await screen.findByText(
      "HF_TOKEN is required to provision the selected gated Cosmos Student base.",
    );
    expect(screen.getByText("Training setup is incomplete")).toBeInTheDocument();
    expect(screen.getByTestId("training-start")).toBeDisabled();
  });

  it("requires confirmation and submits the exact four-job validation suite", async () => {
    renderPage();
    await screen.findByText("Ready to create the training jobs");

    fireEvent.click(screen.getByTestId("training-start"));
    expect(mockCreateSuite).not.toHaveBeenCalled();
    expect(await screen.findByText("Start 4 TAO jobs?")).toBeInTheDocument();
    expect(screen.getByTestId("training-confirmation-summary")).toHaveTextContent(
      "Full precision + FP8 Dynamic",
    );
    expect(screen.getByTestId("training-confirmation-summary")).toHaveTextContent(
      "413 usable · 48 held out",
    );
    expect(screen.getByTestId("training-confirmation-summary")).toHaveTextContent(
      "1 train · 1 baseline evaluate · 1 quantize · 1 quantized evaluate",
    );
    expect(screen.getByTestId("training-confirmation-summary")).toHaveTextContent(
      "may incur compute, storage, and egress charges",
    );

    fireEvent.click(screen.getByTestId("training-confirm-start"));
    await waitFor(() =>
      expect(mockCreateSuite).toHaveBeenCalledWith(
        "pid-1",
        expect.objectContaining({
          student_base_model_config_ids: ["mc-2b"],
          training_preset: "quick",
          include_auto_labeled: true,
          enable_lora: true,
          quantization_schemes: ["FP8_DYNAMIC"],
        }),
      ),
    );
    await screen.findByTestId("monitor-page");
  });

  it("keeps the confirmation visibly locked while suite creation is pending", async () => {
    let resolveSuite:
      | ((value: Awaited<ReturnType<typeof createTrainingSuite>>) => void)
      | undefined;
    mockCreateSuite.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          resolveSuite = resolve;
        }),
    );

    renderPage();
    await screen.findByText("Ready to create the training jobs");
    fireEvent.click(screen.getByTestId("training-start"));
    fireEvent.click(await screen.findByTestId("training-confirm-start"));

    expect(screen.getByText("Start 4 TAO jobs?")).toBeInTheDocument();
    await waitFor(() =>
      expect(screen.getByTestId("training-confirm-start")).toHaveTextContent(
        "Starting…",
      ),
    );
    expect(screen.getByTestId("training-confirm-start")).toBeDisabled();
    expect(screen.getByTestId("training-confirm-cancel")).toBeDisabled();

    resolveSuite?.({
      training_suite_id: "ts-pending",
      project_id: "pid-1",
      idempotency_key: "idem-pending",
      guidance_id: "g-1",
      training_preset: "quick",
      export_field_mode: "all",
      include_auto_labeled: true,
      quantization_schemes: ["FP8_DYNAMIC"],
      training_dataset_export_id: "de-train",
      evaluation_dataset_export_id: "de-eval",
      selected_student_base_model_config_ids: ["mc-2b"],
      chain_ids_ordered: ["chain-2b"],
      chains: [],
      provisioning_run_id: null,
      provisioning_model_names: [],
      setup_error_ref: null,
      status: "initialized",
      created_at: "2026-07-29T00:00:00Z",
      started_at: null,
      completed_at: null,
    });
    await screen.findByTestId("monitor-page");
  });

  it("keeps the Training UI on the qualified LoRA path", async () => {
    renderPage();
    await screen.findByText("Ready to create the training jobs");

    expect(screen.getByTestId("training-method-lora")).toHaveTextContent("LoRA");
    expect(screen.queryByText("Full-weight")).not.toBeInTheDocument();
    fireEvent.click(screen.getByTestId("training-start"));
    expect(
      await screen.findByTestId("training-confirmation-summary"),
    ).toHaveTextContent("LoRA");
    fireEvent.click(screen.getByTestId("training-confirm-start"));
    await waitFor(() =>
      expect(mockCreateSuite).toHaveBeenCalledWith(
        "pid-1",
        expect.objectContaining({ enable_lora: true }),
      ),
    );
  });

  it("shows a data-sufficiency warning without claiming a tiny run is production-ready", async () => {
    renderPage();
    const warning = await screen.findByTestId("small-training-data-warning");
    expect(warning).toHaveTextContent(/fewer than 150 Verified images/i);
    expect(warning).toHaveTextContent("not evidence of a production-quality model");
  });

  it("does not expose Cosmos 3 Super returned by the catalog API", async () => {
    mockListBases.mockResolvedValueOnce({
      items: [
        ...BASES.items,
        {
          ...BASES.items[0],
          model_config_id: "mc-super",
          model_name: "nvidia/cosmos3-super-reasoner",
          tao_base_experiment_id: "base-super",
        },
      ],
      next_cursor: null,
    });

    renderPage();
    await screen.findByText("Ready to create the training jobs");
    expect(
      screen.queryByTestId("base-model-checkbox-mc-super"),
    ).not.toBeInTheDocument();
    expect(screen.queryByText("Cosmos 3 Super (Reasoner)")).not.toBeInTheDocument();

    fireEvent.click(screen.getByTestId("training-intent-compare"));
    await waitFor(() =>
      expect(mockRunPreflight).toHaveBeenLastCalledWith(
        "pid-1",
        ["mc-8b", "mc-2b"],
        true,
        true,
        ["FP8_DYNAMIC", "W4A16"],
      ),
    );
  });

  it("turns a backend conflict into plain-language copy without raw status JSON", async () => {
    mockCreateSuite.mockRejectedValueOnce(
      new ApiError(
        409,
        JSON.stringify({
          detail: "validation: TAO workspace is not bootstrapped. Request setup.",
        }),
      ),
    );
    renderPage();
    await screen.findByText("Ready to create the training jobs");
    fireEvent.click(screen.getByTestId("training-start"));
    fireEvent.click(await screen.findByTestId("training-confirm-start"));

    const error = await screen.findByTestId("training-submit-error");
    expect(error).toHaveTextContent("TAO workspace is not bootstrapped");
    expect(error).not.toHaveTextContent("409");
    expect(error).not.toHaveTextContent('{"detail"');
    expect(screen.queryByTestId("monitor-page")).toBeNull();
  });
});
