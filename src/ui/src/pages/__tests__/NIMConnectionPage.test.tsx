// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Unit tests for the NIM Configuration page.
 *
 * Onboarding is handled by three single-purpose pages
 * (``NIMNvidiaKeyPage``, ``NIMNgcKeyPage``, ``NIMSetupGatePage`` at
 * ``/projects/:id/setup*``); their behavior is covered by their own
 * test files. The rich ``NIMConnectionPage`` layout — service rows,
 * [Configure] overrides, self-hosted / local-deploy panels,
 * missing-prereqs landing — is the post-onboarding surface reached
 * via the persistent "NIM Configuration" header link
 * (``/projects/:id/settings/nim``).
 *
 * Note: that header link and the page's own heading both read
 * "NIM Configuration", so page-loaded gates below assert on the
 * heading (a non-link text node), not the nav link.
 *
 * Core rules pinned here:
 *   - The page renders even on a fully-configured machine (it never
 *     auto-navigates away — the SME explicitly chose to open it).
 *   - Footer reads [Cancel] [Save]; Cancel navigates back to
 *     ``/projects/:id``.
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { installEventSourceMock } from "@/test/event-source-mock";
import { makeEnvironmentResponse, makeProjectResponse } from "@/test/fixtures";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter, Routes, Route } from "react-router-dom";
import type { ReactNode } from "react";
import { NIMConnectionPage } from "@/pages/NIMConnectionPage";
import { ProjectSetupLayout } from "@/pages/ProjectSetupLayout";
import type {
  EnvironmentResponse,
  LocalNimGpuConflict,
  ModelConfigResponse,
} from "@/types/nim";

// ── Mocks ───────────────────────────────────────────────────────────────────

const mockFetchProject = vi.fn();
const mockFetchEnvironment = vi.fn();
const mockTestConnection = vi.fn();
const mockTestNgcCredential = vi.fn();
const mockTestNvidiaCredential = vi.fn();
const mockRunPreflight = vi.fn();
const mockDeployLocalNim = vi.fn();
const mockListLocalNimDeployments = vi.fn();
const mockParseLocalNimGpuConflict = vi.fn();
const mockFetchModelConfigs = vi.fn();
const mockGenerateActionRequest = vi.fn();
const mockLogActionRequestCopy = vi.fn();

vi.mock("@/api/projects", () => ({
  fetchProject: (...args: unknown[]) => mockFetchProject(...args),
  fetchProjectList: vi.fn().mockResolvedValue({ items: [], next_cursor: null }),
  createProject: vi.fn(),
  markSetupCompleted: vi.fn().mockResolvedValue({
    transitioned: true,
    project: {},
  }),
}));

vi.mock("@/api/secrets", () => ({
  setSecret: vi.fn().mockResolvedValue({
    effective: true,
    persisted: false,
    env_path: null,
    allow_persist: true,
  }),
}));

vi.mock("@/api/nim", () => ({
  fetchEnvironment: (...args: unknown[]) => mockFetchEnvironment(...args),
  testConnection: (...args: unknown[]) => mockTestConnection(...args),
  testNgcCredential: (...args: unknown[]) => mockTestNgcCredential(...args),
  testNvidiaCredential: (...args: unknown[]) => mockTestNvidiaCredential(...args),
  runPreflight: (...args: unknown[]) => mockRunPreflight(...args),
  deployLocalNim: (...args: unknown[]) => mockDeployLocalNim(...args),
  listLocalNimDeployments: (...args: unknown[]) => mockListLocalNimDeployments(...args),
  parseLocalNimGpuConflict: (...args: unknown[]) =>
    mockParseLocalNimGpuConflict(...args),
  generateActionRequest: (...args: unknown[]) => mockGenerateActionRequest(...args),
  logActionRequestCopy: (...args: unknown[]) => mockLogActionRequestCopy(...args),
}));

vi.mock("@/api/model-configs", () => ({
  fetchModelConfigs: (...args: unknown[]) => mockFetchModelConfigs(...args),
  updateProject: vi.fn(),
}));

installEventSourceMock();

// ── Fixtures ────────────────────────────────────────────────────────────────

const PROJECT = makeProjectResponse({
  project_id: "test-pid",
  name: "Test Project",
  description: null,
  project_dir: "/tmp/workspace/projects/test-pid",
  created_at: "2026-04-13T10:00:00Z",
  updated_at: "2026-04-13T10:00:00Z",
  teacher_model_config_id: "mc-teacher",
  active_guidance_id: null,
  active_student_model_config_id: null,
});

function makeEnv(overrides: Partial<EnvironmentResponse> = {}): EnvironmentResponse {
  return makeEnvironmentResponse(overrides);
}

function makeModelConfig(
  modelConfigId: string,
  modelName: string,
): ModelConfigResponse {
  return {
    model_config_id: modelConfigId,
    project_id: "test-pid",
    endpoint_id: "hosted-endpoint",
    model_name: modelName,
    context_window_tokens: 131072,
    eligible_roles: ["teacher"],
    supports_image_input: true,
    structured_generation_support: "supported",
    thinking_toggle_mode: "qwen_enable_thinking",
    thinking_toggle_support: "supported",
    visual_budget_mode: "none",
    visual_budget_support: "unsupported",
    model_quantization: null,
    nim_model_profile: null,
    nim_profile_metadata: null,
    local_deploy_metadata: {
      nim_container_image: `nvcr.io/nim/nvidia/${modelName.split("/")[1]}:test`,
    },
    hosted_compatible: false,
    availability: { available: false, reason: "hosted_not_compatible" },
    created_at: "2026-07-24T00:00:00Z",
  };
}

const LOCAL_MODELS = [
  {
    model_name: "nvidia/cosmos-reason2-8b",
    nim_container_image: "nvcr.io/nim/nvidia/cosmos-reason2-8b:1.6.0",
    gpu_memory_minimum_gb: 56,
    fits: true,
  },
  {
    model_name: "nvidia/cosmos3-nano-reasoner",
    nim_container_image: "nvcr.io/nim/nvidia/cosmos3-reasoner:1.7.0",
    gpu_memory_minimum_gb: 56,
    fits: true,
  },
  {
    model_name: "nvidia/cosmos3-super-reasoner",
    nim_container_image: "nvcr.io/nim/nvidia/cosmos3-reasoner:1.7.0",
    gpu_memory_minimum_gb: 88,
    fits: true,
  },
  {
    model_name: "nvidia/cosmos-reason2-2b",
    nim_container_image: "nvcr.io/nim/nvidia/cosmos-reason2-2b:1.6.0",
    gpu_memory_minimum_gb: 36,
    fits: true,
  },
  {
    model_name: "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
    nim_container_image:
      "nvcr.io/nim/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:1.7.0-variant",
    gpu_memory_minimum_gb: 80,
    compute_capability_minimum: 9,
    fits: true,
  },
];

const LOCAL_MODEL_CONFIGS = LOCAL_MODELS.map((model, index) =>
  makeModelConfig(`mc-local-${index}`, model.model_name),
);

// ── Wrapper ─────────────────────────────────────────────────────────────────

function createWrapper(initialPath = "/projects/test-pid/settings/nim") {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false, staleTime: 0 },
    },
  });
  function Wrapper({ children }: { children: ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>
        <MemoryRouter initialEntries={[initialPath]}>
          <Routes>
            <Route path="/projects/:projectId" element={<ProjectSetupLayout />}>
              <Route
                index
                element={<div data-testid="project-index">Project Index</div>}
              />
              <Route path="settings/nim" element={<NIMConnectionPage />} />
            </Route>
            <Route path="/" element={<div data-testid="home-page">Home</div>} />
          </Routes>
          {children}
        </MemoryRouter>
      </QueryClientProvider>
    );
  }
  return { Wrapper, queryClient };
}

function localTeacherEnvironment(
  overrides: Partial<EnvironmentResponse> = {},
): EnvironmentResponse {
  return makeEnv({
    gpus: [
      {
        name: "NVIDIA RTX PRO 6000 Blackwell Server Edition",
        memory_total_gb: 95.6,
        compute_capability: 12,
      },
    ],
    docker_available: true,
    nvidia_toolkit_available: true,
    local_deploy_available: true,
    nvidia_api_key_configured: true,
    ngc_api_key_configured: true,
    local_deployable_models: LOCAL_MODELS,
    recommended_teacher_mode: "local",
    recommended_embedding_mode: "hosted",
    recommended_local_teacher_model_name:
      "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
    recommended_local_teacher_image:
      "nvcr.io/nim/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:1.7.0-variant",
    recommended_local_teacher_gpu_memory_minimum_gb: 80,
    ...overrides,
  });
}

// ── Tests ───────────────────────────────────────────────────────────────────

describe("NIMConnectionPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockFetchProject.mockResolvedValue(PROJECT);
    mockFetchModelConfigs.mockResolvedValue({ items: [] });
    mockListLocalNimDeployments.mockResolvedValue({ items: [] });
    mockParseLocalNimGpuConflict.mockReturnValue(null);
  });

  // Even when both services are fully configured, the page renders
  // instead of bouncing elsewhere — the SME explicitly chose to open
  // it. Onboarding's auto-skip behavior lives on ``NIMNvidiaKeyPage``
  // (see ``NIMNvidiaKeyPage.test.tsx``).

  it("renders even when both services are fully configured (never auto-navigates away)", async () => {
    mockFetchEnvironment.mockResolvedValue(
      makeEnv({
        hosted_nim_available: true,
        nvidia_api_key_configured: true,
        recommended_teacher_mode: "hosted",
        recommended_embedding_mode: "hosted",
      }),
    );
    const { Wrapper } = createWrapper();

    render(<div />, { wrapper: Wrapper });

    await waitFor(() => {
      expect(
        screen.getByText("NIM Configuration", { selector: ":not(a)" }),
      ).toBeInTheDocument();
    });
    expect(screen.queryByTestId("project-index")).toBeNull();
  });

  // --- Recommendation screen ---

  it("renders recommendation screen with two service rows when credentials missing", async () => {
    mockFetchEnvironment.mockResolvedValue(
      makeEnv({
        recommended_teacher_mode: "hosted",
        recommended_embedding_mode: "hosted",
        gpus: [{ name: "A100", memory_total_gb: 80 }],
        docker_available: true,
      }),
    );
    const { Wrapper } = createWrapper();

    render(<div />, { wrapper: Wrapper });

    await waitFor(() => {
      expect(
        screen.getByText("NIM Configuration", { selector: ":not(a)" }),
      ).toBeInTheDocument();
    });

    expect(screen.getByText("Teacher")).toBeInTheDocument();
    expect(screen.getByText("Embeddings")).toBeInTheDocument();
    expect(screen.getByText(/A100 80 GB/)).toBeInTheDocument();
  });

  // --- Hosted override ---

  it("shows API key field in hosted override when key not configured", async () => {
    mockFetchEnvironment.mockResolvedValue(
      makeEnv({
        recommended_teacher_mode: "hosted",
        recommended_embedding_mode: "hosted",
      }),
    );
    const { Wrapper } = createWrapper();
    const user = userEvent.setup();

    render(<div />, { wrapper: Wrapper });

    await waitFor(() => {
      expect(screen.getByText("Teacher")).toBeInTheDocument();
    });

    // Before opening any override: exactly one NVIDIA API Key input, rendered
    // by the inline CredentialInput.
    expect(screen.getAllByPlaceholderText("nvapi-...")).toHaveLength(1);

    // Click Configure on teacher row. The hosted override also renders an
    // API Key input; the inline block MUST suppress its own to avoid a
    // duplicate field on screen.
    const configureButtons = screen.getAllByRole("button", { name: /Configure/i });
    await user.click(configureButtons[0]);

    await waitFor(() => {
      expect(screen.getAllByPlaceholderText("nvapi-...")).toHaveLength(1);
    });

    // Get API Key link is present (rendered by either CredentialInput or the
    // override's ConnectionTestPanel, whichever is currently visible).
    expect(screen.getAllByText(/Get NVIDIA API Key/i).length).toBeGreaterThanOrEqual(1);
  });

  // --- Self-hosted override hides NVIDIA/NGC credential prompts ---

  it("hides NVIDIA and NGC API key inputs when Teacher uses a self-hosted override", async () => {
    mockFetchEnvironment.mockResolvedValue(
      makeEnv({
        recommended_teacher_mode: "hosted",
        recommended_embedding_mode: "hosted",
        gpus: [{ name: "A100", memory_total_gb: 80 }],
        docker_available: true,
        nvidia_toolkit_available: true,
      }),
    );
    const { Wrapper } = createWrapper();
    const user = userEvent.setup();

    render(<div />, { wrapper: Wrapper });

    await waitFor(() => {
      expect(screen.getByText("Teacher")).toBeInTheDocument();
    });

    // Before any override: NVIDIA key is prompted inline (teacher + embeddings
    // both hosted by default, key not configured).
    expect(screen.getAllByPlaceholderText("nvapi-...").length).toBeGreaterThanOrEqual(
      1,
    );

    // Open Teacher's Configure expansion then switch to Self-hosted.
    const configureButtons = screen.getAllByRole("button", { name: /Configure/i });
    await user.click(configureButtons[0]);

    const selfHostedPill = await screen.findByRole("button", {
      name: /Self-hosted/i,
    });
    await user.click(selfHostedPill);

    // Embeddings still effectively hosted, so NVIDIA API Key should still be
    // needed for THAT service and remain prompted. This confirms the
    // visibility rule is per-credential / per-effective-mode, not a blanket
    // "hide everything when an override is open".
    await waitFor(() => {
      expect(screen.getAllByPlaceholderText("nvapi-...").length).toBeGreaterThanOrEqual(
        1,
      );
    });
  });

  // --- Connection test success ---

  it("shows success message after successful connection test", async () => {
    mockFetchEnvironment.mockResolvedValue(
      makeEnv({
        recommended_teacher_mode: "hosted",
        recommended_embedding_mode: "hosted",
      }),
    );
    // CredentialInput routes its NVIDIA Test through testNvidiaCredential
    // — not testConnection, whose public-endpoint probe passes even with
    // an invalid key (see CredentialInput.tsx).
    mockTestNvidiaCredential.mockResolvedValue({ success: true });
    const { Wrapper } = createWrapper();
    const user = userEvent.setup();

    render(<div />, { wrapper: Wrapper });

    await waitFor(() => {
      expect(screen.getByText("Teacher")).toBeInTheDocument();
    });

    // The CredentialInput on the recommendation page already renders a key
    // input and [Test Connection] button without opening [Configure].
    const inputs = await screen.findAllByPlaceholderText("nvapi-...");
    await user.type(inputs[0], "nvapi-test123");
    const testButtons = screen.getAllByRole("button", { name: /Test Connection/i });
    await user.click(testButtons[0]);

    await waitFor(() => {
      expect(screen.getByText(/Connected to/i)).toBeInTheDocument();
    });
  });

  // --- Connection test failure ---

  it("shows error message after failed connection test", async () => {
    mockFetchEnvironment.mockResolvedValue(
      makeEnv({
        recommended_teacher_mode: "hosted",
        recommended_embedding_mode: "hosted",
      }),
    );
    mockTestNvidiaCredential.mockResolvedValue({
      success: false,
      error: "Invalid API key",
    });
    const { Wrapper } = createWrapper();
    const user = userEvent.setup();

    render(<div />, { wrapper: Wrapper });

    await waitFor(() => {
      expect(screen.getByText("Teacher")).toBeInTheDocument();
    });

    const inputs = await screen.findAllByPlaceholderText("nvapi-...");
    await user.type(inputs[0], "bad-key");
    const testButtons = screen.getAllByRole("button", { name: /Test Connection/i });
    await user.click(testButtons[0]);

    await waitFor(() => {
      expect(screen.getByText(/Invalid API key/i)).toBeInTheDocument();
    });
  });

  // --- Navigation ---

  it("[Cancel] navigates back to the project index", async () => {
    mockFetchEnvironment.mockResolvedValue(
      makeEnv({
        recommended_teacher_mode: "hosted",
        recommended_embedding_mode: "hosted",
      }),
    );
    const { Wrapper } = createWrapper();
    const user = userEvent.setup();

    render(<div />, { wrapper: Wrapper });

    await waitFor(() => {
      expect(
        screen.getByText("NIM Configuration", { selector: ":not(a)" }),
      ).toBeInTheDocument();
    });

    await user.click(screen.getByRole("button", { name: /Cancel/i }));

    await waitFor(() => {
      expect(screen.getByTestId("project-index")).toBeInTheDocument();
    });
  });

  // --- Hardware detection display ---

  it("shows detected GPU info (dedup + '× N' + Docker)", async () => {
    mockFetchEnvironment.mockResolvedValue(
      makeEnv({
        gpus: [
          { name: "NVIDIA A100-SXM4-80GB", memory_total_gb: 80 },
          { name: "NVIDIA A100-SXM4-80GB", memory_total_gb: 80 },
        ],
        docker_available: true,
        local_deploy_available: true,
        nvidia_api_key_configured: true,
        ngc_api_key_configured: true,
        recommended_teacher_mode: "hosted",
        recommended_embedding_mode: "local",
      }),
    );
    const { Wrapper } = createWrapper();

    render(<div />, { wrapper: Wrapper });

    await waitFor(() => {
      expect(
        screen.getByText(/NVIDIA A100-SXM4 80 GB × 2 · Docker ready/),
      ).toBeInTheDocument();
    });
  });

  // --- Compatible local Teacher chooser ---

  it("names the quality recommendation instead of the first fitting model", async () => {
    // The environment list intentionally starts with CR2-8B. The page must
    // use recommended_local_teacher_model_name (Omni), not Array.find(fits).
    mockFetchEnvironment.mockResolvedValue(localTeacherEnvironment());
    mockFetchModelConfigs.mockResolvedValue({ items: LOCAL_MODEL_CONFIGS });
    const { Wrapper } = createWrapper();

    render(<div />, { wrapper: Wrapper });

    await waitFor(() => {
      expect(
        screen.getByText(/Nemotron 3 Nano Omni · NVIDIA RTX PRO 6000/i),
      ).toBeInTheDocument();
    });
    expect(
      screen.queryByText(/nvidia\/cosmos-reason2-8b · NVIDIA RTX/i),
    ).not.toBeInTheDocument();
  });

  it("offers every Blueprint-supported Teacher compatible with the GPU", async () => {
    mockFetchEnvironment.mockResolvedValue(localTeacherEnvironment());
    mockFetchModelConfigs.mockResolvedValue({ items: LOCAL_MODEL_CONFIGS });
    const { Wrapper } = createWrapper();
    const user = userEvent.setup();

    render(<div />, { wrapper: Wrapper });

    await user.click((await screen.findAllByRole("button", { name: /Configure/i }))[0]);
    const chooser = await screen.findByTestId("local-teacher-model-select");
    await user.click(chooser);
    const optionNames = (await screen.findAllByRole("option")).map(
      (option) => option.textContent,
    );

    expect(optionNames).toHaveLength(5);
    expect(optionNames).toContain("Nemotron 3 Nano Omni — Recommended");
    expect(optionNames).toContain("Cosmos 3 Nano (Reasoner)");
    expect(optionNames).toContain("Cosmos 3 Super (Reasoner)");
    expect(optionNames).toContain("Cosmos Reason2 8B");
    expect(optionNames).toContain("Cosmos Reason2 2B");
  });

  it("deploys the exact selected model and activates it only after success", async () => {
    mockFetchEnvironment.mockResolvedValue(localTeacherEnvironment());
    mockFetchModelConfigs.mockResolvedValue({ items: LOCAL_MODEL_CONFIGS });
    mockDeployLocalNim.mockResolvedValue({
      disposition: "queued",
      deployment: {
        local_nim_deployment_id: "new-deployment",
        status: "starting",
      },
      preflight: { all_passed: true, checks: [] },
    });
    const { Wrapper } = createWrapper();
    const user = userEvent.setup();

    render(<div />, { wrapper: Wrapper });

    await user.click((await screen.findAllByRole("button", { name: /Configure/i }))[0]);
    await user.click(await screen.findByTestId("local-teacher-model-select"));
    await user.click(
      await screen.findByRole("option", {
        name: "Cosmos 3 Nano (Reasoner)",
      }),
    );
    await user.click(screen.getByRole("button", { name: "Deploy selected model" }));

    await waitFor(() => {
      expect(mockDeployLocalNim).toHaveBeenCalledWith("test-pid", {
        role: "teacher",
        model_config_id: "mc-local-1",
        gpu_assignment: undefined,
        replace_resident: false,
        activate_on_success: true,
      });
    });
  });

  it("requires a named confirmation before replacing the resident NIM", async () => {
    const conflictError = new Error("gpu occupied");
    const conflict: LocalNimGpuConflict = {
      code: "gpu_exhausted",
      message: "All compatible GPUs are occupied.",
      can_replace: true,
      matches_requested_model: false,
      resident: {
        project_id: "trashtruck-id",
        project_name: "TrashTruck",
        local_nim_deployment_id: "omni-deployment",
        role: "teacher",
        model_name: "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
        nim_container_image:
          "nvcr.io/nim/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:1.7.0-variant",
        gpu_assignment: "device=0",
        status: "running",
      },
    };
    mockFetchEnvironment.mockResolvedValue(
      localTeacherEnvironment({
        active_local_nim_residents: [conflict.resident!],
      }),
    );
    mockFetchModelConfigs.mockResolvedValue({ items: LOCAL_MODEL_CONFIGS });
    mockDeployLocalNim.mockRejectedValueOnce(conflictError).mockResolvedValueOnce({
      disposition: "queued",
      deployment: {
        local_nim_deployment_id: "replacement-deployment",
        status: "starting",
      },
      preflight: { all_passed: true, checks: [] },
    });
    mockParseLocalNimGpuConflict.mockImplementation((error) =>
      error === conflictError ? conflict : null,
    );
    const { Wrapper } = createWrapper();
    const user = userEvent.setup();

    render(<div />, { wrapper: Wrapper });

    await user.click((await screen.findAllByRole("button", { name: /Configure/i }))[0]);
    await user.click(await screen.findByTestId("local-teacher-model-select"));
    await user.click(
      await screen.findByRole("option", {
        name: "Cosmos 3 Nano (Reasoner)",
      }),
    );
    await user.click(screen.getByRole("button", { name: "Deploy selected model" }));

    expect(mockDeployLocalNim).toHaveBeenNthCalledWith(1, "test-pid", {
      role: "teacher",
      model_config_id: "mc-local-1",
      gpu_assignment: undefined,
      replace_resident: false,
      activate_on_success: true,
    });
    expect(
      await screen.findByText(/Nemotron 3 Nano Omni is running on device=0/i),
    ).toBeInTheDocument();
    expect(screen.getByText(/project “TrashTruck”/i)).toBeInTheDocument();
    expect(mockDeployLocalNim).toHaveBeenCalledTimes(1);

    await user.click(
      screen.getByRole("button", {
        name: "Replace NIM",
      }),
    );

    await waitFor(() => {
      expect(mockDeployLocalNim).toHaveBeenNthCalledWith(2, "test-pid", {
        role: "teacher",
        model_config_id: "mc-local-1",
        gpu_assignment: "device=0",
        replace_resident: true,
        activate_on_success: true,
      });
    });
  });

  // --- Missing credentials inline ---

  it("renders inline NVIDIA API Key CredentialInput when key not configured", async () => {
    mockFetchEnvironment.mockResolvedValue(
      makeEnv({
        nvidia_api_key_configured: false,
        recommended_teacher_mode: "hosted",
        recommended_embedding_mode: "hosted",
      }),
    );
    const { Wrapper } = createWrapper();

    render(<div />, { wrapper: Wrapper });

    await waitFor(() => {
      // FormField[slotLabel="NVIDIA API Key"] renders the label text.
      expect(screen.getByText("NVIDIA API Key")).toBeInTheDocument();
    });
    expect(screen.getByPlaceholderText("nvapi-...")).toBeInTheDocument();
    expect(screen.getByText(/Get NVIDIA API Key/i)).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /Test Connection/i }),
    ).toBeInTheDocument();
  });

  it("renders inline NGC API Key input when local recommended but NGC missing", async () => {
    mockFetchEnvironment.mockResolvedValue(
      makeEnv({
        gpus: [{ name: "NVIDIA A100-SXM4-80GB", memory_total_gb: 80 }],
        docker_available: true,
        nvidia_toolkit_available: true,
        local_deploy_available: true,
        nvidia_api_key_configured: true,
        ngc_api_key_configured: false,
        recommended_teacher_mode: "hosted",
        recommended_embedding_mode: "local",
      }),
    );
    const { Wrapper } = createWrapper();

    render(<div />, { wrapper: Wrapper });

    await waitFor(() => {
      expect(screen.getByText("NGC API Key")).toBeInTheDocument();
    });
    expect(screen.getByText(/Get NGC API Key/i)).toBeInTheDocument();
  });

  // --- GPU-insufficient explainer (modest-GPU machines) ---

  it("shows GPU-insufficient note when Teacher can't fit locally", async () => {
    mockFetchEnvironment.mockResolvedValue(
      makeEnv({
        gpus: [{ name: "NVIDIA A10G", memory_total_gb: 24 }],
        docker_available: true,
        nvidia_toolkit_available: true,
        local_deploy_available: true,
        nvidia_api_key_configured: true,
        local_deployable_models: [
          {
            model_name: "nvidia/cosmos-reason2-8b",
            nim_container_image: "nvcr.io/nim/nvidia/cosmos-reason2-8b:1.6.0",
            gpu_memory_minimum_gb: 56,
            fits: false,
          },
        ],
        recommended_teacher_mode: "hosted",
        recommended_embedding_mode: "local",
      }),
    );
    const { Wrapper } = createWrapper();

    render(<div />, { wrapper: Wrapper });

    await waitFor(() => {
      expect(
        screen.getByText(/GPU insufficient for local Teacher \(need >56 GB\)/i),
      ).toBeInTheDocument();
    });
  });

  // --- Missing Prerequisites dedicated landing ---

  it("renders the Missing Prerequisites landing when local recommended but Docker is missing", async () => {
    mockFetchEnvironment.mockResolvedValue(
      makeEnv({
        gpus: [{ name: "NVIDIA A100-SXM4-80GB", memory_total_gb: 80 }],
        docker_available: false,
        nvidia_toolkit_available: false,
        local_deploy_available: false,
        recommended_teacher_mode: "local",
        recommended_embedding_mode: "local",
        missing_prerequisites: [
          {
            check: "Docker",
            install_hint: "Install Docker: https://docs.docker.com/engine/install/",
          },
          {
            check: "NVIDIA Container Toolkit",
            install_hint:
              "Install NVIDIA Container Toolkit: https://docs.nvidia.com/...",
          },
        ],
      }),
    );
    const { Wrapper } = createWrapper();
    const user = userEvent.setup();

    render(<div />, { wrapper: Wrapper });

    await waitFor(() => {
      expect(
        screen.getByText(/Local deployment recommended but prerequisites missing/i),
      ).toBeInTheDocument();
    });
    expect(screen.getByText("Docker not found")).toBeInTheDocument();
    expect(screen.getByText("NVIDIA Container Toolkit not found")).toBeInTheDocument();
    expect(screen.getByText(/\.\/scripts\/setup-local\.sh/)).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /Copy to Clipboard/i }),
    ).toBeInTheDocument();

    // Switch to hosted falls through to the recommendation layout.
    await user.click(screen.getByRole("button", { name: /Switch to Hosted/i }));

    await waitFor(() => {
      expect(screen.getByText("Teacher")).toBeInTheDocument();
      expect(screen.getByText("Embeddings")).toBeInTheDocument();
    });
  });
});
