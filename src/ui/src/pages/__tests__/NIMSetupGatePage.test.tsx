// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Tests for NIMSetupGatePage — the end-of-setup summary/gate screen.
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { installEventSourceMock } from "@/test/event-source-mock";
import { makeEnvironmentResponse, makeProjectResponse } from "@/test/fixtures";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import type { ReactNode } from "react";

import { ApiError } from "@/api/client";
import { projectKeys } from "@/api/query-keys";
import { NIMSetupGatePage } from "@/pages/NIMSetupGatePage";
import { ProjectSetupLayout } from "@/pages/ProjectSetupLayout";
import type { EnvironmentResponse } from "@/types/nim";
import type { SetupChainState } from "@/types/setupChain";

// ── Mocks ───────────────────────────────────────────────────────────────────

const mockFetchProject = vi.fn();
const mockFetchEnvironment = vi.fn();
const mockMarkSetupCompleted = vi.fn();
const mockDeployLocalNim = vi.fn();
const mockFetchModelConfigs = vi.fn();
const mockListLocalNimDeployments = vi.fn();

vi.mock("@/api/projects", () => ({
  fetchProject: (...args: unknown[]) => mockFetchProject(...args),
  fetchProjectList: vi.fn().mockResolvedValue({ items: [], next_cursor: null }),
  createProject: vi.fn(),
  archiveProject: vi.fn(),
  markSetupCompleted: (...args: unknown[]) => mockMarkSetupCompleted(...args),
}));

vi.mock("@/api/nim", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/api/nim")>();
  return {
    ...actual,
    fetchEnvironment: (...args: unknown[]) => mockFetchEnvironment(...args),
    deployLocalNim: (...args: unknown[]) => mockDeployLocalNim(...args),
    listLocalNimDeployments: (...args: unknown[]) =>
      mockListLocalNimDeployments(...args),
  };
});

const mockUpdateProject = vi.fn();
vi.mock("@/api/model-configs", () => ({
  fetchModelConfigs: (...args: unknown[]) => mockFetchModelConfigs(...args),
  updateProject: (...args: unknown[]) => mockUpdateProject(...args),
}));

installEventSourceMock();

// ── Fixtures ────────────────────────────────────────────────────────────────

const PROJECT = makeProjectResponse({
  active_guidance_id: null,
  setup_completed_at: null,
});

function makeEnv(overrides: Partial<EnvironmentResponse> = {}): EnvironmentResponse {
  return makeEnvironmentResponse({
    hosted_nim_available: true,
    local_deploy_available: true,
    docker_available: true,
    nvidia_toolkit_available: true,
    nvidia_api_key_configured: true,
    gpus: [{ name: "NVIDIA A100-SXM4-80GB", memory_total_gb: 80 }],
    embedding_deployment: {
      model_name: "nvidia/llama-nemotron-embed-vl-1b-v2",
      nim_container_image: "nvcr.io/nim/nvidia/llama-nemotron-embed-vl-1b-v2:2.0.0",
      gpu_memory_minimum_gb: 24,
      fits: true,
      provider: "self_hosted_nvclip",
    },
    recommended_teacher_mode: "hosted",
    recommended_embedding_mode: "local",
    ...overrides,
  });
}

/** Points fetchModelConfigs at a catalog containing the Cosmos 8B entry. */
function mockCosmosConfig(eligibleRoles: string[] = ["teacher"]) {
  mockFetchModelConfigs.mockResolvedValue({
    items: [
      {
        model_config_id: "mc-cosmos-8b",
        model_name: "nvidia/cosmos-reason2-8b",
        eligible_roles: eligibleRoles,
      },
    ],
    next_cursor: null,
  });
}

/** LocalNimDeployment row for the local Cosmos Teacher. Defaults to the
 *  just-persisted "starting" state; override status/status_reason/
 *  deployed_at for the running / failed variants. */
function makeDeployment(overrides: Partial<Record<string, unknown>> = {}) {
  return {
    local_nim_deployment_id: "ld-1",
    project_id: "test-pid",
    role: "teacher",
    model_config_id: "mc-cosmos-8b",
    nim_container_image: "nvcr.io/nim/nvidia/cosmos-reason2-8b:1.6.0",
    container_name: "vlm-teacher-test-pid",
    host_port: 8001,
    endpoint_url: "http://localhost:8001/v1",
    gpu_assignment: "device=0",
    status: "starting",
    status_reason: null,
    deployed_at: null,
    stopped_at: null,
    created_at: "2026-05-19T20:05:44Z",
    displaced_by_deployment_id: null,
    displaced_at: null,
    ...overrides,
  };
}

// ── Stub child routes capturing nav state ───────────────────────────────────

function ConfirmStub(): JSX.Element {
  return <div data-testid="confirm-defaults-page" />;
}

function createWrapper(initialState?: Partial<SetupChainState>) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false, staleTime: 0 } },
  });
  const entry = {
    pathname: "/projects/test-pid/setup/done",
    state: initialState,
  };
  function Wrapper({ children }: { children: ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>
        <MemoryRouter initialEntries={[entry]}>
          <Routes>
            {/* Project-list stub at "/" — the app's real route table has
                no other list path, so back-to-projects affordances must
                land here. */}
            <Route path="/" element={<div data-testid="project-list-page" />} />
            <Route path="/projects/:projectId" element={<ProjectSetupLayout />}>
              <Route path="setup/done" element={<NIMSetupGatePage />} />
              <Route path="confirm-defaults" element={<ConfirmStub />} />
            </Route>
          </Routes>
          {children}
        </MemoryRouter>
      </QueryClientProvider>
    );
  }
  return { Wrapper, queryClient };
}

// ── Tests ───────────────────────────────────────────────────────────────────

describe("NIMSetupGatePage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockFetchProject.mockResolvedValue(PROJECT);
    mockMarkSetupCompleted.mockResolvedValue({
      transitioned: true,
      project: PROJECT,
    });
    mockDeployLocalNim.mockResolvedValue({
      deployment: { status: "starting" },
      preflight: { all_passed: true, checks: [] },
    });
    mockFetchModelConfigs.mockResolvedValue({
      items: [],
      next_cursor: null,
    });
    // Default: no deployments in the DB yet. Individual tests
    // override this with `{ items: [...] }` to simulate starting /
    // running / failed states.
    mockListLocalNimDeployments.mockResolvedValue({ items: [] });
    mockUpdateProject.mockResolvedValue(PROJECT);
  });

  it("auto-skips to confirm-defaults when cameFromAutoSkip=true (unattended chain)", async () => {
    mockFetchEnvironment.mockResolvedValue(makeEnv());
    const { Wrapper } = createWrapper({ cameFromAutoSkip: true });
    render(<div />, { wrapper: Wrapper });

    await screen.findByTestId("confirm-defaults-page");
    expect(mockMarkSetupCompleted).toHaveBeenCalledWith(
      "test-pid",
      expect.objectContaining({
        auto_skip: true,
        teacher_mode: "hosted",
        embedding_mode: "local",
        embedding_provider: "self_hosted_nvclip",
      }),
    );
  });

  // ── Setup-completion cache invalidation ──────────────────────────
  // The NIM Configuration header link gates on the cached project's
  // setup_completed_at. Both stamp paths MUST invalidate the project
  // detail query after mark_setup_completed resolves, or the link
  // stays hidden until an unrelated refetch.

  it("auto-skip invalidates the project detail cache after stamping setup completion", async () => {
    mockFetchEnvironment.mockResolvedValue(makeEnv());
    const { Wrapper, queryClient } = createWrapper({ cameFromAutoSkip: true });
    const invalidateSpy = vi.spyOn(queryClient, "invalidateQueries");
    render(<div />, { wrapper: Wrapper });

    await screen.findByTestId("confirm-defaults-page");
    await waitFor(() => {
      expect(invalidateSpy).toHaveBeenCalledWith({
        queryKey: projectKeys.detail("test-pid"),
      });
    });
  });

  it("manual [Start labeling] invalidates the project detail cache after the stamp lands", async () => {
    mockFetchEnvironment.mockResolvedValue(makeEnv());
    const { Wrapper, queryClient } = createWrapper({ cameFromAutoSkip: false });
    const invalidateSpy = vi.spyOn(queryClient, "invalidateQueries");
    const user = userEvent.setup();
    render(<div />, { wrapper: Wrapper });

    await user.click(await screen.findByRole("button", { name: /Start labeling/i }));

    await screen.findByTestId("confirm-defaults-page");
    await waitFor(() => {
      expect(invalidateSpy).toHaveBeenCalledWith({
        queryKey: projectKeys.detail("test-pid"),
      });
    });
  });

  it("renders the summary when cameFromAutoSkip is false (or absent)", async () => {
    mockFetchEnvironment.mockResolvedValue(makeEnv());
    const { Wrapper } = createWrapper({ cameFromAutoSkip: false });
    render(<div />, { wrapper: Wrapper });

    await waitFor(() => {
      expect(screen.getByText(/You're set up/i)).toBeInTheDocument();
    });
    expect(screen.getByTestId("setup-gate-card")).toHaveClass(
      "glass-card",
      "glass-card--elevated",
    );
    expect(screen.getByText(/Step 3\.7 Flash/)).toBeInTheDocument();
    expect(screen.getByText(/NeMo Retriever VL/)).toBeInTheDocument();
    expect(screen.queryByTestId("confirm-defaults-page")).toBeNull();
  });

  it("[Start labeling] calls mark_setup_completed(auto_skip:false) then navigates to confirm-defaults", async () => {
    mockFetchEnvironment.mockResolvedValue(makeEnv());
    const { Wrapper } = createWrapper({ cameFromAutoSkip: false });
    const user = userEvent.setup();
    render(<div />, { wrapper: Wrapper });

    await user.click(await screen.findByRole("button", { name: /Start labeling/i }));

    await screen.findByTestId("confirm-defaults-page");
    expect(mockMarkSetupCompleted).toHaveBeenCalledWith(
      "test-pid",
      expect.objectContaining({ auto_skip: false }),
    );
  });

  it("embedding row OMITTED when no embedding NIM will run (no NVIDIA key + single-GPU + Teacher queued)", async () => {
    // With the Teacher queued on a single GPU, dispatchLocalDeploys
    // silently skips the embedding deploy under the one-NIM-per-GPU
    // policy. With no NVIDIA key either, NO embedding NIM is going
    // to run — so the page must not claim "Local NIM · {GPU}" for
    // NeMo Retriever VL; the row is omitted entirely. Image variety
    // falls back to perceptual fingerprints on the backend; the SME
    // doesn't need to see that here.
    //
    // Env matches the placement-aware backend: the planned local
    // Teacher reserves the host's only GPU, so the embedding NIM has
    // no candidate device (fits=false → recommendation "hosted").
    mockFetchEnvironment.mockResolvedValue(
      makeEnv({
        nvidia_api_key_configured: false,
        gpus: [{ name: "NVIDIA A100", memory_total_gb: 80 }],
        embedding_deployment: {
          model_name: "nvidia/llama-nemotron-embed-vl-1b-v2",
          nim_container_image: "nvcr.io/nim/nvidia/llama-nemotron-embed-vl-1b-v2:2.0.0",
          gpu_memory_minimum_gb: 24,
          fits: false,
          provider: "none",
        },
        recommended_embedding_mode: "hosted",
        recommended_local_teacher_model_name: "nvidia/cosmos-reason2-8b",
      }),
    );
    const { Wrapper } = createWrapper({
      cameFromAutoSkip: false,
      activePath: "local",
      // Teacher is queued for local deploy → embedding deploy is
      // about to be skipped → page must NOT promise NeMo Retriever VL.
      localDeployQueued: ["nvidia/cosmos-reason2-8b"],
    });
    render(<div />, { wrapper: Wrapper });

    await waitFor(() => {
      expect(screen.getByText(/You're set up/i)).toBeInTheDocument();
    });
    // Teacher row still renders.
    expect(screen.getByText(/Cosmos Reason2 8B/)).toBeInTheDocument();
    // Embedding row is OMITTED — no model name, no eyebrow.
    expect(screen.queryByText(/NeMo Retriever VL/)).toBeNull();
    expect(screen.queryByText(/EMBEDDING MODEL/)).toBeNull();
  });

  it("embedding row shows hosted when NVIDIA key configured (even on single-GPU + Teacher queued)", async () => {
    // The hybrid path: single-GPU + local Teacher + NVIDIA key →
    // embedding routes hosted via the key. The local deploy is
    // skipped but the embedding still runs (just remotely), so the
    // row renders with "Hosted on build.nvidia.com".
    mockFetchEnvironment.mockResolvedValue(
      makeEnv({
        nvidia_api_key_configured: true,
        gpus: [{ name: "NVIDIA A100", memory_total_gb: 80 }],
        recommended_embedding_mode: "local",
        recommended_local_teacher_model_name: "nvidia/cosmos-reason2-8b",
      }),
    );
    const { Wrapper } = createWrapper({
      cameFromAutoSkip: false,
      activePath: "hybrid",
      localDeployQueued: ["nvidia/cosmos-reason2-8b"],
    });
    render(<div />, { wrapper: Wrapper });

    await waitFor(() => {
      expect(screen.getByText(/You're set up/i)).toBeInTheDocument();
    });
    // Embedding row renders with hosted meta. Other rows may also
    // reference build.nvidia.com (e.g. the hybrid's "Default · Hosted"
    // Teacher row), so scope by the row's eyebrow + name and check
    // at least one "Hosted on build.nvidia.com" appears.
    expect(screen.getByText(/NeMo Retriever VL/)).toBeInTheDocument();
    const hosted = screen.getAllByText(/Hosted on build\.nvidia\.com/);
    expect(hosted.length).toBeGreaterThan(0);
  });

  it("single-GPU local path + NVIDIA key: TEACHER row is the local NIM and EMBEDDING row is hosted (hybrid mode)", async () => {
    // Case A both-checked lands here: the SME chose the local Teacher
    // (activePath="local" — its row promised "as your Teacher") AND
    // pasted an NVIDIA key. One GPU runs the Teacher NIM; embeddings
    // route to build.nvidia.com. The Teacher row must NOT fall back to
    // the hosted default, and the embedding row must NOT claim a local NIM.
    //
    // The backend recommendation stays "local" here (with the key
    // configured it plans a hosted Teacher, leaving the GPU free) —
    // only THIS chain knows it queued a local Teacher on the single
    // GPU, so the page's own guard must produce the hosted row.
    mockFetchEnvironment.mockResolvedValue(
      makeEnv({
        nvidia_api_key_configured: true,
        gpus: [{ name: "NVIDIA A100", memory_total_gb: 80 }],
        recommended_embedding_mode: "local",
        recommended_local_teacher_model_name: "nvidia/cosmos-reason2-8b",
      }),
    );
    const { Wrapper } = createWrapper({
      cameFromAutoSkip: false,
      activePath: "local",
      localDeployQueued: ["nvidia/cosmos-reason2-8b"],
    });
    render(<div />, { wrapper: Wrapper });

    await waitFor(() => {
      expect(screen.getByText(/You're set up/i)).toBeInTheDocument();
    });
    expect(screen.getByText(/TEACHER MODEL/)).toBeInTheDocument();
    expect(screen.getByText(/Cosmos Reason2 8B/)).toBeInTheDocument();
    expect(screen.getByText(/Local NIM · NVIDIA A100/)).toBeInTheDocument();
    expect(screen.queryByText(/Step 3\.7 Flash/)).toBeNull();
    expect(screen.getByText(/NeMo Retriever VL/)).toBeInTheDocument();
    expect(screen.getByText(/Hosted on build\.nvidia\.com/)).toBeInTheDocument();
  });

  it("navigates to confirm-defaults even when mark_setup_completed throws", async () => {
    mockFetchEnvironment.mockResolvedValue(makeEnv());
    mockMarkSetupCompleted.mockRejectedValue(new Error("backend offline"));
    const { Wrapper } = createWrapper({ cameFromAutoSkip: false });
    const user = userEvent.setup();
    render(<div />, { wrapper: Wrapper });

    await user.click(await screen.findByRole("button", { name: /Start labeling/i }));

    await screen.findByTestId("confirm-defaults-page");
  });

  // ── Path-aware summary + dispatch ────────────────────────────────────────

  it("local path: summary names Cosmos as the Teacher with live 'deploying in background' meta", async () => {
    mockFetchEnvironment.mockResolvedValue(
      makeEnv({
        recommended_teacher_mode: "local",
        recommended_local_teacher_model_name: "nvidia/cosmos-reason2-8b",
        recommended_local_teacher_image: "nvcr.io/nim/nvidia/cosmos-reason2-8b:1.6.0",
        recommended_local_teacher_gpu_memory_minimum_gb: 56,
      }),
    );
    mockCosmosConfig(["teacher", "student_base"]);
    // Backend just persisted the row with status="starting" — page
    // should report "deploying in background".
    mockListLocalNimDeployments.mockResolvedValue({ items: [makeDeployment()] });
    const { Wrapper } = createWrapper({
      cameFromAutoSkip: false,
      activePath: "local",
      localDeployQueued: [
        "nvidia/cosmos-reason2-8b",
        "nvidia/llama-nemotron-embed-vl-1b-v2",
      ],
    });
    render(<div />, { wrapper: Wrapper });

    await waitFor(() => {
      expect(screen.getByText(/You're set up/i)).toBeInTheDocument();
    });
    expect(screen.getByText(/Cosmos Reason2 8B/)).toBeInTheDocument();
    // Status meta resolves after the local-NIM list query lands.
    await waitFor(() => {
      expect(screen.getByText(/deploying in background/i)).toBeInTheDocument();
    });
    // Subtitle changes to the background-deploy framing.
    expect(
      screen.getByText(/Local NIMs deploy in the background after you continue/i),
    ).toBeInTheDocument();
  });

  it("local path: deployment status='running' flips the Teacher meta to '(running)' (the user-reported bug fix)", async () => {
    // The meta polls LocalNimDeployment.status and shows "(running)"
    // once the backend's health-poll flips the row — otherwise the
    // /done page keeps saying "deploying in background" indefinitely
    // after the container is already healthy.
    mockFetchEnvironment.mockResolvedValue(
      makeEnv({
        recommended_teacher_mode: "local",
        recommended_local_teacher_model_name: "nvidia/cosmos-reason2-8b",
      }),
    );
    mockCosmosConfig();
    mockListLocalNimDeployments.mockResolvedValue({
      items: [
        makeDeployment({ status: "running", deployed_at: "2026-05-19T20:11:26Z" }),
      ],
    });
    const { Wrapper } = createWrapper({
      cameFromAutoSkip: false,
      activePath: "local",
      localDeployQueued: ["nvidia/cosmos-reason2-8b"],
    });
    render(<div />, { wrapper: Wrapper });

    await waitFor(() => {
      expect(screen.getByText(/You're set up/i)).toBeInTheDocument();
    });
    expect(screen.getByText(/Cosmos Reason2 8B/)).toBeInTheDocument();
    // Meta says "(running)" — NOT the stale "deploying in
    // background" string.
    await waitFor(() => {
      expect(screen.getByText(/\(running\)/i)).toBeInTheDocument();
    });
    expect(screen.queryByText(/deploying in background/i)).toBeNull();
  });

  it("matching host resident is named as reusable before Start labeling", async () => {
    mockFetchEnvironment.mockResolvedValue(
      makeEnv({
        recommended_teacher_mode: "local",
        recommended_local_teacher_model_name: "nvidia/cosmos-reason2-8b",
        active_local_nim_residents: [
          {
            project_id: "existing-pid",
            project_name: "Existing project",
            local_nim_deployment_id: "dep-existing",
            role: "teacher",
            model_name: "nvidia/cosmos-reason2-8b",
            nim_container_image: "nvcr.io/nim/nvidia/cosmos-reason2-8b:1.6.0",
            gpu_assignment: "device=0",
            status: "running",
          },
        ],
      }),
    );
    mockCosmosConfig();
    mockListLocalNimDeployments.mockResolvedValue({ items: [] });
    const { Wrapper } = createWrapper({
      cameFromAutoSkip: false,
      activePath: "local",
      localDeployQueued: ["nvidia/cosmos-reason2-8b"],
    });
    render(<div />, { wrapper: Wrapper });

    await screen.findByText(/already running for Existing project; will be reused/i);
    expect(
      screen.getByText(/no second container or model reload is needed/i),
    ).toBeInTheDocument();
  });

  it("local path: deployment status='failed' surfaces the error reason in the meta", async () => {
    mockFetchEnvironment.mockResolvedValue(
      makeEnv({
        recommended_teacher_mode: "local",
        recommended_local_teacher_model_name: "nvidia/cosmos-reason2-8b",
      }),
    );
    mockFetchModelConfigs.mockResolvedValue({ items: [], next_cursor: null });
    mockListLocalNimDeployments.mockResolvedValue({
      items: [
        makeDeployment({
          status: "failed",
          status_reason: "Health check timed out after 1200s",
        }),
      ],
    });
    const { Wrapper } = createWrapper({
      cameFromAutoSkip: false,
      activePath: "local",
      localDeployQueued: ["nvidia/cosmos-reason2-8b"],
    });
    render(<div />, { wrapper: Wrapper });

    await waitFor(() => {
      expect(screen.getByText(/You're set up/i)).toBeInTheDocument();
    });
    await waitFor(() => {
      expect(screen.getByText(/deploy failed/i)).toBeInTheDocument();
    });
    expect(screen.getByText(/Health check timed out/i)).toBeInTheDocument();
  });

  it("hybrid path: starts hosted and says the verified local Teacher will take over", async () => {
    mockFetchEnvironment.mockResolvedValue(
      makeEnv({
        recommended_local_teacher_model_name: "nvidia/cosmos-reason2-8b",
      }),
    );
    mockCosmosConfig();
    const { Wrapper } = createWrapper({
      cameFromAutoSkip: false,
      activePath: "hybrid",
      localDeployQueued: ["nvidia/cosmos-reason2-8b"],
    });
    render(<div />, { wrapper: Wrapper });

    const user = userEvent.setup();
    await screen.findByText(/You're set up/i);
    expect(
      screen.getByText(
        /Start labeling immediately with the hosted Teacher.*becomes active automatically once verified/i,
      ),
    ).toBeInTheDocument();
    expect(screen.getByText(/Step 3\.7 Flash/)).toBeInTheDocument();
    expect(
      screen.getByText(/Starting now · Hosted until the local Teacher is ready/i),
    ).toBeInTheDocument();
    expect(screen.getByText(/NEXT · LOCAL TEACHER/i)).toBeInTheDocument();
    expect(screen.getByText(/Cosmos Reason2 8B/)).toBeInTheDocument();
    expect(
      screen.getByText(/will become active automatically once verified/i),
    ).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /Start labeling/i }));
    await screen.findByTestId("confirm-defaults-page");
    expect(mockDeployLocalNim).toHaveBeenCalledWith("test-pid", {
      role: "teacher",
      model_config_id: "mc-cosmos-8b",
      replace_resident: false,
      activate_on_success: true,
    });
  });

  it("Start labeling dispatches :deploy per queued model + threads local_deploy_queued into mark_setup_completed", async () => {
    // Placement-aware env no longer queues the embedding NIM on a
    // single-GPU local path, so a both-queued single-GPU chain only
    // arrives via deep-link or a stale assessment — the dispatcher's
    // NIM-on-NIM skip is the defense-in-depth that still refuses it.
    mockFetchEnvironment.mockResolvedValue(
      makeEnv({
        recommended_teacher_mode: "local",
        recommended_local_teacher_model_name: "nvidia/cosmos-reason2-8b",
        recommended_embedding_mode: "hosted",
      }),
    );
    mockCosmosConfig();
    const { Wrapper } = createWrapper({
      cameFromAutoSkip: false,
      activePath: "local",
      localDeployQueued: [
        "nvidia/cosmos-reason2-8b",
        "nvidia/llama-nemotron-embed-vl-1b-v2",
      ],
    });
    const user = userEvent.setup();
    render(<div />, { wrapper: Wrapper });

    await user.click(await screen.findByRole("button", { name: /Start labeling/i }));

    await screen.findByTestId("confirm-defaults-page");

    // Teacher deploy with the resolved model_config_id.
    expect(mockDeployLocalNim).toHaveBeenCalledWith("test-pid", {
      role: "teacher",
      model_config_id: "mc-cosmos-8b",
      replace_resident: false,
    });
    // Single-GPU host: embedding deploy is SKIPPED because two NIMs
    // cannot coexist on a single GPU regardless of VRAM math
    // (README.md "One-NIM-per-GPU policy"). Embedding falls back to
    // pHash diversity.
    expect(mockDeployLocalNim).not.toHaveBeenCalledWith(
      "test-pid",
      expect.objectContaining({ role: "embedding" }),
    );

    // mark_setup_completed receives the queue.
    expect(mockMarkSetupCompleted).toHaveBeenCalledWith(
      "test-pid",
      expect.objectContaining({
        auto_skip: false,
        teacher_mode: "local",
        local_deploy_queued: [
          "nvidia/cosmos-reason2-8b",
          "nvidia/llama-nemotron-embed-vl-1b-v2",
        ],
      }),
    );
  });

  it("multi-GPU host: embedding deploy omits gpu_assignment (backend auto-places on a different GPU)", async () => {
    mockFetchEnvironment.mockResolvedValue(
      makeEnv({
        recommended_teacher_mode: "local",
        recommended_local_teacher_model_name: "nvidia/cosmos-reason2-8b",
        gpus: [
          { name: "NVIDIA A100-SXM4-80GB", memory_total_gb: 80 },
          { name: "NVIDIA A100-SXM4-80GB", memory_total_gb: 80 },
        ],
      }),
    );
    mockCosmosConfig();
    const { Wrapper } = createWrapper({
      cameFromAutoSkip: false,
      activePath: "local",
      localDeployQueued: [
        "nvidia/cosmos-reason2-8b",
        "nvidia/llama-nemotron-embed-vl-1b-v2",
      ],
    });
    const user = userEvent.setup();
    render(<div />, { wrapper: Wrapper });

    await user.click(await screen.findByRole("button", { name: /Start labeling/i }));
    await screen.findByTestId("confirm-defaults-page");

    // No gpu_assignment when there's more than one GPU — backend's
    // auto-placer picks a different device for the embedding.
    expect(mockDeployLocalNim).toHaveBeenCalledWith("test-pid", {
      role: "embedding",
    });
  });

  it("single GPU but no headroom for both: embedding deploy is SKIPPED (falls back to pHash)", async () => {
    mockFetchEnvironment.mockResolvedValue(
      makeEnv({
        recommended_teacher_mode: "local",
        recommended_local_teacher_model_name: "nvidia/cosmos-reason2-8b",
        // Teacher needs more than the GPU holds even before embedding.
        recommended_local_teacher_gpu_memory_minimum_gb: 80,
        gpus: [{ name: "Tiny GPU", memory_total_gb: 80 }],
        // Placement-aware backend: the planned local Teacher reserves
        // the only GPU, so the embedding NIM has no candidate device.
        embedding_deployment: {
          model_name: "nvidia/llama-nemotron-embed-vl-1b-v2",
          nim_container_image: "nvcr.io/nim/nvidia/llama-nemotron-embed-vl-1b-v2:2.0.0",
          gpu_memory_minimum_gb: 24,
          fits: false,
          provider: "none",
        },
        recommended_embedding_mode: "hosted",
      }),
    );
    mockCosmosConfig();
    const { Wrapper } = createWrapper({
      cameFromAutoSkip: false,
      activePath: "local",
      localDeployQueued: [
        "nvidia/cosmos-reason2-8b",
        "nvidia/llama-nemotron-embed-vl-1b-v2",
      ],
    });
    const user = userEvent.setup();
    render(<div />, { wrapper: Wrapper });

    await user.click(await screen.findByRole("button", { name: /Start labeling/i }));
    await screen.findByTestId("confirm-defaults-page");

    // Teacher still dispatched. Embedding NOT — no memory headroom.
    expect(mockDeployLocalNim).toHaveBeenCalledWith("test-pid", {
      role: "teacher",
      model_config_id: "mc-cosmos-8b",
      replace_resident: false,
    });
    expect(mockDeployLocalNim).not.toHaveBeenCalledWith(
      "test-pid",
      expect.objectContaining({ role: "embedding" }),
    );
  });

  it("single small GPU + embedding-only queue: embedding deploy IS dispatched (hosted Teacher + local embeddings)", async () => {
    // The dispatcher's NIM-on-NIM skip requires a QUEUED TEACHER
    // (`teacherQueued && singleGpu`) — an embedding-only queue on a
    // single GPU must pass through. This is the small-GPU host class:
    // the GPU fits no local Teacher but is at/above the embedding
    // floor, so the Teacher stays hosted and the local embedding NIM
    // (the default provider when a suitable GPU exists) gets the GPU.
    mockFetchEnvironment.mockResolvedValue(
      makeEnv({
        gpus: [{ name: "NVIDIA RTX 5000 Ada", memory_total_gb: 32 }],
        recommended_local_teacher_model_name: null,
      }),
    );
    const { Wrapper } = createWrapper({
      cameFromAutoSkip: false,
      activePath: "hosted",
      localDeployQueued: ["nvidia/llama-nemotron-embed-vl-1b-v2"],
    });
    const user = userEvent.setup();
    render(<div />, { wrapper: Wrapper });

    await user.click(await screen.findByRole("button", { name: /Start labeling/i }));
    await screen.findByTestId("confirm-defaults-page");

    expect(mockDeployLocalNim).toHaveBeenCalledWith("test-pid", {
      role: "embedding",
    });
    expect(mockDeployLocalNim).not.toHaveBeenCalledWith(
      "test-pid",
      expect.objectContaining({ role: "teacher" }),
    );
    expect(mockMarkSetupCompleted).toHaveBeenCalledWith(
      "test-pid",
      expect.objectContaining({
        auto_skip: false,
        teacher_mode: "hosted",
        local_deploy_queued: ["nvidia/llama-nemotron-embed-vl-1b-v2"],
      }),
    );
  });

  it("single small GPU + embedding-only queue: summary shows hosted Teacher + LOCAL embedding row", async () => {
    // The /done summary for the small-GPU host class must describe
    // what will actually happen: the configured default stays hosted and
    // the queued NeMo Retriever VL deploy renders as a local NIM row
    // (the single-GPU guard only suppresses the local claim when a
    // Teacher is queued alongside).
    mockFetchEnvironment.mockResolvedValue(
      makeEnv({
        gpus: [{ name: "NVIDIA RTX 5000 Ada", memory_total_gb: 32 }],
        recommended_local_teacher_model_name: null,
      }),
    );
    const { Wrapper } = createWrapper({
      cameFromAutoSkip: false,
      activePath: "hosted",
      localDeployQueued: ["nvidia/llama-nemotron-embed-vl-1b-v2"],
    });
    render(<div />, { wrapper: Wrapper });

    await waitFor(() => {
      expect(screen.getByText(/You're set up/i)).toBeInTheDocument();
    });
    expect(screen.getByText(/Step 3\.7 Flash/)).toBeInTheDocument();
    expect(screen.getByText(/Hosted on build\.nvidia\.com/)).toBeInTheDocument();
    expect(screen.getByText(/NeMo Retriever VL/)).toBeInTheDocument();
    expect(screen.getByText(/Local NIM · NVIDIA RTX 5000 Ada/)).toBeInTheDocument();
  });

  it("embedding row omitted on a no-key chain with nothing queued, even when the backend recommends local", async () => {
    // A recommendation alone deploys nothing: the summary only claims
    // "Local NIM" when the deploy is actually queued for dispatch.
    // Without an NVIDIA key there's no hosted fallback either, so the
    // row is omitted (pHash diversity on the backend).
    mockFetchEnvironment.mockResolvedValue(
      makeEnv({
        nvidia_api_key_configured: false,
        gpus: [{ name: "NVIDIA RTX 5000 Ada", memory_total_gb: 32 }],
        recommended_embedding_mode: "local",
      }),
    );
    const { Wrapper } = createWrapper({
      cameFromAutoSkip: false,
      activePath: "hosted",
      localDeployQueued: [],
    });
    render(<div />, { wrapper: Wrapper });

    await waitFor(() => {
      expect(screen.getByText(/You're set up/i)).toBeInTheDocument();
    });
    expect(screen.queryByText(/NeMo Retriever VL/)).toBeNull();
    expect(screen.queryByText(/EMBEDDING MODEL/)).toBeNull();
  });

  it("dispatch failure does not strand the SME — navigation still happens", async () => {
    mockFetchEnvironment.mockResolvedValue(
      makeEnv({
        recommended_teacher_mode: "local",
        recommended_local_teacher_model_name: "nvidia/cosmos-reason2-8b",
      }),
    );
    mockCosmosConfig();
    mockDeployLocalNim.mockRejectedValueOnce(new Error("preflight failed"));
    const { Wrapper } = createWrapper({
      cameFromAutoSkip: false,
      activePath: "local",
      localDeployQueued: ["nvidia/cosmos-reason2-8b"],
    });
    const user = userEvent.setup();
    render(<div />, { wrapper: Wrapper });

    await user.click(await screen.findByRole("button", { name: /Start labeling/i }));
    await screen.findByTestId("confirm-defaults-page");
  });

  it("auto-skip path includes local_deploy_queued in the mark_setup_completed payload", async () => {
    mockFetchEnvironment.mockResolvedValue(makeEnv());
    const { Wrapper } = createWrapper({
      cameFromAutoSkip: true,
      activePath: "hybrid",
      localDeployQueued: ["nvidia/cosmos-reason2-8b"],
    });
    render(<div />, { wrapper: Wrapper });

    await screen.findByTestId("confirm-defaults-page");
    expect(mockMarkSetupCompleted).toHaveBeenCalledWith(
      "test-pid",
      expect.objectContaining({
        auto_skip: true,
        local_deploy_queued: ["nvidia/cosmos-reason2-8b"],
      }),
    );
  });

  // ── One-NIM-per-GPU dispatch-failure UX ──────────────────────────

  it("different resident offers explicit replacement and retries with replace_resident", async () => {
    mockFetchEnvironment.mockResolvedValue(
      makeEnv({
        recommended_teacher_mode: "local",
        recommended_local_teacher_model_name: "nvidia/cosmos-reason2-8b",
      }),
    );
    mockCosmosConfig();
    mockDeployLocalNim.mockRejectedValueOnce(
      new ApiError(
        409,
        JSON.stringify({
          detail: {
            code: "gpu_occupied",
            message: "GPU device=0 is occupied",
            can_replace: true,
            matches_requested_model: false,
            resident: {
              project_id: "other-pid",
              project_name: "Existing project",
              local_nim_deployment_id: "dep-old",
              role: "teacher",
              model_name: "nvidia/cosmos3-nano-reasoner",
              nim_container_image: "nvcr.io/nim/nvidia/cosmos3-reasoner:1.7.0",
              gpu_assignment: "device=0",
              status: "running",
            },
          },
        }),
      ),
    );
    mockDeployLocalNim.mockResolvedValueOnce({
      deployment: { status: "starting" },
      preflight: { all_passed: true, checks: [] },
      disposition: "queued",
    });

    const { Wrapper } = createWrapper({
      cameFromAutoSkip: false,
      activePath: "local",
      localDeployQueued: ["nvidia/cosmos-reason2-8b"],
    });
    const user = userEvent.setup();
    render(<div />, { wrapper: Wrapper });

    await waitFor(() => {
      expect(screen.getByText(/You're set up/i)).toBeInTheDocument();
    });

    await user.click(await screen.findByRole("button", { name: /Start labeling/i }));

    const alert = await screen.findByTestId("teacher-deploy-blocked");
    expect(alert).toHaveTextContent(/Replace the NIM on device=0/i);
    expect(alert).toHaveTextContent(/Cosmos 3 Nano \(Reasoner\)/i);
    expect(alert).toHaveTextContent(/Existing project/i);
    expect(alert).toHaveTextContent(/cannot use its local model/i);
    expect(screen.queryByTestId("confirm-defaults-page")).toBeNull();
    expect(mockMarkSetupCompleted).not.toHaveBeenCalled();

    await user.click(
      screen.getByRole("button", {
        name: /Stop current & start new NIM/i,
      }),
    );

    await screen.findByTestId("confirm-defaults-page");
    expect(mockDeployLocalNim).toHaveBeenLastCalledWith("test-pid", {
      role: "teacher",
      model_config_id: "mc-cosmos-8b",
      replace_resident: true,
    });
    expect(mockMarkSetupCompleted).toHaveBeenCalled();
  });

  it("same model already starting asks the SME to wait instead of replacing it", async () => {
    mockFetchEnvironment.mockResolvedValue(
      makeEnv({
        recommended_teacher_mode: "local",
        recommended_local_teacher_model_name: "nvidia/cosmos-reason2-8b",
      }),
    );
    mockCosmosConfig();
    mockDeployLocalNim.mockRejectedValue(
      new ApiError(
        409,
        JSON.stringify({
          detail: {
            code: "resident_starting",
            message: "GPU device=0 is occupied",
            can_replace: false,
            matches_requested_model: true,
            resident: {
              project_id: "other-pid",
              project_name: "Existing project",
              local_nim_deployment_id: "dep-old",
              role: "teacher",
              model_name: "nvidia/cosmos-reason2-8b",
              nim_container_image: "nvcr.io/nim/nvidia/cosmos-reason2-8b:1.6.0",
              gpu_assignment: "device=0",
              status: "starting",
            },
          },
        }),
      ),
    );

    const { Wrapper } = createWrapper({
      cameFromAutoSkip: false,
      activePath: "local",
      localDeployQueued: ["nvidia/cosmos-reason2-8b"],
    });
    const user = userEvent.setup();
    render(<div />, { wrapper: Wrapper });

    await waitFor(() => {
      expect(screen.getByText(/You're set up/i)).toBeInTheDocument();
    });
    await user.click(await screen.findByRole("button", { name: /Start labeling/i }));

    const alert = await screen.findByTestId("teacher-deploy-blocked");
    expect(alert).toHaveTextContent(/Cosmos Reason2 8B is already starting/i);
    expect(alert).toHaveTextContent(/reuse it automatically/i);
    expect(screen.getByRole("button", { name: /Check again/i })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Stop current & start/i })).toBeNull();
    expect(mockMarkSetupCompleted).not.toHaveBeenCalled();
  });

  it("GPU conflict [Keep current NIM] lands on the project list", async () => {
    mockFetchEnvironment.mockResolvedValue(
      makeEnv({
        recommended_teacher_mode: "local",
        recommended_local_teacher_model_name: "nvidia/cosmos-reason2-8b",
      }),
    );
    mockCosmosConfig();
    mockDeployLocalNim.mockRejectedValue(
      new ApiError(
        409,
        JSON.stringify({
          detail: {
            code: "gpu_occupied",
            message: "GPU device=0 is occupied",
            can_replace: true,
            matches_requested_model: false,
            resident: {
              project_id: "other-pid",
              project_name: "Existing project",
              local_nim_deployment_id: "dep-old",
              role: "teacher",
              model_name: "nvidia/cosmos3-nano-reasoner",
              nim_container_image: "nvcr.io/nim/nvidia/cosmos3-reasoner:1.7.0",
              gpu_assignment: "device=0",
              status: "running",
            },
          },
        }),
      ),
    );

    const { Wrapper } = createWrapper({
      cameFromAutoSkip: false,
      activePath: "local",
      localDeployQueued: ["nvidia/cosmos-reason2-8b"],
    });
    const user = userEvent.setup();
    render(<div />, { wrapper: Wrapper });

    await waitFor(() => {
      expect(screen.getByText(/You're set up/i)).toBeInTheDocument();
    });
    await user.click(await screen.findByRole("button", { name: /Start labeling/i }));

    await screen.findByTestId("teacher-deploy-blocked");
    await user.click(screen.getByRole("button", { name: /Keep current NIM/i }));

    await screen.findByTestId("project-list-page");
  });

  it("non-Teacher (embedding) 409 does NOT block navigation (falls back to pHash)", async () => {
    // Embedding deploys fail safely — the system falls back to pHash
    // diversity. Only Teacher 409s block. This test pins that the
    // embedding-only failure path continues to navigate normally.
    mockFetchEnvironment.mockResolvedValue(
      makeEnv({
        recommended_teacher_mode: "local",
        recommended_local_teacher_model_name: "nvidia/cosmos-reason2-8b",
        gpus: [
          { name: "NVIDIA A100", memory_total_gb: 80 },
          { name: "NVIDIA A100", memory_total_gb: 80 },
        ],
      }),
    );
    mockCosmosConfig();
    // Teacher deploy SUCCEEDS; embedding deploy FAILS.
    mockDeployLocalNim.mockImplementation((_pid: string, body: { role: string }) => {
      if (body.role === "embedding") {
        return Promise.reject(new ApiError(409, '{"code":"gpu_exhausted"}'));
      }
      return Promise.resolve({
        deployment: { status: "starting" },
        preflight: { all_passed: true, checks: [] },
      });
    });

    const { Wrapper } = createWrapper({
      cameFromAutoSkip: false,
      activePath: "local",
      localDeployQueued: [
        "nvidia/cosmos-reason2-8b",
        "nvidia/llama-nemotron-embed-vl-1b-v2",
      ],
    });
    const user = userEvent.setup();
    render(<div />, { wrapper: Wrapper });

    await waitFor(() => {
      expect(screen.getByText(/You're set up/i)).toBeInTheDocument();
    });
    await user.click(await screen.findByRole("button", { name: /Start labeling/i }));

    // Navigation proceeded — markSetupCompleted was called.
    await waitFor(() => {
      expect(mockMarkSetupCompleted).toHaveBeenCalled();
    });
    // No inline alert.
    expect(screen.queryByTestId("teacher-deploy-blocked")).toBeNull();
  });
});
