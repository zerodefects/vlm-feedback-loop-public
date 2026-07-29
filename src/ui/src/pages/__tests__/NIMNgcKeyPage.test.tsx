// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Tests for NIMNgcKeyPage — the FTU NGC key screen (second step of the
 * three-screen setup chain). Only hosted-path flows reach it; NGC is
 * always optional here.
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { installEventSourceMock } from "@/test/event-source-mock";
import { makeEnvironmentResponse, makeProjectResponse } from "@/test/fixtures";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter, Route, Routes, useLocation } from "react-router-dom";
import type { ReactNode } from "react";

import { NIMNgcKeyPage } from "@/pages/NIMNgcKeyPage";
import { ProjectSetupLayout } from "@/pages/ProjectSetupLayout";
import type { EnvironmentResponse } from "@/types/nim";
import type { SetupChainState } from "@/types/setupChain";

// ── Mocks ───────────────────────────────────────────────────────────────────

const mockFetchProject = vi.fn();
const mockFetchEnvironment = vi.fn();
const mockSetSecret = vi.fn();

vi.mock("@/api/projects", () => ({
  fetchProject: (...args: unknown[]) => mockFetchProject(...args),
  fetchProjectList: vi.fn().mockResolvedValue({ items: [], next_cursor: null }),
  createProject: vi.fn(),
  archiveProject: vi.fn(),
  markSetupCompleted: vi.fn(),
}));

vi.mock("@/api/secrets", () => ({
  setSecret: (...args: unknown[]) => mockSetSecret(...args),
}));

vi.mock("@/api/nim", () => ({
  fetchEnvironment: (...args: unknown[]) => mockFetchEnvironment(...args),
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
  setup_completed_at: null,
  archived_at: null,
});

/** The host class this screen serves after the local-embeddings-by-
 *  default policy: a GPU too small for any local Teacher (so the flow
 *  is hosted-path Case C) but at/above the embedding NIM floor, so the
 *  placement-aware backend reports ``embedding_deployment.fits`` and
 *  recommends local embeddings. */
function makeEnv(overrides: Partial<EnvironmentResponse> = {}): EnvironmentResponse {
  return makeEnvironmentResponse({
    hosted_nim_available: true,
    local_deploy_available: true,
    docker_available: true,
    nvidia_toolkit_available: true,
    nvidia_api_key_configured: true,
    gpus: [{ name: "NVIDIA RTX 5000 Ada", memory_total_gb: 32 }],
    embedding_deployment: {
      model_name: "nvidia/llama-nemotron-embed-vl-1b-v2",
      nim_container_image: "nvcr.io/nim/nvidia/llama-nemotron-embed-vl-1b-v2:2.0.0",
      gpu_memory_minimum_gb: 10,
      fits: true,
      provider: "none",
    },
    recommended_teacher_mode: "hosted",
    recommended_embedding_mode: "local",
    ...overrides,
  });
}

const EMBEDDING_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2";

// ── Stub child routes capturing nav state ───────────────────────────────────

function DoneStub(): JSX.Element {
  const location = useLocation();
  const state = (location.state ?? null) as SetupChainState | null;
  return (
    <div
      data-testid="done-page"
      data-came-from-auto-skip={String(state?.cameFromAutoSkip ?? "undefined")}
      data-active-path={state?.activePath ?? "undefined"}
      data-local-deploy-queued={state?.localDeployQueued?.join(",") ?? "undefined"}
    />
  );
}

function createWrapper(
  initialPath = "/projects/test-pid/setup/ngc",
  initialState?: Partial<SetupChainState>,
) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false, staleTime: 0 } },
  });
  const entries =
    initialState !== undefined
      ? [{ pathname: initialPath, state: initialState }]
      : [initialPath];
  function Wrapper({ children }: { children: ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>
        <MemoryRouter initialEntries={entries}>
          <Routes>
            <Route path="/projects/:projectId" element={<ProjectSetupLayout />}>
              <Route path="setup/ngc" element={<NIMNgcKeyPage />} />
              <Route path="setup/done" element={<DoneStub />} />
            </Route>
          </Routes>
          {children}
        </MemoryRouter>
      </QueryClientProvider>
    );
  }
  return { Wrapper };
}

// ── Tests ───────────────────────────────────────────────────────────────────

describe("NIMNgcKeyPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockFetchProject.mockResolvedValue(PROJECT);
    mockSetSecret.mockResolvedValue({
      effective: true,
      persisted: true,
      env_path: "/home/me/.vlm_feedback_loop/.env",
      allow_persist: true,
    });
  });

  it("auto-skips with cameFromAutoSkip=true when both NGC configured and upstream auto-skipped", async () => {
    mockFetchEnvironment.mockResolvedValue(makeEnv({ ngc_api_key_configured: true }));
    const { Wrapper } = createWrapper("/projects/test-pid/setup/ngc", {
      cameFromAutoSkip: true,
    });
    render(<div />, { wrapper: Wrapper });

    const done = await screen.findByTestId("done-page");
    expect(done.getAttribute("data-came-from-auto-skip")).toBe("true");
  });

  it("auto-skips with cameFromAutoSkip=false when the setup-choice screen rendered upstream", async () => {
    mockFetchEnvironment.mockResolvedValue(makeEnv({ ngc_api_key_configured: true }));
    // No incoming state — defaults to ``cameFromAutoSkip=false``.
    const { Wrapper } = createWrapper();
    render(<div />, { wrapper: Wrapper });

    const done = await screen.findByTestId("done-page");
    expect(done.getAttribute("data-came-from-auto-skip")).toBe("false");
  });

  it("auto-skips when local_deploy_available is false (no GPU/Docker — NGC not useful)", async () => {
    mockFetchEnvironment.mockResolvedValue(
      makeEnv({ local_deploy_available: false, ngc_api_key_configured: false }),
    );
    const { Wrapper } = createWrapper("/projects/test-pid/setup/ngc", {
      cameFromAutoSkip: true,
    });
    render(<div />, { wrapper: Wrapper });

    const done = await screen.findByTestId("done-page");
    expect(done.getAttribute("data-came-from-auto-skip")).toBe("true");
  });

  it("renders the form when NGC missing and local deploy is available", async () => {
    mockFetchEnvironment.mockResolvedValue(makeEnv());
    const { Wrapper } = createWrapper();
    render(<div />, { wrapper: Wrapper });

    await waitFor(() => {
      expect(screen.getByText(/Want faster local embeddings/i)).toBeInTheDocument();
    });
    expect(screen.getByPlaceholderText(/Paste your NGC API key/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Use my GPU/i })).toBeDisabled();
    // [Back] is the escape affordance — visible alongside the primary
    // CTA (there is no "Skip" text-link).
    expect(screen.getByTestId("ngc-back-button")).toBeInTheDocument();
  });

  it("[Use my GPU] applies NGC_API_KEY persist:true and navigates to done with cameFromAutoSkip=false", async () => {
    mockFetchEnvironment.mockResolvedValue(makeEnv());
    const { Wrapper } = createWrapper();
    const user = userEvent.setup();
    render(<div />, { wrapper: Wrapper });

    const input = await screen.findByPlaceholderText(/Paste your NGC API key/i);
    await user.type(input, "ngc-good");
    await user.click(screen.getByRole("button", { name: /Use my GPU/i }));

    const done = await screen.findByTestId("done-page");
    expect(done.getAttribute("data-came-from-auto-skip")).toBe("false");
    expect(mockSetSecret).toHaveBeenCalledWith({
      name: "NGC_API_KEY",
      value: "ngc-good",
      persist: true,
    });
  });

  it("[Back] does not mutate NGC and never lands on the done page", async () => {
    mockFetchEnvironment.mockResolvedValue(makeEnv());
    const { Wrapper } = createWrapper();
    const user = userEvent.setup();
    render(<div />, { wrapper: Wrapper });

    await user.click(await screen.findByTestId("ngc-back-button"));

    // No NGC mutation, no done-page navigation — Back is a pure
    // history pop with no skip-downgrade side effects. (The
    // MemoryRouter has no upstream entry, so we don't assert a
    // specific destination.)
    expect(mockSetSecret).not.toHaveBeenCalled();
    expect(screen.queryByTestId("done-page")).toBeNull();
  });

  it("surfaces the 'add to .env' hint when setSecret returns 403", async () => {
    mockFetchEnvironment.mockResolvedValue(makeEnv());
    mockSetSecret.mockRejectedValue(new Error("403: ui_secret_persist_disabled"));
    const { Wrapper } = createWrapper();
    const user = userEvent.setup();
    render(<div />, { wrapper: Wrapper });

    const input = await screen.findByPlaceholderText(/Paste your NGC API key/i);
    await user.type(input, "ngc-x");
    await user.click(screen.getByRole("button", { name: /Use my GPU/i }));

    await waitFor(() => {
      expect(screen.getByRole("alert")).toHaveTextContent(
        /add NGC_API_KEY=\.\.\. to ~\/\.vlm_feedback_loop\/\.env and restart/i,
      );
    });
    expect(screen.queryByTestId("done-page")).toBeNull();
  });

  // ── Chain-state forwarding + embedding-deploy queueing ──────────────
  // Local embeddings are the DEFAULT provider whenever the host GPU
  // fits the embedding NIM (hosted is the fallback). This screen is
  // where hosted-Teacher chains pick that deploy up: both exits append
  // the embedding NIM to localDeployQueued when
  // ``local_deploy_available && embedding_deployment.fits``; activePath
  // and cameFromAutoSkip are forwarded unchanged.

  it("[Use my GPU] queues the local embedding deploy for the hosted-path chain", async () => {
    // The screen promises "we'll run NeMo Retriever VL on {gpu}" —
    // clicking [Use my GPU] must make that real by appending the
    // embedding NIM to the deploy queue the gate dispatches. This is
    // the small-GPU host class: GPU below every Teacher floor but
    // at/above the embedding floor → hosted Teacher + local
    // embeddings.
    mockFetchEnvironment.mockResolvedValue(makeEnv());
    const { Wrapper } = createWrapper("/projects/test-pid/setup/ngc", {
      activePath: "hosted",
      cameFromAutoSkip: false,
      localDeployQueued: [],
    });
    const user = userEvent.setup();
    render(<div />, { wrapper: Wrapper });

    const input = await screen.findByPlaceholderText(/Paste your NGC API key/i);
    await user.type(input, "ngc-good");
    await user.click(screen.getByRole("button", { name: /Use my GPU/i }));

    const done = await screen.findByTestId("done-page");
    expect(done.getAttribute("data-active-path")).toBe("hosted");
    expect(done.getAttribute("data-local-deploy-queued")).toBe(EMBEDDING_MODEL);
    expect(mockSetSecret).toHaveBeenCalledWith({
      name: "NGC_API_KEY",
      value: "ngc-good",
      persist: true,
    });
  });

  it("[Use my GPU] does not duplicate an embedding deploy already queued upstream", async () => {
    // Deep-linked chains can arrive with the embedding NIM already in
    // the queue; the append is idempotent — one deploy per model, and
    // the rest of the chain state is forwarded unchanged.
    mockFetchEnvironment.mockResolvedValue(makeEnv());
    const { Wrapper } = createWrapper("/projects/test-pid/setup/ngc", {
      activePath: "hybrid",
      cameFromAutoSkip: false,
      localDeployQueued: ["nvidia/cosmos-reason2-8b", EMBEDDING_MODEL],
    });
    const user = userEvent.setup();
    render(<div />, { wrapper: Wrapper });

    const input = await screen.findByPlaceholderText(/Paste your NGC API key/i);
    await user.type(input, "ngc-good");
    await user.click(screen.getByRole("button", { name: /Use my GPU/i }));

    const done = await screen.findByTestId("done-page");
    expect(done.getAttribute("data-active-path")).toBe("hybrid");
    expect(done.getAttribute("data-local-deploy-queued")).toBe(
      `nvidia/cosmos-reason2-8b,${EMBEDDING_MODEL}`,
    );
  });

  it("auto-skip (NGC already configured) also queues the embedding deploy when the GPU fits it", async () => {
    // A second project on a host whose NGC key is already persisted
    // never sees this screen — the auto-skip must still apply the
    // local-embeddings-by-default policy, or that host class silently
    // loses the deploy the first project got.
    mockFetchEnvironment.mockResolvedValue(makeEnv({ ngc_api_key_configured: true }));
    const { Wrapper } = createWrapper("/projects/test-pid/setup/ngc", {
      activePath: "hosted",
      cameFromAutoSkip: false,
      localDeployQueued: [],
    });
    render(<div />, { wrapper: Wrapper });

    const done = await screen.findByTestId("done-page");
    expect(done.getAttribute("data-active-path")).toBe("hosted");
    expect(done.getAttribute("data-local-deploy-queued")).toBe(EMBEDDING_MODEL);
  });

  it("auto-skips WITHOUT queueing when the embedding NIM does not fit the free GPU", async () => {
    // When the placement-aware assessment says the embedding NIM
    // won't fit the GPU it would actually get, the screen's offer
    // can't be delivered: don't render the promise (auto-skip) and
    // don't queue a deploy that would 409/OOM — the queue is
    // forwarded unchanged.
    mockFetchEnvironment.mockResolvedValue(
      makeEnv({
        ngc_api_key_configured: false,
        embedding_deployment: {
          model_name: EMBEDDING_MODEL,
          nim_container_image: "nvcr.io/nim/nvidia/llama-nemotron-embed-vl-1b-v2:2.0.0",
          gpu_memory_minimum_gb: 10,
          fits: false,
          provider: "none",
        },
        recommended_embedding_mode: "hosted",
      }),
    );
    const { Wrapper } = createWrapper("/projects/test-pid/setup/ngc", {
      activePath: "hosted",
      cameFromAutoSkip: false,
      localDeployQueued: [],
    });
    render(<div />, { wrapper: Wrapper });

    const done = await screen.findByTestId("done-page");
    expect(done.getAttribute("data-came-from-auto-skip")).toBe("false");
    expect(done.getAttribute("data-local-deploy-queued")).toBe("");
    expect(mockSetSecret).not.toHaveBeenCalled();
  });
});
