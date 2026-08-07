// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Tests for NIMNvidiaKeyPage — the NVIDIA API key setup screen.
 *
 * Four-case matrix:
 *
 *   - Case A: no key + GPU + local Teacher fits → local primary card +
 *     hosted peer card.
 *   - Case B: key configured + GPU + local Teacher fits → explicit hosted
 *     choice + optional local alternate. Does NOT auto-skip.
 *   - Case C: no key + (no GPU OR no fitting local Teacher) → today's
 *     hosted CTA.
 *   - Case D: key configured + no GPU → auto-skip to the NGC key step.
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { installEventSourceMock } from "@/test/event-source-mock";
import { makeEnvironmentResponse, makeProjectResponse } from "@/test/fixtures";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter, Route, Routes, useLocation } from "react-router-dom";
import type { ReactNode } from "react";

import { NIMNvidiaKeyPage } from "@/pages/NIMNvidiaKeyPage";
import { ProjectSetupLayout } from "@/pages/ProjectSetupLayout";
import type { EnvironmentResponse } from "@/types/nim";
import type { SetupChainState } from "@/types/setupChain";

const mockFetchProject = vi.fn();
const mockFetchEnvironment = vi.fn();
const mockFetchModelConfigs = vi.fn();
const mockTestConnection = vi.fn();
const mockTestNgcCredential = vi.fn();
const mockTestNvidiaCredential = vi.fn();
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

vi.mock("@/api/model-configs", () => ({
  fetchModelConfigs: (...args: unknown[]) => mockFetchModelConfigs(...args),
}));

vi.mock("@/api/nim", () => ({
  fetchEnvironment: (...args: unknown[]) => mockFetchEnvironment(...args),
  testConnection: (...args: unknown[]) => mockTestConnection(...args),
  testNgcCredential: (...args: unknown[]) => mockTestNgcCredential(...args),
  testNvidiaCredential: (...args: unknown[]) => mockTestNvidiaCredential(...args),
}));

installEventSourceMock();

const PROJECT = makeProjectResponse({
  active_guidance_id: null,
  setup_completed_at: null,
});

function makeEnv(overrides: Partial<EnvironmentResponse> = {}): EnvironmentResponse {
  return makeEnvironmentResponse({
    recommended_teacher_mode: "hosted",
    recommended_embedding_mode: "hosted",
    ...overrides,
  });
}

/** Records navigated path + state for assertions. Rendered at both
 *  ``setup/ngc`` and ``setup/done`` so tests don't have to know which
 *  URL the page chooses (Case A + Case B continue to ``setup/done``
 *  directly; Case C goes through ``setup/ngc``). */
function NextStub(): JSX.Element {
  const location = useLocation();
  const state = (location.state ?? null) as SetupChainState | null;
  return (
    <div
      data-testid="ngc-page"
      data-came-from-auto-skip={String(state?.cameFromAutoSkip ?? "undefined")}
      data-active-path={state?.activePath ?? "undefined"}
      data-local-deploy-queued={state?.localDeployQueued?.join(",") ?? "undefined"}
    />
  );
}

function createWrapper(initialPath = "/projects/test-pid/setup") {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false, staleTime: 0 } },
  });
  function Wrapper({ children }: { children: ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>
        <MemoryRouter initialEntries={[initialPath]}>
          <Routes>
            <Route path="/projects/:projectId" element={<ProjectSetupLayout />}>
              <Route path="setup" element={<NIMNvidiaKeyPage />} />
              <Route path="setup/ngc" element={<NextStub />} />
              <Route path="setup/done" element={<NextStub />} />
            </Route>
          </Routes>
          {children}
        </MemoryRouter>
      </QueryClientProvider>
    );
  }
  return { Wrapper };
}

const GPU_80GB = { name: "NVIDIA A100", memory_total_gb: 80 };
const COSMOS_8B_FIELDS = {
  recommended_local_teacher_model_name: "nvidia/cosmos-reason2-8b",
  recommended_local_teacher_image: "nvcr.io/nim/nvidia/cosmos-reason2-8b:1.6.0",
  recommended_local_teacher_gpu_memory_minimum_gb: 56,
};

/** Environment for a local-deploy-capable host: Docker + toolkit + a GPU
 *  the recommended Cosmos Teacher fits on. Case A (no key, local
 *  recommended) by default; pass overrides for the key-configured,
 *  multi-GPU, or hosted-recommended variants. */
function makeCaseAEnv(
  overrides: Partial<EnvironmentResponse> = {},
): EnvironmentResponse {
  return makeEnv({
    local_deploy_available: true,
    docker_available: true,
    nvidia_toolkit_available: true,
    gpus: [GPU_80GB],
    recommended_teacher_mode: "local",
    ...COSMOS_8B_FIELDS,
    ...overrides,
  });
}

/** Embedding NIM catalog entry that fits the local GPU — used by the
 *  Case A variants that also queue the local embedding deploy. */
const EMBEDDING_FITS: EnvironmentResponse["embedding_deployment"] = {
  model_name: "nvidia/llama-nemotron-embed-vl-1b-v2",
  nim_container_image: "nvcr.io/nim/nvidia/llama-nemotron-embed-vl-1b-v2:2.0.0",
  gpu_memory_minimum_gb: 24,
  fits: true,
  provider: "none",
};

describe("NIMNvidiaKeyPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockFetchProject.mockResolvedValue(PROJECT);
    mockFetchModelConfigs.mockResolvedValue({ items: [], next_cursor: null });
    mockSetSecret.mockResolvedValue({
      effective: true,
      persisted: true,
      env_path: "/home/me/.vlm_feedback_loop/.env",
      allow_persist: true,
    });
    // Defaults: both proactive + click-time probes succeed. Individual
    // tests override for bad-key / error paths. ``testConnection`` stays
    // mocked too because some adjacent components (ConnectionTestPanel)
    // use it, even though NIMNvidiaKeyPage itself routes NVIDIA
    // validation through ``testNvidiaCredential``.
    mockTestNgcCredential.mockResolvedValue({ success: true });
    mockTestNvidiaCredential.mockResolvedValue({ success: true });
  });

  it("auto-selects and skips setup when project creation reused the running Teacher", async () => {
    const superProject = makeProjectResponse({
      teacher_model_config_id: "mc-super",
      active_guidance_id: null,
      setup_completed_at: null,
    });
    mockFetchProject.mockResolvedValue(superProject);
    mockFetchModelConfigs.mockResolvedValue({
      items: [
        {
          model_config_id: "mc-super",
          model_name: "nvidia/cosmos3-super-reasoner",
        },
      ],
      next_cursor: null,
    });
    mockFetchEnvironment.mockResolvedValue(
      makeCaseAEnv({
        nvidia_api_key_configured: true,
        ngc_api_key_configured: true,
        gpus: [{ name: "RTX PRO 6000 Blackwell", memory_total_gb: 96 }],
        recommended_teacher_mode: "local",
        recommended_local_teacher_model_name: "nvidia/cosmos3-super-reasoner",
        recommended_local_teacher_image: "nvcr.io/nim/nvidia/cosmos3-reasoner:1.7.0",
        recommended_local_teacher_gpu_memory_minimum_gb: 88,
        active_local_nim_residents: [
          {
            project_id: "owner-project",
            project_name: "Trash",
            local_nim_deployment_id: "deployment-1",
            role: "teacher",
            model_name: "nvidia/cosmos3-super-reasoner",
            nim_container_image: "nvcr.io/nim/nvidia/cosmos3-reasoner:1.7.0",
            gpu_assignment: "device=0",
            status: "running",
          },
        ],
      }),
    );
    const { Wrapper } = createWrapper();
    render(<div />, { wrapper: Wrapper });

    const next = await screen.findByTestId("ngc-page");
    expect(next).toHaveAttribute("data-active-path", "local");
    expect(next).toHaveAttribute("data-came-from-auto-skip", "true");
    expect(next).toHaveAttribute("data-local-deploy-queued", "");
    expect(mockFetchModelConfigs).toHaveBeenCalledWith("test-pid", "teacher");
    expect(mockTestNgcCredential).not.toHaveBeenCalled();
    expect(mockTestNvidiaCredential).not.toHaveBeenCalled();
  });

  it("keeps the embedding setup step when a reused Teacher leaves another GPU free", async () => {
    mockFetchProject.mockResolvedValue(
      makeProjectResponse({
        teacher_model_config_id: "mc-super",
        active_guidance_id: null,
        setup_completed_at: null,
      }),
    );
    mockFetchModelConfigs.mockResolvedValue({
      items: [
        {
          model_config_id: "mc-super",
          model_name: "nvidia/cosmos3-super-reasoner",
        },
      ],
      next_cursor: null,
    });
    mockFetchEnvironment.mockResolvedValue(
      makeCaseAEnv({
        nvidia_api_key_configured: true,
        ngc_api_key_configured: false,
        gpus: [{ name: "RTX PRO 6000 Blackwell", memory_total_gb: 96 }, GPU_80GB],
        recommended_teacher_mode: "local",
        recommended_embedding_mode: "local",
        embedding_deployment: EMBEDDING_FITS,
        recommended_local_teacher_model_name: "nvidia/cosmos3-super-reasoner",
        active_local_nim_residents: [
          {
            project_id: "owner-project",
            project_name: "Trash",
            local_nim_deployment_id: "deployment-1",
            role: "teacher",
            model_name: "nvidia/cosmos3-super-reasoner",
            nim_container_image: "nvcr.io/nim/nvidia/cosmos3-reasoner:1.7.0",
            gpu_assignment: "device=0",
            status: "running",
          },
        ],
      }),
    );
    const { Wrapper } = createWrapper();
    render(<div />, { wrapper: Wrapper });

    const next = await screen.findByTestId("ngc-page");
    expect(next).toHaveAttribute("data-active-path", "local");
    expect(next).toHaveAttribute("data-came-from-auto-skip", "false");
    expect(next).toHaveAttribute("data-local-deploy-queued", "");
  });

  // ── Case A: combined card with local + hosted toggle rows ─────────────────

  it("Case A: renders one combined card with local and hosted rows both checked by default", async () => {
    mockFetchEnvironment.mockResolvedValue(makeCaseAEnv());
    const { Wrapper } = createWrapper();
    render(<div />, { wrapper: Wrapper });

    const setupCard = await screen.findByTestId("combined-card-local-and-hosted");
    expect(setupCard).toHaveClass("glass-card", "glass-card--elevated");
    const localCheckbox = screen.getByTestId("local-row-checkbox");
    const hostedCheckbox = screen.getByTestId("hosted-row-checkbox");
    expect(localCheckbox).toBeChecked();
    // The hosted NVIDIA API key row is checked by default.
    expect(hostedCheckbox).toBeChecked();
    // Local row names the recommended variant and the recommendation marker.
    expect(screen.getByTestId("local-row-label")).toHaveTextContent(
      /Run Cosmos Reason2 8B locally/i,
    );
    expect(screen.getByTestId("local-row-label")).toHaveTextContent(/RECOMMENDED/);
    // Hosted row is checked by default, so its key input is revealed.
    expect(screen.getByTestId("hosted-key-input")).toBeInTheDocument();
    // Separate primary/peer cards must not render — Case A uses the combined card.
    expect(screen.queryByTestId("primary-card-local")).toBeNull();
    expect(screen.queryByTestId("peer-card-hosted")).toBeNull();
  });

  it("Case A: default (local-only) [Set up & continue] queues Cosmos + embedding and lands activePath=local", async () => {
    // Multi-GPU host: the placement-aware backend reserves one GPU for
    // the planned local Teacher and the embedding NIM fits the other,
    // so the recommendation is "local" and BOTH deploys are queued.
    // (A single-GPU host never gets this pairing — see the
    // Cosmos-only variant below.)
    mockFetchEnvironment.mockResolvedValue(
      makeCaseAEnv({
        gpus: [GPU_80GB, GPU_80GB],
        recommended_embedding_mode: "local",
        embedding_deployment: EMBEDDING_FITS,
      }),
    );
    const { Wrapper } = createWrapper();
    const user = userEvent.setup();
    render(<div />, { wrapper: Wrapper });

    await screen.findByTestId("combined-card-local-and-hosted");
    // Hosted is checked by default; uncheck it for the local-only path.
    await user.click(screen.getByTestId("hosted-row-checkbox"));
    // The inline NGC field is required for the embedding NIM pull —
    // shown whenever Case A renders and env has no NGC key yet.
    await user.type(screen.getByTestId("ngc-key-input"), "ngc-good");
    await user.click(screen.getByRole("button", { name: /Set up & continue/i }));

    const ngc = await screen.findByTestId("ngc-page");
    expect(ngc.getAttribute("data-active-path")).toBe("local");
    expect(ngc.getAttribute("data-local-deploy-queued")).toBe(
      "nvidia/cosmos-reason2-8b,nvidia/llama-nemotron-embed-vl-1b-v2",
    );
    // No hosted-key validation happened because the hosted row was unchecked.
    expect(mockTestConnection).not.toHaveBeenCalled();
    expect(mockTestNvidiaCredential).not.toHaveBeenCalled();
    // NGC was persisted via setSecret.
    expect(mockSetSecret).toHaveBeenCalledWith({
      name: "NGC_API_KEY",
      value: "ngc-good",
      persist: true,
    });
  });

  it("Case A (single-GPU): [Set up & continue] queues Cosmos ONLY (the planned Teacher reserves the GPU)", async () => {
    // Placement-aware backend: on a single-GPU host the planned local
    // Teacher reserves the only device, so the embedding NIM has no
    // candidate (fits=false → recommendation "hosted") and the queue
    // carries just the Teacher. Embeddings fall back per the gate's
    // rules (hosted with a key, else pHash diversity).
    mockFetchEnvironment.mockResolvedValue(makeCaseAEnv());
    const { Wrapper } = createWrapper();
    const user = userEvent.setup();
    render(<div />, { wrapper: Wrapper });

    await screen.findByTestId("combined-card-local-and-hosted");
    // Hosted is checked by default; uncheck it for the local-only path.
    await user.click(screen.getByTestId("hosted-row-checkbox"));
    await user.type(screen.getByTestId("ngc-key-input"), "ngc-good");
    await user.click(screen.getByRole("button", { name: /Set up & continue/i }));

    const ngc = await screen.findByTestId("ngc-page");
    expect(ngc.getAttribute("data-active-path")).toBe("local");
    expect(ngc.getAttribute("data-local-deploy-queued")).toBe(
      "nvidia/cosmos-reason2-8b",
    );
  });

  it("Case A: hosted checked by default reveals the key input; an empty key keeps the button disabled and unchecking hides the input", async () => {
    mockFetchEnvironment.mockResolvedValue(makeCaseAEnv());
    const { Wrapper } = createWrapper();
    const user = userEvent.setup();
    render(<div />, { wrapper: Wrapper });

    await screen.findByTestId("combined-card-local-and-hosted");
    // Checked by default → input shown, and the empty key blocks continue.
    expect(screen.getByTestId("hosted-key-input")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Set up & continue/i })).toBeDisabled();

    // Unchecking hosted hides the key input and re-enables continue (local
    // is still checked).
    await user.click(screen.getByTestId("hosted-row-checkbox"));
    expect(screen.queryByTestId("hosted-key-input")).toBeNull();
  });

  it("Case A: both checked + valid key → activePath=local (local row promised 'as your Teacher'; the key adds hosted embeddings + alternates, it does not demote the local pick)", async () => {
    // Multi-GPU host, so the local-embedding recommendation is real
    // (see the queue-construction test above).
    mockFetchEnvironment.mockResolvedValue(
      makeCaseAEnv({
        gpus: [GPU_80GB, GPU_80GB],
        recommended_embedding_mode: "local",
        embedding_deployment: EMBEDDING_FITS,
      }),
    );
    mockTestNvidiaCredential.mockResolvedValue({ success: true });
    const { Wrapper } = createWrapper();
    const user = userEvent.setup();
    render(<div />, { wrapper: Wrapper });

    await screen.findByTestId("combined-card-local-and-hosted");
    // Both rows are checked by default; just fill in the two keys.
    await user.type(screen.getByTestId("hosted-key-input"), "nvapi-good");
    await user.type(screen.getByTestId("ngc-key-input"), "ngc-good");
    await user.click(screen.getByRole("button", { name: /Set up & continue/i }));

    const ngc = await screen.findByTestId("ngc-page");
    expect(ngc.getAttribute("data-active-path")).toBe("local");
    expect(ngc.getAttribute("data-local-deploy-queued")).toBe(
      "nvidia/cosmos-reason2-8b,nvidia/llama-nemotron-embed-vl-1b-v2",
    );
    expect(mockTestNvidiaCredential).toHaveBeenCalledWith("nvapi-good");
    expect(mockSetSecret).toHaveBeenCalledWith({
      name: "NVIDIA_API_KEY",
      value: "nvapi-good",
      persist: true,
    });
    // Both credentials persisted in the same atomic continue.
    expect(mockSetSecret).toHaveBeenCalledWith({
      name: "NGC_API_KEY",
      value: "ngc-good",
      persist: true,
    });
  });

  it("Case A: hosted-only (local unchecked, hosted checked) → activePath=hosted with no local deploys queued", async () => {
    // Multi-GPU host where the backend recommends local embeddings —
    // unchecking Local is an explicit opt-out of "use my GPU", so the
    // recommendation queues nothing.
    mockFetchEnvironment.mockResolvedValue(
      makeCaseAEnv({
        gpus: [GPU_80GB, GPU_80GB],
        recommended_embedding_mode: "local",
        embedding_deployment: EMBEDDING_FITS,
      }),
    );
    mockTestConnection.mockResolvedValue({ success: true });
    const { Wrapper } = createWrapper();
    const user = userEvent.setup();
    render(<div />, { wrapper: Wrapper });

    await screen.findByTestId("combined-card-local-and-hosted");
    await user.click(screen.getByTestId("local-row-checkbox")); // uncheck local
    // Hosted is already checked by default — leave it checked.
    // The NGC input lives in the Local row, so it is only visible when
    // Local is checked. Hosted-only means the SME has opted out of
    // local NIM containers entirely — the embedding
    // falls back to whatever the backend configures from env (hosted
    // NV-CLIP via the NVIDIA API key, or `none`). So we type only the
    // NVIDIA key and expect NGC NOT to be queried.
    expect(screen.queryByTestId("ngc-key-input")).toBeNull();
    await user.type(screen.getByTestId("hosted-key-input"), "nvapi-good");
    await user.click(screen.getByRole("button", { name: /Set up & continue/i }));

    const ngc = await screen.findByTestId("ngc-page");
    expect(ngc.getAttribute("data-active-path")).toBe("hosted");
    // No local deploys queued — Local is unchecked, so neither the local
    // Teacher nor the local embedding NIM are queued.
    expect(ngc.getAttribute("data-local-deploy-queued")).toBe("");
    // NVIDIA key was persisted; NGC was not.
    expect(mockSetSecret).toHaveBeenCalledWith({
      name: "NVIDIA_API_KEY",
      value: "nvapi-good",
      persist: true,
    });
    expect(mockSetSecret).not.toHaveBeenCalledWith(
      expect.objectContaining({ name: "NGC_API_KEY" }),
    );
  });

  it("Case A: bad hosted key shows inline error and blocks navigation (local deploy NOT queued)", async () => {
    mockFetchEnvironment.mockResolvedValue(makeCaseAEnv());
    mockTestNvidiaCredential.mockResolvedValue({
      success: false,
      error: "Invalid API key",
    });
    const { Wrapper } = createWrapper();
    const user = userEvent.setup();
    render(<div />, { wrapper: Wrapper });

    await screen.findByTestId("combined-card-local-and-hosted");
    // Hosted is checked by default; just paste the (bad) key.
    await user.type(screen.getByTestId("hosted-key-input"), "nvapi-bad");
    // NGC is also required; fill it so the button is clickable. The
    // bad hosted key still blocks navigation atomically — we expect NO
    // setSecret call for either credential because validation runs in
    // sequence and aborts on the NVIDIA failure.
    await user.type(screen.getByTestId("ngc-key-input"), "ngc-good");
    await user.click(screen.getByRole("button", { name: /Set up & continue/i }));

    expect(await screen.findByRole("alert")).toHaveTextContent(/Invalid API key/i);
    expect(screen.queryByTestId("ngc-page")).toBeNull();
    expect(mockSetSecret).not.toHaveBeenCalled();
  });

  it("Case A: persisted-bad NGC reveals input + error on mount (no click required)", async () => {
    mockFetchEnvironment.mockResolvedValue(
      makeCaseAEnv({ ngc_api_key_configured: true }),
    );
    mockTestNgcCredential.mockResolvedValue({
      success: false,
      error:
        "NGC key rejected by nvcr.io. The key needs the NGC Catalog and Private Registry scopes…",
    });
    const { Wrapper } = createWrapper();
    render(<div />, { wrapper: Wrapper });

    // The on-mount probe fires automatically. The input + error should
    // appear without any click. Use a wait to absorb the ~tick the
    // async useEffect takes.
    await screen.findByTestId("ngc-key-input");
    expect(await screen.findByRole("alert")).toHaveTextContent(
      /NGC key rejected by nvcr\.io/i,
    );
    // Backend was called without a credential value (effective-key mode) and
    // receives the abort signal that bounds this best-effort preflight.
    expect(mockTestNgcCredential).toHaveBeenCalledWith(
      undefined,
      expect.any(AbortSignal),
    );
  });

  it("Case D: persisted-bad NVIDIA blocks auto-skip + renders replacement card", async () => {
    mockFetchEnvironment.mockResolvedValue(
      makeEnv({
        nvidia_api_key_configured: true,
        local_deploy_available: false,
        gpus: [],
        recommended_teacher_mode: "hosted",
        recommended_embedding_mode: "hosted",
      }),
    );
    mockTestNvidiaCredential.mockResolvedValue({
      success: false,
      error: "NVIDIA API key was rejected by build.nvidia.com. Generate a new key…",
    });
    const { Wrapper } = createWrapper();
    render(<div />, { wrapper: Wrapper });

    // Auto-skip should NOT fire. Instead the replacement card renders.
    expect(await screen.findByTestId("nvidia-replacement-card")).toBeInTheDocument();
    expect(await screen.findByRole("alert")).toHaveTextContent(
      /NVIDIA API key was rejected/i,
    );
    // Ensure we did not navigate.
    expect(screen.queryByTestId("ngc-page")).toBeNull();
  });

  it("Case A: bad NGC key blocks navigation with inline error from the nvcr.io probe", async () => {
    mockFetchEnvironment.mockResolvedValue(makeCaseAEnv());
    // Simulate the live "wrong scope" path: probe says success=false with
    // the friendly backend error copy.
    mockTestNgcCredential.mockResolvedValue({
      success: false,
      error:
        "NGC key rejected by nvcr.io. The key needs the NGC Catalog and Private Registry scopes. Regenerate at https://org.ngc.nvidia.com/setup/api-key with both services enabled, then paste the new key.",
    });
    const { Wrapper } = createWrapper();
    const user = userEvent.setup();
    render(<div />, { wrapper: Wrapper });

    await screen.findByTestId("combined-card-local-and-hosted");
    // Uncheck hosted so the local-only NGC path drives the button.
    await user.click(screen.getByTestId("hosted-row-checkbox"));
    await user.type(screen.getByTestId("ngc-key-input"), "nvapi-wrongscope");
    await user.click(screen.getByRole("button", { name: /Set up & continue/i }));

    // Inline error surfaces; NGC key NOT persisted; no navigation.
    expect(await screen.findByRole("alert")).toHaveTextContent(
      /NGC key rejected by nvcr\.io/i,
    );
    expect(mockSetSecret).not.toHaveBeenCalledWith(
      expect.objectContaining({ name: "NGC_API_KEY" }),
    );
    expect(screen.queryByTestId("ngc-page")).toBeNull();
    // Probe was called with what the SME typed.
    expect(mockTestNgcCredential).toHaveBeenCalledWith("nvapi-wrongscope");
  });

  it("Case A: unchecking both options disables the button and shows helper text", async () => {
    mockFetchEnvironment.mockResolvedValue(makeCaseAEnv());
    const { Wrapper } = createWrapper();
    const user = userEvent.setup();
    render(<div />, { wrapper: Wrapper });

    await screen.findByTestId("combined-card-local-and-hosted");
    // Both rows start checked; uncheck both to reach the empty selection.
    await user.click(screen.getByTestId("local-row-checkbox"));
    await user.click(screen.getByTestId("hosted-row-checkbox"));

    expect(screen.getByRole("button", { name: /Set up & continue/i })).toBeDisabled();
    expect(screen.getByTestId("combined-helper")).toHaveTextContent(
      /Pick at least one option to continue/i,
    );
  });

  // ── Case B: key + GPU + local Teacher fits — local-first hybrid ─────────

  it("Case B (multi-GPU): recommends local for speed and explains the hosted bridge", async () => {
    mockFetchEnvironment.mockResolvedValue(
      makeCaseAEnv({
        nvidia_api_key_configured: true,
        gpus: [GPU_80GB, GPU_80GB],
        recommended_teacher_mode: "hosted",
      }),
    );
    const { Wrapper } = createWrapper();
    render(<div />, { wrapper: Wrapper });

    expect(await screen.findByText("Set up your Teacher")).toBeInTheDocument();

    const recommended = await screen.findByTestId("primary-card-local-recommended");
    expect(recommended).toHaveTextContent(/RECOMMENDED · FASTEST RESPONSES/);
    expect(recommended).toHaveTextContent(/Run Cosmos Reason2 8B locally/i);
    expect(recommended).toHaveTextContent(/Local responses are typically seconds/i);
    expect(recommended).not.toHaveTextContent(/hosted requests/i);
    expect(recommended).toHaveTextContent(/start with Step 3\.7 Flash now/i);
    expect(recommended).toHaveTextContent(
      /HOSTED FIRST · SWITCHES TO LOCAL AFTER VERIFICATION/i,
    );
    expect(recommended).toHaveTextContent(/Start now & deploy local Teacher/i);

    const hostedOnly = screen.getByTestId("hosted-only-card");
    expect(hostedOnly).toHaveTextContent(/ALTERNATIVE · HOSTED ONLY/);
    expect(hostedOnly).toHaveTextContent(/Keep Step 3\.7 Flash hosted/i);
    expect(hostedOnly).toHaveTextContent(/Use hosted only/i);

    // No navigation has fired — the SME must choose.
    expect(screen.queryByTestId("ngc-page")).toBeNull();
  });

  it("Case B (multi-GPU): accepting local collects NGC, persists it, and queues Cosmos behind hosted", async () => {
    mockFetchEnvironment.mockResolvedValue(
      makeCaseAEnv({
        nvidia_api_key_configured: true,
        gpus: [GPU_80GB, GPU_80GB],
        recommended_teacher_mode: "hosted",
      }),
    );
    const { Wrapper } = createWrapper();
    const user = userEvent.setup();
    render(<div />, { wrapper: Wrapper });

    // The recommended local card shows an inline NGC field because the
    // credential is missing, and its action stays disabled until filled.
    await screen.findByTestId("primary-card-local-recommended");
    expect(
      screen.getByRole("button", { name: /Start now & deploy local Teacher/i }),
    ).toBeDisabled();

    await user.type(screen.getByTestId("caseb-ngc-key-input"), "ngc-good");
    await user.click(
      screen.getByRole("button", { name: /Start now & deploy local Teacher/i }),
    );

    const ngc = await screen.findByTestId("ngc-page");
    expect(ngc.getAttribute("data-active-path")).toBe("hybrid");
    expect(ngc.getAttribute("data-local-deploy-queued")).toContain(
      "nvidia/cosmos-reason2-8b",
    );
    expect(mockSetSecret).toHaveBeenCalledWith({
      name: "NGC_API_KEY",
      value: "ngc-good",
      persist: true,
    });
  });

  it("Case B (multi-GPU): hosted-only sets activePath=hosted with an empty queue", async () => {
    mockFetchEnvironment.mockResolvedValue(
      makeCaseAEnv({
        nvidia_api_key_configured: true,
        gpus: [GPU_80GB, GPU_80GB],
        recommended_teacher_mode: "hosted",
      }),
    );
    const { Wrapper } = createWrapper();
    const user = userEvent.setup();
    render(<div />, { wrapper: Wrapper });

    await screen.findByTestId("hosted-only-card");
    await user.click(screen.getByRole("button", { name: /Use hosted only/i }));

    const ngc = await screen.findByTestId("ngc-page");
    expect(ngc.getAttribute("data-active-path")).toBe("hosted");
    expect(ngc.getAttribute("data-local-deploy-queued")).toBe("");
  });

  // ── One-NIM-per-GPU UI surface on single-GPU hosts ───────────────

  it("Case B (single-GPU + key): recommends local and keeps hosted-only available", async () => {
    mockFetchEnvironment.mockResolvedValue(
      makeCaseAEnv({
        nvidia_api_key_configured: true,
        recommended_teacher_mode: "hosted",
      }),
    );
    const { Wrapper } = createWrapper();
    render(<div />, { wrapper: Wrapper });

    const primary = await screen.findByTestId("primary-card-local-recommended");
    expect(primary).toHaveTextContent(/Run Cosmos Reason2 8B locally/i);
    expect(primary).toHaveTextContent(/RECOMMENDED · FASTEST RESPONSES/);
    expect(primary).not.toHaveTextContent(/HYBRID/);

    const hostedOnly = screen.getByTestId("hosted-only-card");
    expect(hostedOnly).toHaveTextContent(/ALTERNATIVE · HOSTED ONLY/);
    expect(hostedOnly).toHaveTextContent(/Keep Step 3\.7 Flash hosted/i);
  });

  it("Case B (single-GPU + key): local recommendation starts hosted and queues the local model", async () => {
    mockFetchEnvironment.mockResolvedValue(
      makeCaseAEnv({
        nvidia_api_key_configured: true,
        recommended_teacher_mode: "hosted",
      }),
    );
    const { Wrapper } = createWrapper();
    const user = userEvent.setup();
    render(<div />, { wrapper: Wrapper });

    await screen.findByTestId("primary-card-local-recommended");
    await user.type(screen.getByTestId("caseb-ngc-key-input"), "ngc-good");
    await user.click(
      screen.getByRole("button", { name: /Start now & deploy local Teacher/i }),
    );

    const ngc = await screen.findByTestId("ngc-page");
    expect(ngc.getAttribute("data-active-path")).toBe("hybrid");
    expect(ngc.getAttribute("data-local-deploy-queued")).toContain(
      "nvidia/cosmos-reason2-8b",
    );
  });

  it("Case B (single-GPU + key + nvidiaReplacementMode): replacement card wins over Teacher choices", async () => {
    // The persisted-bad-NVIDIA-key path must not render the
    // hybrid-recommended framing — the SME has a more urgent
    // problem (bad key) and needs the replacement input first.
    mockTestNvidiaCredential.mockResolvedValue({
      success: false,
      message: "NVIDIA API key was rejected",
    });
    mockFetchEnvironment.mockResolvedValue(
      makeCaseAEnv({
        nvidia_api_key_configured: true,
        recommended_teacher_mode: "hosted",
      }),
    );
    const { Wrapper } = createWrapper();
    render(<div />, { wrapper: Wrapper });

    // Wait for the on-mount probe to flip nvidiaReplacementMode.
    await waitFor(() => {
      expect(mockTestNvidiaCredential).toHaveBeenCalled();
    });
    await waitFor(() => {
      // The replacement card renders; the Teacher choices do not.
      expect(screen.queryByTestId("primary-card-local-recommended")).toBeNull();
    });
  });

  it("Case A (single-GPU): Local row describes Cosmos only (NeMo Retriever VL is NOT promised on a single GPU)", async () => {
    // On a single-GPU host the NeMo Retriever VL NIM can't co-locate
    // with Cosmos (one-NIM-per-GPU), so the row body branches on
    // env.gpus.length and must not promise it.
    mockFetchEnvironment.mockResolvedValue(makeCaseAEnv());
    const { Wrapper } = createWrapper();
    render(<div />, { wrapper: Wrapper });

    const row = await screen.findByTestId("local-row-label");
    expect(row).toHaveTextContent(/Cosmos Reason2 8B as your Teacher/i);
    expect(row).toHaveTextContent(/Local responses are typically seconds/i);
    expect(row).not.toHaveTextContent(/hosted requests/i);
    // Truthful copy: single-GPU does NOT promise NeMo Retriever VL.
    expect(row).not.toHaveTextContent(/NeMo Retriever VL/i);
  });

  it("Case A (multi-GPU): Local row promises Cosmos + NeMo Retriever VL + image-variety benefit", async () => {
    // The both-deploys promise requires the backend's local-embedding
    // recommendation — the row copy must match what buildLocalQueue
    // will actually queue, not just the GPU count.
    mockFetchEnvironment.mockResolvedValue(
      makeCaseAEnv({
        gpus: [GPU_80GB, GPU_80GB],
        recommended_embedding_mode: "local",
        embedding_deployment: EMBEDDING_FITS,
      }),
    );
    const { Wrapper } = createWrapper();
    render(<div />, { wrapper: Wrapper });

    const row = await screen.findByTestId("local-row-label");
    expect(row).toHaveTextContent(/Cosmos Reason2 8B as your Teacher/i);
    expect(row).toHaveTextContent(/NeMo Retriever VL/i);
    expect(row).toHaveTextContent(/Local embeddings improve image variety/i);
    expect(row).toHaveTextContent(/each on its own GPU/i);
  });

  it("Case A (multi-GPU, embedding NIM doesn't fit): Local row does NOT promise NeMo Retriever VL", async () => {
    // Heterogeneous host (e.g. 80 GB + 8 GB): the Teacher reserves
    // the big GPU and the leftover device is below the embedding
    // floor, so the placement-aware backend recommends hosted
    // embeddings. buildLocalQueue won't queue the embedding NIM, so
    // the row copy must not promise it — GPU count alone doesn't earn
    // the both-deploys pitch.
    mockFetchEnvironment.mockResolvedValue(
      makeCaseAEnv({
        gpus: [GPU_80GB, { name: "NVIDIA T4", memory_total_gb: 8 }],
        recommended_embedding_mode: "hosted",
      }),
    );
    const { Wrapper } = createWrapper();
    render(<div />, { wrapper: Wrapper });

    const row = await screen.findByTestId("local-row-label");
    expect(row).toHaveTextContent(/Cosmos Reason2 8B as your Teacher/i);
    expect(row).not.toHaveTextContent(/NeMo Retriever VL/i);
  });

  it("Case A: Hosted row explains free hosted models and embeddings without consuming the GPU", async () => {
    mockFetchEnvironment.mockResolvedValue(makeCaseAEnv());
    const { Wrapper } = createWrapper();
    render(<div />, { wrapper: Wrapper });

    const row = await screen.findByTestId("hosted-row-label");
    expect(row).toHaveTextContent(/A free NVIDIA API key/i);
    expect(row).toHaveTextContent(/hosted Teacher models/i);
    expect(row).toHaveTextContent(/hosted image embeddings/i);
    expect(row).toHaveTextContent(/improving review variety/i);
    expect(row).toHaveTextContent(/without using your GPU/i);
    expect(row).toHaveTextContent(/about 30 seconds/i);
    expect(row).not.toHaveTextContent("—");
    // Keep internal selector vocabulary out of the SME-facing value statement.
    expect(row).not.toHaveTextContent(/CLIP/);
    expect(row).not.toHaveTextContent(/perceptual/i);
    expect(row).not.toHaveTextContent(/pHash/i);
    expect(row).not.toHaveTextContent(/semantic/i);
  });

  // ── Case C: no key + no GPU (today's hosted-only) ────────────────────────

  it("Case C: renders today's hosted CTA when no key + no GPU", async () => {
    mockFetchEnvironment.mockResolvedValue(makeEnv());
    const { Wrapper } = createWrapper();
    render(<div />, { wrapper: Wrapper });

    await waitFor(() => {
      expect(
        screen.getByText(/Paste your NVIDIA API key to get started/i),
      ).toBeInTheDocument();
    });
    expect(screen.getByPlaceholderText("nvapi-...")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /^Go/i })).toBeDisabled();

    // Neither the local primary nor the hosted-only alternate should
    // render — there's no GPU.
    expect(screen.queryByTestId("primary-card-local")).toBeNull();
    expect(screen.queryByTestId("hosted-only-card")).toBeNull();
  });

  // ── Case D: key + no GPU ─────────────────────────────────────────────────

  it("Case D: auto-skips after validating the configured hosted key", async () => {
    mockFetchEnvironment.mockResolvedValue(
      makeEnv({
        nvidia_api_key_configured: true,
        default_teacher_model_name: "stepfun-ai/step-3.7-flash",
      }),
    );
    const { Wrapper } = createWrapper();
    render(<div />, { wrapper: Wrapper });

    const ngc = await screen.findByTestId("ngc-page");
    expect(ngc.getAttribute("data-came-from-auto-skip")).toBe("true");
    expect(ngc.getAttribute("data-active-path")).toBe("hosted");
  });

  it("Case D: shows progress instead of a blank page while the saved-key probe is pending", async () => {
    mockFetchEnvironment.mockResolvedValue(
      makeEnv({
        nvidia_api_key_configured: true,
        default_teacher_model_name: "stepfun-ai/step-3.7-flash",
      }),
    );
    mockTestNvidiaCredential.mockReturnValue(new Promise(() => {}));
    const { Wrapper } = createWrapper();
    render(<div />, { wrapper: Wrapper });

    expect(await screen.findByTestId("nvidia-key-probe-transition")).toHaveTextContent(
      /Checking your saved NVIDIA API key/i,
    );
    expect(screen.getByRole("status")).toBeInTheDocument();
  });

  // ── Hosted-key entry on Case A peer card and Case C primary ─────────────

  it("on hosted-key success (Case C): tests connection, sets secret, navigates with activePath=hosted", async () => {
    mockFetchEnvironment.mockResolvedValue(makeEnv());
    mockTestNvidiaCredential.mockResolvedValue({ success: true });
    const { Wrapper } = createWrapper();
    const user = userEvent.setup();
    render(<div />, { wrapper: Wrapper });

    const input = await screen.findByPlaceholderText("nvapi-...");
    await user.type(input, "nvapi-good");
    await user.click(screen.getByRole("button", { name: /^Go/i }));

    const ngc = await screen.findByTestId("ngc-page");
    expect(ngc.getAttribute("data-active-path")).toBe("hosted");
    expect(mockTestNvidiaCredential).toHaveBeenCalledWith("nvapi-good");
    expect(mockSetSecret).toHaveBeenCalledWith({
      name: "NVIDIA_API_KEY",
      value: "nvapi-good",
      persist: true,
    });
  });

  it("on hosted test failure: inline error renders, field value preserved, no setSecret/navigation", async () => {
    mockFetchEnvironment.mockResolvedValue(makeEnv());
    mockTestNvidiaCredential.mockResolvedValue({
      success: false,
      error: "Invalid API key",
    });
    const { Wrapper } = createWrapper();
    const user = userEvent.setup();
    render(<div />, { wrapper: Wrapper });

    const input = await screen.findByPlaceholderText("nvapi-...");
    await user.type(input, "nvapi-bad");
    await user.click(screen.getByRole("button", { name: /^Go/i }));

    await waitFor(() => {
      expect(screen.getByRole("alert")).toHaveTextContent(/Invalid API key/i);
    });
    expect((input as HTMLInputElement).value).toBe("nvapi-bad");
    expect(mockSetSecret).not.toHaveBeenCalled();
    expect(screen.queryByTestId("ngc-page")).toBeNull();
  });

  it("surfaces the 'add to .env' hint when setSecret returns 403", async () => {
    mockFetchEnvironment.mockResolvedValue(makeEnv());
    mockTestNvidiaCredential.mockResolvedValue({ success: true });
    mockSetSecret.mockRejectedValue(new Error("403: ui_secret_persist_disabled"));
    const { Wrapper } = createWrapper();
    const user = userEvent.setup();
    render(<div />, { wrapper: Wrapper });

    const input = await screen.findByPlaceholderText("nvapi-...");
    await user.type(input, "nvapi-good");
    await user.click(screen.getByRole("button", { name: /^Go/i }));

    await waitFor(() => {
      expect(screen.getByRole("alert")).toHaveTextContent(
        /add NVIDIA_API_KEY=\.\.\. to ~\/\.vlm_feedback_loop\/\.env and restart/i,
      );
    });
    expect(mockSetSecret).toHaveBeenCalled();
    expect(screen.queryByTestId("ngc-page")).toBeNull();
  });

  it("editing the field after a failed hosted Go clears the error", async () => {
    mockFetchEnvironment.mockResolvedValue(makeEnv());
    mockTestNvidiaCredential.mockResolvedValue({
      success: false,
      error: "Invalid API key",
    });
    const { Wrapper } = createWrapper();
    const user = userEvent.setup();
    render(<div />, { wrapper: Wrapper });

    const input = await screen.findByPlaceholderText("nvapi-...");
    await user.type(input, "bad");
    await user.click(screen.getByRole("button", { name: /^Go/i }));
    await screen.findByRole("alert");

    await user.type(input, "x");
    expect(screen.queryByRole("alert")).toBeNull();
  });
});
