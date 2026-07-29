// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Unit tests for the Confirm Model Defaults page.
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { installEventSourceMock } from "@/test/event-source-mock";
import { makeEnvironmentResponse, makeProjectResponse } from "@/test/fixtures";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter, Routes, Route, useLocation } from "react-router-dom";
import type { ReactNode } from "react";
import { ConfirmDefaultsPage } from "@/pages/ConfirmDefaultsPage";
import { ProjectSetupLayout } from "@/pages/ProjectSetupLayout";
import { projectKeys } from "@/api/query-keys";

// ── Mocks ───────────────────────────────────────────────────────────────────

const mockFetchProject = vi.fn();
const mockFetchEnvironment = vi.fn();
const mockFetchModelConfigs = vi.fn();
const mockUpdateProject = vi.fn();

vi.mock("@/api/projects", () => ({
  fetchProject: (...args: unknown[]) => mockFetchProject(...args),
  fetchProjectList: vi.fn().mockResolvedValue({ items: [], next_cursor: null }),
  createProject: vi.fn(),
  markSetupCompleted: vi.fn().mockResolvedValue({
    transitioned: true,
    project: {},
  }),
}));

vi.mock("@/api/nim", () => ({
  fetchEnvironment: (...args: unknown[]) => mockFetchEnvironment(...args),
}));

vi.mock("@/api/model-configs", () => ({
  fetchModelConfigs: (...args: unknown[]) => mockFetchModelConfigs(...args),
  updateProject: (...args: unknown[]) => mockUpdateProject(...args),
}));

installEventSourceMock();

// ── Fixtures ────────────────────────────────────────────────────────────────

const PROJECT = makeProjectResponse({
  teacher_model_config_id: "mc-cosmos-8b",
  active_guidance_id: null,
});

const ENVIRONMENT = makeEnvironmentResponse({
  hosted_nim_available: true,
  nvidia_api_key_configured: true,
  recommended_teacher_mode: "hosted",
  recommended_embedding_mode: "hosted",
  // The preselect fallback follows this backend-provided default name
  // (not a hardcoded UI copy). Pinned to Step 3.7 Flash here because it
  // has a distinctive badge set (always-on reasoning → "thinking" pill,
  // no visual budget) that lets the preselect assertions below verify
  // the wiring end to end.
  default_teacher_model_name: "stepfun-ai/step-3.7-flash",
});

const MODEL_CONFIGS = [
  {
    model_config_id: "mc-qwen-3-5",
    project_id: "test-pid",
    endpoint_id: "ep-1",
    model_name: "qwen/qwen3.5-397b-a17b",
    context_window_tokens: 262144,
    eligible_roles: ["teacher"],
    supports_image_input: true,
    structured_generation_support: "unknown" as const,
    thinking_toggle_mode: "qwen_enable_thinking",
    thinking_toggle_support: "unknown" as const,
    visual_budget_mode: "none",
    visual_budget_support: "unsupported" as const,
    model_quantization: null,
    nim_model_profile: null,
    nim_profile_metadata: null,
    local_deploy_metadata: null,
    created_at: "2026-04-23T10:00:00Z",
  },
  {
    model_config_id: "mc-kimi-k2-5",
    project_id: "test-pid",
    endpoint_id: "ep-1",
    model_name: "moonshotai/kimi-k2.5",
    context_window_tokens: 262144,
    eligible_roles: ["teacher"],
    supports_image_input: true,
    structured_generation_support: "unknown" as const,
    thinking_toggle_mode: "kimi_thinking",
    thinking_toggle_support: "unknown" as const,
    visual_budget_mode: "none",
    visual_budget_support: "unsupported" as const,
    model_quantization: null,
    nim_model_profile: null,
    nim_profile_metadata: null,
    local_deploy_metadata: null,
    created_at: "2026-04-23T10:00:00Z",
  },
  {
    model_config_id: "mc-cosmos-8b",
    project_id: "test-pid",
    endpoint_id: "ep-1",
    model_name: "nvidia/cosmos-reason2-8b",
    context_window_tokens: 256000,
    eligible_roles: ["teacher", "student_base"],
    supports_image_input: true,
    structured_generation_support: "unknown" as const,
    thinking_toggle_mode: "qwen_enable_thinking",
    thinking_toggle_support: "unknown" as const,
    visual_budget_mode: "mm_processor_size",
    visual_budget_support: "unknown" as const,
    model_quantization: null,
    nim_model_profile: null,
    nim_profile_metadata: null,
    local_deploy_metadata: null,
    created_at: "2026-04-13T10:00:00Z",
  },
  {
    model_config_id: "mc-mistral",
    project_id: "test-pid",
    endpoint_id: "ep-1",
    model_name: "mistralai/mistral-large-3-675b-instruct-2512",
    context_window_tokens: 262144,
    eligible_roles: ["teacher"],
    supports_image_input: true,
    structured_generation_support: "unknown" as const,
    thinking_toggle_mode: "none",
    thinking_toggle_support: "unsupported" as const,
    visual_budget_mode: "none",
    visual_budget_support: "unsupported" as const,
    model_quantization: null,
    nim_model_profile: null,
    nim_profile_metadata: null,
    local_deploy_metadata: null,
    created_at: "2026-04-13T10:00:00Z",
  },
  {
    model_config_id: "mc-step",
    project_id: "test-pid",
    endpoint_id: "ep-1",
    model_name: "stepfun-ai/step-3.7-flash",
    context_window_tokens: 262144,
    eligible_roles: ["teacher"],
    supports_image_input: true,
    structured_generation_support: "supported" as const,
    thinking_toggle_mode: "always_on_reasoning",
    thinking_toggle_support: "unsupported" as const,
    visual_budget_mode: "none",
    visual_budget_support: "unsupported" as const,
    model_quantization: null,
    nim_model_profile: null,
    nim_profile_metadata: null,
    local_deploy_metadata: null,
    created_at: "2026-07-21T10:00:00Z",
  },
  {
    model_config_id: "mc-kimi",
    project_id: "test-pid",
    endpoint_id: "ep-1",
    model_name: "moonshotai/kimi-k2-thinking",
    context_window_tokens: 256000,
    // Teacher-eligible but text-only — exercises the dropdown's
    // supports_image_input filter (it must be excluded).
    eligible_roles: ["teacher"],
    supports_image_input: false,
    structured_generation_support: "unknown" as const,
    thinking_toggle_mode: "kimi_thinking",
    thinking_toggle_support: "unknown" as const,
    visual_budget_mode: "none",
    visual_budget_support: "unsupported" as const,
    model_quantization: null,
    nim_model_profile: null,
    nim_profile_metadata: null,
    local_deploy_metadata: null,
    created_at: "2026-04-13T10:00:00Z",
  },
];

// ── Wrapper ─────────────────────────────────────────────────────────────────

/** Landing-route stub exposing the router state the page arrived with,
 *  so tests can assert on the setupAutoSkip acknowledgment payload. */
function LandingStub({ testId }: { testId: string }) {
  const location = useLocation();
  return (
    <div data-testid={testId}>
      <div data-testid="landing-state">{JSON.stringify(location.state ?? null)}</div>
    </div>
  );
}

function createWrapper(initialPath = "/projects/test-pid/confirm-defaults") {
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
              <Route path="setup" element={<div data-testid="setup-page">Setup</div>} />
              <Route path="confirm-defaults" element={<ConfirmDefaultsPage />} />
              <Route path="ready" element={<LandingStub testId="ready-page" />} />
              <Route path="labeling" element={<LandingStub testId="labeling-page" />} />
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

describe("ConfirmDefaultsPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockFetchProject.mockResolvedValue(PROJECT);
    mockFetchEnvironment.mockResolvedValue(ENVIRONMENT);
    mockFetchModelConfigs.mockResolvedValue({ items: MODEL_CONFIGS });
  });

  // --- Confirm screen renders (AC-2) ---

  it("renders the Teacher dropdown when config needs confirmation", async () => {
    // Make defaults "invalid" by providing a project with no matching teacher
    mockFetchProject.mockResolvedValue({
      ...PROJECT,
      teacher_model_config_id: "nonexistent",
    });
    const { Wrapper } = createWrapper();

    render(<div />, { wrapper: Wrapper });

    await waitFor(() => {
      expect(screen.getByText("Confirm Model Defaults")).toBeInTheDocument();
    });

    expect(screen.getByText("Teacher")).toBeInTheDocument();
  });

  // --- Teacher dropdown filters correctly (AC-2) ---

  it("teacher dropdown only shows models with teacher role and image input", async () => {
    mockFetchProject.mockResolvedValue({
      ...PROJECT,
      teacher_model_config_id: "nonexistent",
    });
    const { Wrapper } = createWrapper();

    render(<div />, { wrapper: Wrapper });

    await waitFor(() => {
      expect(screen.getByText("Confirm Model Defaults")).toBeInTheDocument();
    });

    // Teacher dropdown should include nvidia/cosmos-reason2-8b and mistral (both have teacher + image)
    // but NOT kimi-k2-thinking (no image input)
    const user = userEvent.setup();
    const teacherTrigger = screen.getByTestId("teacher-select");
    await user.click(teacherTrigger);
    const options = await screen.findAllByRole("option");
    const optionNames = options.map((o) => o.textContent);
    expect(optionNames).toContain("nvidia/cosmos-reason2-8b");
    expect(optionNames).toContain("mistralai/mistral-large-3-675b-instruct-2512");
    expect(optionNames).not.toContain("moonshotai/kimi-k2-thinking");
  });

  // --- Continue saves (AC-4) ---

  it("[Continue] calls updateProject with the selected model ID", async () => {
    mockFetchProject.mockResolvedValue({
      ...PROJECT,
      teacher_model_config_id: "nonexistent",
    });
    mockUpdateProject.mockResolvedValue(PROJECT);
    const { Wrapper } = createWrapper();
    const user = userEvent.setup();

    render(<div />, { wrapper: Wrapper });

    await waitFor(() => {
      expect(screen.getByText("Confirm Model Defaults")).toBeInTheDocument();
    });

    await user.click(screen.getByRole("button", { name: /Continue/i }));

    // The orphan teacher ID falls back to the seeded default (Step 3.7
    // Flash, mc-step).
    await waitFor(() => {
      expect(mockUpdateProject).toHaveBeenCalledWith(
        "test-pid",
        expect.objectContaining({
          teacher_model_config_id: "mc-step",
        }),
      );
    });
  });

  // --- Back navigation (AC-5) ---

  it("[Back] navigates to setup page", async () => {
    mockFetchProject.mockResolvedValue({
      ...PROJECT,
      teacher_model_config_id: "nonexistent",
    });
    const { Wrapper } = createWrapper();
    const user = userEvent.setup();

    render(<div />, { wrapper: Wrapper });

    await waitFor(() => {
      expect(screen.getByText("Confirm Model Defaults")).toBeInTheDocument();
    });

    await user.click(screen.getByRole("button", { name: /Back/i }));

    await waitFor(() => {
      expect(screen.getByTestId("setup-page")).toBeInTheDocument();
    });
  });

  // --- Capability badges ---

  it("shows capability badges for selected model", async () => {
    // A valid teacher ID would auto-skip, so force the confirm screen
    // via an orphan ID — the preselect falls back to Step 3.7 Flash
    // (vision-capable) and its badges render.
    mockFetchProject.mockResolvedValue({
      ...PROJECT,
      teacher_model_config_id: "nonexistent",
    });
    const { Wrapper } = createWrapper();

    render(<div />, { wrapper: Wrapper });

    await waitFor(() => {
      expect(screen.getByText("Confirm Model Defaults")).toBeInTheDocument();
    });

    // Look for capability badges from Step 3.7 Flash (the preselected teacher)
    expect(screen.getAllByText("vision").length).toBeGreaterThan(0);
  });

  // --- Preselect fallback ---
  // When project state has a null model ID, the form MUST preselect the
  // seeded default by the model name the backend reports in
  // ``environment.default_teacher_model_name`` (here pinned to
  // stepfun-ai/step-3.7-flash by the ENVIRONMENT fixture). The
  // preselect-by-name logic still has to work (projects with null fields
  // exist after partial init).

  it("preselects the seeded default by model name when project state has a null model ID", async () => {
    mockFetchProject.mockResolvedValue({
      ...PROJECT,
      teacher_model_config_id: null,
    });
    const { Wrapper } = createWrapper();

    render(<div />, { wrapper: Wrapper });

    await waitFor(() => {
      expect(screen.getByText("Confirm Model Defaults")).toBeInTheDocument();
    });

    // Step 3.7 Flash is the default the ENVIRONMENT fixture pins here:
    // vision-capable, reasoning always-on (mode="always_on_reasoning" →
    // the thinking pill DOES render, matching the Omni precedent), no
    // visual-budget controls. Exactly one badge row renders (the selected
    // model's); the "visual budget" pill belongs to other catalog entries
    // (Cosmos 8B) and MUST NOT appear at preselect time.
    await waitFor(() => {
      expect(screen.getAllByText("vision").length).toBe(1);
    });
    expect(screen.queryAllByText("thinking").length).toBe(1);
    expect(screen.queryAllByText("visual budget").length).toBe(0);
  });

  // --- Preselect fallback for orphan stored IDs ---
  // Shown when defaults are missing, invalid, or the user connected a
  // custom endpoint where seeded models may not exist. When
  // the stored ID is a valid UUID string but no matching entry exists in
  // the role-filtered options (e.g. the previously-selected model was
  // archived, or a custom endpoint replaced the seeded catalog), the page
  // MUST fall back to the seeded default rather than letting KUI Select
  // render the raw UUID and hide the capability-pill row.

  it("preselects the seeded default when the stored model ID is orphan (not in role-filtered options)", async () => {
    // Project has a stored ID that does not match anything in MODEL_CONFIGS.
    mockFetchProject.mockResolvedValue({
      ...PROJECT,
      teacher_model_config_id: "deadbeef-orphan-teacher",
    });
    const { Wrapper } = createWrapper();

    render(<div />, { wrapper: Wrapper });

    await waitFor(() => {
      expect(screen.getByText("Confirm Model Defaults")).toBeInTheDocument();
    });

    // The dropdown MUST resolve to Step 3.7 Flash's pills (vision +
    // always-on thinking) — not the raw orphan UUID.
    await waitFor(() => {
      expect(screen.getAllByText("vision").length).toBe(1);
    });
    expect(screen.queryAllByText("thinking").length).toBe(1);
    expect(screen.queryByText(/deadbeef-orphan/)).not.toBeInTheDocument();
  });

  // --- Auto-skip acknowledgment payload ---
  // A fully-auto-skipped setup chain never shows the SME a setup
  // screen, so the auto-skip navigation MUST forward the auto-selected
  // configuration through router state for the landing screen's
  // acknowledgment banner. The default PROJECT fixture has a valid
  // teacher ID, so the page auto-skips on mount.

  it("auto-skip forwards the setupAutoSkip acknowledgment payload to the landing screen", async () => {
    const { Wrapper } = createWrapper();

    render(<div />, { wrapper: Wrapper });

    // active_guidance_id is null → lands on ready.
    await screen.findByTestId("ready-page");
    const raw = screen.getByTestId("landing-state").textContent ?? "null";
    const state = JSON.parse(raw) as {
      setupAutoSkip?: Record<string, unknown>;
    } | null;
    expect(state?.setupAutoSkip).toEqual({
      teacherMode: "hosted",
      embeddingMode: "hosted",
      teacherName: "nvidia/cosmos-reason2-8b",
    });
  });

  it("manual [Continue] does NOT forward the acknowledgment payload (the SME saw this screen)", async () => {
    mockFetchProject.mockResolvedValue({
      ...PROJECT,
      teacher_model_config_id: "nonexistent",
    });
    mockUpdateProject.mockResolvedValue(PROJECT);
    const { Wrapper } = createWrapper();
    const user = userEvent.setup();

    render(<div />, { wrapper: Wrapper });

    await waitFor(() => {
      expect(screen.getByText("Confirm Model Defaults")).toBeInTheDocument();
    });
    await user.click(screen.getByRole("button", { name: /Continue/i }));

    await screen.findByTestId("ready-page");
    const raw = screen.getByTestId("landing-state").textContent ?? "null";
    const state = JSON.parse(raw) as {
      setupAutoSkip?: Record<string, unknown>;
    } | null;
    expect(state?.setupAutoSkip).toBeUndefined();
  });

  // --- Setup-completion cache invalidation ---
  // The NIM Configuration header link gates on the cached project's
  // setup_completed_at. Completing setup MUST invalidate the project
  // detail query AFTER the stamp lands, or the link stays hidden on
  // the landing screen until an unrelated refetch.

  it("auto-skip invalidates the project detail cache after stamping setup completion", async () => {
    const { Wrapper, queryClient } = createWrapper();
    const invalidateSpy = vi.spyOn(queryClient, "invalidateQueries");

    render(<div />, { wrapper: Wrapper });

    await screen.findByTestId("ready-page");
    await waitFor(() => {
      expect(invalidateSpy).toHaveBeenCalledWith({
        queryKey: projectKeys.detail("test-pid"),
      });
    });
  });
});
