// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, it, expect, vi, beforeEach } from "vitest";
import { installEventSourceMock } from "@/test/event-source-mock";
import { makeProjectResponse } from "@/test/fixtures";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter, Routes, Route, Outlet } from "react-router-dom";

import { BatchPreRunPage } from "../BatchPreRunPage";

// ── Mocks ──────────────────────────────────────────────────────────────────

vi.mock("@/api/batch", () => ({
  createBatchLabelRun: vi.fn(),
}));

vi.mock("@/api/evaluation", () => ({
  fetchScaleUpGate: vi.fn(),
}));

vi.mock("@/api/guidance", () => ({
  fetchIclCount: vi.fn(),
  fetchGuidance: vi.fn(),
}));

vi.mock("@/api/model-configs", () => ({
  fetchModelConfigs: vi.fn(),
}));

vi.mock("@/pages/ProjectSetupLayout", () => ({
  useSetupContext: vi.fn(),
}));

installEventSourceMock();

import { createBatchLabelRun } from "@/api/batch";
import { fetchScaleUpGate } from "@/api/evaluation";
import { fetchGuidance, fetchIclCount } from "@/api/guidance";
import { fetchModelConfigs } from "@/api/model-configs";
import { useSetupContext } from "@/pages/ProjectSetupLayout";

const mockCreate = createBatchLabelRun as ReturnType<typeof vi.fn>;
const mockFetchGate = fetchScaleUpGate as ReturnType<typeof vi.fn>;
const mockFetchIclCount = fetchIclCount as ReturnType<typeof vi.fn>;
const mockFetchModelConfigs = fetchModelConfigs as ReturnType<typeof vi.fn>;
const mockFetchGuidance = fetchGuidance as ReturnType<typeof vi.fn>;
const mockSetup = useSetupContext as ReturnType<typeof vi.fn>;

// The page reads example-state counts straight from the setup context's
// project detail (``ProjectResponse.counts``) — no project-list fetch.
const PROJECT = makeProjectResponse({
  project_id: "pid-1",
  name: "Test",
  active_guidance_id: "g-1",
  teacher_model_config_id: "mc-1",
  labeling_generation_preset_key: "precise",
  thinking_default_on: true,
  visual_budget_preset_key: "balanced",
  counts: {
    unlabeled: 100,
    auto_labeled: 50,
    verified: 20,
    omitted: 5,
    pending_relabel: 0,
  },
});

/** Re-stub the setup context with count overrides for empty-state tests. */
function setupWithCounts(counts: Partial<typeof PROJECT.counts>) {
  mockSetup.mockReturnValue({
    projectId: "pid-1",
    project: { ...PROJECT, counts: { ...PROJECT.counts, ...counts } },
    environment: {},
  });
}

let qc: QueryClient;

beforeEach(() => {
  vi.clearAllMocks();
  qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  mockSetup.mockReturnValue({
    projectId: "pid-1",
    project: PROJECT,
    environment: {},
  });
  // Default gate=ready; individual tests override for gate-not-ready
  // scenarios.
  mockFetchGate.mockResolvedValue({
    gate_status: "ready",
    criteria: [],
    evaluated_at: "2026-04-16T00:00:00Z",
  });
  mockFetchIclCount.mockResolvedValue({ eligible_count: 12 });
  mockFetchModelConfigs.mockResolvedValue({
    items: [
      {
        model_config_id: "mc-1",
        model_name: "nvidia/cosmos-reason2-8b",
        endpoint_id: "ep-1",
        eligible_roles: ["teacher"],
      },
    ],
    next_cursor: null,
  });
  mockFetchGuidance.mockResolvedValue({
    guidance_id: "g-1",
    version_number: 3,
    description: "test",
    rules: "",
    schema: { fields: [] },
    created_at: "2026-04-16T00:00:00Z",
  });
  mockCreate.mockResolvedValue({
    run_id: "run-1",
    run_type: "batch_label_run",
    status: "queued",
    examples_total: 100,
    created_at: "2026-04-16T00:00:00Z",
  });
});

function renderPage(routerState?: Record<string, unknown>) {
  return render(
    <QueryClientProvider client={qc}>
      <MemoryRouter
        initialEntries={[
          { pathname: "/projects/pid-1/batch-prerun", state: routerState ?? null },
        ]}
      >
        <Routes>
          <Route path="/projects/:projectId" element={<Outlet />}>
            <Route path="batch-prerun" element={<BatchPreRunPage />} />
            <Route
              path="batch-status/:runId"
              element={<div data-testid="status-page" />}
            />
          </Route>
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

// ── Tests ──────────────────────────────────────────────────────────────────

describe("BatchPreRunPage", () => {
  it("shows configuration snapshot with resolved Teacher name and Guidance version", async () => {
    renderPage();
    await waitFor(() => {
      expect(screen.getByText("Configuration")).toBeInTheDocument();
      // The snapshot shows the model name (not the UUID) and v{N}.
      expect(screen.getByText("nvidia/cosmos-reason2-8b")).toBeInTheDocument();
      expect(screen.getByText("v3")).toBeInTheDocument();
      // Preset values display title-cased (shared titleCasePreset).
      expect(screen.getByText("Precise")).toBeInTheDocument();
    });
  });

  it("shows auto-labeled notice", async () => {
    renderPage();
    await waitFor(() => {
      expect(
        screen.getByText(/Auto-Labeled outputs.*not ground truth/),
      ).toBeInTheDocument();
    });
  });

  it("shows input count from project", async () => {
    renderPage();
    await waitFor(() => {
      expect(screen.getByText(/100 Unlabeled/)).toBeInTheDocument();
    });
  });

  it("launches batch run and navigates", async () => {
    renderPage();
    // The Launch button is always rendered — wait for the gate
    // query to resolve so the button is enabled before we click.
    await waitFor(() => {
      expect(screen.getByTestId("launch-batch")).not.toBeDisabled();
    });
    fireEvent.click(screen.getByTestId("launch-batch"));
    await waitFor(() => {
      expect(mockCreate).toHaveBeenCalledWith("pid-1", {
        include_auto_labeled: false,
      });
    });
    // Should navigate to status page
    await waitFor(() => {
      expect(screen.getByTestId("status-page")).toBeInTheDocument();
    });
  });

  it("restart-with-prompt-only carries the forced mode into the launch request", async () => {
    // The run-status screen's structured-generation-rejection recovery
    // navigates here with router state; the relaunch must post that
    // mode instead of falling back to the project default — the exact
    // mode that just failed.
    renderPage({ structured_generation_mode: "prompt_only" });
    await waitFor(() => {
      expect(screen.getByTestId("launch-batch")).not.toBeDisabled();
    });
    // The locked-config snapshot surfaces the forced mode for
    // verification before launch.
    expect(screen.getByText("Structured Generation")).toBeInTheDocument();
    expect(screen.getByText(/Prompt-only/)).toBeInTheDocument();
    fireEvent.click(screen.getByTestId("launch-batch"));
    await waitFor(() => {
      expect(mockCreate).toHaveBeenCalledWith("pid-1", {
        include_auto_labeled: false,
        structured_generation_mode: "prompt_only",
      });
    });
  });

  it("shows no-unlabeled warning when count is zero", async () => {
    setupWithCounts({ unlabeled: 0 });
    renderPage();
    await waitFor(() => {
      expect(screen.getByTestId("no-unlabeled-warning")).toBeInTheDocument();
    });
  });

  it("renders the three-button footer on the no-unlabeled empty state", async () => {
    setupWithCounts({ unlabeled: 0, auto_labeled: 341 });
    renderPage();
    await waitFor(() => {
      expect(screen.getByTestId("batch-add-images")).toBeInTheDocument();
    });
    // All three buttons must be present, with Run visibly disabled.
    expect(screen.getByRole("button", { name: /cancel/i })).toBeInTheDocument();
    const run = screen.getByTestId("launch-batch");
    expect(run).toBeInTheDocument();
    expect(run).toBeDisabled();
  });

  it("shows '0 Unlabeled images · N Auto-Labeled available' when Unlabeled=0 and Auto-Labeled>0", async () => {
    setupWithCounts({ unlabeled: 0, auto_labeled: 341 });
    renderPage();
    await waitFor(() => {
      expect(
        screen.getByText(/0 Unlabeled images · 341 Auto-Labeled available/),
      ).toBeInTheDocument();
    });
  });

  // ── Scale-Up gate + ICL line ──────────────────────────────────────

  it("disables [Run Batch Labeling] when the Scale-Up gate is not ready", async () => {
    mockFetchGate.mockResolvedValueOnce({
      gate_status: "not_ready",
      criteria: [],
      evaluated_at: "2026-04-16T00:00:00Z",
    });
    renderPage();
    // Wait until the gate-driven helper appears — the Launch button is
    // always rendered, so waiting only on its presence is not enough
    // to guarantee the gate query has resolved.
    await waitFor(() => {
      expect(screen.getByTestId("batch-gate-helper")).toBeInTheDocument();
    });
    expect(screen.getByTestId("launch-batch")).toBeDisabled();
    expect(screen.getByTestId("batch-gate-helper")).toHaveTextContent(/Gate not ready/);
  });

  it("includes the ICL line in the config card", async () => {
    mockFetchIclCount.mockResolvedValueOnce({ eligible_count: 12 });
    renderPage();
    await waitFor(() => {
      expect(screen.getByText(/12 edits/)).toBeInTheDocument();
    });
    // Matches wireframe's "ICL: 12 edits" formatting.
    expect(screen.getByText(/^ICL$/)).toBeInTheDocument();
  });
});
