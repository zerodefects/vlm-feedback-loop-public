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

vi.mock("@/api/nim", () => ({
  fetchNimEndpoints: vi.fn(),
}));

vi.mock("@/pages/setup-context", () => ({
  useSetupContext: vi.fn(),
}));

installEventSourceMock();

import { createBatchLabelRun } from "@/api/batch";
import { fetchScaleUpGate } from "@/api/evaluation";
import { fetchGuidance, fetchIclCount } from "@/api/guidance";
import { fetchModelConfigs } from "@/api/model-configs";
import { fetchNimEndpoints } from "@/api/nim";
import { useSetupContext } from "@/pages/setup-context";

const mockCreate = createBatchLabelRun as ReturnType<typeof vi.fn>;
const mockFetchGate = fetchScaleUpGate as ReturnType<typeof vi.fn>;
const mockFetchIclCount = fetchIclCount as ReturnType<typeof vi.fn>;
const mockFetchModelConfigs = fetchModelConfigs as ReturnType<typeof vi.fn>;
const mockFetchGuidance = fetchGuidance as ReturnType<typeof vi.fn>;
const mockFetchNimEndpoints = fetchNimEndpoints as ReturnType<typeof vi.fn>;
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
        structured_generation_support: "supported",
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
  mockFetchNimEndpoints.mockResolvedValue({
    items: [
      {
        endpoint_id: "ep-1",
        endpoint_mode: "self_hosted",
        source_kind: "user_configured",
        usage_policy: "operator_managed",
      },
    ],
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

  it("does not claim current Guidance exists when none is active", async () => {
    mockSetup.mockReturnValue({
      projectId: "pid-1",
      project: { ...PROJECT, active_guidance_id: null },
      environment: {},
    });
    renderPage();
    fireEvent.click(await screen.findByTestId("advanced-toggle"));

    expect(
      screen.getByText(
        "Activate Guidance before re-labeling existing Auto-Labeled results.",
      ),
    ).toBeInTheDocument();
    expect(screen.queryByText(/Re-run with current Guidance/)).toBeNull();
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
        structured_generation_mode: "auto",
        icl_mode: "enabled",
      });
    });
    // Should navigate to status page
    await waitFor(() => {
      expect(screen.getByTestId("status-page")).toBeInTheDocument();
    });
  });

  it("requires explicit evaluation confirmation for the NVIDIA API Catalog endpoint", async () => {
    mockFetchNimEndpoints.mockResolvedValueOnce({
      items: [
        {
          endpoint_id: "ep-1",
          endpoint_mode: "hosted",
          source_kind: "seeded_hosted",
          usage_policy: "evaluation_only",
        },
      ],
    });
    renderPage();

    const launch = screen.getByTestId("launch-batch");
    await waitFor(() => expect(launch).not.toBeDisabled());
    expect(screen.getByText(/NVIDIA API Catalog · evaluation only/i)).toBeVisible();

    fireEvent.click(launch);
    const warning = await screen.findByTestId("evaluation-endpoint-warning");
    expect(warning).toHaveTextContent(/up to 100 images/i);
    expect(warning).toHaveTextContent(/additional trial credits.*do not authorize/i);
    expect(mockCreate).not.toHaveBeenCalled();

    fireEvent.click(screen.getByTestId("confirm-evaluation-batch"));
    await waitFor(() => expect(mockCreate).toHaveBeenCalledTimes(1));
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
        icl_mode: "enabled",
      });
    });
  });

  it("lets the SME snapshot the advanced run limit, structured mode, and ICL mode", async () => {
    renderPage();
    fireEvent.click(await screen.findByTestId("advanced-toggle"));

    fireEvent.change(screen.getByTestId("batch-run-limit"), {
      target: { value: "20" },
    });
    fireEvent.change(screen.getByTestId("batch-ingested-after"), {
      target: { value: "2026-08-01T00:00" },
    });
    fireEvent.change(screen.getByTestId("batch-ingested-before"), {
      target: { value: "2026-08-03T23:59" },
    });
    fireEvent.change(screen.getByTestId("batch-structured-generation-mode"), {
      target: { value: "prompt_only" },
    });
    fireEvent.change(screen.getByTestId("batch-icl-mode"), {
      target: { value: "disabled" },
    });

    const launch = screen.getByTestId("launch-batch");
    await waitFor(() => expect(launch).not.toBeDisabled());
    fireEvent.click(launch);

    await waitFor(() => {
      expect(mockCreate).toHaveBeenCalledWith("pid-1", {
        include_auto_labeled: false,
        run_limit: 20,
        ingested_after: "2026-08-01T00:00:00Z",
        ingested_before: "2026-08-03T23:59:00Z",
        structured_generation_mode: "prompt_only",
        icl_mode: "disabled",
      });
    });
  });

  it("rejects an invalid run limit before submitting", async () => {
    renderPage();
    fireEvent.click(await screen.findByTestId("advanced-toggle"));
    fireEvent.change(screen.getByTestId("batch-run-limit"), {
      target: { value: "0" },
    });

    expect(screen.getByTestId("batch-run-limit-helper")).toHaveTextContent(
      "Enter a whole number of 1 or more.",
    );
    expect(screen.getByTestId("launch-batch")).toBeDisabled();
    expect(mockCreate).not.toHaveBeenCalled();
  });

  it("rejects an inverted ingestion range before submitting", async () => {
    renderPage();
    fireEvent.click(await screen.findByTestId("advanced-toggle"));
    fireEvent.change(screen.getByTestId("batch-ingested-after"), {
      target: { value: "2026-08-04T12:00" },
    });
    fireEvent.change(screen.getByTestId("batch-ingested-before"), {
      target: { value: "2026-08-04T11:00" },
    });

    expect(screen.getByTestId("batch-ingested-range-helper")).toHaveTextContent(
      "The end must be the same as or later than the start.",
    );
    expect(screen.getByTestId("launch-batch")).toBeDisabled();
    expect(mockCreate).not.toHaveBeenCalled();
  });

  it("pins structured generation to prompt-only for an unsupported Teacher", async () => {
    mockFetchModelConfigs.mockResolvedValueOnce({
      items: [
        {
          model_config_id: "mc-1",
          model_name: "zero-schema-teacher",
          endpoint_id: "ep-1",
          eligible_roles: ["teacher"],
          structured_generation_support: "unsupported",
        },
      ],
      next_cursor: null,
    });
    renderPage();
    fireEvent.click(await screen.findByTestId("advanced-toggle"));

    const structuredMode = screen.getByTestId("batch-structured-generation-mode");
    await waitFor(() => expect(structuredMode).toBeDisabled());
    expect(structuredMode).toHaveValue("prompt_only");
    expect(
      screen.getByText(/does not support schema-constrained output/i),
    ).toBeVisible();

    const launch = screen.getByTestId("launch-batch");
    await waitFor(() => expect(launch).not.toBeDisabled());
    fireEvent.click(launch);
    await waitFor(() =>
      expect(mockCreate).toHaveBeenCalledWith(
        "pid-1",
        expect.objectContaining({ structured_generation_mode: "prompt_only" }),
      ),
    );
  });

  it("shows the backend reason when a gate race rejects launch", async () => {
    const { ApiError } = await import("@/api/client");
    mockFetchGate
      .mockResolvedValueOnce({
        gate_status: "ready",
        criteria: [],
        evaluated_at: "2026-04-16T00:00:00Z",
      })
      .mockResolvedValue({
        gate_status: "not_ready",
        criteria: [],
        evaluated_at: "2026-04-16T00:01:00Z",
      });
    mockCreate.mockRejectedValueOnce(
      new ApiError(
        409,
        JSON.stringify({
          detail:
            "Scale-Up readiness changed before launch. Review the current criteria.",
        }),
      ),
    );
    renderPage();
    const launch = screen.getByTestId("launch-batch");
    await waitFor(() => expect(launch).not.toBeDisabled());
    fireEvent.click(launch);

    expect(await screen.findByTestId("batch-launch-error")).toHaveTextContent(
      "Could not start batch run: Scale-Up readiness changed before launch. Review the current criteria.",
    );
    expect(screen.getByTestId("review-scaleup-after-launch-error")).toBeVisible();
    await waitFor(() => expect(screen.getByTestId("launch-batch")).toBeDisabled());
    expect(mockFetchGate).toHaveBeenCalledTimes(2);
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
      expect(screen.getByText("12 edits")).toBeInTheDocument();
    });
    // Matches wireframe's "ICL: 12 edits" formatting.
    expect(screen.getByText(/^ICL$/)).toBeInTheDocument();
  });
});
