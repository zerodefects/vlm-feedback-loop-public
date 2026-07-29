// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, it, expect, vi, beforeEach } from "vitest";
import { installEventSourceMock } from "@/test/event-source-mock";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter, Routes, Route, Outlet } from "react-router-dom";

import { BatchRunStatusPage } from "../BatchRunStatusPage";

// ── Mocks ──────────────────────────────────────────────────────────────────

vi.mock("@/api/batch", () => ({
  getBatchLabelRun: vi.fn(),
  resumeBatchLabelRun: vi.fn(),
  cancelBatchLabelRun: vi.fn(),
  createDatasetExport: vi.fn(),
  getDatasetExport: vi.fn(),
  listDatasetExports: vi.fn(),
}));

vi.mock("@/pages/ProjectSetupLayout", () => ({
  useSetupContext: vi.fn(),
}));

installEventSourceMock();

import {
  getBatchLabelRun,
  resumeBatchLabelRun,
  cancelBatchLabelRun,
  createDatasetExport,
  getDatasetExport,
  listDatasetExports,
} from "@/api/batch";
import { useSetupContext } from "@/pages/ProjectSetupLayout";

const mockGetRun = getBatchLabelRun as ReturnType<typeof vi.fn>;
const mockResume = resumeBatchLabelRun as ReturnType<typeof vi.fn>;
const mockCancel = cancelBatchLabelRun as ReturnType<typeof vi.fn>;
const mockExport = createDatasetExport as ReturnType<typeof vi.fn>;
const mockGetExport = getDatasetExport as ReturnType<typeof vi.fn>;
const mockListExports = listDatasetExports as ReturnType<typeof vi.fn>;

function makeExport(overrides: Record<string, unknown> = {}) {
  return {
    dataset_export_id: "de-1",
    status: "running",
    status_reason: null,
    progress: null,
    artifact_refs: null,
    selection_definition_snapshot: { batch_label_run_id: "run-1" },
    ...overrides,
  };
}
const mockSetup = useSetupContext as ReturnType<typeof vi.fn>;

function makeRun(overrides: Record<string, unknown> = {}) {
  return {
    run_id: "run-1",
    run_type: "batch_label_run",
    status: "running",
    status_reason: null,
    paused_reason: null,
    guidance_id: "g-1",
    model_config_id: "mc-1",
    generation_preset_key: "precise",
    thinking_mode_effective: "on",
    visual_budget_preset_key: "balanced",
    structured_generation_mode_effective: "auto",
    progress: { processed: 50, total: 100 },
    examples_succeeded: 45,
    examples_schema_invalid: 3,
    examples_timeout: 1,
    examples_endpoint_error: 1,
    examples_total: 100,
    created_at: "2026-04-16T00:00:00Z",
    started_at: "2026-04-16T00:01:00Z",
    completed_at: null,
    cancel_requested_at: null,
    recovered_from_restart: false,
    ...overrides,
  };
}

let qc: QueryClient;

beforeEach(() => {
  mockListExports.mockResolvedValue({ items: [], next_cursor: null });
  vi.clearAllMocks();
  qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  mockSetup.mockReturnValue({
    projectId: "pid-1",
    project: {
      project_id: "pid-1",
      name: "Test",
      active_guidance_id: "g-1",
      teacher_model_config_id: "mc-1",
      export_field_mode: "core_only",
    },
    environment: {},
  });
  mockResume.mockResolvedValue({ run_id: "run-1", status: "queued" });
  mockCancel.mockResolvedValue({
    run_id: "run-1",
    status: "canceling",
    cancel_requested_at: "2026-04-16T00:10:00Z",
  });
  mockExport.mockResolvedValue({
    dataset_export_id: "de-1",
    example_count: 45,
    created_at: "2026-04-16T01:00:00Z",
  });
});

function renderPage() {
  return render(
    <QueryClientProvider client={qc}>
      <MemoryRouter initialEntries={["/projects/pid-1/batch-status/run-1"]}>
        <Routes>
          <Route path="/projects/:projectId" element={<Outlet />}>
            <Route path="batch-status/:runId" element={<BatchRunStatusPage />} />
            <Route path="scale-up" element={<div data-testid="scaleup-page" />} />
          </Route>
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

// ── Tests ──────────────────────────────────────────────────────────────────

describe("BatchRunStatusPage", () => {
  it("shows running state with progress", async () => {
    mockGetRun.mockResolvedValue(makeRun());
    renderPage();
    await waitFor(() => {
      expect(screen.getByTestId("status-badge")).toHaveTextContent("Running");
      expect(screen.getByText(/50 of 100/)).toBeInTheDocument();
    });
  });

  it("shows paused state with circuit breaker banner", async () => {
    mockGetRun.mockResolvedValue(
      makeRun({
        status: "paused",
        paused_reason: "circuit_breaker_threshold_reached",
        examples_timeout: 10,
      }),
    );
    renderPage();
    await waitFor(() => {
      expect(screen.getByTestId("status-badge")).toHaveTextContent("Paused");
      expect(screen.getByTestId("circuit-breaker-banner")).toBeInTheDocument();
      expect(screen.getByText(/Endpoint appears unreachable/)).toBeInTheDocument();
    });
  });

  it("shows completed state with export button", async () => {
    mockGetRun.mockResolvedValue(
      makeRun({
        status: "completed",
        completed_at: "2026-04-16T01:00:00Z",
      }),
    );
    renderPage();
    await waitFor(() => {
      expect(screen.getByTestId("status-badge")).toHaveTextContent("Completed");
      expect(screen.getByTestId("completed-banner")).toBeInTheDocument();
      expect(screen.getByTestId("export-dataset-btn")).toBeInTheDocument();
    });
  });

  it("shows failed state with error details", async () => {
    mockGetRun.mockResolvedValue(
      makeRun({
        status: "failed",
        status_reason: "unhandled_exception",
        completed_at: "2026-04-16T01:00:00Z",
      }),
    );
    renderPage();
    await waitFor(() => {
      expect(screen.getByTestId("status-badge")).toHaveTextContent("Failed");
      expect(screen.getByTestId("failed-banner")).toBeInTheDocument();
    });
  });

  it("shows structured gen rejected with restart button", async () => {
    mockGetRun.mockResolvedValue(
      makeRun({
        status: "failed",
        status_reason: "structured_generation_rejected",
        completed_at: "2026-04-16T01:00:00Z",
      }),
    );
    renderPage();
    await waitFor(() => {
      expect(screen.getByTestId("structured-gen-rejected-banner")).toBeInTheDocument();
      expect(screen.getByTestId("restart-prompt-only")).toBeInTheDocument();
    });
    // The banner explains what "prompt-only" means so the SME can judge the
    // recovery action without leaving the page.
    expect(screen.getByTestId("structured-gen-rejected-banner")).toHaveTextContent(
      "Prompt-only asks for JSON in the prompt instead of enforcing a schema.",
    );
    // The recommended recovery carries the primary treatment — it is the
    // one action the failure state is steering the SME toward. KUI renders
    // the ``kind`` prop as an ``nv-button--kind-*`` class.
    expect(screen.getByTestId("restart-prompt-only")).toHaveClass(
      "nv-button--kind-primary",
    );
  });

  it("calls resume API when Resume clicked", async () => {
    mockGetRun.mockResolvedValue(
      makeRun({ status: "paused", paused_reason: "circuit_breaker_threshold_reached" }),
    );
    renderPage();
    await waitFor(() => screen.getByTestId("resume-btn"));
    fireEvent.click(screen.getByTestId("resume-btn"));
    await waitFor(() => expect(mockResume).toHaveBeenCalledWith("pid-1", "run-1"));
  });

  it("calls cancel API when Cancel clicked", async () => {
    mockGetRun.mockResolvedValue(makeRun());
    renderPage();
    await waitFor(() => screen.getByTestId("cancel-btn"));
    fireEvent.click(screen.getByTestId("cancel-btn"));
    await waitFor(() => expect(mockCancel).toHaveBeenCalledWith("pid-1", "run-1"));
  });

  it("calls export API when Export Dataset clicked", async () => {
    mockGetRun.mockResolvedValue(
      makeRun({ status: "completed", completed_at: "2026-04-16T01:00:00Z" }),
    );
    renderPage();
    await waitFor(() => screen.getByTestId("export-dataset-btn"));
    fireEvent.click(screen.getByTestId("export-dataset-btn"));
    await waitFor(() =>
      expect(mockExport).toHaveBeenCalledWith("pid-1", {
        dataset_intent: "training",
        label_tier_filter: "auto_labeled_only",
        export_field_mode: "core_only",
        batch_label_run_id: "run-1",
      }),
    );
  });

  it("shows honest copy for a schema-evolution-canceled run (no 'retained' claim)", async () => {
    mockGetRun.mockResolvedValue(
      makeRun({
        status: "failed",
        status_reason: "schema_evolution_canceled",
        examples_succeeded: 5,
        completed_at: "2026-04-16T01:00:00Z",
      }),
    );
    renderPage();
    await waitFor(() => screen.getByTestId("failed-banner"));
    const banner = screen.getByTestId("failed-banner");
    expect(banner).toHaveTextContent(
      /Guidance schema changed .* while this run was in flight/,
    );
    expect(banner).toHaveTextContent(/labels were reset.*outputs were discarded/);
    expect(banner).not.toHaveTextContent(/retained/);
    expect(banner).not.toHaveTextContent(/schema_evolution_canceled/);
  });

  it("tracks the background export to completion and reports success", async () => {
    // The create response is only the running record — the archive
    // builds in the background. The page must poll the export record to
    // terminal state, or the SME never learns the outcome.
    mockGetRun.mockResolvedValue(
      makeRun({ status: "completed", completed_at: "2026-04-16T01:00:00Z" }),
    );
    mockExport.mockResolvedValue(makeExport());
    mockGetExport.mockResolvedValue(
      makeExport({
        status: "completed",
        progress: { images_written: 42, images_total: 42 },
        artifact_refs: { archive_path: "/x/de-1.tar.gz", checksum_sha256: "c" },
      }),
    );
    renderPage();
    await waitFor(() => screen.getByTestId("export-dataset-btn"));
    fireEvent.click(screen.getByTestId("export-dataset-btn"));
    await waitFor(() =>
      expect(screen.getByTestId("export-status-banner")).toHaveTextContent(
        /exported successfully — 42 images/,
      ),
    );
    // The export button is gone once the export completed.
    expect(screen.queryByTestId("export-dataset-btn")).toBeNull();
  });

  it("surfaces a failed background export and offers export again", async () => {
    // A build failure (missing image file, backend restart) previously
    // left a permanent affirmative 'started' banner.
    mockGetRun.mockResolvedValue(
      makeRun({ status: "completed", completed_at: "2026-04-16T01:00:00Z" }),
    );
    mockExport.mockResolvedValue(makeExport());
    mockGetExport.mockResolvedValue(
      makeExport({ status: "failed", status_reason: "backend_restart_interrupted" }),
    );
    renderPage();
    await waitFor(() => screen.getByTestId("export-dataset-btn"));
    fireEvent.click(screen.getByTestId("export-dataset-btn"));
    await waitFor(() =>
      expect(screen.getByTestId("export-failed-banner")).toHaveTextContent(
        /export failed: backend_restart_interrupted/i,
      ),
    );
    // The button returns so the SME can export again after fixing the cause.
    expect(screen.getByTestId("export-dataset-btn")).toBeInTheDocument();
  });

  it("adopts an export still building after navigation and blocks a duplicate", async () => {
    // The background export outlives the page: on mount, a running
    // export is adopted (banner + poll resume) and the Export button is
    // withheld so the natural re-click cannot double the build.
    mockGetRun.mockResolvedValue(
      makeRun({ status: "completed", completed_at: "2026-04-16T01:00:00Z" }),
    );
    mockListExports.mockResolvedValue({
      items: [makeExport({ dataset_export_id: "de-adopted" })],
      next_cursor: null,
    });
    mockGetExport.mockResolvedValue(makeExport({ dataset_export_id: "de-adopted" }));
    renderPage();
    await waitFor(() =>
      expect(screen.getByTestId("export-status-banner")).toBeInTheDocument(),
    );
    expect(mockGetExport).toHaveBeenCalledWith("pid-1", "de-adopted");
    expect(screen.queryByTestId("export-dataset-btn")).toBeNull();
  });

  it("reconciles the export list when remounted with the same query cache", async () => {
    mockGetRun.mockResolvedValue(
      makeRun({ status: "completed", completed_at: "2026-04-16T01:00:00Z" }),
    );
    mockListExports
      .mockResolvedValueOnce({ items: [], next_cursor: null })
      .mockResolvedValue({
        items: [makeExport({ dataset_export_id: "de-remounted" })],
        next_cursor: null,
      });
    mockExport.mockResolvedValue(makeExport({ dataset_export_id: "de-remounted" }));
    mockGetExport.mockResolvedValue(makeExport({ dataset_export_id: "de-remounted" }));

    const firstMount = renderPage();
    await waitFor(() => screen.getByTestId("export-dataset-btn"));
    fireEvent.click(screen.getByTestId("export-dataset-btn"));
    await waitFor(() =>
      expect(screen.getByTestId("export-status-banner")).toBeInTheDocument(),
    );
    firstMount.unmount();

    renderPage();

    await waitFor(() => expect(mockListExports).toHaveBeenCalledTimes(2));
    await waitFor(() =>
      expect(screen.getByTestId("export-status-banner")).toBeInTheDocument(),
    );
    expect(mockGetExport).toHaveBeenCalledWith("pid-1", "de-remounted");
    expect(screen.queryByTestId("export-dataset-btn")).toBeNull();
  });

  it("does not attribute another Batch run's export to this run", async () => {
    mockGetRun.mockResolvedValue(
      makeRun({ status: "completed", completed_at: "2026-04-16T01:00:00Z" }),
    );
    mockListExports.mockResolvedValue({
      items: [
        makeExport({
          dataset_export_id: "de-other-run",
          selection_definition_snapshot: { batch_label_run_id: "run-2" },
        }),
      ],
      next_cursor: null,
    });

    renderPage();

    await waitFor(() => screen.getByTestId("export-dataset-btn"));
    expect(mockGetExport).not.toHaveBeenCalled();
    expect(screen.queryByTestId("export-status-banner")).toBeNull();
  });

  it("polls a running export to its terminal state", async () => {
    // The poll is load-bearing: without refetchInterval the record is
    // fetched once as 'running' and the terminal state never renders.
    mockGetRun.mockResolvedValue(
      makeRun({ status: "completed", completed_at: "2026-04-16T01:00:00Z" }),
    );
    mockExport.mockResolvedValue(makeExport());
    mockGetExport.mockResolvedValueOnce(makeExport()).mockResolvedValue(
      makeExport({
        status: "completed",
        progress: { images_written: 7, images_total: 7 },
      }),
    );
    renderPage();
    await waitFor(() => screen.getByTestId("export-dataset-btn"));
    fireEvent.click(screen.getByTestId("export-dataset-btn"));
    await waitFor(
      () =>
        expect(screen.getByTestId("export-status-banner")).toHaveTextContent(
          /exported successfully — 7 images/,
        ),
      { timeout: 8000 },
    );
  });

  // ── Action failures must be visible ─────────────────────────────────
  // This is the walk-away screen: a rejected Resume/Cancel/Export with no
  // rendered feedback reads as "the button did nothing".

  it("shows the backend detail when Resume fails", async () => {
    const { ApiError } = await import("@/api/client");
    mockGetRun.mockResolvedValue(
      makeRun({ status: "paused", paused_reason: "circuit_breaker_threshold_reached" }),
    );
    mockResume.mockRejectedValue(
      new ApiError(409, JSON.stringify({ detail: "Run is not paused." })),
    );
    renderPage();
    await waitFor(() => screen.getByTestId("resume-btn"));
    fireEvent.click(screen.getByTestId("resume-btn"));
    const err = await screen.findByTestId("action-error");
    expect(err).toHaveTextContent("Could not resume the run: Run is not paused.");
  });

  it("shows an error line when Cancel fails", async () => {
    mockGetRun.mockResolvedValue(makeRun());
    mockCancel.mockRejectedValue(new Error("fetch failed"));
    renderPage();
    await waitFor(() => screen.getByTestId("cancel-btn"));
    fireEvent.click(screen.getByTestId("cancel-btn"));
    const err = await screen.findByTestId("action-error");
    expect(err).toHaveTextContent("Could not cancel the run: fetch failed");
  });

  it("shows an error line when Export fails and no success notice renders", async () => {
    mockGetRun.mockResolvedValue(
      makeRun({ status: "completed", completed_at: "2026-04-16T01:00:00Z" }),
    );
    mockExport.mockRejectedValue(new Error("disk full"));
    renderPage();
    await waitFor(() => screen.getByTestId("export-dataset-btn"));
    fireEvent.click(screen.getByTestId("export-dataset-btn"));
    const err = await screen.findByTestId("action-error");
    expect(err).toHaveTextContent("Could not export the dataset: disk full");
    expect(screen.queryByText("Dataset exported successfully.")).toBeNull();
  });
});
