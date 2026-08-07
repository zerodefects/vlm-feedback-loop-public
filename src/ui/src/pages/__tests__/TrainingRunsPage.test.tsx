// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { beforeEach, describe, expect, it, vi } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Outlet, Route, Routes } from "react-router-dom";

import type { TrainingSuite } from "@/types/training";
import { TrainingRunsPage } from "../TrainingRunsPage";

const mockListTrainingSuites = vi.fn();

vi.mock("@/api/training", () => ({
  listTrainingSuites: (...args: unknown[]) => mockListTrainingSuites(...args),
}));

vi.mock("@/pages/setup-context", () => ({
  useSetupContext: () => ({ projectId: "pid-1", project: {}, environment: {} }),
}));

function makeSuite(
  trainingSuiteId: string,
  status: TrainingSuite["status"],
  overrides: Partial<TrainingSuite> = {},
): TrainingSuite {
  return {
    training_suite_id: trainingSuiteId,
    project_id: "pid-1",
    idempotency_key: `idem-${trainingSuiteId}`,
    guidance_id: "g-1",
    training_preset: "quick",
    export_field_mode: "all",
    include_auto_labeled: true,
    quantization_schemes: ["FP8_DYNAMIC"],
    training_dataset_export_id: "de-train",
    evaluation_dataset_export_id: "de-eval",
    training_example_count: 100,
    evaluation_example_count: 20,
    evaluation_dataset_checksum_sha256: "sha256:test",
    selected_student_base_model_config_ids: ["mc-2b"],
    chain_ids_ordered: ["chain-1"],
    chains: [
      {
        chain_id: "chain-1",
        student_base_model_config_id: "mc-2b",
        base_model_name: "nvidia/cosmos-reason2-2b",
        jobs: [
          {
            tao_job_id: "job-1",
            action: "train",
            chain_sequence: 1,
            status: status === "completed" ? "succeeded" : "running",
            tao_external_job_id: "ext-1",
            chain_halted_reason: null,
            outputs_fetch_status: status === "completed" ? "completed" : "pending",
            outputs_fetch_error_ref: null,
          },
        ],
      },
    ],
    student_model_ids: [],
    provisioning_run_id: null,
    provisioning_model_names: [],
    setup_error_ref: null,
    status,
    created_at: "2026-08-03T10:00:00Z",
    started_at: "2026-08-03T10:01:00Z",
    completed_at: status === "completed" ? "2026-08-03T10:30:00Z" : null,
    ...overrides,
  };
}

function renderPage() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={["/projects/pid-1/training-runs"]}>
        <Routes>
          <Route path="/projects/:projectId" element={<Outlet />}>
            <Route path="training-runs" element={<TrainingRunsPage />} />
            <Route
              path="training/:trainingSuiteId"
              element={<div data-testid="training-run-detail" />}
            />
            <Route path="training" element={<div data-testid="training-page" />} />
            <Route path="overview" element={<div data-testid="overview-page" />} />
          </Route>
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

beforeEach(() => {
  vi.clearAllMocks();
});

describe("TrainingRunsPage", () => {
  it("renders newest-first suite metadata and the correct active and terminal actions", async () => {
    const newerActive = makeSuite("suite-newer", "running", {
      created_at: "2026-08-03T11:00:00Z",
      started_at: "2026-08-03T11:01:00Z",
      training_preset: "standard",
      chains: [
        {
          chain_id: "chain-newer",
          student_base_model_config_id: "mc-8b",
          base_model_name: "nvidia/cosmos-reason2-8b",
          jobs: [
            {
              tao_job_id: "job-succeeded",
              action: "train",
              chain_sequence: 1,
              status: "succeeded",
              tao_external_job_id: "ext-succeeded",
              chain_halted_reason: null,
              outputs_fetch_status: "completed",
              outputs_fetch_error_ref: null,
            },
            {
              tao_job_id: "job-running",
              action: "evaluate",
              chain_sequence: 2,
              status: "running",
              tao_external_job_id: "ext-running",
              chain_halted_reason: null,
              outputs_fetch_status: "pending",
              outputs_fetch_error_ref: null,
            },
          ],
        },
      ],
    });
    const olderTerminal = makeSuite("suite-older", "completed", {
      created_at: "2026-08-03T10:00:00Z",
      started_at: "2026-08-03T10:01:00Z",
    });
    mockListTrainingSuites.mockResolvedValue({
      // The API contract returns suites newest-first; the page must preserve it.
      items: [newerActive, olderTerminal],
      next_cursor: null,
    });

    renderPage();

    const newerCard = await screen.findByTestId("training-run-suite-newer");
    const olderCard = screen.getByTestId("training-run-suite-older");
    expect(
      newerCard.compareDocumentPosition(olderCard) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
    expect(newerCard).toHaveTextContent("Cosmos Reason2 8B");
    expect(newerCard).toHaveTextContent("Standard preset · 1 of 2 jobs completed");
    expect(newerCard).toHaveTextContent("Started");
    expect(newerCard).toHaveTextContent("Running");
    expect(newerCard).toHaveTextContent("Resume");
    expect(olderCard).toHaveTextContent("Cosmos Reason2 2B");
    expect(olderCard).toHaveTextContent("Quick preset · 1 of 1 jobs completed");
    expect(olderCard).toHaveTextContent("Completed");
    expect(olderCard).toHaveTextContent("View details");
  });

  it("lists history without offering standalone artifact downloads", async () => {
    mockListTrainingSuites.mockResolvedValue({
      items: [makeSuite("suite-completed", "completed")],
      next_cursor: null,
    });
    renderPage();

    await screen.findByText("Cosmos Reason2 2B");
    expect(screen.getByText("Completed")).toBeInTheDocument();
    expect(
      screen.getByText(/Portable deployment downloads are available only/),
    ).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /download/i })).toBeNull();
  });

  it("opens the selected run details", async () => {
    mockListTrainingSuites.mockResolvedValue({
      items: [makeSuite("suite-completed", "completed")],
      next_cursor: null,
    });
    renderPage();

    fireEvent.click(await screen.findByTestId("view-training-run-suite-completed"));
    await waitFor(() =>
      expect(screen.getByTestId("training-run-detail")).toBeInTheDocument(),
    );
  });
});
