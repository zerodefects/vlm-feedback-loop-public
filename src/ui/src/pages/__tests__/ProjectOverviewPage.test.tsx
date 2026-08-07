// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { beforeEach, describe, expect, it, vi } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Outlet, Route, Routes } from "react-router-dom";

import { makeProjectResponse } from "@/test/fixtures";
import { ProjectOverviewPage } from "../ProjectOverviewPage";

const mockListTrainingSuites = vi.fn();
const mockListStudentModels = vi.fn();
const mockSetup = vi.fn();

vi.mock("@/api/training", () => ({
  listTrainingSuites: (...args: unknown[]) => mockListTrainingSuites(...args),
}));

vi.mock("@/api/students", () => ({
  listStudentModels: (...args: unknown[]) => mockListStudentModels(...args),
}));

vi.mock("@/pages/setup-context", () => ({
  useSetupContext: () => mockSetup(),
}));

function renderPage() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={["/projects/pid-1/overview"]}>
        <Routes>
          <Route path="/projects/:projectId" element={<Outlet />}>
            <Route path="overview" element={<ProjectOverviewPage />} />
            <Route path="labeling" element={<div data-testid="labeling-page" />} />
            <Route path="ready" element={<div data-testid="ready-page" />} />
            <Route path="scale-up" element={<div data-testid="scale-up-page" />} />
            <Route path="compare" element={<div data-testid="compare-page" />} />
            <Route
              path="training-runs"
              element={<div data-testid="training-runs-page" />}
            />
          </Route>
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

beforeEach(() => {
  vi.clearAllMocks();
  mockSetup.mockReturnValue({
    projectId: "pid-1",
    project: makeProjectResponse({
      project_id: "pid-1",
      name: "Bottle Sorter",
      counts: {
        verified: 84,
        unlabeled: 16,
        auto_labeled: 200,
        omitted: 3,
        pending_relabel: 0,
        prior_relabeled: 0,
      },
    }),
    environment: {},
  });
  mockListTrainingSuites.mockResolvedValue({
    items: [{ training_suite_id: "suite-1", status: "completed" }],
    next_cursor: null,
  });
  mockListStudentModels.mockResolvedValue({
    items: [
      {
        student_model_id: "student-1",
        quality_status: "validated",
        serving_status: "validated",
      },
      {
        student_model_id: "student-2",
        quality_status: "validated",
        serving_status: "not_attempted",
      },
    ],
    next_cursor: null,
  });
});

describe("ProjectOverviewPage", () => {
  it("separates labeling, model results, and training history", async () => {
    renderPage();

    await screen.findByText("Bottle Sorter");
    expect(screen.getByText("2 Student variants")).toBeInTheDocument();
    expect(
      screen.getByText("2 quality validated · 1 serving validated"),
    ).toBeInTheDocument();
    expect(screen.getByText("1 run")).toBeInTheDocument();
    expect(screen.getByTestId("overview-model-results")).not.toBeDisabled();
    expect(screen.getByTestId("overview-training-runs")).not.toBeDisabled();
    expect(screen.queryByText("Scale-Up")).not.toBeInTheDocument();
    expect(document.querySelectorAll(".nvidia-green-button")).toHaveLength(3);
  });

  it("routes each mature-project intent to its dedicated screen", async () => {
    renderPage();
    await screen.findByText("Bottle Sorter");

    fireEvent.click(screen.getByTestId("overview-model-results"));
    await waitFor(() => expect(screen.getByTestId("compare-page")).toBeInTheDocument());
  });

  it("adapts when history exists but produced no Student", async () => {
    mockListStudentModels.mockResolvedValue({ items: [], next_cursor: null });
    renderPage();

    await screen.findByText(
      "No Student model is available from the recorded Training Runs.",
    );
    expect(screen.getByTestId("overview-model-results")).toBeDisabled();
    expect(screen.getByTestId("overview-training-runs")).not.toBeDisabled();
    expect(screen.getByTestId("overview-continue-labeling")).not.toBeDisabled();
  });
});
