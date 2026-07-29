// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Unit tests for ProjectSetupLayout — pins the failure-state contract:
 * a backend that is down (network error) or erroring (ApiError) must
 * produce an actionable screen with a working Retry, never a dead end.
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { makeEnvironmentResponse, makeProjectResponse } from "@/test/fixtures";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter, Routes, Route } from "react-router-dom";
import { ApiError } from "@/api/client";
import { ProjectSetupLayout } from "@/pages/ProjectSetupLayout";

const mockFetchProject = vi.fn();
const mockFetchEnvironment = vi.fn();

vi.mock("@/api/projects", () => ({
  fetchProject: (...args: unknown[]) => mockFetchProject(...args),
}));

vi.mock("@/api/nim", () => ({
  fetchEnvironment: (...args: unknown[]) => mockFetchEnvironment(...args),
}));

const PROJECT = makeProjectResponse({
  project_id: "p1",
  name: "Test Project",
  setup_completed_at: null,
});

function renderLayout() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={["/projects/p1"]}>
        <Routes>
          <Route path="/projects/:projectId" element={<ProjectSetupLayout />}>
            <Route index element={<div data-testid="outlet-child" />} />
          </Route>
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

beforeEach(() => {
  vi.clearAllMocks();
});

describe("ProjectSetupLayout", () => {
  it("renders the outlet when both queries succeed", async () => {
    mockFetchProject.mockResolvedValue(PROJECT);
    mockFetchEnvironment.mockResolvedValue(makeEnvironmentResponse());

    renderLayout();

    await waitFor(() => expect(screen.getByTestId("outlet-child")).toBeInTheDocument());
  });

  // A network-level failure (fetch throws TypeError, not ApiError) means
  // the backend is unreachable — the screen must say so and offer a way
  // to act on it, because "backend process died" is a state the SME can
  // fix and retry without reloading the app.
  it("shows the unreachable-backend message and recovers via Retry", async () => {
    mockFetchProject.mockRejectedValueOnce(new TypeError("fetch failed"));
    mockFetchEnvironment.mockRejectedValueOnce(new TypeError("fetch failed"));

    renderLayout();

    await waitFor(() =>
      expect(screen.getByText("Cannot reach the backend")).toBeInTheDocument(),
    );
    expect(
      screen.getByText("Check that the backend is running, then retry."),
    ).toBeInTheDocument();

    // Backend comes back; Retry must refetch both queries and proceed
    // to the outlet without a page reload.
    mockFetchProject.mockResolvedValue(PROJECT);
    mockFetchEnvironment.mockResolvedValue(makeEnvironmentResponse());
    await userEvent.click(screen.getByTestId("setup-layout-retry"));

    await waitFor(() => expect(screen.getByTestId("outlet-child")).toBeInTheDocument());
  });

  it("shows the backend's error detail for API failures", async () => {
    mockFetchProject.mockRejectedValue(
      new ApiError(404, JSON.stringify({ detail: "Project not found" })),
    );
    mockFetchEnvironment.mockResolvedValue(makeEnvironmentResponse());

    renderLayout();

    await waitFor(() =>
      expect(
        screen.getByText("Failed to load project or environment data"),
      ).toBeInTheDocument(),
    );
    expect(screen.getByText("Project not found")).toBeInTheDocument();
    expect(screen.getByTestId("setup-layout-retry")).toBeInTheDocument();
  });

  // Once setup is complete, the header exposes a single "NIM Configuration"
  // link into the NIM Connection edit screen — no kebab, and no in-header
  // Archive affordance (archiving lives on the Project List screen).
  it("renders the NIM Configuration header link, and no Archive affordance, after setup", async () => {
    mockFetchProject.mockResolvedValue({
      ...PROJECT,
      setup_completed_at: "2026-07-01T00:00:00Z",
    });
    mockFetchEnvironment.mockResolvedValue(makeEnvironmentResponse());

    renderLayout();
    await waitFor(() => expect(screen.getByTestId("outlet-child")).toBeInTheDocument());

    const link = screen.getByTestId("project-nim-config-link");
    expect(link).toHaveTextContent("NIM Configuration");
    expect(link).toHaveAttribute("href", "/projects/p1/settings/nim");

    expect(screen.queryByText("Archive Project")).not.toBeInTheDocument();
    expect(screen.queryByTestId("project-kebab-trigger")).not.toBeInTheDocument();
  });

  // Before setup completes, no project chrome is portalled into the header.
  it("does not render the NIM Configuration link before setup completes", async () => {
    mockFetchProject.mockResolvedValue(PROJECT);
    mockFetchEnvironment.mockResolvedValue(makeEnvironmentResponse());

    renderLayout();
    await waitFor(() => expect(screen.getByTestId("outlet-child")).toBeInTheDocument());

    expect(screen.queryByTestId("project-nim-config-link")).not.toBeInTheDocument();
  });
});
