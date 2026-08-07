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

function renderLayout(
  initialEntry:
    | string
    | { pathname: string; state?: Record<string, unknown> } = "/projects/p1",
) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={[initialEntry]}>
        <Routes>
          <Route path="/projects/:projectId" element={<ProjectSetupLayout />}>
            <Route index element={<div data-testid="outlet-child" />} />
            <Route path="overview" element={<div data-testid="overview-child" />} />
            <Route path="*" element={<div data-testid="outlet-child" />} />
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
  it("renders an ordinary project route without waiting for environment", async () => {
    mockFetchProject.mockResolvedValue(PROJECT);

    renderLayout();

    await waitFor(() => expect(screen.getByTestId("outlet-child")).toBeInTheDocument());
    expect(mockFetchEnvironment).not.toHaveBeenCalled();
  });

  it("loads the environment before rendering a capability-dependent route", async () => {
    mockFetchProject.mockResolvedValue(PROJECT);
    mockFetchEnvironment.mockResolvedValue(makeEnvironmentResponse());

    renderLayout("/projects/p1/setup");

    await waitFor(() => expect(screen.getByTestId("outlet-child")).toBeInTheDocument());
    expect(mockFetchEnvironment).toHaveBeenCalledOnce();
  });

  // A network-level failure (fetch throws TypeError, not ApiError) means
  // the backend is unreachable — the screen must say so and offer a way
  // to act on it, because "backend process died" is a state the SME can
  // fix and retry without reloading the app.
  it("shows the unreachable-backend message and recovers via Retry", async () => {
    mockFetchProject.mockRejectedValueOnce(new TypeError("fetch failed"));

    renderLayout();

    await waitFor(() =>
      expect(screen.getByText("Cannot reach the backend")).toBeInTheDocument(),
    );
    expect(
      screen.getByText("Check that the backend is running, then retry."),
    ).toBeInTheDocument();

    // Backend comes back; Retry refetches the project and proceeds without
    // starting an unrelated hardware assessment.
    mockFetchProject.mockResolvedValue(PROJECT);
    await userEvent.click(screen.getByTestId("setup-layout-retry"));

    await waitFor(() => expect(screen.getByTestId("outlet-child")).toBeInTheDocument());
    expect(mockFetchEnvironment).not.toHaveBeenCalled();
  });

  it("shows the backend's error detail for API failures", async () => {
    mockFetchProject.mockRejectedValue(
      new ApiError(404, JSON.stringify({ detail: "Project not found" })),
    );

    renderLayout();

    await waitFor(() =>
      expect(screen.getByText("Failed to load project data")).toBeInTheDocument(),
    );
    expect(screen.getByText("Project not found")).toBeInTheDocument();
    expect(screen.getByTestId("setup-layout-retry")).toBeInTheDocument();
  });

  it("surfaces environment failures on capability-dependent routes", async () => {
    mockFetchProject.mockResolvedValue(PROJECT);
    mockFetchEnvironment.mockRejectedValue(
      new ApiError(503, JSON.stringify({ detail: "Assessment unavailable" })),
    );

    renderLayout("/projects/p1/setup");

    await waitFor(() =>
      expect(
        screen.getByText("Failed to load project or environment data"),
      ).toBeInTheDocument(),
    );
    expect(screen.getByText("Assessment unavailable")).toBeInTheDocument();
  });

  // Once setup is complete, the header keeps the mature-project overview and
  // NIM Configuration reachable without duplicating contextual destinations.
  it("renders persistent project destinations after setup", async () => {
    mockFetchProject.mockResolvedValue({
      ...PROJECT,
      setup_completed_at: "2026-07-01T00:00:00Z",
    });
    mockFetchEnvironment.mockResolvedValue(makeEnvironmentResponse());

    renderLayout();
    await waitFor(() => expect(screen.getByTestId("outlet-child")).toBeInTheDocument());

    expect(screen.getByTestId("project-context-name")).toHaveTextContent(
      "Project: Test Project",
    );
    expect(screen.getByTestId("project-context-name")).toHaveAttribute(
      "title",
      "Project: Test Project",
    );
    expect(screen.getByTestId("project-overview-link")).toHaveAttribute(
      "href",
      "/projects/p1/overview",
    );
    expect(screen.queryByTestId("project-model-results-link")).not.toBeInTheDocument();
    expect(screen.getByTestId("project-nim-config-link")).toHaveAttribute(
      "href",
      "/projects/p1/settings/nim",
    );

    expect(screen.queryByText("Archive Project")).not.toBeInTheDocument();
    expect(screen.queryByTestId("project-kebab-trigger")).not.toBeInTheDocument();
  });

  it.each(["setup", "setup/ngc", "setup/done", "confirm-defaults"])(
    "redirects a completed project's copied /%s onboarding URL to its overview",
    async (suffix) => {
      mockFetchProject.mockResolvedValue({
        ...PROJECT,
        setup_completed_at: "2026-07-01T00:00:00Z",
      });
      mockFetchEnvironment.mockResolvedValue(makeEnvironmentResponse());

      renderLayout(`/projects/p1/${suffix}`);

      await waitFor(() =>
        expect(screen.getByTestId("overview-child")).toBeInTheDocument(),
      );
      expect(screen.queryByTestId("outlet-child")).not.toBeInTheDocument();
    },
  );

  it("preserves a completed setup transition that carries its model-path state", async () => {
    mockFetchProject.mockResolvedValue({
      ...PROJECT,
      setup_completed_at: "2026-07-01T00:00:00Z",
    });
    mockFetchEnvironment.mockResolvedValue(makeEnvironmentResponse());

    renderLayout({
      pathname: "/projects/p1/confirm-defaults",
      state: {
        activePath: "local",
        cameFromAutoSkip: false,
        localDeployQueued: ["nvidia/example-local-teacher"],
      },
    });

    await waitFor(() => expect(screen.getByTestId("outlet-child")).toBeInTheDocument());
    expect(screen.queryByTestId("overview-child")).not.toBeInTheDocument();
  });

  // Before setup completes, no project chrome is portalled into the header.
  it("does not render the NIM Configuration link before setup completes", async () => {
    mockFetchProject.mockResolvedValue(PROJECT);
    mockFetchEnvironment.mockResolvedValue(makeEnvironmentResponse());

    renderLayout();
    await waitFor(() => expect(screen.getByTestId("outlet-child")).toBeInTheDocument());

    expect(screen.queryByTestId("project-nim-config-link")).not.toBeInTheDocument();
    expect(screen.queryByTestId("project-overview-link")).not.toBeInTheDocument();
    expect(screen.queryByTestId("project-context-name")).not.toBeInTheDocument();
  });
});
