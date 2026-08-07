// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Tests for ProjectIndexRedirect — the first-run setup gate.
 *
 * The gate is "setup_completed_at is null?" rather than "any seeded
 * model_config_id missing?" because seeded defaults always exist
 * post-creation, so a missing-model gate would never fire.
 */

import { beforeEach, describe, expect, it, vi } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { render, screen, waitFor } from "@testing-library/react";

import { ProjectIndexRedirect } from "@/pages/ProjectIndexRedirect";
import { ProjectSetupLayout } from "@/pages/ProjectSetupLayout";
import { makeEnvironmentResponse, makeProjectResponse } from "@/test/fixtures";

const mockFetchProject = vi.fn();
const mockFetchEnvironment = vi.fn();
const mockListTrainingSuites = vi.fn();
const mockListStudentModels = vi.fn();

vi.mock("@/api/projects", () => ({
  fetchProject: (...args: unknown[]) => mockFetchProject(...args),
  fetchProjectList: vi.fn().mockResolvedValue({ items: [], next_cursor: null }),
  createProject: vi.fn(),
  archiveProject: vi.fn(),
  markSetupCompleted: vi.fn(),
}));

vi.mock("@/api/nim", () => ({
  fetchEnvironment: (...args: unknown[]) => mockFetchEnvironment(...args),
}));

vi.mock("@/api/training", () => ({
  listTrainingSuites: (...args: unknown[]) => mockListTrainingSuites(...args),
}));

vi.mock("@/api/students", () => ({
  listStudentModels: (...args: unknown[]) => mockListStudentModels(...args),
}));

const ENV = makeEnvironmentResponse({
  embedding_deployment: {
    model_name: "nvidia/llama-nemotron-embed-vl-1b-v2",
    nim_container_image: "nvcr.io/nim/x:1.0",
    gpu_memory_minimum_gb: 24,
    fits: false,
    provider: "none",
  },
});

function baseProject(setupCompletedAt: string | null) {
  return makeProjectResponse({
    active_guidance_id: null,
    setup_completed_at: setupCompletedAt,
  });
}

function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false, staleTime: 0 } },
  });
  return ({ children: _children }: { children?: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={["/projects/test-pid"]}>
        <Routes>
          <Route path="/projects/:projectId" element={<ProjectSetupLayout />}>
            <Route index element={<ProjectIndexRedirect />} />
            <Route path="setup" element={<div data-testid="setup-page">Setup</div>} />
            <Route path="ready" element={<div data-testid="ready-page">Ready</div>} />
            <Route
              path="labeling"
              element={<div data-testid="labeling-page">Labeling</div>}
            />
            <Route
              path="create-guidance"
              element={<div data-testid="create-guidance-page">Guidance</div>}
            />
            <Route
              path="training/:trainingSuiteId"
              element={<div data-testid="training-jobs-page">Training Jobs</div>}
            />
            <Route
              path="overview"
              element={<div data-testid="project-overview-page">Overview</div>}
            />
            <Route
              path="compare"
              element={<div data-testid="compare-page">Models</div>}
            />
          </Route>
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>
  );
}

describe("ProjectIndexRedirect", () => {
  beforeEach(() => {
    mockFetchEnvironment.mockResolvedValue(ENV);
    mockListTrainingSuites.mockResolvedValue({ items: [], next_cursor: null });
    mockListStudentModels.mockResolvedValue({ items: [], next_cursor: null });
  });

  it("routes to /setup when setup_completed_at is null", async () => {
    mockFetchProject.mockResolvedValue(baseProject(null));
    const Wrapper = createWrapper();
    render(<div />, { wrapper: Wrapper });

    await waitFor(() => {
      expect(screen.getByTestId("setup-page")).toBeInTheDocument();
    });
  });

  it("routes to /ready when setup is complete but no examples ingested", async () => {
    mockFetchProject.mockResolvedValue(baseProject("2026-05-12T17:00:00Z"));
    const Wrapper = createWrapper();
    render(<div />, { wrapper: Wrapper });

    await waitFor(() => {
      expect(screen.getByTestId("ready-page")).toBeInTheDocument();
    });
  });

  it("routes to /create-guidance when examples exist but no active guidance", async () => {
    const p = baseProject("2026-05-12T17:00:00Z");
    p.counts.unlabeled = 42;
    mockFetchProject.mockResolvedValue(p);
    const Wrapper = createWrapper();
    render(<div />, { wrapper: Wrapper });

    await waitFor(() => {
      expect(screen.getByTestId("create-guidance-page")).toBeInTheDocument();
    });
  });

  it("routes to /labeling when everything is set up", async () => {
    const p = baseProject("2026-05-12T17:00:00Z");
    p.counts.unlabeled = 42;
    p.active_guidance_id = "guidance-1";
    mockFetchProject.mockResolvedValue(p);
    const Wrapper = createWrapper();
    render(<div />, { wrapper: Wrapper });

    await waitFor(() => {
      expect(screen.getByTestId("labeling-page")).toBeInTheDocument();
    });
  });

  it("resumes the newest non-terminal Training Suite for a project in training", async () => {
    const p = baseProject("2026-05-12T17:00:00Z");
    p.counts.unlabeled = 42;
    p.active_guidance_id = "guidance-1";
    mockFetchProject.mockResolvedValue(p);
    mockListTrainingSuites.mockResolvedValue({
      items: [
        {
          training_suite_id: "suite-completed",
          status: "completed",
        },
        {
          training_suite_id: "suite-running",
          status: "running",
        },
        {
          training_suite_id: "suite-older-running",
          status: "running",
        },
      ],
      next_cursor: null,
    });

    const Wrapper = createWrapper();
    render(<div />, { wrapper: Wrapper });

    await waitFor(() => {
      expect(screen.getByTestId("training-jobs-page")).toBeInTheDocument();
    });
    expect(mockListTrainingSuites).toHaveBeenCalledWith("test-pid");
  });

  it("opens the project overview when terminal Training Suite history exists", async () => {
    const p = baseProject("2026-05-12T17:00:00Z");
    p.counts.unlabeled = 42;
    p.active_guidance_id = "guidance-1";
    mockFetchProject.mockResolvedValue(p);
    mockListTrainingSuites.mockResolvedValue({
      items: [
        {
          training_suite_id: "suite-failed",
          status: "failed",
        },
        {
          training_suite_id: "suite-canceled",
          status: "canceled",
        },
      ],
      next_cursor: null,
    });

    const Wrapper = createWrapper();
    render(<div />, { wrapper: Wrapper });

    await waitFor(() => {
      expect(screen.getByTestId("project-overview-page")).toBeInTheDocument();
    });
  });

  it("resumes Models & Results while Student serving validation is active", async () => {
    const p = baseProject("2026-05-12T17:00:00Z");
    p.counts.unlabeled = 42;
    p.active_guidance_id = "guidance-1";
    mockFetchProject.mockResolvedValue(p);
    mockListStudentModels.mockResolvedValue({
      items: [{ student_model_id: "student-1", serving_status: "pending" }],
      next_cursor: null,
    });

    const Wrapper = createWrapper();
    render(<div />, { wrapper: Wrapper });

    await waitFor(() => {
      expect(screen.getByTestId("compare-page")).toBeInTheDocument();
    });
  });

  it("opens the project overview when a Student exists without suite history", async () => {
    const p = baseProject("2026-05-12T17:00:00Z");
    p.counts.unlabeled = 42;
    p.active_guidance_id = "guidance-1";
    mockFetchProject.mockResolvedValue(p);
    mockListStudentModels.mockResolvedValue({
      items: [{ student_model_id: "student-1", serving_status: "not_attempted" }],
      next_cursor: null,
    });

    const Wrapper = createWrapper();
    render(<div />, { wrapper: Wrapper });

    await waitFor(() => {
      expect(screen.getByTestId("project-overview-page")).toBeInTheDocument();
    });
  });
});
