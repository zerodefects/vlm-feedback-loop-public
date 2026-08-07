// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Tests for LocalDeployBanner (F-amendment NIM-FTU-Local-Peer).
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import type { ReactNode } from "react";

import { LocalDeployBanner } from "@/components/LocalDeployBanner";

const mockListLocalNimDeployments = vi.fn();
const mockListStudentModels = vi.fn();

vi.mock("@/api/nim", () => ({
  listLocalNimDeployments: (...args: unknown[]) => mockListLocalNimDeployments(...args),
}));

vi.mock("@/api/students", () => ({
  listStudentModels: (...args: unknown[]) => mockListStudentModels(...args),
}));

function wrap(path: string) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false, staleTime: 0 } },
  });
  function Wrapper({ children }: { children: ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>
        <MemoryRouter initialEntries={[path]}>
          <Routes>
            <Route
              path="/projects/:projectId/compare"
              element={<div data-testid="compare-route" />}
            />
            <Route path="/projects/:projectId/*" element={children} />
            <Route path="/" element={children} />
          </Routes>
        </MemoryRouter>
      </QueryClientProvider>
    );
  }
  return Wrapper;
}

function failedDeployment(overrides: Record<string, unknown> = {}) {
  return {
    local_nim_deployment_id: "f1",
    project_id: "test-pid",
    model_config_id: "mc",
    role: "teacher",
    nim_container_image: "nvcr.io/nim/nvidia/cosmos3-reasoner:1.7.0",
    container_name: "c",
    container_id: null,
    host_port: 8000,
    endpoint_url: "http://localhost:8000",
    gpu_assignment: "0",
    status: "failed",
    status_reason: "Health check timed out after 1200s",
    deployed_at: null,
    stopped_at: null,
    created_at: new Date().toISOString(),
    matches_active_role_config: true,
    ...overrides,
  };
}

describe("LocalDeployBanner", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    window.localStorage.clear();
    mockListStudentModels.mockResolvedValue({ items: [], next_cursor: null });
  });

  it("renders nothing on non-project routes", async () => {
    const Wrapper = wrap("/");
    const { container } = render(<LocalDeployBanner />, { wrapper: Wrapper });
    // Wait a tick to be sure no async render happened.
    await new Promise((r) => setTimeout(r, 0));
    expect(container.querySelector('[data-testid="local-deploy-banner"]')).toBeNull();
    expect(mockListLocalNimDeployments).not.toHaveBeenCalled();
  });

  it("renders nothing when no deployments are in 'starting' state", async () => {
    mockListLocalNimDeployments.mockResolvedValue({
      items: [{ status: "running", nim_container_image: "x", created_at: "z" }],
    });
    const Wrapper = wrap("/projects/test-pid/labeling");
    render(<LocalDeployBanner />, { wrapper: Wrapper });
    await waitFor(() =>
      expect(mockListLocalNimDeployments).toHaveBeenCalledWith("test-pid"),
    );
    expect(screen.queryByTestId("local-deploy-banner")).toBeNull();
  });

  it("renders the deploying banner with model name when a starting deployment exists", async () => {
    mockListLocalNimDeployments.mockResolvedValue({
      items: [
        {
          local_nim_deployment_id: "d1",
          project_id: "test-pid",
          model_config_id: "mc",
          role: "teacher",
          nim_container_image: "nvcr.io/nim/nvidia/cosmos-reason2-8b:1.6.0",
          container_name: "c",
          container_id: null,
          host_port: 8000,
          endpoint_url: "http://localhost:8000",
          gpu_assignment: "0",
          status: "starting",
          status_reason: null,
          deployed_at: null,
          stopped_at: null,
          created_at: new Date().toISOString(),
        },
      ],
    });
    const Wrapper = wrap("/projects/test-pid/labeling");
    render(<LocalDeployBanner />, { wrapper: Wrapper });

    const banner = await screen.findByTestId("local-deploy-banner");
    expect(banner).toHaveTextContent(/cosmos-reason2-8b deploying/i);
    expect(banner).toHaveTextContent(/switchable from the Teacher picker/i);
  });

  it("describes a temporary Student NIM as serving validation, not a Teacher", async () => {
    mockListLocalNimDeployments.mockResolvedValue({
      items: [
        {
          ...failedDeployment(),
          local_nim_deployment_id: "student-starting",
          role: "student",
          status: "starting",
          status_reason: null,
        },
      ],
    });
    const Wrapper = wrap("/projects/test-pid/training/suite-1");
    render(<LocalDeployBanner />, { wrapper: Wrapper });

    const banner = await screen.findByTestId("local-deploy-banner");
    expect(banner).toHaveTextContent(/Student NIM deploying/i);
    expect(banner).toHaveTextContent(/temporary serving validation is running/i);
    expect(banner).not.toHaveTextContent(/Teacher picker/i);
  });

  it("routes a failed Student validation to Compare", async () => {
    mockListLocalNimDeployments.mockResolvedValue({
      items: [failedDeployment({ role: "student" })],
    });
    const Wrapper = wrap("/projects/test-pid/training/suite-1");
    render(<LocalDeployBanner />, { wrapper: Wrapper });

    const cta = await screen.findByTestId("local-deploy-banner-fix-cta");
    expect(cta).toHaveTextContent("Open Compare");
    fireEvent.click(cta);
    expect(await screen.findByTestId("compare-route")).toBeInTheDocument();
  });

  it("does not show an older Student deployment failure during durable lifecycle preflight", async () => {
    mockListLocalNimDeployments.mockResolvedValue({
      items: [failedDeployment({ role: "student", local_nim_deployment_id: "old" })],
    });
    mockListStudentModels.mockResolvedValue({
      items: [{ student_model_id: "student-new", serving_status: "pending" }],
      next_cursor: null,
    });
    const Wrapper = wrap("/projects/test-pid/training/suite-1");
    render(<LocalDeployBanner />, { wrapper: Wrapper });

    await waitFor(() =>
      expect(mockListStudentModels).toHaveBeenCalledWith("test-pid", { limit: 200 }),
    );
    expect(screen.queryByTestId("local-deploy-banner")).toBeNull();
  });

  it("renders the failure banner with reason when the failed config is still the active Teacher", async () => {
    mockListLocalNimDeployments.mockResolvedValue({
      items: [failedDeployment()],
    });
    const Wrapper = wrap("/projects/test-pid/labeling");
    render(<LocalDeployBanner />, { wrapper: Wrapper });

    const banner = await screen.findByTestId("local-deploy-banner");
    expect(banner).toHaveAttribute("data-banner-variant", "failed");
    expect(banner).toHaveTextContent(/cosmos3-reasoner deploy failed/i);
    expect(banner).toHaveTextContent(/Health check timed out/i);
    expect(screen.getByTestId("local-deploy-banner-dismiss")).toBeInTheDocument();
  });

  it("suppresses a failure whose model config the active Teacher no longer references", async () => {
    mockListLocalNimDeployments.mockResolvedValue({
      items: [failedDeployment({ matches_active_role_config: false })],
    });
    const Wrapper = wrap("/projects/test-pid/labeling");
    render(<LocalDeployBanner />, { wrapper: Wrapper });

    await waitFor(() =>
      expect(mockListLocalNimDeployments).toHaveBeenCalledWith("test-pid"),
    );
    expect(screen.queryByTestId("local-deploy-banner")).toBeNull();
  });

  it("dismiss hides the failure persistently, but a NEW failure id surfaces again", async () => {
    mockListLocalNimDeployments.mockResolvedValue({
      items: [failedDeployment({ local_nim_deployment_id: "old-failure" })],
    });
    const Wrapper = wrap("/projects/test-pid/labeling");
    const { unmount } = render(<LocalDeployBanner />, { wrapper: Wrapper });

    await screen.findByTestId("local-deploy-banner");
    fireEvent.click(screen.getByTestId("local-deploy-banner-dismiss"));
    await waitFor(() => expect(screen.queryByTestId("local-deploy-banner")).toBeNull());

    // Remount (fresh page load): the dismissal persists via localStorage…
    unmount();
    const Wrapper2 = wrap("/projects/test-pid/labeling");
    render(<LocalDeployBanner />, { wrapper: Wrapper2 });
    await waitFor(() => expect(mockListLocalNimDeployments).toHaveBeenCalled());
    expect(screen.queryByTestId("local-deploy-banner")).toBeNull();

    // …but a NEW failed deployment (different id) is not covered by it.
    mockListLocalNimDeployments.mockResolvedValue({
      items: [failedDeployment({ local_nim_deployment_id: "new-failure" })],
    });
    const Wrapper3 = wrap("/projects/test-pid/labeling");
    render(<LocalDeployBanner />, { wrapper: Wrapper3 });
    expect(await screen.findByTestId("local-deploy-banner")).toHaveAttribute(
      "data-banner-variant",
      "failed",
    );
  });
});
