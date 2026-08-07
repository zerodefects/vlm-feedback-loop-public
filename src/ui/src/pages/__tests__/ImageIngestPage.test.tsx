// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Unit tests for the Image Ingestion page — verifies its rendering
 * states (file browser, ingestion progress, completion summary, and
 * the path/permission/browse-disabled error states).
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { installEventSourceMock } from "@/test/event-source-mock";
import { suppressActWarnings } from "@/test/suppress-act-warnings";
import { makeEnvironmentResponse, makeProjectResponse } from "@/test/fixtures";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider, useQuery } from "@tanstack/react-query";
import {
  MemoryRouter,
  Routes,
  Route,
  Outlet,
  useOutletContext,
} from "react-router-dom";
import type { ReactNode } from "react";
import { ImageIngestPage } from "@/pages/ImageIngestPage";
import { projectKeys } from "@/api/query-keys";
import type { EnvironmentResponse } from "@/types/nim";

// ---------------------------------------------------------------------------
// Mock API
// ---------------------------------------------------------------------------

const mockBrowseFilesystem = vi.fn();
const mockScanDirectory = vi.fn();
const mockIngestExamples = vi.fn();

vi.mock("@/api/filesystem", () => ({
  browseFilesystem: (...args: unknown[]) => mockBrowseFilesystem(...args),
  scanDirectory: (...args: unknown[]) => mockScanDirectory(...args),
  ingestExamples: (...args: unknown[]) => mockIngestExamples(...args),
}));

const mockFetchProject = vi.fn();
const mockFetchEnvironment = vi.fn();

vi.mock("@/api/projects", () => ({
  fetchProject: (...args: unknown[]) => mockFetchProject(...args),
}));

vi.mock("@/api/nim", () => ({
  fetchEnvironment: (...args: unknown[]) => mockFetchEnvironment(...args),
}));

// Mock EventSource
installEventSourceMock();

// Known-benign act warnings: the batch-ingest loop resumes after each
// deferred-mock `ingestExamples` resolution in microtasks between
// `waitFor` polls, so its per-batch progress setters fire outside act
// (see "shows [Start labeling] in progress card after first batch").
suppressActWarnings();

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const PROJECT = makeProjectResponse({
  project_id: "test-pid",
  name: "Test Project",
  description: null,
  project_dir: "/tmp/workspace/projects/test-pid",
  created_at: "2026-04-14T00:00:00Z",
  updated_at: "2026-04-14T00:00:00Z",
  teacher_model_config_id: "mc-1",
  active_guidance_id: null,
  active_student_model_config_id: null,
  phash_algorithm: "dct_phash_64",
  embedding_provider: "none",
  embedding_model_id: null,
  embedding_dim: null,
  counts: {
    verified: 0,
    unlabeled: 2,
    auto_labeled: 0,
    omitted: 0,
    pending_relabel: 0,
    prior_relabeled: 0,
  },
});

const ENVIRONMENT = {
  ...makeEnvironmentResponse(),
  embedding_deployment: null,
};

const BROWSE_ROOT = {
  path: "/",
  parent: null,
  entries: [
    { name: "data", type: "directory" as const, path: "/data", size_bytes: null },
    { name: "tmp", type: "directory" as const, path: "/tmp", size_bytes: null },
  ],
};

const BROWSE_CONFIGURED_ROOT = {
  path: "/data/images",
  parent: null,
  entries: [
    {
      name: "paper",
      type: "directory" as const,
      path: "/data/images/paper",
      size_bytes: null,
    },
  ],
};

const SCAN_RESULT = {
  path: "/data/images",
  images: [
    {
      storage_ref: "/data/images/img_001.jpg",
      suggested_example_key: "img1",
      size_bytes: 2048576,
      key_status: "available" as const,
      existing_storage_ref: null,
    },
    {
      storage_ref: "/data/images/img_002.png",
      suggested_example_key: "img2",
      size_bytes: 1400000,
      key_status: "available" as const,
      existing_storage_ref: null,
    },
  ],
  skipped: [{ path: "/data/images/notes.txt", reason: "unsupported_format" }],
  total_images: 2,
  total_skipped: 1,
  total_collisions: 0,
};

const INGEST_RESPONSE = {
  results: [
    {
      example_key: "img1",
      status: "created" as const,
      error: null,
      error_code: null,
      warnings: [],
      example: { example_key: "img1", state: "Unlabeled" },
    },
    {
      example_key: "img2",
      status: "created" as const,
      error: null,
      error_code: null,
      warnings: [],
      example: { example_key: "img2", state: "Unlabeled" },
    },
  ],
};

// ---------------------------------------------------------------------------
// Wrapper — renders inside ProjectSetupLayout context
// ---------------------------------------------------------------------------

function createWrapper(projectOverrides?: Record<string, unknown>) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false, staleTime: 0 } },
  });

  const projectData = { ...PROJECT, ...projectOverrides };

  // Provide the SetupContext via a mock Outlet
  function MockSetupLayout() {
    return (
      <Outlet
        context={{
          projectId: "test-pid",
          project: projectData,
          environment: ENVIRONMENT,
        }}
      />
    );
  }

  function Wrapper({ children }: { children: ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>
        <MemoryRouter initialEntries={["/projects/test-pid/ready"]}>
          <Routes>
            <Route path="/projects/:projectId" element={<MockSetupLayout />}>
              <Route path="ready" element={children} />
              <Route path="labeling" element={<div data-testid="labeling-page" />} />
              <Route
                path="create-guidance"
                element={<div data-testid="create-guidance-page" />}
              />
            </Route>
          </Routes>
        </MemoryRouter>
      </QueryClientProvider>
    );
  }
  return { Wrapper, queryClient };
}

function createReactiveWrapper(
  initialProject: typeof PROJECT,
  environment: EnvironmentResponse = makeEnvironmentResponse(),
) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false, staleTime: Infinity } },
  });

  function MockSetupLayout() {
    const { data: project } = useQuery({
      queryKey: projectKeys.detail("test-pid"),
      queryFn: () => mockFetchProject(),
      initialData: initialProject,
      staleTime: Infinity,
    });
    return (
      <Outlet
        context={{
          projectId: "test-pid",
          project,
          environment,
        }}
      />
    );
  }

  function LabelingProjectCount() {
    const { project } = useOutletContext<{ project: typeof PROJECT }>();
    return <div data-testid="labeling-unlabeled-count">{project.counts.unlabeled}</div>;
  }

  function Wrapper({ children }: { children: ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>
        <MemoryRouter initialEntries={["/projects/test-pid/ready"]}>
          <Routes>
            <Route path="/projects/:projectId" element={<MockSetupLayout />}>
              <Route path="ready" element={children} />
              <Route path="labeling" element={<LabelingProjectCount />} />
            </Route>
          </Routes>
        </MemoryRouter>
      </QueryClientProvider>
    );
  }

  return { Wrapper, queryClient };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

beforeEach(() => {
  vi.clearAllMocks();
  localStorage.clear();
});

describe("ImageIngestPage", () => {
  // ── File Browser ──────────────────────────────────────────────────────

  it("renders the file browser with directory entries", async () => {
    mockBrowseFilesystem.mockResolvedValueOnce(BROWSE_ROOT);
    const { Wrapper } = createWrapper();

    render(<ImageIngestPage />, { wrapper: Wrapper });

    await waitFor(() => {
      expect(screen.getByTestId("image-ingest-page")).toBeInTheDocument();
    });

    expect(screen.getByText("Ingest Images")).toBeInTheDocument();
    expect(screen.getByRole("textbox", { name: "Path" })).toBe(
      screen.getByTestId("path-input"),
    );

    await waitFor(() => {
      expect(screen.getByTestId("file-browser")).toBeInTheDocument();
    });
    expect(mockBrowseFilesystem).toHaveBeenCalledWith(undefined);
    expect(screen.getByTestId("browse-parent-button")).toBeDisabled();
    expect(screen.getByTestId("browse-root-boundary")).toHaveTextContent(
      "filesystem root",
    );
  });

  it("moves up one folder from the explicit filesystem control", async () => {
    mockBrowseFilesystem
      .mockResolvedValueOnce(BROWSE_CONFIGURED_ROOT)
      .mockResolvedValueOnce({
        path: "/data/images/paper",
        parent: "/data/images",
        entries: [
          {
            name: "paper-001.png",
            type: "file" as const,
            path: "/data/images/paper/paper-001.png",
            size_bytes: 1024,
          },
        ],
        bundled_sample_path: null,
      })
      .mockResolvedValueOnce(BROWSE_CONFIGURED_ROOT);
    const { Wrapper } = createWrapper();
    const user = userEvent.setup();

    render(<ImageIngestPage />, { wrapper: Wrapper });

    await user.click(await screen.findByRole("button", { name: "paper/" }));
    const upButton = screen.getByRole("button", { name: "Up one folder" });
    expect(upButton).toBeEnabled();
    await user.click(upButton);

    await waitFor(() =>
      expect(mockBrowseFilesystem).toHaveBeenLastCalledWith("/data/images"),
    );
    expect(await screen.findByRole("button", { name: "paper/" })).toBeInTheDocument();
  });

  it("explains the configured image-root boundary when no parent is available", async () => {
    mockBrowseFilesystem.mockResolvedValueOnce(BROWSE_CONFIGURED_ROOT);
    const { Wrapper } = createWrapper();

    render(<ImageIngestPage />, { wrapper: Wrapper });

    expect(await screen.findByTestId("browse-parent-button")).toBeDisabled();
    expect(screen.getByTestId("browse-root-boundary")).toHaveTextContent(
      /configured image root.*change IMAGE_ROOT/i,
    );
  });

  it("starts beside the shipped sample so its root folder is selectable", async () => {
    const samplePath = "/repo/deploy/example-images";
    mockBrowseFilesystem
      .mockResolvedValueOnce({
        ...BROWSE_ROOT,
        bundled_sample_path: samplePath,
      })
      .mockResolvedValueOnce({
        path: "/repo/deploy",
        parent: "/repo",
        bundled_sample_path: samplePath,
        entries: [
          {
            name: "example-images",
            type: "directory" as const,
            path: samplePath,
            size_bytes: null,
          },
        ],
      });
    const { Wrapper } = createWrapper();

    render(<ImageIngestPage />, { wrapper: Wrapper });

    await waitFor(() =>
      expect(mockBrowseFilesystem).toHaveBeenLastCalledWith("/repo/deploy"),
    );
    expect(
      await screen.findByRole("checkbox", { name: "Select directory example-images" }),
    ).toBeInTheDocument();
    expect(screen.queryByText(/use bundled sample/i)).toBeNull();
  });

  it("hides recent paths that fall outside the backend-selected root", async () => {
    localStorage.setItem(
      "vlm-ingest-recent-paths",
      JSON.stringify(["/data/images/paper", "/mnt/old-dataset"]),
    );
    mockBrowseFilesystem.mockResolvedValueOnce(BROWSE_CONFIGURED_ROOT);
    const { Wrapper } = createWrapper();

    render(<ImageIngestPage />, { wrapper: Wrapper });

    await waitFor(() => expect(screen.getByTestId("file-browser")).toBeInTheDocument());
    expect(screen.getByRole("button", { name: "/data/images/paper" })).toBeVisible();
    expect(screen.queryByRole("button", { name: "/mnt/old-dataset" })).toBeNull();
  });

  // ── Footer selection summary ──────────────────────────────────────────
  //
  // The summary sits beside the [Ingest Selected] CTA so the disabled
  // state self-explains: with nothing checked the footer reads
  // "0 selected", and checking items updates the count in place.

  it("footer shows '0 selected' beside the disabled CTA when nothing is checked", async () => {
    mockBrowseFilesystem.mockResolvedValueOnce(BROWSE_ROOT);
    const { Wrapper } = createWrapper();

    render(<ImageIngestPage />, { wrapper: Wrapper });

    await waitFor(() => expect(screen.getByTestId("file-browser")).toBeInTheDocument());
    expect(screen.getByText("0 selected")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Ingest Selected" })).toBeDisabled();
  });

  it("footer summary updates to the checked count with a folder/file breakdown", async () => {
    mockBrowseFilesystem.mockResolvedValueOnce(BROWSE_ROOT);
    const { Wrapper } = createWrapper();
    const user = userEvent.setup();

    render(<ImageIngestPage />, { wrapper: Wrapper });

    await waitFor(() => expect(screen.getByTestId("file-browser")).toBeInTheDocument());
    const checkbox = screen.getAllByRole("checkbox")[0];
    await user.click(checkbox);

    expect(screen.getByText("1 selected (1 folder)")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Ingest Selected" })).toBeEnabled();
  });

  // ── Ingestion flow ───────────────────────────────────────────────────
  //
  // The ingestion tests below drive the file-browser path: check a
  // directory in the tree, click [Ingest Selected]. ``handleIngestSelected``
  // calls ``scanDirectory`` internally to expand the selected directory;
  // the skipped-files list is discarded so the wall-of-errors UI failure
  // mode is structurally impossible. There is no Scan button or standalone
  // scan-preview screen.

  // ── [Start labeling] appears mid-ingest ─────────────────────────────
  //
  // The backend's :ingest endpoint creates skeleton rows + returns
  // 202, so any accepted image is already labelable. The SME should be
  // able to hop out as soon as the first batch lands — they should NOT
  // have to wait for the entire progress bar to fill.

  it("refreshes project counts before mid-ingest [Start labeling] navigation", async () => {
    // Enough images to require multiple ramped batches.
    const TWO_BATCH_SCAN = {
      ...SCAN_RESULT,
      images: Array.from({ length: 250 }, (_, i) => ({
        storage_ref: `/data/images/img_${i}.png`,
        suggested_example_key: `img_${i}--hash`,
        size_bytes: 1000,
        key_status: "available" as const,
        existing_storage_ref: null,
      })),
      total_images: 250,
      total_skipped: 0,
      total_collisions: 0,
    };

    // Deferred-resolution mock: each ingestExamples call holds open
    // until the test resolves it. The first call we resolve immediately
    // (first batch lands → button should appear); the second we leave
    // pending so the loop is still active when we assert.
    type Deferred = {
      resolve: (v: { results: Array<Record<string, unknown>> }) => void;
      done: Promise<{ results: Array<Record<string, unknown>> }>;
    };
    const deferreds: Deferred[] = [];
    mockBrowseFilesystem.mockResolvedValueOnce(BROWSE_ROOT);
    mockScanDirectory.mockResolvedValueOnce(TWO_BATCH_SCAN);
    mockIngestExamples.mockImplementation(
      (_projectId: string, body: { examples: Array<{ example_key: string }> }) => {
        let resolveFn!: (v: { results: Array<Record<string, unknown>> }) => void;
        const done = new Promise<{ results: Array<Record<string, unknown>> }>(
          (resolve) => {
            resolveFn = resolve;
          },
        );
        const d: Deferred = { resolve: resolveFn, done };
        deferreds.push(d);
        // First call: resolve immediately so the first batch lands.
        if (deferreds.length === 1) {
          d.resolve({
            results: body.examples.map((e) => ({
              example_key: e.example_key,
              status: "created",
              error: null,
              error_code: null,
              warnings: [],
              example: { example_key: e.example_key, state: "Unlabeled" },
            })),
          });
        }
        return d.done;
      },
    );

    const { Wrapper } = createReactiveWrapper({
      ...PROJECT,
      active_guidance_id: "g-1",
      embedding_provider: "hosted_nvclip",
    });
    const user = userEvent.setup();
    render(<ImageIngestPage />, { wrapper: Wrapper });

    await waitFor(() => expect(screen.getByTestId("file-browser")).toBeInTheDocument());
    const checkbox = screen.getAllByRole("checkbox")[0];
    await user.click(checkbox);
    await user.click(screen.getByText("Ingest Selected"));

    // Wait for the first batch to settle. Once accepted > 0 the
    // in-progress [Start labeling] CTA should be visible.
    await waitFor(
      () =>
        expect(screen.getByTestId("in-progress-start-labeling")).toBeInTheDocument(),
      { timeout: 3000 },
    );
    // We are still in the "ingesting" state (the second batch is
    // pending), not in the "queued" summary screen.
    expect(screen.getByTestId("ingestion-progress")).toBeInTheDocument();
    expect(screen.queryByTestId("ingestion-summary")).toBeNull();
    expect(
      screen.getByText("Computing CLIP embeddings via NVIDIA hosted NIM…"),
    ).toBeInTheDocument();

    await user.click(screen.getByTestId("in-progress-start-labeling"));
    expect(await screen.findByTestId("labeling-unlabeled-count")).toHaveTextContent(
      "12",
    );
    // Drain the held batch, then let later ramp batches resolve immediately
    // so no async ingestion work leaks into the next test.
    await waitFor(() => expect(deferreds.length).toBeGreaterThan(1));
    mockIngestExamples.mockResolvedValue({ results: [] });
    deferreds[1].resolve({ results: [] });
  });

  // ── Completion Summary ────────────────────────────────────────────────

  it("shows completion summary with counts", async () => {
    mockBrowseFilesystem.mockResolvedValueOnce(BROWSE_ROOT);
    mockScanDirectory.mockResolvedValueOnce(SCAN_RESULT);
    mockIngestExamples.mockResolvedValueOnce(INGEST_RESPONSE);
    const { Wrapper } = createWrapper();
    const user = userEvent.setup();

    render(<ImageIngestPage />, { wrapper: Wrapper });

    await waitFor(() => expect(screen.getByTestId("file-browser")).toBeInTheDocument());
    const checkbox = screen.getAllByRole("checkbox")[0];
    await user.click(checkbox);
    await user.click(screen.getByText("Ingest Selected"));

    await waitFor(() =>
      expect(screen.getByTestId("ingestion-summary")).toBeInTheDocument(),
    );

    expect(screen.getByText("Ingestion Complete")).toBeInTheDocument();
    expect(screen.getByText(/Accepted: 2/)).toBeInTheDocument();
    // There is no "queued for processing" body note — it is
    // redundant with the CLIP-embeddings line below. Background
    // pHash sweep + CLIP compute still happen — they just don't get
    // a dedicated UI line.
    expect(screen.queryByTestId("ingestion-background-note")).toBeNull();
    // The embedding line is provider-aware. Fixture has
    // ``embedding_provider: "none"`` so the truth-telling copy is the
    // "Embeddings unavailable" advisory plus the [Configure] CTA.
    expect(
      screen.getByText(/Embeddings unavailable — diversity selection uses pHash\./),
    ).toBeInTheDocument();
    expect(screen.getByTestId("configure-embeddings-cta")).toBeInTheDocument();
    expect(screen.getByText("Add More Images")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Start labeling/ })).toBeInTheDocument();
  });

  it("keeps hosted embedding activation neutral until none transitions to hosted", async () => {
    mockBrowseFilesystem.mockResolvedValueOnce(BROWSE_ROOT);
    mockScanDirectory.mockResolvedValueOnce(SCAN_RESULT);
    mockIngestExamples.mockResolvedValueOnce(INGEST_RESPONSE);
    let resolveProject!: (project: typeof PROJECT) => void;
    mockFetchProject.mockReturnValue(
      new Promise<typeof PROJECT>((resolve) => {
        resolveProject = resolve;
      }),
    );
    const hostedEnvironment = makeEnvironmentResponse({
      nvidia_api_key_configured: true,
      hosted_nim_available: true,
    });
    const { Wrapper } = createReactiveWrapper(PROJECT, hostedEnvironment);
    const user = userEvent.setup();

    render(<ImageIngestPage />, { wrapper: Wrapper });
    await waitFor(() => expect(screen.getByTestId("file-browser")).toBeInTheDocument());
    await user.click(screen.getAllByRole("checkbox")[0]);
    await user.click(screen.getByText("Ingest Selected"));

    expect(
      await screen.findByText("Confirming the configured embedding provider…"),
    ).toBeInTheDocument();
    expect(screen.queryByText(/Embeddings unavailable/)).toBeNull();

    resolveProject({
      ...PROJECT,
      embedding_provider: "hosted_nvclip",
      embedding_model_id: "nvidia/llama-nemotron-embed-vl-1b-v2",
      embedding_dim: 2048,
    });

    expect(
      await screen.findByText("Computing CLIP embeddings via NVIDIA hosted NIM…"),
    ).toBeInTheDocument();
    expect(screen.queryByTestId("configure-embeddings-cta")).toBeNull();
  });

  it("reconciles none to the active local embedding provider", async () => {
    mockBrowseFilesystem.mockResolvedValueOnce(BROWSE_ROOT);
    mockScanDirectory.mockResolvedValueOnce(SCAN_RESULT);
    mockIngestExamples.mockResolvedValueOnce(INGEST_RESPONSE);
    mockFetchProject.mockResolvedValue({
      ...PROJECT,
      embedding_provider: "self_hosted_nvclip",
      embedding_model_id: "nvidia/llama-nemotron-embed-vl-1b-v2",
      embedding_dim: 2048,
    });
    const localEnvironment = makeEnvironmentResponse({
      local_deploy_available: true,
      recommended_embedding_mode: "local",
      embedding_deployment: {
        ...makeEnvironmentResponse().embedding_deployment,
        fits: true,
        provider: "self_hosted_nvclip",
      },
    });
    const { Wrapper } = createReactiveWrapper(PROJECT, localEnvironment);
    const user = userEvent.setup();

    render(<ImageIngestPage />, { wrapper: Wrapper });
    await waitFor(() => expect(screen.getByTestId("file-browser")).toBeInTheDocument());
    await user.click(screen.getAllByRole("checkbox")[0]);
    await user.click(screen.getByText("Ingest Selected"));

    expect(
      await screen.findByText("Computing CLIP embeddings via local embedding NIM…"),
    ).toBeInTheDocument();
    expect(screen.queryByText(/Embeddings unavailable/)).toBeNull();
    expect(screen.queryByTestId("configure-embeddings-cta")).toBeNull();
  });

  // ── Low Image Count Warning ──────────────────────────────────────────

  it("shows low image count warning when fewer than 150 images are accepted", async () => {
    mockBrowseFilesystem.mockResolvedValueOnce(BROWSE_ROOT);
    mockScanDirectory.mockResolvedValueOnce(SCAN_RESULT); // only 2 images
    mockIngestExamples.mockResolvedValueOnce(INGEST_RESPONSE);
    const { Wrapper } = createWrapper();
    const user = userEvent.setup();

    render(<ImageIngestPage />, { wrapper: Wrapper });

    await waitFor(() => expect(screen.getByTestId("file-browser")).toBeInTheDocument());
    const checkbox = screen.getAllByRole("checkbox")[0];
    await user.click(checkbox);
    await user.click(screen.getByText("Ingest Selected"));

    await waitFor(() =>
      expect(screen.getByTestId("ingestion-summary")).toBeInTheDocument(),
    );

    // Low count warning should be visible (< 150 images)
    expect(screen.getByTestId("low-count-warning")).toBeInTheDocument();
    expect(screen.getByText(/at least 150 images/)).toBeInTheDocument();
  });

  it("identifies the bundled sample as a walkthrough rather than a scale-up dataset", async () => {
    const samplePath = "/repo/deploy/example-images";
    mockBrowseFilesystem.mockResolvedValueOnce({
      path: samplePath,
      parent: "/repo/deploy",
      bundled_sample_path: samplePath,
      entries: [
        {
          name: "rock",
          type: "directory" as const,
          path: `${samplePath}/rock`,
          size_bytes: null,
        },
      ],
    });
    mockScanDirectory.mockResolvedValueOnce(SCAN_RESULT);
    mockIngestExamples.mockResolvedValueOnce(INGEST_RESPONSE);
    const { Wrapper } = createWrapper();
    const user = userEvent.setup();

    render(<ImageIngestPage />, { wrapper: Wrapper });
    await user.click((await screen.findAllByRole("checkbox"))[0]);
    await user.click(screen.getByText("Ingest Selected"));

    expect(
      await screen.findByText(
        /bundled sample is a walkthrough, not a Scale-Up-ready dataset/,
      ),
    ).toBeInTheDocument();
  });

  // ── Path Not Found ───────────────────────────────────────────────────

  it("shows path not found error", async () => {
    const { ApiError } = await import("@/api/client");
    mockBrowseFilesystem.mockResolvedValueOnce(BROWSE_ROOT); // initial browse
    // Pressing Enter on a typed path drives ``doBrowse``; surface a 404.
    mockBrowseFilesystem.mockRejectedValueOnce(
      new ApiError(
        404,
        JSON.stringify({ detail: "Directory not found: /nonexistent" }),
      ),
    );
    const { Wrapper } = createWrapper();
    const user = userEvent.setup();

    render(<ImageIngestPage />, { wrapper: Wrapper });

    await waitFor(() => expect(screen.getByTestId("path-input")).toBeInTheDocument());
    const input = screen.getByTestId("path-input");
    await user.clear(input);
    await user.type(input, "/nonexistent");
    await user.keyboard("{Enter}");

    await waitFor(() => {
      expect(screen.getByTestId("path-error")).toBeInTheDocument();
    });

    expect(screen.getByText(/Directory not found/)).toBeInTheDocument();
  });

  // ── Permission Denied ────────────────────────────────────────────────

  it("shows permission denied error", async () => {
    const { ApiError } = await import("@/api/client");
    mockBrowseFilesystem.mockResolvedValueOnce(BROWSE_ROOT); // initial browse
    mockBrowseFilesystem.mockRejectedValueOnce(
      new ApiError(403, JSON.stringify({ detail: "Permission denied: /restricted" })),
    );
    const { Wrapper } = createWrapper();
    const user = userEvent.setup();

    render(<ImageIngestPage />, { wrapper: Wrapper });

    await waitFor(() => expect(screen.getByTestId("path-input")).toBeInTheDocument());
    const input = screen.getByTestId("path-input");
    await user.clear(input);
    await user.type(input, "/restricted");
    await user.keyboard("{Enter}");

    await waitFor(() => {
      expect(screen.getByTestId("path-error")).toBeInTheDocument();
    });

    expect(screen.getByText(/Permission denied/)).toBeInTheDocument();
  });

  it("preserves an IMAGE_ROOT boundary error instead of calling it permission denied", async () => {
    const { ApiError } = await import("@/api/client");
    mockBrowseFilesystem.mockResolvedValueOnce(BROWSE_ROOT);
    mockBrowseFilesystem.mockRejectedValueOnce(
      new ApiError(
        403,
        JSON.stringify({ detail: "Path is outside IMAGE_ROOT: /home" }),
      ),
    );
    const { Wrapper } = createWrapper();
    const user = userEvent.setup();

    render(<ImageIngestPage />, { wrapper: Wrapper });

    await waitFor(() => expect(screen.getByTestId("path-input")).toBeInTheDocument());
    const input = screen.getByTestId("path-input");
    await user.clear(input);
    await user.type(input, "/home");
    await user.keyboard("{Enter}");

    await waitFor(() => {
      expect(screen.getByText("Path is outside IMAGE_ROOT: /home")).toBeInTheDocument();
    });
    expect(screen.queryByText("Permission denied: /home")).not.toBeInTheDocument();
  });

  // ── Browse Disabled ──────────────────────────────────────────────────

  it("shows browse disabled message", async () => {
    const { ApiError } = await import("@/api/client");
    mockBrowseFilesystem.mockRejectedValueOnce(
      new ApiError(
        403,
        JSON.stringify({
          detail:
            "Filesystem browsing is disabled. Configure IMAGE_ROOT to allow browsing when the backend is network-accessible.",
        }),
      ),
    );
    const { Wrapper } = createWrapper();

    render(<ImageIngestPage />, { wrapper: Wrapper });

    await waitFor(() => {
      expect(screen.getByTestId("browse-disabled")).toBeInTheDocument();
    });

    const disabledEl = screen.getByTestId("browse-disabled");
    expect(disabledEl).toBeInTheDocument();
    expect(disabledEl.textContent).toContain("Filesystem browsing is disabled");
    expect(disabledEl.textContent).toContain("IMAGE_ROOT");
    expect(screen.getByText("Previous screen")).toBeInTheDocument();
  });

  // ── Batch dispatch invariant ─────────────────────────────────────────
  //
  // Batches dispatch sequentially, never in concurrent windows:
  // concurrent ingest POSTs fight the pHash sweeper + CLIP worker for
  // the single project DB's SQLite write lock, some hitting
  // busy_timeout and returning 500. This test pins the sequential
  // behavior: never more than 1 in flight at any moment, while the
  // batch SHAPE is preserved (small first batch + 200-batches).
  it("runs ingestExamples sequentially (stability tune)", async () => {
    const TOTAL_IMAGES = 1000;
    const EXPECTED_MAX_CONCURRENCY = 1;

    // Build a large scan result with TOTAL_IMAGES "available" rows.
    const largeScan = {
      ...SCAN_RESULT,
      images: Array.from({ length: TOTAL_IMAGES }, (_, i) => ({
        storage_ref: `/data/images/img_${i}.png`,
        suggested_example_key: `img_${i}--hash`,
        size_bytes: 1000,
        key_status: "available" as const,
        existing_storage_ref: null,
      })),
      total_images: TOTAL_IMAGES,
      total_skipped: 0,
      total_collisions: 0,
    };

    // Tracking spy on the ingest mock: increments on entry, holds 20ms,
    // decrements on exit. ``maxInFlight`` records the high-water mark.
    let inFlight = 0;
    let maxInFlight = 0;
    mockBrowseFilesystem.mockResolvedValueOnce(BROWSE_ROOT);
    mockScanDirectory.mockResolvedValueOnce(largeScan);
    mockIngestExamples.mockImplementation(
      async (
        _projectId: string,
        body: { examples: Array<{ example_key: string }> },
      ) => {
        inFlight += 1;
        if (inFlight > maxInFlight) maxInFlight = inFlight;
        // Yield + brief hold so the chunker has time to dispatch the
        // full window before any call resolves.
        await new Promise<void>((resolve) => setTimeout(resolve, 20));
        inFlight -= 1;
        return {
          results: body.examples.map((e) => ({
            example_key: e.example_key,
            status: "created" as const,
            error: null,
            error_code: null,
            warnings: [],
            example: { example_key: e.example_key, state: "Unlabeled" },
          })),
        };
      },
    );

    const { Wrapper } = createWrapper();
    const user = userEvent.setup();
    render(<ImageIngestPage />, { wrapper: Wrapper });

    await waitFor(() => expect(screen.getByTestId("file-browser")).toBeInTheDocument());
    const checkbox = screen.getAllByRole("checkbox")[0];
    await user.click(checkbox);
    await user.click(screen.getByText("Ingest Selected"));

    // Wait for completion: the summary screen is the terminal state.
    await waitFor(
      () => expect(screen.getByTestId("ingestion-summary")).toBeInTheDocument(),
      { timeout: 5000 },
    );

    // Batch layout ramps from a small first batch up to the full size:
    // 1000 images → [10,20,40,80,160,200,200,200,90] = 9 batches dispatched.
    expect(mockIngestExamples).toHaveBeenCalledTimes(9);
    // Stability invariant: only one batch in flight at any moment so
    // there is no concurrent POST fight for the SQLite write lock.
    expect(maxInFlight).toBe(EXPECTED_MAX_CONCURRENCY);
  });

  // ── Immediate scanning feedback ──────────────────────────────────────
  //
  // Clicking [Ingest Selected] must give feedback BEFORE the directory
  // scans resolve: a recursive scan of a large tree takes seconds, and
  // with no state change the click looks like a no-op, so SMEs press
  // the button again or assume the app is broken.
  it("shows the scanning state immediately while directory scans are pending", async () => {
    mockBrowseFilesystem.mockResolvedValueOnce(BROWSE_ROOT);
    // Hold the scan open until the test resolves it.
    let resolveScan!: (v: typeof SCAN_RESULT) => void;
    mockScanDirectory.mockImplementationOnce(
      () => new Promise<typeof SCAN_RESULT>((resolve) => (resolveScan = resolve)),
    );
    mockIngestExamples.mockResolvedValue(INGEST_RESPONSE);

    const { Wrapper } = createWrapper();
    const user = userEvent.setup();
    render(<ImageIngestPage />, { wrapper: Wrapper });

    await waitFor(() => expect(screen.getByTestId("file-browser")).toBeInTheDocument());
    await user.click(screen.getAllByRole("checkbox")[0]);
    await user.click(screen.getByText("Ingest Selected"));

    // The scan is still pending — the scanning card must already be up
    // and the browse card (with its CTA) gone.
    expect(screen.getByTestId("ingest-scanning")).toBeInTheDocument();
    expect(screen.getByText(/Scanning 1 folder for images/)).toBeInTheDocument();
    expect(screen.queryByTestId("file-browser")).toBeNull();

    resolveScan(SCAN_RESULT);
    await waitFor(() =>
      expect(screen.getByTestId("ingestion-summary")).toBeInTheDocument(),
    );
  });

  it("returns to browse with an inline error when the selection yields no images", async () => {
    mockBrowseFilesystem.mockResolvedValueOnce(BROWSE_ROOT);
    mockScanDirectory.mockResolvedValueOnce({
      ...SCAN_RESULT,
      images: [],
      total_images: 0,
    });

    const { Wrapper } = createWrapper();
    const user = userEvent.setup();
    render(<ImageIngestPage />, { wrapper: Wrapper });

    await waitFor(() => expect(screen.getByTestId("file-browser")).toBeInTheDocument());
    await user.click(screen.getAllByRole("checkbox")[0]);
    await user.click(screen.getByText("Ingest Selected"));

    await waitFor(() => expect(screen.getByTestId("scan-error")).toBeInTheDocument());
    expect(
      screen.getByText("No ingestible images were found in the selection."),
    ).toBeInTheDocument();
    // Unlike path errors, scan errors keep the tree visible so the
    // selection can be adjusted and retried.
    expect(screen.getByTestId("file-browser")).toBeInTheDocument();
    expect(mockIngestExamples).not.toHaveBeenCalled();
  });

  it("surfaces failed scans instead of silently doing nothing", async () => {
    mockBrowseFilesystem.mockResolvedValueOnce(BROWSE_ROOT);
    mockScanDirectory.mockRejectedValueOnce(new TypeError("fetch failed"));

    const { Wrapper } = createWrapper();
    const user = userEvent.setup();
    render(<ImageIngestPage />, { wrapper: Wrapper });

    await waitFor(() => expect(screen.getByTestId("file-browser")).toBeInTheDocument());
    await user.click(screen.getAllByRole("checkbox")[0]);
    await user.click(screen.getByText("Ingest Selected"));

    await waitFor(() => expect(screen.getByTestId("scan-error")).toBeInTheDocument());
    expect(screen.getByText(/Could not scan 1 selected folder\./)).toBeInTheDocument();
    expect(mockIngestExamples).not.toHaveBeenCalled();
  });

  // ── Post-ingest routing ──────────────────────────────────────────────
  //
  // Both "Start labeling" CTAs exit via handleContinue. Only the FTUE
  // first run (no Guidance yet) should continue to Create Guidance;
  // re-entering ingest from a mature project ([Add Images], queue-empty)
  // must land back in labeling — otherwise the steady-state add-images
  // loop detours into a blank guidance editor whose Save would replace
  // the active Guidance.

  async function ingestToSummary(projectOverrides?: Record<string, unknown>) {
    mockBrowseFilesystem.mockResolvedValueOnce(BROWSE_ROOT);
    mockScanDirectory.mockResolvedValueOnce(SCAN_RESULT);
    mockIngestExamples.mockResolvedValueOnce(INGEST_RESPONSE);
    const { Wrapper } = createWrapper(projectOverrides);
    const user = userEvent.setup();
    render(<ImageIngestPage />, { wrapper: Wrapper });

    await waitFor(() => expect(screen.getByTestId("file-browser")).toBeInTheDocument());
    await user.click(screen.getAllByRole("checkbox")[0]);
    await user.click(screen.getByText("Ingest Selected"));
    await waitFor(() =>
      expect(screen.getByTestId("ingestion-summary")).toBeInTheDocument(),
    );
    return { user };
  }

  it("summary [Start labeling] returns to labeling when active Guidance exists", async () => {
    const { user } = await ingestToSummary({ active_guidance_id: "g-1" });

    await user.click(screen.getByRole("button", { name: /Start labeling/ }));

    await screen.findByTestId("labeling-page");
    expect(screen.queryByTestId("create-guidance-page")).toBeNull();
  });

  it("summary [Start labeling] continues to Create Guidance on the first run", async () => {
    const { user } = await ingestToSummary({ active_guidance_id: null });

    await user.click(screen.getByRole("button", { name: /Start labeling/ }));

    await screen.findByTestId("create-guidance-page");
  });

  // ── Backend key authority for individually selected files ───────────
  //
  // The backend's scan endpoint owns the example-key scheme (slug +
  // canonical-path hash with key_status dedupe). Individually checked files
  // must reuse it via a parent-directory scan — a client-side slug of
  // the basename collides on same-named files from different folders
  // (backend rejects the second with example_key_collision).

  it("individually selected files ingest with backend-generated keys from a parent-dir scan", async () => {
    const BROWSE_WITH_FILES = {
      path: "/data/images",
      parent: "/data",
      entries: [
        {
          name: "img 001.png",
          type: "file" as const,
          path: "/data/images/img 001.png",
          size_bytes: 1000,
        },
        {
          name: "img_002.png",
          type: "file" as const,
          path: "/data/images/img_002.png",
          size_bytes: 1000,
        },
      ],
    };
    // The parent scan returns rows for the whole directory — including a
    // file the SME did NOT select, which must be filtered out.
    const PARENT_SCAN = {
      path: "/data/images",
      images: [
        {
          storage_ref: "/data/images/img 001.png",
          suggested_example_key: "img_001_png--aaa111bbb222",
          size_bytes: 1000,
          key_status: "available" as const,
          existing_storage_ref: null,
        },
        {
          storage_ref: "/data/images/img_002.png",
          suggested_example_key: "img_002_png--ccc333ddd444",
          size_bytes: 1000,
          key_status: "available" as const,
          existing_storage_ref: null,
        },
        {
          storage_ref: "/data/images/unselected.png",
          suggested_example_key: "unselected_png--eee555fff666",
          size_bytes: 1000,
          key_status: "available" as const,
          existing_storage_ref: null,
        },
      ],
      skipped: [],
      total_images: 3,
      total_skipped: 0,
      total_collisions: 0,
    };
    mockBrowseFilesystem.mockResolvedValueOnce(BROWSE_WITH_FILES);
    mockScanDirectory.mockResolvedValueOnce(PARENT_SCAN);
    mockIngestExamples.mockResolvedValueOnce({
      results: PARENT_SCAN.images.slice(0, 2).map((img) => ({
        example_key: img.suggested_example_key,
        status: "created" as const,
        error: null,
        error_code: null,
        warnings: [],
        example: { example_key: img.suggested_example_key, state: "Unlabeled" },
      })),
    });

    const { Wrapper } = createWrapper();
    const user = userEvent.setup();
    render(<ImageIngestPage />, { wrapper: Wrapper });

    await waitFor(() => expect(screen.getByTestId("file-browser")).toBeInTheDocument());
    const checkboxes = screen.getAllByRole("checkbox");
    await user.click(checkboxes[0]);
    await user.click(checkboxes[1]);
    await user.click(screen.getByText("Ingest Selected"));

    await waitFor(() =>
      expect(screen.getByTestId("ingestion-summary")).toBeInTheDocument(),
    );

    // One non-recursive scan of the shared parent directory.
    expect(mockScanDirectory).toHaveBeenCalledTimes(1);
    expect(mockScanDirectory).toHaveBeenCalledWith("/data/images", false, "test-pid");
    // Ingest carries the backend's suggested keys — hash disambiguator
    // included — and only the selected files.
    expect(mockIngestExamples).toHaveBeenCalledWith("test-pid", {
      examples: [
        {
          example_key: "img_001_png--aaa111bbb222",
          storage_ref: "/data/images/img 001.png",
        },
        {
          example_key: "img_002_png--ccc333ddd444",
          storage_ref: "/data/images/img_002.png",
        },
      ],
    });
  });
});
