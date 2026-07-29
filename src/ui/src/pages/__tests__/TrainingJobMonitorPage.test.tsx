// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Tests for the Training Job Monitor screen — per-status job cards, the
 * Halted/Failed distinction, progress + outputs, Action Requests, job
 * cancellation, and the full chain display.
 */

import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter, Outlet, Route, Routes } from "react-router-dom";

import { TAO_JOB_STATUSES } from "@/types/training";
import type { TAOJob, TAOJobStatus, TrainingSuite } from "@/types/training";
import { statusDisplay } from "@/lib/training/statusDisplay";
import { TrainingJobMonitorPage } from "../TrainingJobMonitorPage";

// ── Mocks ──────────────────────────────────────────────────────────────────

vi.mock("@/api/training", () => ({
  getTrainingSuite: vi.fn(),
  getTAOJob: vi.fn(),
  cancelTAOJob: vi.fn(),
  cancelTrainingSuite: vi.fn(),
}));

vi.mock("@/api/nim", () => ({
  generateActionRequest: vi.fn(),
  logActionRequestCopy: vi.fn(),
}));

vi.mock("@/pages/ProjectSetupLayout", () => ({
  useSetupContext: vi.fn(),
}));

vi.mock("@/hooks/useProjectSSE", () => ({
  useProjectSSE: vi.fn(),
}));

import {
  cancelTAOJob,
  cancelTrainingSuite,
  getTAOJob,
  getTrainingSuite,
} from "@/api/training";
import { generateActionRequest } from "@/api/nim";
import { useProjectSSE } from "@/hooks/useProjectSSE";
import { useSetupContext } from "@/pages/ProjectSetupLayout";

const mockGetSuite = getTrainingSuite as ReturnType<typeof vi.fn>;
const mockGetJob = getTAOJob as ReturnType<typeof vi.fn>;
const mockCancelJob = cancelTAOJob as ReturnType<typeof vi.fn>;
const mockCancelSuite = cancelTrainingSuite as ReturnType<typeof vi.fn>;
const mockGenerateActionRequest = generateActionRequest as ReturnType<typeof vi.fn>;
const mockSetup = useSetupContext as ReturnType<typeof vi.fn>;
const mockUseSSE = useProjectSSE as ReturnType<typeof vi.fn>;

// ── Fixtures ──────────────────────────────────────────────────────────────

function makeJob(
  tao_job_id: string,
  action: TAOJob["action"],
  chain_sequence: number,
  status: TAOJobStatus,
  extras: Partial<TAOJob> = {},
): TAOJob {
  return {
    tao_job_id,
    project_id: "pid-1",
    status,
    tao_status_raw: null,
    action,
    training_backend: "cosmos_rl_tao_vlm",
    training_policy_type: "sft",
    student_base_model_config_id: "mc",
    dataset_export_ids: ["de"],
    job_config: {},
    tao_create_job_request: {},
    tao_external_job_id: "ext",
    progress: null,
    outputs: null,
    parent_tao_job_id: null,
    chain_id: "chain-8b",
    chain_sequence,
    chain_halted_reason: null,
    preflight_result: null,
    error_ref: null,
    poll_error_ref: null,
    created_at: "2026-04-17T00:00:00Z",
    started_at: null,
    completed_at: null,
    last_polled_at: null,
    ...extras,
  };
}

function makeSuite(
  chains: Array<{
    chain_id: string;
    baseModelName: string;
    modelConfigId: string;
    jobs: Array<[string, TAOJob["action"], number, TAOJobStatus, string | null]>;
  }>,
): TrainingSuite {
  return {
    training_suite_id: "ts-1",
    project_id: "pid-1",
    idempotency_key: "idem",
    guidance_id: "g-1",
    training_preset: "standard",
    export_field_mode: "all",
    include_auto_labeled: true,
    quantization_schemes: ["FP8_DYNAMIC", "W4A16"],
    training_dataset_export_id: "de-train",
    evaluation_dataset_export_id: "de-eval",
    selected_student_base_model_config_ids: chains.map((c) => c.modelConfigId),
    chain_ids_ordered: chains.map((c) => c.chain_id),
    chains: chains.map((c) => ({
      chain_id: c.chain_id,
      student_base_model_config_id: c.modelConfigId,
      base_model_name: c.baseModelName,
      jobs: c.jobs.map(([id, action, seq, status, haltedReason]) => ({
        tao_job_id: id,
        action,
        chain_sequence: seq,
        status,
        tao_external_job_id: status === "not_started" ? null : `ext-${id}`,
        chain_halted_reason: haltedReason,
      })),
    })),
    provisioning_run_id: null,
    provisioning_model_names: [],
    setup_error_ref: null,
    status: "running",
    created_at: "2026-04-17T00:00:00Z",
    started_at: null,
    completed_at: null,
  };
}

let qc: QueryClient;

beforeEach(() => {
  vi.clearAllMocks();
  qc = new QueryClient({
    defaultOptions: {
      queries: { retry: false, gcTime: 0, staleTime: 0 },
    },
  });
  mockSetup.mockReturnValue({
    projectId: "pid-1",
    project: {},
    environment: {},
  });
  mockUseSSE.mockReturnValue({ lastEvent: null });
  mockGetJob.mockImplementation((_pid: string, jobId: string) =>
    Promise.resolve(makeJob(jobId, "train", 1, "queued")),
  );
  mockGenerateActionRequest.mockResolvedValue({
    request_type: "tao_issue",
    generated_at: "2026-04-17T00:00:00Z",
    project_name: "Test",
    technical_requirements: {},
    current_environment: {},
    rendered_text: "TAO ISSUE REQUEST",
  });
});

function renderPage() {
  return render(
    <QueryClientProvider client={qc}>
      <MemoryRouter initialEntries={["/projects/pid-1/training/ts-1"]}>
        <Routes>
          <Route path="/" element={<div data-testid="projects-page">Projects</div>} />
          <Route path="/projects/:projectId" element={<Outlet />}>
            <Route
              path="training/:trainingSuiteId"
              element={<TrainingJobMonitorPage />}
            />
          </Route>
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

// ══════════════════════════════════════════════════════════════════════════
// Screen 14 acceptance items
// ══════════════════════════════════════════════════════════════════════════

describe("TrainingJobMonitorPage", () => {
  it("shows selected missing bases as one green provisioning step", async () => {
    const suite = {
      ...makeSuite([]),
      training_dataset_export_id: null,
      evaluation_dataset_export_id: null,
      provisioning_run_id: "prov-1",
      provisioning_model_names: [
        "nvidia/cosmos3-nano-reasoner",
        "nvidia/cosmos3-super-reasoner",
      ],
      status: "provisioning",
    };
    mockGetSuite.mockResolvedValue(suite);

    renderPage();

    const step = await screen.findByTestId("training-provisioning-step");
    expect(step).toHaveTextContent("Provision Student Bases");
    expect(step).toHaveTextContent("Cosmos 3 Nano (Reasoner)");
    expect(step).toHaveTextContent("Cosmos 3 Super (Reasoner)");
    const status = screen.getByTestId("training-provisioning-status");
    expect(status).toHaveTextContent("Running");
    expect(status).toHaveStyle({ color: "var(--accent-green, #76b900)" });
    expect(screen.getByTestId("monitor-compare-students")).toBeDisabled();
  });

  it("omits the provisioning step when every selected base was already ready", async () => {
    mockGetSuite.mockResolvedValue(
      makeSuite([
        {
          chain_id: "chain-8b",
          baseModelName: "nvidia/cosmos-reason2-8b",
          modelConfigId: "mc-8b",
          jobs: [["j-1", "train", 1, "running", null]],
        },
      ]),
    );

    renderPage();
    await screen.findByTestId("training-job-monitor-page");
    expect(screen.queryByTestId("training-provisioning-step")).toBeNull();
  });

  it("shows a canceled provisioning step as canceled rather than completed", async () => {
    mockGetSuite.mockResolvedValue({
      ...makeSuite([]),
      training_dataset_export_id: null,
      evaluation_dataset_export_id: null,
      provisioning_run_id: "prov-canceled",
      provisioning_model_names: ["nvidia/cosmos3-nano-reasoner"],
      status: "canceled",
    });

    renderPage();

    expect(await screen.findByTestId("training-provisioning-status")).toHaveTextContent(
      "Canceled",
    );
  });

  // ── Status rendering — each canonical status goes through the product
  // display mapping. Expected labels derive from statusDisplay; the literal
  // status→label contract is pinned in
  // lib/training/__tests__/statusDisplay.test.ts.
  it.each(TAO_JOB_STATUSES.map((s) => [s, statusDisplay(s, null).label] as const))(
    "renders %s as '%s' and never shows the raw canonical",
    async (status, label) => {
      const suite = makeSuite([
        {
          chain_id: "chain-8b",
          baseModelName: "nvidia/cosmos-reason2-8b",
          modelConfigId: "mc-8b",
          jobs: [["j-1", "train", 1, status, null]],
        },
      ]);
      mockGetSuite.mockResolvedValue(suite);
      mockGetJob.mockResolvedValueOnce(makeJob("j-1", "train", 1, status));
      renderPage();
      await screen.findByTestId("training-job-monitor-page");

      const badge = await screen.findByTestId("training-job-status-j-1");
      expect(badge.textContent).toContain(label);
      // Canonical string NOT visible as badge text.
      expect(badge.textContent).not.toContain(status);
      // "Succeeded" never appears.
      expect(badge.textContent).not.toMatch(/Succeeded/);
    },
  );

  // ── Halted vs Failed distinction ──
  it("renders failed + chain_halted_reason as 'Halted' and shows the chain-halted banner", async () => {
    const suite = makeSuite([
      {
        chain_id: "chain-8b",
        baseModelName: "nvidia/cosmos-reason2-8b",
        modelConfigId: "mc-8b",
        jobs: [
          ["train-1", "train", 1, "succeeded", null],
          ["eval-1", "evaluate", 2, "failed", null],
          [
            "quant-1",
            "quantize",
            3,
            "failed",
            "Chain halted: evaluate (seq 2, id=eval-1) reached terminal 'failed'",
          ],
          [
            "eval-q-1",
            "evaluate",
            4,
            "failed",
            "Chain halted: evaluate (seq 2, id=eval-1) reached terminal 'failed'",
          ],
        ],
      },
    ]);
    mockGetSuite.mockResolvedValue(suite);
    mockGetJob.mockImplementation((_pid: string, jobId: string) => {
      const map: Record<string, TAOJob> = {
        "train-1": makeJob("train-1", "train", 1, "succeeded"),
        "eval-1": makeJob("eval-1", "evaluate", 2, "failed", {
          error_ref: "CUDA out of memory",
        }),
        "quant-1": makeJob("quant-1", "quantize", 3, "failed", {
          chain_halted_reason: "Chain halted: evaluate (seq 2, id=eval-1)",
        }),
        "eval-q-1": makeJob("eval-q-1", "evaluate", 4, "failed", {
          chain_halted_reason: "Chain halted: evaluate (seq 2, id=eval-1)",
        }),
      };
      return Promise.resolve(map[jobId]);
    });

    renderPage();
    await screen.findByTestId("training-job-monitor-page");

    // The root failed job shows "Failed" (no halted_reason on the suite entry).
    const rootBadge = await screen.findByTestId("training-job-status-eval-1");
    expect(rootBadge.textContent).toContain("Failed");
    expect(rootBadge.textContent).not.toContain("Halted");

    // Downstream not_started → failed + halted_reason siblings show "Halted".
    const quantBadge = await screen.findByTestId("training-job-status-quant-1");
    expect(quantBadge.textContent).toContain("Halted");
    expect(quantBadge.textContent).not.toContain("Failed");

    // Chain banner surfaces the reason.
    expect(
      (await screen.findByTestId("chain-halted-banner-chain-8b")).textContent,
    ).toContain("Chain halted: evaluate failed.");
  });

  // ── Running card shows epoch progress + ETA + metrics ──
  it("running card renders epoch progress bar, ETA, and metrics", async () => {
    const suite = makeSuite([
      {
        chain_id: "chain-8b",
        baseModelName: "nvidia/cosmos-reason2-8b",
        modelConfigId: "mc-8b",
        jobs: [["r-1", "train", 1, "running", null]],
      },
    ]);
    mockGetSuite.mockResolvedValue(suite);
    mockGetJob.mockImplementation((_pid: string, jobId: string) =>
      Promise.resolve(
        makeJob(jobId, "train", 1, "running", {
          started_at: "2026-04-17T10:00:00Z",
          progress: {
            epoch_current: 2,
            epoch_total: 3,
            eta_seconds: 2700,
            metrics_latest: { loss: 1.234, accuracy: 0.87 },
          },
        }),
      ),
    );
    renderPage();
    await screen.findByTestId("training-job-monitor-page");
    await screen.findByTestId("training-job-progress-fill");
    expect((await screen.findByTestId("training-job-metrics")).textContent).toContain(
      "loss",
    );
    expect((await screen.findByTestId("training-job-metrics")).textContent).toContain(
      "accuracy",
    );
    // ETA text present.
    expect(
      screen.getByTestId("training-job-card-r-1").textContent?.includes("ETA"),
    ).toBe(true);
  });

  // ── Completed card shows outputs ──
  it("completed card lists outputs under friendly labels, not raw TAO keys", async () => {
    const suite = makeSuite([
      {
        chain_id: "chain-8b",
        baseModelName: "nvidia/cosmos-reason2-8b",
        modelConfigId: "mc-8b",
        jobs: [["c-1", "train", 1, "succeeded", null]],
      },
    ]);
    mockGetSuite.mockResolvedValue(suite);
    mockGetJob.mockImplementation((_pid: string, jobId: string) =>
      Promise.resolve(
        makeJob(jobId, "train", 1, "succeeded", {
          started_at: "2026-04-17T10:00:00Z",
          completed_at: "2026-04-17T11:45:00Z",
          progress: {
            epoch_current: 3,
            epoch_total: 3,
            eta_seconds: null,
            metrics_latest: { loss: 0.84 },
          },
          outputs: {
            artifacts: [
              { name: "best_model", artifact_ref: "/art/best.pth" },
              { name: "latest_model", artifact_ref: "/art/latest.pth" },
              { name: "training_config", artifact_ref: "/art/config.json" },
              // Arbitrary TAO file paths have no friendly mapping and
              // must pass through unchanged.
              { name: "evaluate_results.tar.gz", artifact_ref: "/art/eval.tar.gz" },
            ],
            logs_ref: null,
          },
        }),
      ),
    );
    renderPage();
    await screen.findByTestId("training-job-monitor-page");
    const outputs = await screen.findByTestId("training-job-outputs");
    expect(outputs.textContent).toContain("Best model");
    expect(outputs.textContent).toContain("Latest model");
    expect(outputs.textContent).toContain("Training config");
    expect(outputs.textContent).toContain("evaluate_results.tar.gz");
    // Raw snake_case keys never surface as user-facing labels.
    expect(outputs.textContent).not.toContain("best_model");
    expect(outputs.textContent).not.toContain("latest_model");
    expect(outputs.textContent).not.toContain("training_config");
    // The refs themselves still render.
    expect(outputs.textContent).toContain("/art/best.pth");
  });

  // ── Failed card surfaces [Report TAO Issue] → Action Request ──
  it("failed card opens the TAO issue Action Request on click", async () => {
    const suite = makeSuite([
      {
        chain_id: "chain-8b",
        baseModelName: "nvidia/cosmos-reason2-8b",
        modelConfigId: "mc-8b",
        jobs: [["f-1", "train", 1, "failed", null]],
      },
    ]);
    mockGetSuite.mockResolvedValue(suite);
    mockGetJob.mockImplementation((_pid: string, jobId: string) =>
      Promise.resolve(
        makeJob(jobId, "train", 1, "failed", {
          error_ref: "CUDA out of memory on GPU 3.",
        }),
      ),
    );
    renderPage();
    await screen.findByTestId("training-job-monitor-page");
    const reportBtn = await screen.findByTestId("training-job-report-issue");
    fireEvent.click(reportBtn);
    await waitFor(() =>
      expect(mockGenerateActionRequest).toHaveBeenCalledWith(
        "pid-1",
        expect.objectContaining({ request_type: "tao_issue" }),
      ),
    );
  });

  // ── Paused card's [Cancel Job] calls cancelTAOJob ──
  it("paused card exposes [Cancel Job] that invokes the cancel mutation", async () => {
    const suite = makeSuite([
      {
        chain_id: "chain-8b",
        baseModelName: "nvidia/cosmos-reason2-8b",
        modelConfigId: "mc-8b",
        jobs: [["p-1", "train", 1, "paused", null]],
      },
    ]);
    mockGetSuite.mockResolvedValue(suite);
    mockGetJob.mockImplementation((_pid: string, jobId: string) =>
      Promise.resolve(makeJob(jobId, "train", 1, "paused")),
    );
    mockCancelJob.mockResolvedValue(makeJob("p-1", "train", 1, "canceled"));
    renderPage();
    await screen.findByTestId("training-job-monitor-page");
    const cancelBtn = await screen.findByTestId("training-job-cancel");
    fireEvent.click(cancelBtn);
    await waitFor(() => expect(mockCancelJob).toHaveBeenCalledWith("pid-1", "p-1"));
  });

  it("confirms suite cancellation, releases the project, and preserves warnings", async () => {
    const suite = makeSuite([
      {
        chain_id: "chain-8b",
        baseModelName: "nvidia/cosmos-reason2-8b",
        modelConfigId: "mc-8b",
        jobs: [
          ["j-done", "train", 1, "succeeded", null],
          ["j-running", "evaluate", 2, "running", null],
          ["j-waiting", "quantize", 3, "not_started", null],
        ],
      },
    ]);
    mockGetSuite.mockResolvedValue(suite);
    mockCancelSuite.mockResolvedValue({
      training_suite: {
        ...suite,
        status: "canceled",
        completed_at: "2026-07-27T12:00:00Z",
        chains: suite.chains.map((chain) => ({
          ...chain,
          jobs: chain.jobs.map((job) =>
            job.status === "succeeded" ? job : { ...job, status: "canceled" },
          ),
        })),
      },
      jobs_canceled: 2,
      jobs_already_terminal: 1,
      setup_tasks_canceled: 0,
      remote_cancel_failures: [
        {
          tao_job_id: "j-running",
          error: "TAO did not confirm the cancellation",
        },
      ],
    });

    renderPage();
    await screen.findByTestId("training-job-monitor-page");

    fireEvent.click(screen.getByTestId("monitor-cancel-jobs"));
    expect(await screen.findByText("Cancel remaining jobs?")).toBeInTheDocument();
    expect(screen.getByText(/Completed work is preserved/i)).toBeInTheDocument();

    fireEvent.click(screen.getByTestId("confirm-cancel-training-suite"));

    await waitFor(() => expect(mockCancelSuite).toHaveBeenCalledWith("pid-1", "ts-1"));
    const banner = await screen.findByTestId("training-suite-canceled-banner");
    expect(banner).toHaveTextContent("Training released with TAO warnings.");
    expect(banner).toHaveTextContent(
      "This project will no longer resume Training Jobs.",
    );
    expect(screen.queryByTestId("monitor-cancel-jobs")).not.toBeInTheDocument();

    fireEvent.click(screen.getByTestId("training-canceled-back-to-projects"));
    expect(await screen.findByTestId("projects-page")).toBeInTheDocument();
  });

  // ── Full chain display with grouping + progress line + ordering ──
  it("groups jobs per base model, renders chain progress line, and orders by chain_sequence", async () => {
    const suite = makeSuite([
      {
        chain_id: "chain-8b",
        baseModelName: "nvidia/cosmos-reason2-8b",
        modelConfigId: "mc-8b",
        jobs: [
          // Intentionally shuffled — component MUST sort.
          ["j-8b-3", "quantize", 3, "succeeded", null],
          ["j-8b-1", "train", 1, "succeeded", null],
          ["j-8b-2", "evaluate", 2, "succeeded", null],
          ["j-8b-4", "evaluate", 4, "succeeded", null],
          ["j-8b-5", "quantize", 5, "succeeded", null],
          ["j-8b-6", "evaluate", 6, "succeeded", null],
        ],
      },
      {
        chain_id: "chain-2b",
        baseModelName: "nvidia/cosmos-reason2-2b",
        modelConfigId: "mc-2b",
        jobs: [
          ["j-2b-1", "train", 1, "succeeded", null],
          ["j-2b-2", "evaluate", 2, "succeeded", null],
          ["j-2b-3", "quantize", 3, "succeeded", null],
          ["j-2b-4", "evaluate", 4, "succeeded", null],
          ["j-2b-5", "quantize", 5, "running", null],
          ["j-2b-6", "evaluate", 6, "not_started", null],
        ],
      },
    ]);
    mockGetSuite.mockResolvedValue(suite);
    mockGetJob.mockImplementation((_pid: string, jobId: string) =>
      Promise.resolve(
        makeJob(
          jobId,
          "train",
          1,
          jobId.endsWith("5")
            ? "running"
            : jobId.endsWith("6")
              ? "not_started"
              : "succeeded",
        ),
      ),
    );

    renderPage();
    await screen.findByTestId("training-job-monitor-page");

    // Both chain sections render.
    const chain8b = await screen.findByTestId("training-chain-chain-8b");
    const chain2b = await screen.findByTestId("training-chain-chain-2b");
    expect(chain8b).toBeTruthy();
    expect(chain2b).toBeTruthy();

    // Overall progress line: "8B: done  2B: 4 of 6"
    const overall = await screen.findByTestId("monitor-overall-progress");
    expect(overall.textContent).toContain("8B: done");
    expect(overall.textContent).toContain("2B: 4 of 6");

    // Cards inside the 8B chain are sorted by chain_sequence.
    const cards = within(chain8b).getAllByTestId(/^training-job-card-j-8b-\d$/);
    const seqs = cards.map((el) => Number(el.getAttribute("data-chain-sequence")));
    expect(seqs).toEqual([1, 2, 3, 4, 5, 6]);

    // Compare Students stays disabled until every chain completes.
    const compare = screen.getByTestId("monitor-compare-students") as HTMLButtonElement;
    expect(compare.disabled).toBe(true);
  });

  // SSE terminal event → invalidates caches
  it("invalidates the suite + job caches when a tao_job_completed event arrives", async () => {
    const suite = makeSuite([
      {
        chain_id: "chain-8b",
        baseModelName: "nvidia/cosmos-reason2-8b",
        modelConfigId: "mc-8b",
        jobs: [["j-1", "train", 1, "running", null]],
      },
    ]);
    mockGetSuite.mockResolvedValue(suite);
    mockGetJob.mockImplementation((_pid: string, jobId: string) =>
      Promise.resolve(makeJob(jobId, "train", 1, "running")),
    );
    let ssePayload: { type: string; data: Record<string, unknown> } | null = null;
    mockUseSSE.mockImplementation(() => ({
      lastEvent: ssePayload,
    }));

    const { rerender } = renderPage();
    await screen.findByTestId("training-job-monitor-page");

    const invalidateSpy = vi.spyOn(qc, "invalidateQueries");
    ssePayload = {
      type: "tao_job_completed",
      data: {
        run_id: "j-1",
        tao_job_id: "j-1",
        run_type: "tao_job",
        status: "succeeded",
      },
    };
    act(() => {
      rerender(
        <QueryClientProvider client={qc}>
          <MemoryRouter initialEntries={["/projects/pid-1/training/ts-1"]}>
            <Routes>
              <Route path="/projects/:projectId" element={<Outlet />}>
                <Route
                  path="training/:trainingSuiteId"
                  element={<TrainingJobMonitorPage />}
                />
              </Route>
            </Routes>
          </MemoryRouter>
        </QueryClientProvider>,
      );
    });

    await waitFor(() => {
      const calls = invalidateSpy.mock.calls.map((c) =>
        (c[0] as { queryKey: readonly unknown[] }).queryKey.join("|"),
      );
      expect(calls.some((k) => k.includes("suite") && k.includes("ts-1"))).toBe(true);
      expect(calls.some((k) => k.includes("job") && k.includes("j-1"))).toBe(true);
    });
  });

  it("keeps the Training Jobs monitor focused on its forward action", async () => {
    const suite = makeSuite([
      {
        chain_id: "chain-8b",
        baseModelName: "nvidia/cosmos-reason2-8b",
        modelConfigId: "mc-8b",
        jobs: [["j-1", "train", 1, "succeeded", null]],
      },
    ]);
    mockGetSuite.mockResolvedValue(suite);
    mockGetJob.mockImplementation((_pid: string, jobId: string) =>
      Promise.resolve(makeJob(jobId, "train", 1, "succeeded")),
    );
    renderPage();
    await screen.findByTestId("training-job-monitor-page");
    expect(screen.queryByText("Back to Scale-Up")).not.toBeInTheDocument();
    expect(screen.getByTestId("monitor-compare-students")).toBeInTheDocument();
  });
});
