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
  createTrainingSuite: vi.fn(),
  getTrainingSuite: vi.fn(),
  getTAOJob: vi.fn(),
  cancelTAOJob: vi.fn(),
  cancelTrainingSuite: vi.fn(),
}));

vi.mock("@/api/nim", () => ({
  generateActionRequest: vi.fn(),
  logActionRequestCopy: vi.fn(),
}));

vi.mock("@/pages/setup-context", () => ({
  useSetupContext: vi.fn(),
}));

vi.mock("@/hooks/useProjectSSE", () => ({
  useProjectSSE: vi.fn(),
}));

import {
  cancelTAOJob,
  cancelTrainingSuite,
  createTrainingSuite,
  getTAOJob,
  getTrainingSuite,
} from "@/api/training";
import { generateActionRequest } from "@/api/nim";
import { useProjectSSE } from "@/hooks/useProjectSSE";
import { useSetupContext } from "@/pages/setup-context";

const mockGetSuite = getTrainingSuite as ReturnType<typeof vi.fn>;
const mockGetJob = getTAOJob as ReturnType<typeof vi.fn>;
const mockCancelJob = cancelTAOJob as ReturnType<typeof vi.fn>;
const mockCancelSuite = cancelTrainingSuite as ReturnType<typeof vi.fn>;
const mockCreateSuite = createTrainingSuite as ReturnType<typeof vi.fn>;
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
    outputs_fetch_status: status === "succeeded" ? "completed" : "pending",
    outputs_fetch_error_ref: null,
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
    enable_lora: true,
    quantization_schemes: ["FP8_DYNAMIC", "W4A16"],
    training_dataset_export_id: "de-train",
    evaluation_dataset_export_id: "de-eval",
    training_example_count: 100,
    evaluation_example_count: 20,
    evaluation_dataset_checksum_sha256: "sha256:test",
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
        outputs_fetch_status: status === "succeeded" ? "completed" : "pending",
        outputs_fetch_error_ref: null,
      })),
    })),
    student_model_ids: [],
    provisioning_run_id: null,
    provisioning_model_names: [],
    setup_error_ref: null,
    setup_retryable: false,
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
            <Route
              path="compare"
              element={<div data-testid="compare-page">Compare</div>}
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
  it("explains the post-success artifact handoff that blocks the next job", async () => {
    const suite = makeSuite([
      {
        chain_id: "chain-super",
        baseModelName: "nvidia/cosmos3-super-reasoner",
        modelConfigId: "mc-super",
        jobs: [
          ["train-super", "train", 1, "succeeded", null],
          ["eval-super", "evaluate", 2, "not_started", null],
        ],
      },
    ]);
    suite.chains[0].jobs[0].outputs_fetch_status = "in_progress";
    mockGetSuite.mockResolvedValue(suite);
    mockGetJob.mockImplementation((_pid: string, jobId: string) =>
      Promise.resolve(
        jobId === "train-super"
          ? makeJob(jobId, "train", 1, "succeeded", {
              outputs_fetch_status: "in_progress",
            })
          : makeJob(jobId, "evaluate", 2, "not_started"),
      ),
    );

    renderPage();

    expect(await screen.findByText("Finalizing")).toBeTruthy();
    expect(
      screen.getByText(
        "Waiting for the previous job's artifacts to finish processing.",
      ),
    ).toBeTruthy();
  });

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

  it("retries a failed frozen-dataset upload with the original request", async () => {
    const failedSuite = {
      ...makeSuite([]),
      selected_student_base_model_config_ids: ["mc-8b"],
      enable_lora: false,
      status: "failed" as const,
      setup_retryable: true,
      setup_error_ref:
        "tao_dataset_upload_failed: evaluation export upload timed out; retry with the same idempotency key",
    };
    const retryingSuite = { ...failedSuite, status: "preparing" as const };
    mockGetSuite.mockResolvedValue(failedSuite);
    mockCreateSuite.mockResolvedValue(retryingSuite);

    renderPage();

    expect(await screen.findByTestId("training-setup-error")).toHaveTextContent(
      "Training Jobs setup failed.",
    );
    expect(screen.getByTestId("training-setup-error")).not.toHaveTextContent(
      "tao_dataset_upload_failed:",
    );
    fireEvent.click(screen.getByTestId("training-retry-dataset-upload"));

    await waitFor(() =>
      expect(mockCreateSuite).toHaveBeenCalledWith("pid-1", {
        student_base_model_config_ids: ["mc-8b"],
        training_preset: "standard",
        include_auto_labeled: true,
        enable_lora: false,
        export_field_mode: "all",
        quantization_schemes: ["FP8_DYNAMIC", "W4A16"],
        idempotency_key: "idem",
      }),
    );
  });

  it("does not offer retry when frozen export integrity has failed", async () => {
    mockGetSuite.mockResolvedValue({
      ...makeSuite([]),
      selected_student_base_model_config_ids: ["mc-8b"],
      status: "failed",
      setup_retryable: false,
      setup_error_ref:
        "tao_dataset_upload_failed: Dataset export integrity check failed: annotations.json no longer matches the frozen archive",
    });

    renderPage();

    expect(await screen.findByTestId("training-setup-error")).toBeInTheDocument();
    expect(screen.queryByTestId("training-retry-dataset-upload")).toBeNull();
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

  it("describes a cancellation halt truthfully without its internal job ID", async () => {
    const reason =
      "Chain halted: evaluate (seq 2, id=09de3bac-42e5-41c7-b384-045954fe3f17) canceled by SME";
    const suite = makeSuite([
      {
        chain_id: "chain-8b",
        baseModelName: "nvidia/cosmos-reason2-8b",
        modelConfigId: "mc-8b",
        jobs: [
          ["train-1", "train", 1, "succeeded", null],
          ["eval-1", "evaluate", 2, "canceled", null],
          ["quant-1", "quantize", 3, "failed", reason],
        ],
      },
    ]);
    mockGetSuite.mockResolvedValue(suite);

    renderPage();
    const banner = await screen.findByTestId("chain-halted-banner-chain-8b");
    expect(banner).toHaveTextContent("Chain halted: evaluate canceled by SME.");
    expect(banner).not.toHaveTextContent("09de3bac");
  });

  it("treats an auto-skipped baseline evaluation as progress, not a halted chain", async () => {
    const autoSkipReason =
      "auto-skip: action=evaluate auto-skipped; trained checkpoint is adapter-only";
    const suite = makeSuite([
      {
        chain_id: "chain-2b",
        baseModelName: "nvidia/cosmos-reason2-2b",
        modelConfigId: "mc-2b",
        jobs: [
          ["train-1", "train", 1, "succeeded", null],
          ["eval-1", "evaluate", 2, "canceled", autoSkipReason],
          ["quant-1", "quantize", 3, "running", null],
          ["eval-q-1", "evaluate", 4, "not_started", null],
        ],
      },
    ]);
    mockGetSuite.mockResolvedValue(suite);

    renderPage();
    await screen.findByTestId("training-job-monitor-page");

    expect(screen.queryByTestId("chain-halted-banner-chain-2b")).toBeNull();
    expect(screen.getByTestId("monitor-overall-progress")).toHaveTextContent(
      "2B: 2 of 4",
    );
  });

  it("enables Compare when all executed jobs complete and evaluation legs auto-skip", async () => {
    const autoSkipReason =
      "auto-skip: action=evaluate auto-skipped; trained checkpoint is adapter-only";
    const suite = {
      ...makeSuite([
        {
          chain_id: "chain-2b",
          baseModelName: "nvidia/cosmos-reason2-2b",
          modelConfigId: "mc-2b",
          jobs: [
            ["train-1", "train", 1, "succeeded", null],
            ["eval-1", "evaluate", 2, "canceled", autoSkipReason],
            ["quant-1", "quantize", 3, "succeeded", null],
            ["eval-q-1", "evaluate", 4, "canceled", autoSkipReason],
          ],
        },
      ]),
      status: "completed" as const,
    };
    mockGetSuite.mockResolvedValue(suite);

    renderPage();
    await screen.findByTestId("training-job-monitor-page");

    expect(screen.queryByTestId("chain-halted-banner-chain-2b")).toBeNull();
    expect(screen.getByTestId("monitor-overall-progress")).toHaveTextContent(
      "2B: done",
    );
    expect(screen.getByTestId("monitor-compare-students")).toBeEnabled();
  });

  it("enables Compare for finalized Students after an independent chain fails", async () => {
    const autoSkipReason =
      "auto-skip: action=evaluate auto-skipped; quality uses local NIM";
    const suite = {
      ...makeSuite([
        {
          chain_id: "chain-nano",
          baseModelName: "nvidia/cosmos3-nano-reasoner",
          modelConfigId: "mc-nano",
          jobs: [
            ["nano-train", "train", 1, "succeeded", null],
            ["nano-eval", "evaluate", 2, "succeeded", null],
            ["nano-quant", "quantize", 3, "succeeded", null],
            ["nano-qeval", "evaluate", 4, "canceled", autoSkipReason],
          ],
        },
        {
          chain_id: "chain-super",
          baseModelName: "nvidia/cosmos3-super-reasoner",
          modelConfigId: "mc-super",
          jobs: [
            ["super-train", "train", 1, "failed", null],
            ["super-eval", "evaluate", 2, "failed", "Chain halted: train failed"],
          ],
        },
      ]),
      status: "failed" as const,
    };
    mockGetSuite.mockResolvedValue(suite);

    renderPage();
    await screen.findByTestId("training-job-monitor-page");

    expect(screen.getByTestId("monitor-compare-students")).toBeEnabled();
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

  it("completed card keeps internal artifact references out of the UI", async () => {
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
    await screen.findByText(/^Completed:/);
    expect(screen.queryByText("/art/best.pth")).toBeNull();
    expect(screen.queryByText("evaluate_results.tar.gz")).toBeNull();
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
        expect.objectContaining({
          request_type: "tao_issue",
          context: { tao_job_id: "f-1" },
        }),
      ),
    );
  });

  it("local Student NIM failure opens Compare without generating a TAO issue", async () => {
    const suite = makeSuite([
      {
        chain_id: "chain-8b",
        baseModelName: "nvidia/cosmos-reason2-8b",
        modelConfigId: "mc-8b",
        jobs: [["nim-eval-1", "evaluate", 2, "failed", null]],
      },
    ]);
    mockGetSuite.mockResolvedValue(suite);
    mockGetJob.mockResolvedValue(
      makeJob("nim-eval-1", "evaluate", 2, "failed", {
        training_backend: "student_nim_local",
        error_ref: "student_nim_evaluation_failed",
        outputs: { student_model_id: "student-8b" },
      }),
    );

    renderPage();
    const openCompare = await screen.findByTestId("training-job-open-compare");
    expect(screen.queryByText("Report TAO Issue")).toBeNull();
    fireEvent.click(openCompare);
    expect(await screen.findByTestId("compare-page")).toBeInTheDocument();
    expect(mockGenerateActionRequest).not.toHaveBeenCalled();
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

    // Product copy humanizes the canonical Cosmos-RL enum retained by the API.
    expect(within(chain8b).getByText("Quantize (FP8 Dynamic)")).toBeInTheDocument();
    expect(within(chain8b).getByText("Evaluate (W4A16)")).toBeInTheDocument();

    // Cards inside the 8B chain are sorted by chain_sequence.
    const cards = within(chain8b).getAllByTestId(/^training-job-card-j-8b-\d$/);
    const seqs = cards.map((el) => Number(el.getAttribute("data-chain-sequence")));
    expect(seqs).toEqual([1, 2, 3, 4, 5, 6]);

    // Compare Students stays disabled until every chain completes.
    const compare = screen.getByTestId("monitor-compare-students") as HTMLButtonElement;
    expect(compare.disabled).toBe(true);
  });

  it("uses compact Cosmos 3 names in overall progress", async () => {
    const suite = makeSuite([
      {
        chain_id: "chain-nano",
        baseModelName: "nvidia/cosmos3-nano-reasoner",
        modelConfigId: "mc-nano",
        jobs: [["j-nano-1", "train", 1, "succeeded", null]],
      },
      {
        chain_id: "chain-super",
        baseModelName: "nvidia/cosmos3-super-reasoner",
        modelConfigId: "mc-super",
        jobs: [["j-super-1", "train", 1, "running", null]],
      },
    ]);
    mockGetSuite.mockResolvedValue(suite);

    renderPage();
    await screen.findByTestId("training-job-monitor-page");

    const overall = screen.getByTestId("monitor-overall-progress");
    expect(overall).toHaveTextContent("Nano: done");
    expect(overall).toHaveTextContent("Super: 0 of 1");
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
