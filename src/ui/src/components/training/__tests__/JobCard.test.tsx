// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

import type { TAOJob, TAOJobStatus, TrainingSuiteJob } from "@/types/training";
import {
  humanizeFailureReason,
  humanizeHaltedReason,
  splitClassifiedFailure,
} from "@/lib/training/failure-display";
import { isTAOJobPollingSettled } from "@/lib/run-status-polling";
import { JobCard } from "../JobCard";

vi.mock("@/api/training", () => ({
  getTAOJob: vi.fn(),
  cancelTAOJob: vi.fn(),
}));

import { getTAOJob } from "@/api/training";

const mockGetJob = getTAOJob as ReturnType<typeof vi.fn>;

function makeJob(status: TAOJobStatus, extras: Partial<TAOJob> = {}): TAOJob {
  return {
    tao_job_id: "j-1",
    project_id: "pid-1",
    status,
    tao_status_raw: null,
    action: "train",
    training_backend: "cosmos_rl_tao_vlm",
    training_policy_type: "sft",
    student_base_model_config_id: "mc",
    dataset_export_ids: ["de"],
    job_config: {},
    tao_create_job_request: {},
    tao_external_job_id: "ext-j-1",
    progress: null,
    outputs: null,
    outputs_fetch_status: status === "succeeded" ? "completed" : "pending",
    outputs_fetch_error_ref: null,
    parent_tao_job_id: null,
    chain_id: "chain-8b",
    chain_sequence: 1,
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

function renderCard(
  status: TAOJobStatus,
  extras: Partial<TAOJob> = {},
  onOpenCompare?: () => void,
) {
  mockGetJob.mockResolvedValue(makeJob(status, extras));
  const suiteJob: TrainingSuiteJob = {
    tao_job_id: "j-1",
    action: "train",
    chain_sequence: 1,
    status,
    tao_external_job_id: "ext-j-1",
    chain_halted_reason: null,
    outputs_fetch_status:
      extras.outputs_fetch_status ?? (status === "succeeded" ? "completed" : "pending"),
    outputs_fetch_error_ref: extras.outputs_fetch_error_ref ?? null,
  };
  const qc = new QueryClient({
    defaultOptions: { queries: { retry: false, gcTime: 0, staleTime: 0 } },
  });
  return render(
    <QueryClientProvider client={qc}>
      <JobCard projectId="pid-1" suiteJob={suiteJob} onOpenCompare={onOpenCompare} />
    </QueryClientProvider>,
  );
}

describe("JobCard state bodies", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("paused card renders the amber progress bar at the paused epoch", async () => {
    // Paused shares the running body's epoch math; the bar switches to
    // the amber `paused` variant so pill and bar tell the same story.
    renderCard("paused", {
      progress: {
        epoch_current: 2,
        epoch_total: 4,
        eta_seconds: null,
        metrics_latest: null,
      },
    });
    const fill = await screen.findByTestId("training-job-paused-progress-fill");
    expect(fill.style.width).toBe("50%");
    expect(fill.style.backgroundColor).toContain("--warning-amber");
    expect(screen.getByTestId("training-job-paused-body").textContent).toContain(
      "Progress at pause: epoch 2 / 4",
    );
  });

  it("paused card omits the progress bar when no epoch progress was reported", async () => {
    renderCard("paused");
    await screen.findByTestId("training-job-paused-body");
    expect(screen.queryByTestId("training-job-paused-progress-fill")).toBeNull();
  });

  it("deleted card mutes its title so the audit-only record reads subdued", async () => {
    renderCard("deleted");
    await screen.findByText(/Job removed from TAO/);
    const title = screen.getByText("Train");
    expect(title.style.color).toContain("--text-muted");
  });

  it("active card keeps the full-brightness title", async () => {
    renderCard("running");
    const card = await screen.findByTestId("training-job-card-j-1");
    expect(card).toBeTruthy();
    const title = screen.getByText("Train");
    expect(title.style.color).toBe("");
  });

  it("running card omits ETA and epoch until TAO reports each value", async () => {
    renderCard("running", {
      progress: {
        epoch_current: null,
        epoch_total: 3,
        eta_seconds: null,
        metrics_latest: null,
      },
    });

    const card = await screen.findByTestId("training-job-card-j-1");
    expect(card.textContent).not.toContain("ETA");
    expect(card.textContent).not.toContain("Epoch");
    expect(card.textContent).not.toContain("—");
    expect(card.className).toContain("p-4");
    expect(card.className).toContain("gap-2");
  });

  it("running card renders only telemetry TAO actually reports", async () => {
    renderCard("running", {
      progress: {
        epoch_current: 1,
        epoch_total: 3,
        eta_seconds: 2700,
        metrics_latest: null,
      },
    });

    expect(await screen.findByText("Epoch: 1 / 3")).toBeTruthy();
    expect(screen.getByText("ETA 45m")).toBeTruthy();
    expect(screen.getByTestId("training-job-progress-fill")).toBeTruthy();
  });

  it("canceled card omits partial epoch telemetry instead of showing a dash", async () => {
    renderCard("canceled", {
      progress: {
        epoch_current: null,
        epoch_total: 3,
        eta_seconds: null,
        metrics_latest: null,
      },
    });

    const card = await screen.findByTestId("training-job-card-j-1");
    expect(card.textContent).not.toContain("Progress at cancellation");
    expect(card.textContent).not.toContain("—");
  });

  it("explains an intentionally omitted evaluation instead of calling it canceled", async () => {
    renderCard("canceled", {
      action: "evaluate",
      chain_halted_reason:
        "auto-skip: action=evaluate auto-skipped; Student quality uses local NIM",
    });

    await screen.findByText("Not Required");
    const card = screen.getByTestId("training-job-card-j-1");
    expect(card).toHaveTextContent("Not Required");
    expect(card).toHaveTextContent("local Student NIM quality");
    expect(card).not.toHaveTextContent("Canceled");
  });

  it("does not expose backend training artifact references", async () => {
    renderCard("succeeded", {
      completed_at: "2026-04-17T11:45:00Z",
      outputs: {
        artifacts: [
          { name: "best_model", artifact_ref: "/art/best.pth" },
          // Arbitrary TAO file paths have no mapping — raw passthrough.
          { name: "evaluate_results.tar.gz", artifact_ref: "/art/eval.tar.gz" },
        ],
        logs_ref: null,
        metrics_ref: "/art/metrics.json",
      },
    });
    await screen.findByText(/^Completed:/);
    expect(screen.queryByText("/art/best.pth")).toBeNull();
    expect(screen.queryByText("/art/metrics.json")).toBeNull();
    expect(screen.queryByRole("button", { name: "Download" })).toBeNull();
  });

  it("shows artifact finalization as active after TAO reports success", async () => {
    renderCard("succeeded", {
      outputs_fetch_status: "in_progress",
      completed_at: "2026-04-17T11:45:00Z",
    });

    expect(await screen.findByText("Finalizing")).toBeTruthy();
    expect(
      screen.getByTestId("training-job-finalizing-artifacts").textContent,
    ).toContain("retrieving and processing artifacts");
    expect(screen.queryByText(/^Completed:/)).toBeNull();
  });

  it("surfaces artifact-processing failure separately from TAO success", async () => {
    renderCard("succeeded", {
      outputs_fetch_status: "failed",
      outputs_fetch_error_ref: "workspace S3 download failed",
    });

    expect(await screen.findByText("Artifact Failed")).toBeTruthy();
    const body = screen.getByTestId("training-job-artifact-failed");
    expect(body.textContent).toContain("TAO completed");
    expect(body.textContent).toContain("workspace S3 download failed");
    expect(screen.getByText("Report TAO Issue")).toBeTruthy();
  });

  it("keeps polling a succeeded job until post-success processing is terminal", () => {
    expect(
      isTAOJobPollingSettled(
        makeJob("succeeded", { outputs_fetch_status: "in_progress" }),
      ),
    ).toBe(false);
    expect(
      isTAOJobPollingSettled(
        makeJob("succeeded", { outputs_fetch_status: "completed" }),
      ),
    ).toBe(true);
    expect(
      isTAOJobPollingSettled(makeJob("succeeded", { outputs_fetch_status: "failed" })),
    ).toBe(true);
  });

  it("identifies a Blueprint-merged local Student NIM baseline evaluation", async () => {
    renderCard("succeeded", {
      completed_at: "2026-07-30T19:20:00Z",
      outputs: {
        evaluation_source: "student_nim_local",
        evaluation_run_id: "run-local-baseline",
      },
    });
    const result = await screen.findByTestId("training-job-result");
    expect(result.textContent).toContain("Local Student NIM evaluation");
    expect(result.textContent).toContain("run-local-baseline");
  });

  it("failed card renders a friendly reason for bare-token error_refs, never raw snake_case", async () => {
    renderCard("failed", { error_ref: "submission_interrupted" });
    await screen.findByText(/Submission interrupted/);
    const body = screen.getByTestId("training-job-failed-body");
    expect(body.textContent).not.toContain("submission_interrupted");
  });

  it("routes a Blueprint-local Student NIM failure to Compare instead of reporting TAO", async () => {
    const onOpenCompare = vi.fn();
    renderCard(
      "failed",
      {
        training_backend: "student_nim_local",
        error_ref: "student_nim_evaluation_failed",
        outputs: { student_model_id: "student-1" },
      },
      onOpenCompare,
    );

    const openCompare = await screen.findByTestId("training-job-open-compare");
    expect(screen.getByTestId("training-job-failed-body").textContent).toContain(
      "Student NIM serving validation failed",
    );
    expect(screen.queryByText("Report TAO Issue")).toBeNull();
    openCompare.click();
    expect(onOpenCompare).toHaveBeenCalledOnce();
  });
});

describe("humanizeFailureReason", () => {
  it("maps known tokens to their friendly labels", () => {
    expect(humanizeFailureReason("submission_interrupted")).toContain(
      "Submission interrupted",
    );
  });

  it("renders TAO quality failure tokens as SME-facing copy", () => {
    expect(humanizeFailureReason("tao_evaluate_failed")).toBe(
      "TAO quality evaluation failed.",
    );
  });

  it("humanizes unknown snake_case tokens via underscore→space + sentence case", () => {
    expect(humanizeFailureReason("worker_disk_full")).toBe("Worker disk full");
  });

  it("passes full-sentence provider messages through unchanged", () => {
    const msg = "CUDA out of memory on GPU 3. Reduce batch size or use larger GPU.";
    expect(humanizeFailureReason(msg)).toBe(msg);
  });

  it("removes transport escaping around quotes in provider messages", () => {
    expect(humanizeFailureReason("parameter can\\'t contain a period")).toBe(
      "parameter can't contain a period",
    );
  });
});

describe("humanizeHaltedReason", () => {
  it("removes internal chain sequence and job IDs", () => {
    expect(
      humanizeHaltedReason(
        "Chain halted: train (seq 1, id=721dc8fe-1209-4040-82f0-8ba69c0ff253) reached terminal 'failed'",
      ),
    ).toBe("Chain halted: train failed.");
  });

  it("removes bookkeeping IDs from non-failure halt outcomes", () => {
    expect(
      humanizeHaltedReason(
        "Chain halted: evaluate (seq 2, id=09de3bac-42e5-41c7-b384-045954fe3f17) canceled by SME",
      ),
    ).toBe("Chain halted: evaluate canceled by SME.");
  });
});

describe("splitClassifiedFailure", () => {
  it("returns generic placeholder for null", () => {
    expect(splitClassifiedFailure(null)).toEqual({
      primary: "TAO job failed.",
      hint: null,
    });
  });

  it("returns generic placeholder for undefined", () => {
    expect(splitClassifiedFailure(undefined)).toEqual({
      primary: "TAO job failed.",
      hint: null,
    });
  });

  it("returns whole string as primary when no em-dash separator", () => {
    const raw = "TAO submission failed: HTTP 503";
    expect(splitClassifiedFailure(raw)).toEqual({ primary: raw, hint: null });
  });

  it("splits raw + friendly hint on the canonical em-dash separator", () => {
    // Mirrors classify_tao_failure's hf_gated_repo output.
    const raw =
      "Cosmos-RL SFT training failed: 401 Client Error — " +
      "Cosmos-RL container hit HuggingFace gated-repo (HTTP 401). " +
      "The Cosmos Reason2 family is gated; the worker authenticates " +
      "with HF_TOKEN passed via TAO docker_env_vars. Verify HF_TOKEN " +
      "is set in ~/.vlm_feedback_loop/.env and that the account has " +
      "accepted the Cosmos Reason2 license at " +
      "https://huggingface.co/nvidia/Cosmos-Reason2-2B.";
    const { primary, hint } = splitClassifiedFailure(raw);
    expect(primary).toBe("Cosmos-RL SFT training failed: 401 Client Error");
    expect(hint).not.toBeNull();
    expect(hint).toContain("HF_TOKEN");
    expect(hint).toContain("huggingface.co/nvidia/Cosmos-Reason2-2B");
  });

  it("splits parallelism mismatch into raw + WORLD_SIZE remediation hint", () => {
    const raw =
      "Cosmos-RL SFT training failed: Invalid parallel dims: " +
      "dp_replicate(1) * dp_shard(1) * cp(1) * tp(2) * pp(1) != WORLD_SIZE(1) — " +
      "Cosmos-RL parallelism plan rejected by WORLD_SIZE assertion " +
      "— the cosmos-rl container has fewer GPUs than the spec's " +
      "policy.parallelism.tp_size * dp_shard_size product. Adjust " +
      "num_gpu on the TAO request or override " +
      "policy.parallelism.dp_shard_size to match the allocated GPU " +
      "count.";
    const { primary, hint } = splitClassifiedFailure(raw);
    // We split on the FIRST " — " — the second separator inside the
    // hint is preserved as part of the hint body.
    expect(primary).toContain("Invalid parallel dims");
    expect(hint).toContain("WORLD_SIZE");
    expect(hint).toContain("dp_shard_size");
  });

  it("splits empty-after-separator into empty hint string", () => {
    // Defensive: classify_tao_failure should never emit this shape, but
    // the split must not crash.
    const { primary, hint } = splitClassifiedFailure("X — ");
    expect(primary).toBe("X");
    expect(hint).toBe("");
  });
});
