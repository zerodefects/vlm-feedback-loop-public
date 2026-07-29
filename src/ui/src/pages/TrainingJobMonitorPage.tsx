// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Training Job Monitor screen.
 *
 * Chain-structured live view of every TAOJob per base model.  Jobs
 * within a chain advance automatically — the screen listens for
 * ``tao_job_progress`` / ``tao_job_completed`` / ``run_failed`` SSE
 * events emitted by the TAO background polling loop and refreshes
 * the suite + per-job caches.  ``[Compare Students]`` stays disabled
 * (with an explanatory tooltip) until every chain completes.
 */

import { useEffect, useMemo, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Button, Spinner, Text, Tooltip } from "@kui/react";

import { getTrainingSuite } from "@/api/training";
import { trainingKeys } from "@/api/query-keys";
import { InfoBanner } from "@/components/common/InfoBanner";
import { PageContainer } from "@/components/common/PageContainer";
import { SectionCard } from "@/components/common/SectionCard";
import { SectionHeading } from "@/components/common/SectionHeading";
import { StatusPill } from "@/components/common/StatusPill";
import { useProjectSSE } from "@/hooks/useProjectSSE";
import { useSetupContext } from "@/pages/ProjectSetupLayout";
import { ChainProgressLine } from "@/components/training/ChainProgressLine";
import { CancelTrainingSuiteDialog } from "@/components/training/CancelTrainingSuiteDialog";
import { JobCard } from "@/components/training/JobCard";
import {
  TERMINAL_TRAINING_SUITE_STATUSES,
  TERMINAL_TAO_STATUSES,
  type TrainingSuite,
  type TrainingSuiteCancelResponse,
  type TrainingSuiteChain,
} from "@/types/training";

const POLL_INTERVAL_MS = 5000;

function suiteAllTerminal(suite: TrainingSuite | undefined): boolean {
  if (!suite) return false;
  if (["completed", "failed", "canceled"].includes(suite.status)) return true;
  if (!suite.chains.length) return false;
  return suite.chains.every((chain) =>
    chain.jobs.every((job) => TERMINAL_TAO_STATUSES.has(job.status)),
  );
}

function chainAllSucceeded(chain: TrainingSuiteChain): boolean {
  return chain.jobs.length > 0 && chain.jobs.every((job) => job.status === "succeeded");
}

function countSucceeded(chain: TrainingSuiteChain): number {
  return chain.jobs.filter((job) => job.status === "succeeded").length;
}

function findHaltedReasonJob(
  chain: TrainingSuiteChain,
): { action: string; chainSequence: number } | null {
  // The "root" halted job is the failed one (not the downstream jobs marked
  // with chain_halted_reason).  Find it by scanning for status=failed WITHOUT
  // chain_halted_reason, or — if everything is halted — the lowest sequence
  // that has chain_halted_reason set.
  const rootFailed = chain.jobs.find(
    (j) => j.status === "failed" && !j.chain_halted_reason,
  );
  if (rootFailed)
    return { action: rootFailed.action, chainSequence: rootFailed.chain_sequence };
  const firstHalted = chain.jobs
    .filter((j) => j.chain_halted_reason)
    .sort((a, b) => a.chain_sequence - b.chain_sequence)[0];
  if (firstHalted)
    return {
      action: firstHalted.action,
      chainSequence: firstHalted.chain_sequence,
    };
  return null;
}

// Display-label helpers live in @/lib/model-display so other screens
// (Scale-Up and other summary screens) can reuse them consistently.
import {
  formatModelDisplayName,
  localTeacherDisplayName,
  shortBaseLabel,
} from "@/lib/model-display";
import { runStatusRefetchInterval } from "@/lib/run-status-polling";

/**
 * Build a per-job quantization suffix label for display on Quantize and
 * Evaluate cards that correspond to a specific scheme.  Derives the
 * scheme by pairing consecutive quantize+evaluate pairs after the
 * baseline train+evaluate.
 */
function computeQuantLabels(
  chain: TrainingSuiteChain,
  schemes: string[],
): Record<string, string | null> {
  const result: Record<string, string | null> = {};
  // Expected chain order: train(1) → evaluate baseline(2) →
  // quantize scheme_0(3) → evaluate scheme_0(4) → quantize scheme_1(5) → ...
  const orderedJobs = [...chain.jobs].sort(
    (a, b) => a.chain_sequence - b.chain_sequence,
  );
  for (let i = 0; i < orderedJobs.length; i++) {
    const job = orderedJobs[i];
    if (job.action === "evaluate" && i === 1) {
      result[job.tao_job_id] = "baseline";
      continue;
    }
    if (job.action === "quantize" || job.action === "evaluate") {
      // After the baseline evaluate, each pair corresponds to the next scheme.
      const quantPairIndex = Math.floor((i - 2) / 2);
      const scheme = schemes[quantPairIndex] ?? null;
      result[job.tao_job_id] = scheme;
    } else {
      result[job.tao_job_id] = null;
    }
  }
  return result;
}

export function TrainingJobMonitorPage() {
  const { projectId } = useSetupContext();
  const { trainingSuiteId } = useParams<{ trainingSuiteId: string }>();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [cancelDialogOpen, setCancelDialogOpen] = useState(false);
  const [cancelResult, setCancelResult] = useState<TrainingSuiteCancelResponse | null>(
    null,
  );

  const { lastEvent } = useProjectSSE(projectId);

  const { data: suite, isLoading } = useQuery<TrainingSuite>({
    queryKey: trainingKeys.suite(projectId, trainingSuiteId!),
    queryFn: () => getTrainingSuite(projectId, trainingSuiteId!),
    enabled: !!trainingSuiteId,
    refetchInterval: runStatusRefetchInterval({
      isSettled: suiteAllTerminal,
      activeMs: POLL_INTERVAL_MS,
    }),
  });

  // Live SSE invalidation: whenever a TAO-job event arrives, refresh the
  // suite + the specific job's detail cache so React Query refetches.
  useEffect(() => {
    if (!lastEvent || !trainingSuiteId) return;
    const { type, data } = lastEvent;
    if (
      (type === "tao_job_progress" ||
        type === "tao_job_completed" ||
        type === "run_failed") &&
      data.run_type === "tao_job"
    ) {
      queryClient.invalidateQueries({
        queryKey: trainingKeys.suite(projectId, trainingSuiteId),
      });
      if (typeof data.tao_job_id === "string") {
        queryClient.invalidateQueries({
          queryKey: trainingKeys.job(projectId, data.tao_job_id),
        });
      }
    }
  }, [lastEvent, projectId, trainingSuiteId, queryClient]);

  const allChainsComplete = useMemo(
    () =>
      suite !== undefined &&
      suite.chains.length > 0 &&
      suite.chains.every(chainAllSucceeded),
    [suite],
  );

  if (isLoading || !suite) {
    return (
      <PageContainer data-testid="training-job-monitor-page-loading">
        <div className="flex items-center justify-center py-20">
          <Spinner aria-label="Loading training suite" size="large" />
        </div>
      </PageContainer>
    );
  }

  const quantSchemes = suite.quantization_schemes.map((s) => String(s));
  const provisioningFailed =
    suite.status === "failed" &&
    (suite.setup_error_ref ?? "").startsWith("Base provisioning failed:");
  const provisioningCanceled = suite.status === "canceled";
  const provisioningRunning = suite.status === "provisioning";
  const provisioningComplete =
    suite.provisioning_run_id !== null &&
    !provisioningRunning &&
    !provisioningFailed &&
    !provisioningCanceled;
  const suiteCancelable = !TERMINAL_TRAINING_SUITE_STATUSES.has(suite.status);

  // Overall chain progress line (e.g., "8B: done · 2B: 5 of 6") composed
  // from the ordered chains.  Uses the codebase's established middle-dot
  // separator (see NIMConnectionPage, ScanPreview)
  // so the two per-model segments don't visually collide in caption text.
  const overallProgress = suite.chains
    .map((chain) => {
      const label = shortBaseLabel(chain.base_model_name);
      const succeeded = countSucceeded(chain);
      const total = chain.jobs.length;
      if (succeeded === total) return `${label}: done`;
      return `${label}: ${succeeded} of ${total}`;
    })
    .join(" · ");

  return (
    <PageContainer data-testid="training-job-monitor-page">
      <div className="flex items-center justify-between">
        <Text kind="title/md">Training Jobs</Text>
        {overallProgress && (
          <Text
            kind="body/regular/sm"
            style={{ color: "var(--text-muted)" }}
            data-testid="monitor-overall-progress"
          >
            {overallProgress}
          </Text>
        )}
      </div>

      {(cancelResult || suite.status === "canceled") && (
        <InfoBanner
          tone={
            cancelResult && cancelResult.remote_cancel_failures.length > 0
              ? "warning"
              : "success"
          }
          border="edge"
          heading={
            cancelResult && cancelResult.remote_cancel_failures.length > 0
              ? "Training released with TAO warnings."
              : "Training Jobs canceled."
          }
          body={
            cancelResult && cancelResult.remote_cancel_failures.length > 0
              ? `${cancelResult.remote_cancel_failures.length} job cancellation ${
                  cancelResult.remote_cancel_failures.length === 1 ? "was" : "were"
                } not confirmed by TAO. Check TAO before assuming all remote work stopped. This project will no longer resume Training Jobs.`
              : "Completed work was preserved. This project will no longer resume Training Jobs."
          }
          actions={
            <Button
              kind="tertiary"
              onClick={() => navigate("/")}
              style={{ color: "var(--text-primary)", fontWeight: 600 }}
              data-testid="training-canceled-back-to-projects"
            >
              Back to Projects
            </Button>
          }
          data-testid="training-suite-canceled-banner"
        />
      )}

      {suite.provisioning_run_id && (
        <section className="flex flex-col gap-3" data-testid="training-setup-steps">
          <SectionHeading>SETUP</SectionHeading>
          <SectionCard density="dense" data-testid="training-provisioning-step">
            <div className="flex items-center justify-between gap-3">
              <Text kind="label/bold/sm">Provision Student Bases</Text>
              <StatusPill
                tone={
                  provisioningFailed
                    ? "error"
                    : provisioningCanceled
                      ? "neutral"
                      : "success"
                }
                spinner={provisioningRunning}
                label={
                  provisioningFailed
                    ? "Failed"
                    : provisioningCanceled
                      ? "Canceled"
                      : provisioningComplete
                        ? "Completed"
                        : "Running"
                }
                data-testid="training-provisioning-status"
              />
            </div>
            <Text kind="body/regular/sm" style={{ color: "var(--text-muted)" }}>
              {suite.provisioning_model_names
                .map((name) => localTeacherDisplayName(name))
                .join(" · ")}
            </Text>
            {provisioningFailed && suite.setup_error_ref && (
              <Text
                kind="body/regular/sm"
                className="text-error"
                data-testid="training-provisioning-error"
              >
                {suite.setup_error_ref}
              </Text>
            )}
          </SectionCard>

          {suite.status === "preparing" && (
            <SectionCard density="dense" data-testid="training-preparing-step">
              <div className="flex items-center justify-between gap-3">
                <Text kind="label/bold/sm">Prepare Training Jobs</Text>
                <StatusPill
                  tone="success"
                  spinner
                  label="Running"
                  data-testid="training-preparing-status"
                />
              </div>
              <Text kind="body/regular/sm" style={{ color: "var(--text-muted)" }}>
                Building and uploading the training and evaluation datasets.
              </Text>
            </SectionCard>
          )}

          {suite.status === "failed" &&
            !provisioningFailed &&
            suite.setup_error_ref && (
              <InfoBanner
                tone="error"
                border="edge"
                heading="Training Jobs setup failed."
                body={suite.setup_error_ref}
                data-testid="training-setup-error"
              />
            )}
        </section>
      )}

      {suite.chains.map((chain) => {
        const succeeded = countSucceeded(chain);
        const total = chain.jobs.length;
        const halted = findHaltedReasonJob(chain);
        const quantLabels = computeQuantLabels(chain, quantSchemes);

        return (
          <section
            key={chain.chain_id}
            className="space-y-3"
            data-testid={`training-chain-${chain.chain_id}`}
            data-base-model={chain.base_model_name}
          >
            <SectionHeading
              trailing={<ChainProgressLine succeeded={succeeded} total={total} />}
            >
              {formatModelDisplayName(chain.base_model_name, "upper")}
            </SectionHeading>

            {halted && chain.jobs.some((j) => j.chain_halted_reason) && (
              <InfoBanner
                tone="warning"
                border="edge"
                align="center"
                heading={`Chain halted: ${halted.action} failed.`}
                data-testid={`chain-halted-banner-${chain.chain_id}`}
              />
            )}

            <div className="flex flex-col gap-3">
              {[...chain.jobs]
                .sort((a, b) => a.chain_sequence - b.chain_sequence)
                .map((job) => (
                  <JobCard
                    key={job.tao_job_id}
                    projectId={projectId}
                    suiteJob={job}
                    quantizationScheme={quantLabels[job.tao_job_id]}
                  />
                ))}
            </div>
          </section>
        );
      })}

      <div className="flex items-center justify-end gap-3 pt-4">
        {suiteCancelable && (
          <Button
            kind="secondary"
            onClick={() => setCancelDialogOpen(true)}
            data-testid="monitor-cancel-jobs"
          >
            Cancel Jobs
          </Button>
        )}
        {allChainsComplete ? (
          <Button
            kind="primary"
            className="nvidia-green-button"
            onClick={() => navigate("../compare")}
            data-testid="monitor-compare-students"
          >
            Compare Students
          </Button>
        ) : (
          <Tooltip content="Available once every chain reaches Completed">
            <Button
              kind="primary"
              disabled
              data-testid="monitor-compare-students"
              aria-disabled
            >
              Compare Students
            </Button>
          </Tooltip>
        )}
      </div>

      <CancelTrainingSuiteDialog
        open={cancelDialogOpen}
        projectId={projectId}
        trainingSuiteId={trainingSuiteId!}
        onClose={() => setCancelDialogOpen(false)}
        onCanceled={(result) => {
          queryClient.setQueryData(
            trainingKeys.suite(projectId, trainingSuiteId!),
            result.training_suite,
          );
          void queryClient.invalidateQueries({
            queryKey: trainingKeys.suites(projectId),
          });
          setCancelResult(result);
        }}
      />
    </PageContainer>
  );
}
