// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Polymorphic per-job card for the Training Job Monitor.
 *
 * Fetches per-job detail via React Query with a 3s polling fallback
 * while non-terminal; SSE terminal events invalidate the cache upstream.
 */

import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Button, Text, Tooltip } from "@kui/react";
import { AlertTriangle, ExternalLink, FileText } from "lucide-react";

import { ActionRequestPanel } from "@/components/ActionRequestPanel";
import { ProgressBar } from "@/components/ProgressBar";
import { MiniSpinner } from "@/components/common/MiniSpinner";
import { useCopyToClipboard } from "@/hooks/useCopyToClipboard";
import { trainingKeys } from "@/api/query-keys";
import { cancelTAOJob, getTAOJob } from "@/api/training";
import { formatTimestamp } from "@/lib/format-date";
import { runStatusRefetchInterval } from "@/lib/run-status-polling";
import { formatDuration, formatEta } from "@/lib/training/formatters";
import { TrainingStatusBadge } from "@/components/training/TrainingStatusBadge";
import {
  TERMINAL_TAO_STATUSES,
  type TAOJob,
  type TAOJobAction,
  type TAOJobStatus,
  type TrainingSuiteJob,
} from "@/types/training";

interface JobCardProps {
  projectId: string;
  /** The suite-level job summary (source of canonical ordering). */
  suiteJob: TrainingSuiteJob;
  /** Quantization scheme label for display when applicable. */
  quantizationScheme?: string | null;
  "data-testid"?: string;
}

function actionLabel(action: TAOJobAction | string, suffix?: string | null): string {
  const base =
    {
      train: "Train",
      evaluate: "Evaluate",
      quantize: "Quantize",
      inference: "Inference",
    }[action as TAOJobAction] ?? action;
  return suffix ? `${base} (${suffix})` : base;
}

function logsAction(logsRef: string | null | undefined): {
  kind: "link" | "copy" | "none";
  value: string | null;
} {
  if (!logsRef) return { kind: "none", value: null };
  const trimmed = String(logsRef).trim();
  if (/^https?:\/\//i.test(trimmed)) {
    return { kind: "link", value: trimmed };
  }
  return { kind: "copy", value: trimmed };
}

// Friendly display labels for the well-known TAO artifact keys. Names can
// be arbitrary TAO file paths (e.g. evaluate_results.tar.gz), so unknown
// keys pass through unchanged.
const ARTIFACT_LABELS: Record<string, string> = {
  best_model: "Best model",
  latest_model: "Latest model",
  training_config: "Training config",
  metrics_ref: "Metrics",
};

function renderArtifacts(outputs: TAOJob["outputs"]) {
  const rows: Array<{ key: string; label: string; value: string }> = [];
  if (!outputs) return rows;
  if (Array.isArray(outputs.artifacts)) {
    for (const a of outputs.artifacts) {
      const name = a.name ?? a.kind ?? "artifact";
      const ref = a.artifact_ref ?? a.tao_file_path ?? a.uri ?? a.checksum ?? "(ref)";
      rows.push({
        key: name,
        label: ARTIFACT_LABELS[name] ?? name,
        value: String(ref),
      });
    }
  }
  if (outputs.metrics_ref) {
    rows.push({
      key: "metrics_ref",
      label: ARTIFACT_LABELS.metrics_ref,
      value: String(outputs.metrics_ref),
    });
  }
  return rows;
}

export function JobCard({
  projectId,
  suiteJob,
  quantizationScheme = null,
  "data-testid": testid,
}: JobCardProps) {
  const queryClient = useQueryClient();
  const [showReportIssue, setShowReportIssue] = useState(false);
  const { copied: copiedLogsRef, copy: copyLogsRef } = useCopyToClipboard();

  // Fetch full per-job detail for the fields the suite endpoint does not
  // carry (progress, outputs, error_ref, started_at, completed_at).
  // Poll every 3s while non-terminal.
  const { data: job } = useQuery<TAOJob>({
    queryKey: trainingKeys.job(projectId, suiteJob.tao_job_id),
    queryFn: () => getTAOJob(projectId, suiteJob.tao_job_id),
    refetchInterval: runStatusRefetchInterval({
      isSettled: (j: TAOJob | undefined) => !!j && TERMINAL_TAO_STATUSES.has(j.status),
      activeMs: 3000,
    }),
  });

  // Authoritative status: prefer fresh per-job fetch once available, else
  // fall back to the suite snapshot. SSE/refetch keeps these in sync.
  const status: TAOJobStatus = job?.status ?? suiteJob.status;
  const chainHaltedReason = job?.chain_halted_reason ?? suiteJob.chain_halted_reason;

  const cancelMut = useMutation({
    mutationFn: () => cancelTAOJob(projectId, suiteJob.tao_job_id),
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: trainingKeys.job(projectId, suiteJob.tao_job_id),
      });
    },
  });

  const logs = logsAction(job?.outputs?.logs_ref ?? null);
  const title = actionLabel(suiteJob.action, quantizationScheme);
  const hasFooterActions =
    logs.kind !== "none" ||
    (status === "failed" && !chainHaltedReason) ||
    status === "paused";

  function handleCopyLogsRef() {
    if (logs.kind !== "copy" || !logs.value) return;
    void copyLogsRef(logs.value);
  }

  return (
    <div
      className="glass-card glass-card--static p-4 flex flex-col gap-2"
      data-testid={testid ?? `training-job-card-${suiteJob.tao_job_id}`}
      data-action={suiteJob.action}
      data-chain-sequence={suiteJob.chain_sequence}
    >
      {/* Top row: title + status badge ---------------------------------- */}
      <div className="flex items-center justify-between gap-3">
        {/* Deleted cards are audit-only — mute the title so the whole card
            reads subdued, not just the body line. */}
        <Text
          kind="label/bold/sm"
          style={status === "deleted" ? { color: "var(--text-muted)" } : undefined}
        >
          {title}
        </Text>
        <TrainingStatusBadge
          status={status}
          chainHaltedReason={chainHaltedReason}
          data-testid={`training-job-status-${suiteJob.tao_job_id}`}
        />
      </div>

      {/* Status-specific content ---------------------------------------- */}
      {status === "running" && <RunningBody job={job} />}
      {status === "succeeded" && <CompletedBody job={job} />}
      {status === "failed" && !chainHaltedReason && <FailedBody job={job} />}
      {status === "failed" && chainHaltedReason && (
        <HaltedBody reason={chainHaltedReason} />
      )}
      {status === "paused" && <PausedBody job={job} />}
      {status === "canceled" && <CanceledBody job={job} />}
      {status === "deleted" && (
        <Text kind="body/regular/xs" style={{ color: "var(--text-muted)" }}>
          Job removed from TAO. Record preserved locally for audit.
        </Text>
      )}
      {(status === "not_started" ||
        status === "submitting" ||
        status === "submitted" ||
        status === "queued") && <QueuedBody status={status} />}

      {/* Footer actions row — do not reserve space when no action exists. */}
      {hasFooterActions && (
        <div className="flex items-center gap-3 flex-wrap">
          {logs.kind === "link" && (
            <Button
              kind="secondary"
              onClick={() => window.open(logs.value as string, "_blank")}
              data-testid="training-job-view-logs"
            >
              <ExternalLink size={14} /> View Logs
            </Button>
          )}
          {logs.kind === "copy" && (
            <Tooltip
              content={
                copiedLogsRef
                  ? "Logs reference copied to clipboard"
                  : "Copy TAO logs reference"
              }
            >
              <Button
                kind="secondary"
                onClick={handleCopyLogsRef}
                data-testid="training-job-view-logs"
              >
                <FileText size={14} /> {copiedLogsRef ? "Copied" : "Copy Logs Ref"}
              </Button>
            </Tooltip>
          )}
          {status === "failed" && !chainHaltedReason && (
            <Button
              kind="secondary"
              onClick={() => setShowReportIssue((v) => !v)}
              data-testid="training-job-report-issue"
            >
              <AlertTriangle size={14} />{" "}
              {showReportIssue ? "Hide Request" : "Report TAO Issue"}
            </Button>
          )}
          {status === "paused" && (
            <Button
              kind="secondary"
              onClick={() => cancelMut.mutate()}
              disabled={cancelMut.isPending}
              data-testid="training-job-cancel"
            >
              {cancelMut.isPending ? "Canceling..." : "Cancel Job"}
            </Button>
          )}
        </div>
      )}

      {cancelMut.isError && (
        <Text
          kind="body/regular/sm"
          className="text-error"
          data-testid="training-job-cancel-error"
        >
          Cancel failed: {(cancelMut.error as Error).message}
        </Text>
      )}

      {showReportIssue && (
        <div className="glass-inner-panel rounded-[14px] p-3">
          <ActionRequestPanel
            projectId={projectId}
            requestType="tao_issue"
            onClose={() => setShowReportIssue(false)}
          />
        </div>
      )}
    </div>
  );
}

// ── Variant bodies ────────────────────────────────────────────────────────

/**
 * Key/value metrics grid shared by the running ("Latest metrics") and
 * completed ("Final metrics") bodies. The key column hugs its longest
 * metric name so values sit next to their labels (matching the Outputs
 * rows below). Renders nothing when the job hasn't reported metrics yet.
 */
function MetricsGrid({
  heading,
  metrics,
  testId,
}: {
  heading: string;
  metrics: Record<string, unknown> | null | undefined;
  testId: string;
}) {
  if (!metrics || Object.keys(metrics).length === 0) return null;
  return (
    <div className="flex flex-col gap-0.5" data-testid={testId}>
      <Text kind="label/bold/xs" style={{ color: "var(--text-muted)" }}>
        {heading}
      </Text>
      <dl className="grid grid-cols-[max-content_1fr] gap-x-4 gap-y-0.5">
        {Object.entries(metrics).map(([k, v]) => (
          <div key={k} className="contents">
            <Text kind="body/regular/xs" style={{ color: "var(--text-muted)" }}>
              {k}
            </Text>
            <Text kind="body/regular/xs">
              {typeof v === "number" ? v.toFixed(4) : String(v)}
            </Text>
          </div>
        ))}
      </dl>
    </div>
  );
}

function QueuedBody({ status }: { status: TAOJobStatus }) {
  const msg =
    status === "not_started"
      ? "Waiting for a predecessor to complete."
      : status === "submitting"
        ? "Sending to TAO..."
        : "Waiting for TAO to start...";
  return (
    <div className="flex items-center gap-2">
      {status === "submitting" && <MiniSpinner />}
      <Text kind="body/regular/sm" style={{ color: "var(--text-muted)" }}>
        {msg}
      </Text>
    </div>
  );
}

function RunningBody({ job }: { job: TAOJob | undefined }) {
  const p = job?.progress ?? null;
  const current = p?.epoch_current ?? null;
  const total = p?.epoch_total ?? null;
  const hasEpoch = current !== null && total !== null && total > 0;
  const pct = hasEpoch ? Math.min(100, Math.round((current / total) * 100)) : null;
  const eta = formatEta(p?.eta_seconds);
  const metrics = p?.metrics_latest;
  const hasMetrics = !!metrics && Object.keys(metrics).length > 0;

  if (!hasEpoch && eta === null && !hasMetrics) return null;

  return (
    <div className="flex flex-col gap-2">
      {hasEpoch && (
        <Text kind="body/regular/sm">
          Epoch: {current} / {total}
        </Text>
      )}
      {pct !== null && (
        <ProgressBar
          percent={pct}
          fillTestId="training-job-progress-fill"
          ariaLabel="Training epoch progress"
        />
      )}
      {eta !== null && (
        <Text kind="body/regular/xs" style={{ color: "var(--text-muted)" }}>
          {eta}
        </Text>
      )}
      <MetricsGrid
        heading="Latest metrics"
        metrics={metrics}
        testId="training-job-metrics"
      />
    </div>
  );
}

function CompletedBody({ job }: { job: TAOJob | undefined }) {
  const completed = formatTimestamp(job?.completed_at);
  const duration = formatDuration(job?.started_at, job?.completed_at);
  const metrics = job?.progress?.metrics_latest ?? null;
  const artifacts = renderArtifacts(job?.outputs ?? null);
  const hasMetrics = !!metrics && Object.keys(metrics).length > 0;
  if (!completed && !duration && !hasMetrics && artifacts.length === 0) return null;

  return (
    <div className="flex flex-col gap-2">
      {completed && (
        <Text kind="body/regular/sm">
          Completed: {completed}
          {duration ? ` (${duration})` : ""}
        </Text>
      )}
      {!completed && duration && (
        <Text kind="body/regular/sm">Duration: {duration}</Text>
      )}
      <MetricsGrid
        heading="Final metrics"
        metrics={metrics}
        testId="training-job-completed-metrics"
      />
      {artifacts.length > 0 && (
        <div className="flex flex-col gap-0.5" data-testid="training-job-outputs">
          <Text kind="label/bold/xs" style={{ color: "var(--text-muted)" }}>
            Outputs
          </Text>
          {artifacts.map((a) => (
            <div key={a.key} className="flex items-baseline gap-2">
              <Text kind="body/regular/xs" style={{ color: "var(--text-muted)" }}>
                {a.label}
              </Text>
              <Text kind="body/regular/xs">{a.value}</Text>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/**
 * Split a classified TAO failure message into its raw provider message
 * and the friendly remediation hint.
 *
 * ``tao_job_service.classify_tao_failure`` formats actionable failures
 * as ``"<raw provider message> — <friendly hint with next step>"``. The
 * em-dash separator is the canonical delimiter; when it's missing (the
 * failure didn't match a known pattern) we treat the whole string as
 * the raw message and emit no hint. Plain ``null`` falls back to a
 * generic placeholder.
 */
export function splitClassifiedFailure(err: string | null | undefined): {
  primary: string;
  hint: string | null;
} {
  if (!err) return { primary: "TAO job failed.", hint: null };
  const sepIdx = err.indexOf(" — ");
  if (sepIdx < 0) return { primary: err, hint: null };
  return { primary: err.slice(0, sepIdx), hint: err.slice(sepIdx + 3) };
}

// Friendly display labels for the well-known bare-token error_refs (same
// pattern as ARTIFACT_LABELS). Unmatched bare tokens fall back to
// underscore→space + sentence case so no raw snake_case reaches the SME;
// full-sentence provider messages pass through untouched.
const FAILURE_REASON_LABELS: Record<string, string> = {
  submission_interrupted:
    "Submission interrupted — the backend restarted before TAO confirmed this job.",
};

export function humanizeFailureReason(reason: string): string {
  const mapped = FAILURE_REASON_LABELS[reason];
  if (mapped) return mapped;
  if (/^[a-z0-9]+(?:_[a-z0-9]+)+$/.test(reason)) {
    const spaced = reason.replace(/_/g, " ");
    return spaced.charAt(0).toUpperCase() + spaced.slice(1);
  }
  return reason;
}

function FailedBody({ job }: { job: TAOJob | undefined }) {
  const err = job?.error_ref ?? job?.poll_error_ref ?? null;
  const current = job?.progress?.epoch_current ?? null;
  const total = job?.progress?.epoch_total ?? null;
  const hasEpochLocator =
    current !== null && total !== null && total !== undefined && total > 0;
  const pct = hasEpochLocator
    ? Math.min(100, Math.round(((current as number) / (total as number)) * 100))
    : null;

  const { primary, hint } = splitClassifiedFailure(err);

  return (
    <div className="flex flex-col gap-1" data-testid="training-job-failed-body">
      {hasEpochLocator && (
        <Text kind="body/regular/sm">
          Failed at epoch {current} / {total}
          {pct !== null ? ` (${pct}%)` : ""}
        </Text>
      )}
      <Text kind="body/regular/sm" className="text-error">
        {humanizeFailureReason(primary)}
      </Text>
      {hint && (
        <Text
          kind="body/regular/xs"
          className="opacity-80"
          data-testid="training-job-failure-hint"
        >
          {hint}
        </Text>
      )}
    </div>
  );
}

function HaltedBody({ reason }: { reason: string }) {
  return (
    <Text
      kind="body/regular/sm"
      style={{ color: "var(--warning-amber, #f59e0b)" }}
      data-testid="training-job-halted-body"
    >
      {reason.startsWith("Chain halted") ? reason : `Chain halted: ${reason}`}
    </Text>
  );
}

function PausedBody({ job }: { job: TAOJob | undefined }) {
  const p = job?.progress ?? null;
  const current = p?.epoch_current ?? null;
  const total = p?.epoch_total ?? null;
  const pct =
    current !== null && total !== null && total > 0
      ? Math.min(100, Math.round((current / total) * 100))
      : null;
  return (
    <div className="flex flex-col gap-1" data-testid="training-job-paused-body">
      <Text kind="body/regular/sm">Job paused by TAO.</Text>
      {pct !== null && (
        <ProgressBar
          percent={pct}
          variant="paused"
          ariaLabel="Training epoch progress (paused)"
          fillTestId="training-job-paused-progress-fill"
        />
      )}
      {current !== null && total !== null && (
        <Text kind="body/regular/xs" style={{ color: "var(--text-muted)" }}>
          Progress at pause: epoch {current} / {total}
        </Text>
      )}
    </div>
  );
}

function CanceledBody({ job }: { job: TAOJob | undefined }) {
  const canceled = formatTimestamp(job?.completed_at);
  const current = job?.progress?.epoch_current ?? null;
  const total = job?.progress?.epoch_total ?? null;
  const hasEpoch = current !== null && total !== null && total > 0;
  if (!canceled && !hasEpoch) return null;

  return (
    <div className="flex flex-col gap-1">
      {canceled && (
        <Text kind="body/regular/xs" style={{ color: "var(--text-muted)" }}>
          Canceled: {canceled}
        </Text>
      )}
      {hasEpoch && (
        <Text kind="body/regular/xs" style={{ color: "var(--text-muted)" }}>
          Progress at cancellation: epoch {current} / {total}
        </Text>
      )}
    </div>
  );
}
