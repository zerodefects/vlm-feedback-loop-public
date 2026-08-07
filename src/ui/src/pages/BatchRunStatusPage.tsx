// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Batch Labeling Run Status screen.
 *
 * Supports "walk away and come back": live progress via polling (this
 * page opens no SSE connection), schema-valid/invalid Core rates,
 * circuit breaker pause handling, and dataset export on completion.
 */

import { useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Button, Spinner, Text } from "@kui/react";
import { Download, Play } from "lucide-react";

import {
  cancelBatchLabelRun,
  createDatasetExport,
  datasetExportArchiveUrl,
  getBatchLabelRun,
  getDatasetExport,
  listDatasetExports,
  getSchemaInvalidManifest,
  resumeBatchLabelRun,
} from "@/api/batch";
import { parseApiErrorDetail } from "@/api/client";
import { batchKeys } from "@/api/query-keys";
import { ConfigRow } from "@/components/common/ConfigRow";
import { InfoBanner } from "@/components/common/InfoBanner";
import { PageContainer } from "@/components/common/PageContainer";
import { SectionCard } from "@/components/common/SectionCard";
import { SectionHeading } from "@/components/common/SectionHeading";
import { StatBlock } from "@/components/common/StatBlock";
import { StatusPill } from "@/components/common/StatusPill";
import { ProgressBar, type ProgressBarVariant } from "@/components/ProgressBar";
import { titleCasePreset } from "@/lib/formatPreset";
import { runStatusRefetchInterval } from "@/lib/run-status-polling";
import type { Tone } from "@/lib/tone";
import { useSetupContext } from "@/pages/setup-context";
import type { BatchLabelRunResponse, CommonErrorEntry } from "@/types/batch";

const TERMINAL = new Set(["completed", "canceled", "failed"]);

const PROGRESS_VARIANT_BY_STATUS: Record<string, ProgressBarVariant> = {
  queued: "default",
  running: "default",
  paused: "paused",
  canceling: "neutral",
  completed: "default",
  failed: "error",
  canceled: "neutral",
};

// Batch runs carry no logs_ref, so the failure copy names where the logs
// actually live rather than pointing at an affordance the page doesn't have.
const LOGS_HINT =
  "Check the backend logs (the project's logs/ directory in the workspace) for details.";

const STATUS_REASON_COPY: Record<string, string> = {
  internal_error: `Internal error during batch labeling. ${LOGS_HINT}`,
  prompt_rendering_error: `Internal error during prompt rendering. ${LOGS_HINT}`,
  database_error: `Database error during batch labeling. ${LOGS_HINT}`,
  unhandled_exception: `An unexpected error stopped batch labeling. ${LOGS_HINT}`,
  gate_revoked:
    "Scale-Up gate was revoked while the run was in flight. Rerun after the gate passes.",
  unrecoverable_config: `Unrecoverable configuration issue. ${LOGS_HINT}`,
  schema_evolution_canceled:
    "The Guidance schema changed (labels were reset) while this run was in flight, so its outputs were discarded. Start a new run under the current Guidance.",
  guidance_edited_during_run:
    "The Guidance was edited while this run was in flight, so the run was stopped. Its completed outputs were kept and moved to the new Guidance; start a new run to label the rest.",
};

function humanizeStatusReason(reason: string | null | undefined): string {
  if (!reason) return `Internal error during batch labeling. ${LOGS_HINT}`;
  return STATUS_REASON_COPY[reason] ?? reason;
}

function formatConfigLine(run: BatchLabelRunResponse): string {
  const model = run.model_name ?? "—";
  const guidance =
    run.guidance_version_number != null ? `v${run.guidance_version_number}` : "—";
  const stability = titleCasePreset(run.generation_preset_key ?? "precise");
  const visualBudget = titleCasePreset(run.visual_budget_preset_key ?? "high_detail");
  const thinking =
    run.thinking_mode_effective === "on" ? "Thinking On" : "Thinking Off";
  const icl = run.icl_mode === "disabled" ? "ICL Off" : "ICL On";
  const structured =
    run.structured_generation_mode_effective === "prompt_only"
      ? "Prompt-only"
      : "Structured Auto";
  return `Config: ${model} · Guidance ${guidance} · ${stability} · ${thinking} · ${visualBudget} · ${icl} · ${structured}`;
}

export function BatchRunStatusPage() {
  const { projectId, project } = useSetupContext();
  const { runId } = useParams<{ runId: string }>();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [manifestError, setManifestError] = useState<string | null>(null);
  // Failure line for the Resume / Cancel / Export actions — this is the
  // walk-away screen, so a rejected mutation must not read as "the
  // button did nothing". Cleared when the next attempt starts.
  const [actionError, setActionError] = useState<string | null>(null);

  const actionErrorHandler = (verb: string) => (e: Error) =>
    setActionError(`Could not ${verb}: ${parseApiErrorDetail(e) ?? e.message}`);

  const { data: run, isLoading } = useQuery({
    queryKey: batchKeys.detail(projectId, runId!),
    queryFn: () => getBatchLabelRun(projectId, runId!),
    enabled: !!runId,
    refetchInterval: runStatusRefetchInterval({
      isSettled: (run: BatchLabelRunResponse | undefined) => {
        const status = run?.status;
        return !!status && (TERMINAL.has(status) || status === "paused");
      },
      activeMs: 3000,
    }),
  });

  const resumeMut = useMutation({
    mutationFn: () => resumeBatchLabelRun(projectId, runId!),
    onMutate: () => setActionError(null),
    onSuccess: () =>
      queryClient.invalidateQueries({
        queryKey: batchKeys.detail(projectId, runId!),
      }),
    onError: actionErrorHandler("resume the run"),
  });

  const cancelMut = useMutation({
    mutationFn: () => cancelBatchLabelRun(projectId, runId!),
    onMutate: () => setActionError(null),
    onSuccess: () =>
      queryClient.invalidateQueries({
        queryKey: batchKeys.detail(projectId, runId!),
      }),
    onError: actionErrorHandler("cancel the run"),
  });

  const exportMut = useMutation({
    mutationFn: () =>
      createDatasetExport(projectId, {
        dataset_intent: "training",
        label_tier_filter: "auto_labeled_only",
        export_field_mode: project.export_field_mode,
        batch_label_run_id: runId!,
      }),
    onMutate: () => setActionError(null),
    onError: actionErrorHandler("export the dataset"),
  });

  // A background export outlives navigation: on mount, adopt this run's
  // export so its lifecycle stays visible and the Export button cannot
  // start a duplicate multi-GB build (the backend also refuses with a
  // 409). Never attribute another run's project-level export to this
  // screen. Take a fresh REST snapshot on every mount, even when this
  // QueryClient already cached the pre-export list from an earlier visit.
  // staleTime: Infinity then prevents focus refetches from un-adopting an
  // export that has since completed — the poll query below owns live
  // status. Caution: sse-store's invalidateProject matches this key by
  // predicate and would bypass staleTime — if this page ever mounts
  // useProjectSSE, an unrelated event's refetch could un-adopt a terminal
  // export and vanish its banner.
  const { data: existingExports } = useQuery({
    queryKey: batchKeys.exportList(projectId),
    queryFn: () => listDatasetExports(projectId),
    staleTime: Infinity,
    refetchOnMount: "always",
  });
  // An export created on this screen wins over the mount-time adoption.
  const exportId =
    exportMut.data?.dataset_export_id ??
    existingExports?.items.find(
      (e) =>
        (e.status === "running" || e.status === "completed") &&
        e.selection_definition_snapshot.batch_label_run_id === runId,
    )?.dataset_export_id ??
    null;

  // The archive builds in the background; the create response is only the
  // running record. Poll the export until terminal so a build failure
  // (missing image, backend restart) is surfaced instead of a permanent
  // affirmative banner — REST stays the authoritative reconciliation.
  const { data: exportRecord } = useQuery({
    queryKey: batchKeys.export(projectId, exportId ?? ""),
    queryFn: () => getDatasetExport(projectId, exportId!),
    enabled: exportId != null,
    refetchInterval: (query) => (query.state.data?.status === "running" ? 3000 : false),
  });

  const manifestMut = useMutation({
    mutationFn: async () => {
      const manifest = await getSchemaInvalidManifest(projectId, runId!);
      const blob = new Blob([JSON.stringify(manifest, null, 2)], {
        type: "application/json",
      });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `schema-invalid-manifest-${runId}.json`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    },
    onError: (e: Error) => setManifestError(e.message),
    onSuccess: () => setManifestError(null),
  });

  const downloadCompletedExport = () => {
    if (exportId == null || exportRecord?.status !== "completed") return;
    const link = document.createElement("a");
    link.href = datasetExportArchiveUrl(projectId, exportId);
    // The backend's Content-Disposition owns the stable archive filename.
    link.download = "";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // Derived values
  const status = run?.status ?? "loading";
  const total = run?.examples_total ?? 0;
  const succeeded = run?.examples_succeeded ?? 0;
  const schemaInvalid = run?.examples_schema_invalid ?? 0;
  const timedOut = run?.examples_timeout ?? 0;
  const endpointErr = run?.examples_endpoint_error ?? 0;
  const processed = succeeded + schemaInvalid + timedOut + endpointErr;
  const pct = total > 0 ? Math.round((processed / total) * 100) : 0;
  const validRate =
    processed > 0 ? `${((succeeded / processed) * 100).toFixed(1)}%` : "—";
  const invalidRate =
    processed > 0 ? `${((schemaInvalid / processed) * 100).toFixed(1)}%` : "—";

  const isStructuredGenRejected =
    status === "failed" && run?.status_reason === "structured_generation_rejected";

  // Shared across the failed / structured-gen-rejected / canceled
  // banners: what a partial run leaves behind.
  // A semantic schema-evolution cancel DELETED the partial outputs, so
  // the "retained" line would be factually false only for that reason;
  // a non-semantic guidance edit keeps and re-points them.
  const partialRetainedLine =
    succeeded > 0 && run?.status_reason !== "schema_evolution_canceled"
      ? `${succeeded} Auto-Labeled outputs from partial run retained.`
      : undefined;

  if (isLoading || !run) {
    return (
      <PageContainer data-testid="batch-run-status-page">
        <div className="flex items-center justify-center py-20">
          <Spinner aria-label="Loading run status" />
        </div>
      </PageContainer>
    );
  }

  const commonErrors = run.common_errors ?? [];
  const showProgressLine = total > 0;

  return (
    <PageContainer data-testid="batch-run-status-page">
      {/* ── Header ─────────────────────────────────────────────────── */}
      <div className="flex items-center justify-between">
        <Text kind="title/md">Batch Labeling Run</Text>
        <BatchRunStatusBadge status={status} />
      </div>

      {/* ── Config Snapshot ─────────────────────────────────────────── */}
      <div className="glass-card p-3">
        <Text
          kind="body/regular/sm"
          style={{ color: "var(--text-secondary)" }}
          data-testid="config-snapshot"
        >
          {formatConfigLine(run)}
        </Text>
      </div>

      {/* ── Progress Bar + count line (all states with total > 0) ─────
          Every state shows the bar at the final % so
          operators see how far the run got regardless of whether it
          finished, failed, or was canceled. Variant mirrors the StatusPill
          tone: running→default (green), paused→paused (amber),
          completed→default (green 100%), failed→error (red),
          canceled→neutral (muted). */}
      {showProgressLine && (
        <div className="space-y-2">
          <ProgressBar
            percent={pct}
            variant={PROGRESS_VARIANT_BY_STATUS[status] ?? "default"}
            ariaLabel="Batch labeling progress"
          />
          <Text
            kind="body/regular/sm"
            style={{ color: "var(--text-secondary)" }}
            data-testid={
              TERMINAL.has(status) ? "progress-line-terminal" : "progress-line"
            }
          >
            {TERMINAL.has(status)
              ? `${processed} of ${total}`
              : `${processed} of ${total} (${pct}%)`}
          </Text>
        </div>
      )}

      {/* ── Counters ───────────────────────────────────────────────── */}
      <SectionCard density="dense">
        <div className="flex flex-wrap gap-6">
          <StatBlock
            label="Valid rate"
            value={validRate}
            tone="green"
            data-testid="counters-valid-rate"
          />
          <StatBlock
            label="Invalid"
            value={schemaInvalid}
            data-testid="counters-invalid"
          />
          {timedOut + endpointErr > 0 && (
            <StatBlock
              label="Errors"
              value={timedOut + endpointErr}
              data-testid="counters-errors"
            />
          )}
        </div>
        <div className="flex items-center justify-between gap-4 text-sm">
          <Text kind="body/regular/sm" style={{ color: "var(--text-secondary)" }}>
            Schema-valid Core
          </Text>
          <Text kind="body/regular/sm">
            {succeeded} ({validRate})
          </Text>
        </div>
        <div className="flex items-center justify-between gap-4 text-sm">
          <Text kind="body/regular/sm" style={{ color: "var(--text-secondary)" }}>
            Schema-invalid Core
          </Text>
          {/* Button before value so the numeric value stays flush against
              the card's right edge, aligned with the other rows' values. */}
          <div className="flex items-center gap-3">
            {schemaInvalid > 0 && (
              <Button
                kind="tertiary"
                onClick={() => manifestMut.mutate()}
                disabled={manifestMut.isPending}
                data-testid="download-manifest-btn"
              >
                <Download size={12} />
                {manifestMut.isPending ? "Preparing…" : "Download manifest"}
              </Button>
            )}
            <Text kind="body/regular/sm">
              {schemaInvalid} ({invalidRate})
            </Text>
          </div>
        </div>
        {timedOut > 0 && (
          <div className="flex items-center justify-between gap-4 text-sm">
            <Text kind="body/regular/sm" style={{ color: "var(--text-secondary)" }}>
              Timeout
            </Text>
            <Text kind="body/regular/sm">{timedOut}</Text>
          </div>
        )}
        {endpointErr > 0 && (
          <div className="flex items-center justify-between gap-4 text-sm">
            <Text kind="body/regular/sm" style={{ color: "var(--text-secondary)" }}>
              Endpoint error
            </Text>
            <Text kind="body/regular/sm">{endpointErr}</Text>
          </div>
        )}
      </SectionCard>
      {manifestError && (
        <Text
          kind="body/regular/sm"
          className="text-error"
          data-testid="manifest-download-error"
        >
          Could not download manifest: {manifestError}
        </Text>
      )}

      {/* ── Common errors block ────────────────────────────────────── */}
      <CommonErrorsBlock errors={commonErrors} />

      {/* ── Paused: Circuit Breaker ────────────────────────────────── */}
      {status === "paused" && (
        <InfoBanner
          tone="warning"
          heading="Endpoint appears unreachable."
          body={
            run.paused_reason === "circuit_breaker_threshold_reached"
              ? run.circuit_breaker_threshold != null
                ? `${run.circuit_breaker_threshold} consecutive endpoint failures reached the run's safety threshold.`
                : "The run reached its consecutive endpoint-failure safety threshold."
              : (run.paused_reason ?? "Run paused.")
          }
          data-testid="circuit-breaker-banner"
        />
      )}

      {/* ── Failed ──────────────────────────────────────────────────── */}
      {status === "failed" && !isStructuredGenRejected && (
        <InfoBanner
          tone="error"
          heading="Run failed"
          body={humanizeStatusReason(run.status_reason)}
          extra={partialRetainedLine}
          data-testid="failed-banner"
        />
      )}

      {/* ── Structured Generation Rejected (message only; restart action is in the footer) ── */}
      {isStructuredGenRejected && (
        <InfoBanner
          tone="error"
          heading="Run failed: structured generation rejected."
          body="The model rejected json_schema output for this run. Prompt-only asks for JSON in the prompt instead of enforcing a schema."
          extra={partialRetainedLine}
          data-testid="structured-gen-rejected-banner"
        />
      )}

      {/* ── Canceled ───────────────────────────────────────────────── */}
      {status === "canceled" && (
        <InfoBanner
          tone="info"
          align="center"
          body={
            <>
              {run.paused_reason === "circuit_breaker_threshold_reached"
                ? "Canceled by user after circuit breaker pause."
                : "Canceled by user."}{" "}
              {partialRetainedLine ?? "No outputs retained."}
            </>
          }
        />
      )}

      {/* ── Completed Notice ───────────────────────────────────────── */}
      {status === "completed" && (
        <InfoBanner
          tone="success"
          body={
            <>
              {succeeded} Auto-Labeled outputs ready for export. Schema-invalid and
              errored examples are excluded.
            </>
          }
          data-testid="completed-banner"
        />
      )}

      {/* ── Export lifecycle notice ─────────────────────────────────── */}
      {exportId != null && exportRecord?.status !== "failed" && (
        <InfoBanner
          tone={exportRecord?.status === "completed" ? "success" : "info"}
          align="center"
          icon={
            exportRecord?.status === "completed" ? undefined : (
              <Spinner size="small" aria-label="Building dataset export" />
            )
          }
          data-testid="export-status-banner"
          body={
            exportRecord?.status === "completed"
              ? `Dataset exported successfully — ${
                  exportRecord.progress?.images_written ?? "all"
                } images archived.`
              : `Dataset export in progress — ${
                  exportRecord?.progress?.images_written ?? 0
                } of ${exportRecord?.progress?.images_total ?? exportRecord?.example_count ?? 0} images archived. You can leave this screen; the run will keep tracking it.`
          }
        />
      )}
      {exportRecord?.status === "failed" && (
        <InfoBanner
          tone="error"
          align="center"
          data-testid="export-failed-banner"
          body={`Dataset export failed${
            exportRecord.status_reason ? `: ${exportRecord.status_reason}` : ""
          }. Fix the cause and export again.`}
        />
      )}

      {/* Export records carry private workspace refs. Surface the stable
          logical package contract while keeping those host paths out of the
          browser. */}
      {exportId != null && exportRecord != null && (
        <SectionCard density="dense" data-testid="export-details">
          <SectionHeading>Export details</SectionHeading>
          <ConfigRow
            size="sm"
            label="Format"
            value="Cosmos-RL (annotations.json + images)"
          />
          <ConfigRow
            size="sm"
            label="Fields"
            value={titleCasePreset(exportRecord.export_field_mode)}
          />
          <ConfigRow
            size="sm"
            label="Labels"
            value={titleCasePreset(exportRecord.label_tier_filter)}
          />
          <ConfigRow
            size="sm"
            label="Examples"
            value={`${exportRecord.example_count} images`}
          />
          <ConfigRow
            size="sm"
            label="Manifest"
            value={exportRecord.manifest_ref ? "Included" : "Pending"}
          />
        </SectionCard>
      )}

      {/* ── Actions ───────────────────────────────────────────────── */}
      {actionError && (
        <Text
          kind="body/regular/sm"
          className="text-error text-right"
          data-testid="action-error"
        >
          {actionError}
        </Text>
      )}
      <div className="flex gap-3 justify-end flex-wrap" data-testid="actions">
        {/* Paused: Resume + Cancel */}
        {status === "paused" && (
          <>
            <Button
              kind="secondary"
              disabled={cancelMut.isPending}
              onClick={() => cancelMut.mutate()}
              data-testid="cancel-btn"
            >
              Cancel
            </Button>
            <Button
              kind="primary"
              className="nvidia-green-button"
              disabled={resumeMut.isPending}
              onClick={() => resumeMut.mutate()}
              data-testid="resume-btn"
            >
              <Play size={14} />
              Resume
            </Button>
          </>
        )}

        {/* Running / queued: Cancel only */}
        {(status === "running" || status === "queued") && (
          <Button
            kind="secondary"
            disabled={cancelMut.isPending}
            onClick={() => cancelMut.mutate()}
            data-testid="cancel-btn"
          >
            Cancel
          </Button>
        )}

        {/* Terminal: Back + Export + Restart (structured-gen-rejected).
            The green primary anchors the rightmost slot, matching the
            paused row and BatchPreRunPage: Export Dataset when completed,
            Restart when struct-gen-rejected (the recommended recovery
            there — status is "failed", so Export stays secondary and the
            two primaries never co-occur). */}
        {TERMINAL.has(status) && (
          <>
            <Button
              kind="secondary"
              onClick={() => navigate("../scale-up")}
              data-testid="back-to-scaleup"
            >
              Back to Scale-Up
            </Button>
            {succeeded > 0 &&
              (exportId == null || exportRecord?.status === "failed") && (
                <Button
                  kind={status === "completed" ? "primary" : "secondary"}
                  className={status === "completed" ? "nvidia-green-button" : undefined}
                  disabled={exportMut.isPending}
                  onClick={() => exportMut.mutate()}
                  data-testid="export-dataset-btn"
                >
                  {exportMut.isPending ? (
                    <>
                      <Spinner size="small" aria-label="Requesting dataset export" />
                      Requesting…
                    </>
                  ) : (
                    <>
                      <Download size={14} />
                      {status === "completed" ? "Export Dataset" : "Export Partial"}
                    </>
                  )}
                </Button>
              )}
            {isStructuredGenRejected && (
              <Button
                kind="primary"
                className="nvidia-green-button"
                onClick={() =>
                  navigate("../batch-prerun", {
                    state: { structured_generation_mode: "prompt_only" },
                  })
                }
                data-testid="restart-prompt-only"
              >
                Restart with prompt-only
              </Button>
            )}
            {exportRecord?.status === "completed" && exportId != null && (
              <Button
                kind={isStructuredGenRejected ? "secondary" : "primary"}
                className={isStructuredGenRejected ? undefined : "nvidia-green-button"}
                onClick={downloadCompletedExport}
                data-testid="download-export-archive-btn"
              >
                <Download size={14} />
                Download archive
              </Button>
            )}
          </>
        )}
      </div>
    </PageContainer>
  );
}

// ── Common errors block ──────────────────────────────────────────────────

function CommonErrorsBlock({ errors }: { errors: CommonErrorEntry[] }) {
  if (errors.length === 0) return null;
  return (
    <SectionCard density="dense" data-testid="common-errors-block">
      <SectionHeading>Common errors</SectionHeading>
      <ul className="space-y-1">
        {errors.map((e) => (
          <li
            key={e.code}
            className="flex items-start justify-between gap-3"
            data-testid={`common-error-${e.code}`}
          >
            <Text kind="body/regular/sm" style={{ color: "var(--text-secondary)" }}>
              {formatErrorCode(e)}
            </Text>
            <Text kind="body/regular/sm" style={{ color: "var(--text-muted)" }}>
              ({e.count})
            </Text>
          </li>
        ))}
      </ul>
    </SectionCard>
  );
}

function formatErrorCode(entry: CommonErrorEntry): string {
  // "schema_invalid:primary_damage" → "primary_damage: value not in allowed values"
  if (entry.code.startsWith("schema_invalid:") && entry.sample) {
    return entry.sample;
  }
  if (entry.code === "schema_invalid") {
    return entry.sample ?? "Schema validation failed";
  }
  if (entry.code === "timeout") return "Request timed out";
  if (entry.code === "endpoint_error") return "Could not reach the NIM endpoint";
  if (entry.code === "structured_generation_rejected")
    return "Structured-generation output rejected";
  return entry.sample ?? entry.code;
}

// ── Status Badge ──────────────────────────────────────────────────────────
// Domain-specific thin wrapper around the shared <StatusPill/> — resolves
// a batch-run status string into (tone, label) and delegates all chrome.

// ``spinner`` carries the "still working" motion cue on in-progress
// states so the pill reads live during a walk-away-and-come-back run.
// Matches the Training Job Monitor (see
// src/lib/training/statusDisplay.ts), which already sets spinner=true on
// running/submitting. Both screens share <StatusPill/>; keeping the
// spinner rule aligned is the point of the shared component.
const BATCH_STATUS_DISPLAY: Record<
  string,
  { label: string; tone: Tone; spinner: boolean }
> = {
  queued: { label: "Queued", tone: "neutral", spinner: false },
  running: { label: "Running", tone: "success", spinner: true },
  paused: { label: "Paused", tone: "warning", spinner: false },
  canceling: { label: "Canceling", tone: "neutral", spinner: true },
  completed: { label: "Completed", tone: "success", spinner: false },
  canceled: { label: "Canceled", tone: "neutral", spinner: false },
  failed: { label: "Failed", tone: "error", spinner: false },
};

function BatchRunStatusBadge({ status }: { status: string }) {
  const d = BATCH_STATUS_DISPLAY[status] ?? {
    label: status,
    tone: "neutral" as Tone,
    spinner: false,
  };
  return (
    <StatusPill
      tone={d.tone}
      label={d.label}
      spinner={d.spinner}
      data-testid="status-badge"
      data-status={status}
    />
  );
}
