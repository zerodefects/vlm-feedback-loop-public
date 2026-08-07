// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Batch Labeling pre-run screen.
 *
 * Shows the snapshotted configuration for verification before launch.
 * Advanced controls let the SME bound the run, include previously
 * Auto-Labeled images, and select the run-wide structured generation and ICL
 * contracts that the backend snapshots for reproducibility.
 * Navigates to the Run Status screen on launch.
 */

import { useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Button, Modal, Spinner, Text } from "@kui/react";
import { ChevronDown, ChevronRight, Plus } from "lucide-react";

import { createBatchLabelRun } from "@/api/batch";
import { parseApiErrorDetail } from "@/api/client";
import { fetchScaleUpGate } from "@/api/evaluation";
import { fetchIclCount } from "@/api/guidance";
import { fetchNimEndpoints } from "@/api/nim";
import { ConfigRow } from "@/components/common/ConfigRow";
import { InfoBanner } from "@/components/common/InfoBanner";
import { PageContainer } from "@/components/common/PageContainer";
import { SectionCard } from "@/components/common/SectionCard";
import { SectionHeading } from "@/components/common/SectionHeading";
import { titleCasePreset } from "@/lib/formatPreset";
import { evaluationKeys, guidanceKeys, nimEndpointKeys } from "@/api/query-keys";
import { useSetupContext } from "@/pages/setup-context";
import { useTeacherAndGuidance } from "@/hooks/useTeacherAndGuidance";

export function BatchPreRunPage() {
  const { projectId, project } = useSetupContext();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  // "Restart with prompt-only" (structured-generation rejection
  // recovery on the run-status screen) hands the forced mode over via
  // router state; without it the relaunch falls back to the project
  // default — the exact mode that just failed.
  const { state } = useLocation();
  const forcedPromptOnly =
    (state as { structured_generation_mode?: string } | null)
      ?.structured_generation_mode === "prompt_only";
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [includeAutoLabeled, setIncludeAutoLabeled] = useState(false);
  const [runLimit, setRunLimit] = useState("");
  const [ingestedAfter, setIngestedAfter] = useState("");
  const [ingestedBefore, setIngestedBefore] = useState("");
  const [structuredGenerationMode, setStructuredGenerationMode] = useState<
    "auto" | "prompt_only"
  >(
    forcedPromptOnly || project.structured_generation_mode_default === "prompt_only"
      ? "prompt_only"
      : "auto",
  );
  const [iclMode, setIclMode] = useState<"enabled" | "disabled">("enabled");
  const [showEvaluationWarning, setShowEvaluationWarning] = useState(false);

  // Example counts come from the setup context's project detail —
  // ``ProjectResponse.counts`` carries the same state counts as the
  // project list, so no extra fetch is needed.
  const unlabeledCount = project.counts.unlabeled;
  const autoLabeledCount = project.counts.auto_labeled;
  const inputCount = includeAutoLabeled
    ? unlabeledCount + autoLabeledCount
    : unlabeledCount;
  const hasNoUnlabeled = unlabeledCount === 0;

  // ── Scale-Up Readiness Gate — batch runs require ready ─────────────
  const { data: gate } = useQuery({
    queryKey: evaluationKeys.gate(projectId),
    queryFn: () => fetchScaleUpGate(projectId),
  });
  const gateReady = gate?.gate_status === "ready";

  // ── ICL count — the backend-owned ICL-eligible Edit count ("Verified
  // Edits, non-pool, current Guidance") the batch config snapshot
  // displays.
  const { data: iclCountData } = useQuery({
    queryKey: guidanceKeys.iclCount(projectId),
    queryFn: () => fetchIclCount(projectId),
  });
  const iclCount = iclCountData?.eligible_count ?? 0;

  // ── Teacher name + Guidance version (render
  // `nvidia/cosmos-reason2-8b` and `v3`, not UUIDs). Shares query keys with
  // ScaleUpHubPage so the Scale-Up → Batch Pre-Run navigation hits warm cache.
  const { teacherConfig, teacherName, activeGuidance } = useTeacherAndGuidance(
    projectId,
    project,
  );
  const { data: endpointData, isError: endpointPolicyError } = useQuery({
    queryKey: nimEndpointKeys.list(projectId),
    queryFn: () => fetchNimEndpoints(projectId),
  });
  const teacherEndpoint = endpointData?.items.find(
    (endpoint) => endpoint.endpoint_id === teacherConfig?.endpoint_id,
  );
  const endpointPolicyResolved = teacherEndpoint !== undefined;
  const endpointPolicyUnavailable =
    endpointData !== undefined &&
    teacherConfig !== undefined &&
    teacherEndpoint === undefined;
  const usesEvaluationEndpoint = teacherEndpoint?.usage_policy === "evaluation_only";
  const guidanceLabel = activeGuidance ? `v${activeGuidance.version_number}` : "—";
  const structuredGenerationUnsupported =
    teacherConfig?.structured_generation_support === "unsupported";
  const effectiveStructuredGenerationMode =
    forcedPromptOnly || structuredGenerationUnsupported
      ? "prompt_only"
      : structuredGenerationMode;
  const parsedRunLimit = runLimit === "" ? null : Number(runLimit);
  const runLimitInvalid =
    parsedRunLimit != null && (!Number.isInteger(parsedRunLimit) || parsedRunLimit < 1);
  const dateRangeInvalid =
    ingestedAfter !== "" && ingestedBefore !== "" && ingestedAfter > ingestedBefore;
  const plannedImageCount =
    parsedRunLimit != null && !runLimitInvalid
      ? Math.min(inputCount, parsedRunLimit)
      : inputCount;

  const asUtcTimestamp = (value: string) =>
    value === "" ? null : `${value.length === 16 ? `${value}:00` : value}Z`;

  const launchMutation = useMutation({
    mutationFn: () =>
      createBatchLabelRun(projectId, {
        include_auto_labeled: includeAutoLabeled,
        ...(parsedRunLimit != null && !runLimitInvalid
          ? { run_limit: parsedRunLimit }
          : {}),
        ...(ingestedAfter ? { ingested_after: asUtcTimestamp(ingestedAfter) } : {}),
        ...(ingestedBefore ? { ingested_before: asUtcTimestamp(ingestedBefore) } : {}),
        structured_generation_mode: effectiveStructuredGenerationMode,
        icl_mode: iclMode,
      }),
    onSuccess: (data) => {
      navigate(`../batch-status/${data.run_id}`);
    },
    // A 409 can mean the authoritative gate changed after this screen loaded.
    // Reconcile instead of leaving a green CTA backed by stale readiness.
    onError: () =>
      queryClient.invalidateQueries({ queryKey: evaluationKeys.gate(projectId) }),
  });
  const launchErrorDetail = launchMutation.isError
    ? (parseApiErrorDetail(launchMutation.error) ?? launchMutation.error.message)
    : null;
  const launchRejectedByGate =
    launchErrorDetail != null && /scale-up|readiness gate/i.test(launchErrorDetail);

  function requestLaunch() {
    if (usesEvaluationEndpoint) {
      setShowEvaluationWarning(true);
      return;
    }
    launchMutation.mutate();
  }

  return (
    <PageContainer data-testid="batch-prerun-page">
      <Modal
        open={showEvaluationWarning}
        onOpenChange={setShowEvaluationWarning}
        dismissible
        slotHeading={<Text kind="title/sm">Confirm evaluation use</Text>}
        slotFooter={
          <div className="flex w-full flex-col gap-2">
            <Button kind="secondary" onClick={() => navigate("../settings/nim")}>
              Configure production endpoint
            </Button>
            <div className="flex justify-end gap-3">
              <Button kind="secondary" onClick={() => setShowEvaluationWarning(false)}>
                Cancel
              </Button>
              <Button
                kind="primary"
                className="nvidia-green-button"
                onClick={() => {
                  setShowEvaluationWarning(false);
                  launchMutation.mutate();
                }}
                data-testid="confirm-evaluation-batch"
              >
                Continue evaluation
              </Button>
            </div>
          </div>
        }
        data-testid="evaluation-endpoint-warning"
      >
        <div className="flex flex-col gap-3">
          <Text kind="body/regular/sm" style={{ color: "var(--text-primary)" }}>
            You are about to label up to {plannedImageCount.toLocaleString()} images
            using NVIDIA&apos;s free API Catalog endpoint.
          </Text>
          <Text kind="body/regular/sm" style={{ color: "var(--text-secondary)" }}>
            Catalog credits, including additional trial credits, do not authorize
            production use. Continue only for development or evaluation.
          </Text>
          <Text kind="body/regular/sm" style={{ color: "var(--text-secondary)" }}>
            For production, connect a subscribed provider endpoint or deploy NIM under
            the appropriate NVIDIA AI Enterprise entitlement. Review the{" "}
            <a
              href="https://assets.ngc.nvidia.com/products/api-catalog/legal/NVIDIA%20API%20Trial%20Terms%20of%20Service.pdf"
              target="_blank"
              rel="noreferrer"
              className="underline"
              style={{ color: "var(--accent-green)" }}
            >
              NVIDIA API Trial Terms
            </a>
            .
          </Text>
        </div>
      </Modal>

      <Text kind="title/md">Batch Labeling</Text>

      {/* Row order: Teacher → Guidance → ICL →
           Output Stability → Thinking → Visual Budget → Input. Output
           Stability + Thinking surface the Generation Controls
           that the backend snapshots into every batch
           run so the SME can verify the full locked configuration. */}
      <SectionCard density="dense">
        <SectionHeading>Configuration</SectionHeading>
        <ConfigRow
          size="sm"
          label="Teacher"
          value={teacherName ?? project.teacher_model_config_id ?? "—"}
        />
        {endpointPolicyResolved && (
          <ConfigRow
            size="sm"
            label="Endpoint Use"
            value={
              usesEvaluationEndpoint
                ? "NVIDIA API Catalog · evaluation only"
                : "Operator-managed endpoint"
            }
          />
        )}
        <ConfigRow size="sm" label="Guidance" value={guidanceLabel} />
        <ConfigRow size="sm" label="ICL" value={`${iclCount} edits`} />
        <ConfigRow
          size="sm"
          label="Output Stability"
          value={titleCasePreset(project.labeling_generation_preset_key ?? "precise")}
        />
        <ConfigRow
          size="sm"
          label="Thinking"
          value={project.thinking_default_on ? "ON" : "OFF"}
        />
        {project.visual_budget_preset_key && (
          <ConfigRow
            size="sm"
            label="Visual Budget"
            value={titleCasePreset(project.visual_budget_preset_key)}
          />
        )}
        <ConfigRow
          size="sm"
          label="Structured Generation"
          value={
            forcedPromptOnly
              ? "Prompt-only (forced by restart)"
              : structuredGenerationUnsupported
                ? "Prompt-only (Teacher does not support schemas)"
                : effectiveStructuredGenerationMode === "auto"
                  ? "Auto"
                  : "Prompt-only"
          }
        />
        <ConfigRow
          size="sm"
          label="ICL Mode"
          value={iclMode === "enabled" ? `Enabled (${iclCount} edits)` : "Disabled"}
        />
        {parsedRunLimit != null && !runLimitInvalid && (
          <ConfigRow size="sm" label="Run Limit" value={`${parsedRunLimit} images`} />
        )}
        {ingestedAfter && (
          <ConfigRow
            size="sm"
            label="Ingested After"
            value={`${ingestedAfter.replace("T", " ")} UTC`}
          />
        )}
        {ingestedBefore && (
          <ConfigRow
            size="sm"
            label="Ingested Before"
            value={`${ingestedBefore.replace("T", " ")} UTC`}
          />
        )}
        <ConfigRow
          size="sm"
          label="Input"
          value={
            includeAutoLabeled
              ? `${unlabeledCount} Unlabeled + ${autoLabeledCount} Auto-Labeled images`
              : hasNoUnlabeled && autoLabeledCount > 0
                ? `0 Unlabeled images · ${autoLabeledCount} Auto-Labeled available`
                : `${unlabeledCount} Unlabeled images (excluding Omitted)`
          }
        />
      </SectionCard>

      {/* ── Advanced Filters (above the notice + warning) ──────────────
           Shared-surface collapsible: the toggle is the header row of the
           same bordered container the content expands into, mirroring
           TrainingAdvancedExpander. */}
      <div
        className="border"
        style={{
          borderColor: "var(--glass-border, rgba(255,255,255,0.08))",
          borderRadius: "var(--glass-radius-sm, 14px)",
        }}
      >
        <button
          className="flex w-full items-center gap-1.5 px-4 py-2 text-sm text-left"
          style={{ color: "var(--text-secondary)" }}
          onClick={() => setShowAdvanced((v) => !v)}
          data-testid="advanced-toggle"
        >
          {showAdvanced ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
          Advanced filters
        </button>

        {showAdvanced && (
          <div className="px-4 pb-4 grid gap-4 md:grid-cols-2">
            {/* Label text + helper share one column so both lines keep a
                single left edge past the checkbox. */}
            <label className="flex items-start gap-2 text-sm cursor-pointer md:col-span-2">
              <input
                type="checkbox"
                className="glass-input"
                checked={includeAutoLabeled}
                onChange={(e) => setIncludeAutoLabeled(e.target.checked)}
                data-testid="include-auto-labeled-toggle"
              />
              <div className="flex flex-col gap-1">
                <Text kind="body/regular/sm" style={{ color: "var(--text-primary)" }}>
                  Include previously Auto-Labeled images
                </Text>
                <Text kind="body/regular/sm" style={{ color: "var(--text-secondary)" }}>
                  {activeGuidance
                    ? "Re-run with current Guidance and ICL. Existing Auto-Labeled results will be replaced."
                    : "Activate Guidance before re-labeling existing Auto-Labeled results."}
                </Text>
              </div>
            </label>

            <label className="flex flex-col gap-1.5">
              <Text kind="label/bold/sm">Run limit</Text>
              <input
                type="number"
                className="glass-input w-full"
                min={1}
                step={1}
                value={runLimit}
                onChange={(event) => setRunLimit(event.target.value)}
                placeholder="All eligible images"
                aria-invalid={runLimitInvalid}
                data-testid="batch-run-limit"
              />
              <Text
                kind="body/regular/sm"
                style={{
                  color: runLimitInvalid
                    ? "var(--error-red, #ef4444)"
                    : "var(--text-secondary)",
                }}
                data-testid="batch-run-limit-helper"
              >
                {runLimitInvalid
                  ? "Enter a whole number of 1 or more."
                  : "Leave blank to process every eligible image."}
              </Text>
            </label>

            <label className="flex flex-col gap-1.5">
              <Text kind="label/bold/sm">Ingested at or after (UTC)</Text>
              <input
                type="datetime-local"
                className="glass-input w-full"
                value={ingestedAfter}
                max={ingestedBefore || undefined}
                onChange={(event) => setIngestedAfter(event.target.value)}
                aria-invalid={dateRangeInvalid}
                data-testid="batch-ingested-after"
              />
              <Text kind="body/regular/sm" style={{ color: "var(--text-secondary)" }}>
                Optional lower bound on image ingestion time.
              </Text>
            </label>

            <label className="flex flex-col gap-1.5">
              <Text kind="label/bold/sm">Ingested at or before (UTC)</Text>
              <input
                type="datetime-local"
                className="glass-input w-full"
                value={ingestedBefore}
                min={ingestedAfter || undefined}
                onChange={(event) => setIngestedBefore(event.target.value)}
                aria-invalid={dateRangeInvalid}
                data-testid="batch-ingested-before"
              />
              <Text
                kind="body/regular/sm"
                style={{
                  color: dateRangeInvalid
                    ? "var(--error-red, #ef4444)"
                    : "var(--text-secondary)",
                }}
                data-testid="batch-ingested-range-helper"
              >
                {dateRangeInvalid
                  ? "The end must be the same as or later than the start."
                  : "Optional upper bound on image ingestion time."}
              </Text>
            </label>

            <label className="flex flex-col gap-1.5">
              <Text kind="label/bold/sm">Structured generation</Text>
              <select
                className="glass-input w-full"
                value={effectiveStructuredGenerationMode}
                disabled={forcedPromptOnly || structuredGenerationUnsupported}
                onChange={(event) =>
                  setStructuredGenerationMode(
                    event.target.value as "auto" | "prompt_only",
                  )
                }
                data-testid="batch-structured-generation-mode"
              >
                <option value="auto">Auto (use JSON schema when supported)</option>
                <option value="prompt_only">Prompt-only</option>
              </select>
              <Text kind="body/regular/sm" style={{ color: "var(--text-secondary)" }}>
                {forcedPromptOnly
                  ? "Locked by the failed run's prompt-only restart action."
                  : structuredGenerationUnsupported
                    ? "This Teacher does not support schema-constrained output."
                    : "The selected mode stays fixed for the entire run."}
              </Text>
            </label>

            <label className="flex flex-col gap-1.5">
              <Text kind="label/bold/sm">ICL mode</Text>
              <select
                className="glass-input w-full"
                value={iclMode}
                onChange={(event) =>
                  setIclMode(event.target.value as "enabled" | "disabled")
                }
                data-testid="batch-icl-mode"
              >
                <option value="enabled">Enabled ({iclCount} eligible edits)</option>
                <option value="disabled">Disabled (zero-shot)</option>
              </select>
              <Text kind="body/regular/sm" style={{ color: "var(--text-secondary)" }}>
                Disable only for a Teacher that performs better without examples.
              </Text>
            </label>
          </div>
        )}
      </div>

      {/* ── Auto-Labeled Notice (persistent) ───────────────────────── */}
      <InfoBanner
        tone="info"
        align="center"
        body="Batch Labeling generates Auto-Labeled outputs. These are not ground truth until reviewed."
      />

      {(endpointPolicyError || endpointPolicyUnavailable) && (
        <InfoBanner
          tone="error"
          align="center"
          body="Could not verify the selected Teacher endpoint's usage policy. Reload this page or configure the endpoint before starting a batch."
          data-testid="endpoint-policy-error"
        />
      )}

      {/* ── No Unlabeled Warning (sits below the
           Advanced filters toggle — its own copy refers to the toggle
           "above", which only parses if the warning sits below it) ─── */}
      {hasNoUnlabeled && !includeAutoLabeled && (
        <InfoBanner
          tone="warning"
          heading="No Unlabeled images to process."
          body={
            autoLabeledCount > 0
              ? "To re-label existing Auto-Labeled images with your current setup, expand Advanced filters above."
              : undefined
          }
          data-testid="no-unlabeled-warning"
        />
      )}

      {/* ── Actions ─────────────────────────────────────────────────
           Empty state renders all three buttons:
             [Add Images] [Cancel] [Run Batch Labeling (disabled)]
           keeping the primary CTA visible-but-disabled so the SME sees
           the forward path without having to guess why it disappeared. */}
      <div className="flex flex-col items-end gap-2">
        <div className="flex gap-3 justify-end">
          {hasNoUnlabeled && !includeAutoLabeled && (
            <Button
              kind="secondary"
              onClick={() => navigate("../ready")}
              data-testid="batch-add-images"
            >
              <Plus size={14} />
              Add Images
            </Button>
          )}
          <Button kind="secondary" onClick={() => navigate("../scale-up")}>
            Cancel
          </Button>
          <Button
            kind="primary"
            className="nvidia-green-button"
            disabled={
              inputCount === 0 ||
              runLimitInvalid ||
              dateRangeInvalid ||
              launchMutation.isPending ||
              !gateReady ||
              !endpointPolicyResolved ||
              endpointPolicyError ||
              endpointPolicyUnavailable
            }
            onClick={requestLaunch}
            data-testid="launch-batch"
          >
            {launchMutation.isPending ? (
              <Spinner aria-label="Starting batch run" />
            ) : (
              `Run Batch Labeling`
            )}
          </Button>
        </div>
        {!gateReady && !(hasNoUnlabeled && !includeAutoLabeled) && (
          <Text
            kind="body/regular/sm"
            style={{ color: "var(--text-muted)" }}
            data-testid="batch-gate-helper"
          >
            Gate not ready —{" "}
            <button
              type="button"
              onClick={() => navigate("../scale-up")}
              className="underline"
              style={{ color: "var(--accent-green)", background: "transparent" }}
              data-testid="batch-gate-helper-link"
            >
              see Scale-Up for details
            </button>
            .
          </Text>
        )}
      </div>

      {launchMutation.isError && (
        <InfoBanner
          tone="error"
          align="center"
          body={`Could not start batch run: ${launchErrorDetail}`}
          actions={
            launchRejectedByGate ? (
              <Button
                kind="secondary"
                onClick={() => navigate("../scale-up")}
                data-testid="review-scaleup-after-launch-error"
              >
                Review Scale-Up
              </Button>
            ) : undefined
          }
          data-testid="batch-launch-error"
        />
      )}
    </PageContainer>
  );
}
