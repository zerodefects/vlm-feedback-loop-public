// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Batch Labeling pre-run screen.
 *
 * Shows the snapshotted configuration for verification before launch.
 * Advanced filters let the SME include previously Auto-Labeled images.
 * Navigates to the Run Status screen on launch.
 */

import { useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { useMutation, useQuery } from "@tanstack/react-query";
import { Button, Spinner, Text } from "@kui/react";
import { ChevronDown, ChevronRight, Plus } from "lucide-react";

import { createBatchLabelRun } from "@/api/batch";
import { fetchScaleUpGate } from "@/api/evaluation";
import { fetchIclCount } from "@/api/guidance";
import { ConfigRow } from "@/components/common/ConfigRow";
import { InfoBanner } from "@/components/common/InfoBanner";
import { PageContainer } from "@/components/common/PageContainer";
import { SectionCard } from "@/components/common/SectionCard";
import { SectionHeading } from "@/components/common/SectionHeading";
import { titleCasePreset } from "@/lib/formatPreset";
import { evaluationKeys, guidanceKeys } from "@/api/query-keys";
import { useSetupContext } from "@/pages/ProjectSetupLayout";
import { useTeacherAndGuidance } from "@/hooks/useTeacherAndGuidance";

export function BatchPreRunPage() {
  const { projectId, project } = useSetupContext();
  const navigate = useNavigate();
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
  const { teacherName, activeGuidance } = useTeacherAndGuidance(projectId, project);
  const guidanceLabel = activeGuidance ? `v${activeGuidance.version_number}` : "—";

  const launchMutation = useMutation({
    mutationFn: () =>
      createBatchLabelRun(projectId, {
        include_auto_labeled: includeAutoLabeled,
        ...(forcedPromptOnly ? { structured_generation_mode: "prompt_only" } : {}),
      }),
    onSuccess: (data) => {
      navigate(`../batch-status/${data.run_id}`);
    },
  });

  return (
    <PageContainer data-testid="batch-prerun-page">
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
        {forcedPromptOnly && (
          <ConfigRow
            size="sm"
            label="Structured Generation"
            value="Prompt-only (forced by restart)"
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
          <div className="px-4 pb-4 flex flex-col gap-3">
            {/* Label text + helper share one column so both lines keep a
                single left edge past the checkbox. */}
            <label className="flex items-start gap-2 text-sm cursor-pointer">
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
                  Re-run with current Guidance and ICL. Existing Auto-Labeled results
                  will be replaced.
                </Text>
              </div>
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
            disabled={inputCount === 0 || launchMutation.isPending || !gateReady}
            onClick={() => launchMutation.mutate()}
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
          body="Failed to start batch run. Please try again."
        />
      )}
    </PageContainer>
  );
}
