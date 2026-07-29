// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Evaluation strip — appears on the labeling screen once the Test Pool
 * has members. Shows pool counter, trigger recommendations, evaluation
 * progress, and Auto-Evaluate toggle.
 */

import { Button, Spinner, Text } from "@kui/react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { AlertOctagon, Info, X } from "lucide-react";
import { useState } from "react";

import { parseApiErrorDetail } from "@/api/client";
import {
  cancelEvaluationRun,
  createEvaluationRun,
  dismissTrigger,
  fetchTriggerStatus,
  listEvaluationRuns,
} from "@/api/evaluation";
import { updateProject } from "@/api/model-configs";
import { evaluationKeys, projectKeys } from "@/api/query-keys";
import { formatTimestamp } from "@/lib/format-date";
import { formatDeltaPoints, formatPct } from "@/lib/format-percent";
import { runStatusRefetchInterval } from "@/lib/run-status-polling";
import type {
  EvaluationRunListResponse,
  EvaluationRunResponse,
} from "@/types/evaluation";
import { InfoBanner } from "@/components/common/InfoBanner";
import { SegmentedControl } from "@/components/SegmentedControl";

import { safeMetrics } from "./metricsHelpers";

interface EvaluationStripProps {
  projectId: string;
  poolCount: number;
  /**
   * Growth disclosure: the pool's target size,
   * floor(total_verified × test_pool_fraction), from the Scale-Up gate's
   * min_test_pool_size criterion details. When it exceeds `poolCount`,
   * verifying more labels will grow the holdout — shown on the chip so
   * the benchmark never re-bases silently between evaluations.
   */
  poolTarget?: number | null;
  onShowResults: (run: EvaluationRunResponse) => void;
  /**
   * When `true`, suppress the first-pool-threshold trigger banner
   * so it doesn't compete with a schema refinement reminder
   * that's currently on screen. Schema refinement is higher-stakes because
   * semantic schema changes invalidate existing labels, so it wins the
   * single-nudge slot until the SME dismisses or acts on it.
   *
   * Only the first-pool banner is suppressed. `configuration_change` and
   * `icl_growth` are unaffected — neither fires at the low verified counts
   * where a schema reminder lives.
   */
  suppressFirstPoolTrigger?: boolean;
}

export function EvaluationStrip({
  projectId,
  poolCount,
  poolTarget = null,
  onShowResults,
  suppressFirstPoolTrigger = false,
}: EvaluationStripProps) {
  const queryClient = useQueryClient();
  const [dismissedBanners, setDismissedBanners] = useState<Set<string>>(new Set());

  // ── Queries ────────────────────────────────────────────────────────────

  const { data: triggers } = useQuery({
    queryKey: evaluationKeys.triggerStatus(projectId),
    queryFn: () => fetchTriggerStatus(projectId),
    refetchInterval: 10_000,
  });

  // Faster poll while a run is in flight so small pools (e.g., the
  // first-pool 5-image trigger that completes in ~1.5-2s under
  // EVAL_CONCURRENCY=8) actually surface mid-run progress instead of
  // jumping straight from "0 of 5" to "completed". Once terminal,
  // relax to 5s — the strip just needs to notice eventual transitions.
  // React Query re-evaluates refetchInterval reactively, so a function
  // form picks up status changes automatically without manual state.
  // basis: "gate" — the strip is the SME's current-evaluation surface.
  // Student benchmark runs (§9.5.2) run concurrently with gate-basis
  // evals and were never the strip's "current evaluation"; without the
  // scope, a running benchmark would hijack the strip and its [Cancel].
  const { data: runsData } = useQuery({
    queryKey: evaluationKeys.list(projectId),
    queryFn: () => listEvaluationRuns(projectId, { basis: "gate", limit: 5 }),
    refetchInterval: runStatusRefetchInterval({
      isSettled: (data: EvaluationRunListResponse | undefined) => {
        const status = data?.items?.[0]?.status;
        return !(status === "running" || status === "queued");
      },
      activeMs: 750,
      settledMs: 5_000,
    }),
  });

  const latestRun = runsData?.items?.[0] ?? null;
  const isRunning = latestRun?.status === "running" || latestRun?.status === "queued";
  const isCompleted = latestRun?.status === "completed";
  const isIncomplete = latestRun?.status === "incomplete";
  const isFailed = latestRun?.status === "failed";

  // ── Mutations ──────────────────────────────────────────────────────────

  const startEval = useMutation({
    mutationFn: () => createEvaluationRun(projectId, {}),
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: evaluationKeys.all(projectId),
      });
    },
  });

  // Prompt-only restart from the failed banner. A mutation (not a bare
  // promise chain) so a rejected create keeps the failure banner up and
  // renders its error inline — the banner is only dismissed on success.
  const restartPromptOnly = useMutation({
    mutationFn: () =>
      createEvaluationRun(projectId, { structured_generation_mode: "prompt_only" }),
    onSuccess: () => {
      setDismissedBanners((p) => new Set(p).add(runBannerKey("failed")));
      queryClient.invalidateQueries({
        queryKey: evaluationKeys.all(projectId),
      });
    },
  });

  const cancelEval = useMutation({
    mutationFn: () => cancelEvaluationRun(projectId, latestRun!.run_id),
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: evaluationKeys.all(projectId),
      });
    },
  });

  const dismiss = useMutation({
    mutationFn: (triggerType: string) => dismissTrigger(projectId, triggerType),
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: evaluationKeys.triggerStatus(projectId),
      });
    },
  });

  const toggleAuto = useMutation({
    mutationFn: (enabled: boolean) =>
      updateProject(projectId, { auto_evaluate_enabled: enabled }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: projectKeys.detail(projectId) });
      queryClient.invalidateQueries({
        queryKey: evaluationKeys.triggerStatus(projectId),
      });
    },
  });

  // ── Helpers ────────────────────────────────────────────────────────────

  const autoEnabled = triggers?.auto_evaluate_enabled ?? false;

  function handleDismiss(triggerType: string) {
    setDismissedBanners((prev) => new Set(prev).add(triggerType));
    dismiss.mutate(triggerType);
  }

  // Run-lifecycle banners (complete/incomplete/failed/coverage/
  // settings-changed) are dismissed per run: the dismissal key embeds the
  // run_id, so a NEW run's banners always start visible. Trigger banners
  // keep plain keys — their dismissal persists server-side via
  // dismissTrigger, the local entry only hides them until refetch.
  function runBannerKey(banner: string) {
    return `${latestRun?.run_id ?? "none"}:${banner}`;
  }

  if (poolCount <= 0) return null;

  // Derive running progress for "Evaluating N of M..."
  const progress = latestRun?.progress as
    | { processed: number; total: number }
    | null
    | undefined;

  // Detect settings-changed-since-run-start
  const settingsChangedDuringRun =
    isRunning && triggers?.configuration_change?.is_active === true;

  // Mirrors the config-change nudge's render predicate below, so the
  // completion banner it suppresses reappears the moment the nudge is
  // dismissed.
  const configChangeNudgeVisible =
    !autoEnabled &&
    !isRunning &&
    triggers?.configuration_change.is_active === true &&
    !dismissedBanners.has("configuration_change");

  // ── Render ─────────────────────────────────────────────────────────────

  // Every banner CTA below renders kind="secondary" — the hero card's
  // Save is the labeling screen's sole green primary.
  return (
    // mt-2 separates this glass-info strip from the equally-glass-info top
    // bar above so the two horizontal strips read as distinct cards instead
    // of merging into one stacked column. Inner banner stack uses gap-2 so
    // the same breathing room sits between strip-bar and any child banner.
    <div className="mt-2 flex flex-col gap-2" data-testid="evaluation-strip">
      {/* ── Main strip bar ──────────────────────────────────────────── */}
      <div
        className="glass-info flex items-center justify-between px-4 py-2"
        data-testid="eval-strip-bar"
      >
        <div className="flex items-center gap-3">
          <Text kind="label/bold/sm">
            Test Pool: {poolCount}
            {poolTarget != null && poolTarget > poolCount && (
              <span
                style={{ color: "rgba(255,255,255,0.62)", fontWeight: 400 }}
                title={
                  "The Test Pool grows toward " +
                  "floor(verified × test_pool_fraction) as you verify more " +
                  "labels, which re-bases evaluation between runs. Lower " +
                  "test_pool_fraction in project settings to pin a fixed " +
                  "holdout."
                }
                data-testid="pool-growth-note"
              >
                {" "}
                → grows to {poolTarget} as you verify
              </span>
            )}
          </Text>
          {latestRun && (isCompleted || isIncomplete) && (
            <Button
              kind="secondary"
              size="small"
              onClick={() => onShowResults(latestRun)}
              data-testid="results-btn"
            >
              Results
            </Button>
          )}
        </div>

        <div className="flex items-center gap-3">
          {/* Auto-Evaluate — segmented ON|OFF pill, matches the Thinking
              control on the top bar so the two-state pills on this screen
              share one family. */}
          <div className="flex items-center gap-2" data-testid="auto-evaluate-toggle">
            <Text kind="label/regular/sm" style={{ color: "var(--text-muted)" }}>
              Auto:
            </Text>
            <SegmentedControl
              testId="auto-evaluate-segment"
              options={[
                { key: "off", label: "OFF" },
                { key: "on", label: "ON" },
              ]}
              value={autoEnabled ? "on" : "off"}
              onChange={(k) => toggleAuto.mutate(k === "on")}
            />
          </div>

          <Button
            kind="secondary"
            onClick={() => startEval.mutate()}
            disabled={isRunning || startEval.isPending}
            data-testid="evaluate-btn"
          >
            Evaluate
          </Button>
        </div>
      </div>

      {/* ── Evaluate request failure ───────────────────────────────── */}
      {/* A rejected POST (409 busy gate, 400 config, 5xx) previously gave
          no feedback at all — the button just looked dead. Dismiss resets
          the mutation, so the banner also clears on the next attempt. */}
      {startEval.isError && (
        <Banner
          testId="eval-start-error"
          tone="error"
          icon={<AlertOctagon size={14} />}
          actions={
            <DismissBtn testId="eval-start-error" onClick={() => startEval.reset()} />
          }
        >
          <Text kind="body/regular/sm">
            Could not start evaluation:{" "}
            {parseApiErrorDetail(startEval.error) ?? startEval.error.message}
          </Text>
        </Banner>
      )}

      {/* ── Running progress ───────────────────────────────────────── */}
      {isRunning && latestRun && (
        <>
          <Banner
            testId="eval-running"
            icon={<Spinner size="small" aria-label="Evaluating" />}
            actions={
              // When the settings-changed note is visible it carries its
              // own Cancel affordance; suppressing this one keeps a single
              // Cancel on screen.
              settingsChangedDuringRun &&
              !dismissedBanners.has(runBannerKey("settings-changed")) ? null : (
                <Button
                  kind="tertiary"
                  onClick={() => cancelEval.mutate()}
                  data-testid="cancel-eval-btn"
                >
                  Cancel Evaluation
                </Button>
              )
            }
          >
            <Text kind="body/regular/sm">
              {latestRun.status === "running" && progress
                ? `Evaluating ${progress.processed} of ${progress.total}...`
                : latestRun.status === "running"
                  ? "Evaluating..."
                  : "Evaluating (queued)"}
            </Text>
          </Banner>
          {/* Settings-changed note */}
          {settingsChangedDuringRun &&
            !dismissedBanners.has(runBannerKey("settings-changed")) && (
              <Banner
                testId="eval-settings-changed"
                icon={<Info size={14} />}
                actions={
                  <div className="flex items-center gap-2">
                    <Button
                      kind="tertiary"
                      onClick={() => cancelEval.mutate()}
                      data-testid="cancel-eval-btn"
                    >
                      Cancel Evaluation
                    </Button>
                    <DismissBtn
                      testId="eval-settings-changed"
                      onClick={() =>
                        setDismissedBanners((p) =>
                          new Set(p).add(runBannerKey("settings-changed")),
                        )
                      }
                    />
                  </div>
                }
              >
                <Text kind="body/regular/sm">
                  Running with settings from {formatTimestamp(latestRun.created_at)}.
                  Current settings differ.
                </Text>
              </Banner>
            )}
        </>
      )}

      {/* ── Completion notification ────────────────────────────────── */}
      {/* Suppressed while the configuration-change nudge is on screen —
          once settings changed, the accuracy headline describes a stale
          configuration, and the nudge below carries the actionable
          message instead of stacking identical banners. Dismissing the
          nudge restores this banner and its View Results affordance. */}
      {isCompleted &&
        !configChangeNudgeVisible &&
        (() => {
          const completedMetrics = safeMetrics(latestRun?.metrics);
          const overall = completedMetrics?.overall;
          if (
            !latestRun ||
            !overall ||
            overall.exact_match_rate == null ||
            dismissedBanners.has(runBannerKey("complete"))
          ) {
            return null;
          }
          return (
            <Banner
              testId="eval-complete"
              tone="success"
              actions={
                <div className="flex items-center gap-2">
                  <Button
                    kind="tertiary"
                    onClick={() => onShowResults(latestRun)}
                    data-testid="view-results-btn"
                  >
                    View Results
                  </Button>
                  <DismissBtn
                    testId="eval-complete"
                    onClick={() =>
                      setDismissedBanners((p) =>
                        new Set(p).add(runBannerKey("complete")),
                      )
                    }
                  />
                </div>
              }
            >
              <Text kind="body/regular/sm">
                Evaluation complete: {formatPct(overall.exact_match_rate)} accuracy
                {latestRun.previous_overall_exact_match != null &&
                completedMetrics?.returning?.exact_match_rate != null ? (
                  // The badge claims "on same images", so it must compare the
                  // Returning subset — not the overall rate — against the
                  // previous run (which scored exactly those images). Runs
                  // without a Returning split fall back to the image count.
                  <DeltaBadge
                    current={completedMetrics.returning.exact_match_rate}
                    previous={latestRun.previous_overall_exact_match}
                  />
                ) : (
                  <> ({overall.example_count} images)</>
                )}
                .
              </Text>
            </Banner>
          );
        })()}

      {/* ── Incomplete ─────────────────────────────────────────────── */}
      {isIncomplete &&
        latestRun &&
        !dismissedBanners.has(runBannerKey("incomplete")) &&
        (() => {
          // Renders "Evaluation incomplete: 2 examples
          // failed. Results are diagnostic only." Derive the failed
          // count from progress.total - successful-metric count, mirroring
          // ResultsPanel.tsx. Fall back to "some" when the
          // derivation isn't possible (race conditions in incomplete
          // states where progress or metrics haven't propagated yet)
          // OR when the derived count is 0 — an `incomplete` status
          // with zero failures is logically inconsistent (the run would
          // have terminated as `completed`); rather than render the
          // nonsensical "0 examples failed" copy, fall back to "some".
          const total = (
            latestRun.progress as { processed: number; total: number } | null
          )?.total;
          const succeeded = safeMetrics(latestRun.metrics)?.overall.example_count;
          const failed = total != null && succeeded != null ? total - succeeded : null;
          const message =
            failed != null && failed > 0
              ? `Evaluation incomplete: ${failed} ${failed === 1 ? "example" : "examples"} failed. Results are diagnostic only.`
              : "Evaluation incomplete: some examples failed. Results are diagnostic only.";
          return (
            // Amber warning tone — matches ResultsPanel's treatment of the
            // same diagnostic-only state.
            <Banner
              testId="eval-incomplete"
              tone="warning"
              actions={
                <div className="flex items-center gap-2">
                  <Button kind="tertiary" onClick={() => onShowResults(latestRun)}>
                    View Results
                  </Button>
                  <DismissBtn
                    testId="eval-incomplete"
                    onClick={() =>
                      setDismissedBanners((p) =>
                        new Set(p).add(runBannerKey("incomplete")),
                      )
                    }
                  />
                </div>
              }
            >
              <Text kind="body/regular/sm">{message}</Text>
            </Banner>
          );
        })()}

      {/* ── Failed ─────────────────────────────────────────────────── */}
      {/* The failed state renders with the `x` (error) icon, distinct
          from the incomplete state's amber warning. We use tone="error" + AlertOctagon
          so the banner reads as a hard failure rather than another dismissable
          nudge. Matches ProposalFailure.tsx's toast-error treatment. */}
      {isFailed && latestRun && !dismissedBanners.has(runBannerKey("failed")) && (
        <Banner
          testId="eval-failed"
          tone="error"
          icon={<AlertOctagon size={14} />}
          actions={
            <div className="flex items-center gap-2">
              {latestRun.status_reason === "structured_generation_rejected" && (
                // The banner auto-dismisses on SUCCESS only (mutation
                // onSuccess) — the new run picks up via the running-state
                // UI, no leftover [X] click needed. On failure it stays up
                // with the error line rendered below.
                <Button
                  kind="secondary"
                  onClick={() => restartPromptOnly.mutate()}
                  disabled={restartPromptOnly.isPending}
                  data-testid="restart-prompt-only-btn"
                >
                  Restart with prompt-only
                </Button>
              )}
              <DismissBtn
                testId="eval-failed"
                onClick={() =>
                  setDismissedBanners((p) => new Set(p).add(runBannerKey("failed")))
                }
              />
            </div>
          }
        >
          <div className="flex flex-col gap-1">
            <Text kind="body/regular/sm">
              Evaluation failed
              {latestRun.status_reason === "structured_generation_rejected"
                ? ": structured generation rejected."
                : "."}
            </Text>
            {latestRun.status_reason === "structured_generation_rejected" && (
              <Text kind="body/regular/sm" style={{ color: "var(--text-secondary)" }}>
                The model rejected json_schema output for this run. Prompt-only asks for
                JSON in the prompt instead of enforcing a schema.
              </Text>
            )}
            {restartPromptOnly.isError && (
              <Text kind="body/regular/sm" data-testid="restart-prompt-only-error">
                Could not restart the evaluation:{" "}
                {parseApiErrorDetail(restartPromptOnly.error) ??
                  restartPromptOnly.error.message}
                . Try again.
              </Text>
            )}
          </div>
        </Banner>
      )}

      {/* ── Trigger recommendation banners ──────────────────────────── */}
      {/* Trigger banners render at `body/regular/sm` to match the schema
          refinement reminder (InlineNotices.NoticeBanner) so the SME reads
          them as peers rather than a visually weaker class of nudge. */}
      {!autoEnabled && triggers && (
        <>
          {triggers.first_pool_threshold.is_active &&
            !suppressFirstPoolTrigger &&
            !dismissedBanners.has("first_pool_threshold") && (
              <Banner
                testId="trigger-first-pool"
                icon={<Info size={14} />}
                actions={
                  <>
                    {/* Banner [Evaluate] is the in-context CTA; the strip
                       [Evaluate] above remains the persistent manual
                       trigger. The banner has no further state to
                       communicate once Evaluate is clicked — the run is
                       in flight, the strip-bar above takes over with the
                       running progress UI. Auto-dismiss the banner so the
                       SME isn't left clicking [X] on a dead nudge. */}
                    <Button
                      kind="secondary"
                      onClick={() => {
                        startEval.mutate();
                        handleDismiss("first_pool_threshold");
                      }}
                      disabled={isRunning || startEval.isPending}
                      data-testid="trigger-first-pool-evaluate"
                    >
                      Evaluate
                    </Button>
                    <DismissBtn
                      testId="trigger-first-pool"
                      onClick={() => handleDismiss("first_pool_threshold")}
                    />
                  </>
                }
              >
                <Text kind="body/regular/sm">
                  {triggers.first_pool_threshold.message}
                </Text>
              </Banner>
            )}

          {configChangeNudgeVisible && (
            <Banner
              testId="trigger-config-change"
              icon={<Info size={14} />}
              actions={
                <>
                  {/* Auto-dismiss on click; the strip-bar's running UI
                     takes over. */}
                  <Button
                    kind="secondary"
                    onClick={() => {
                      startEval.mutate();
                      handleDismiss("configuration_change");
                    }}
                    disabled={isRunning || startEval.isPending}
                    data-testid="trigger-config-change-evaluate"
                  >
                    Evaluate
                  </Button>
                  <DismissBtn
                    testId="trigger-config-change"
                    onClick={() => handleDismiss("configuration_change")}
                  />
                </>
              }
            >
              <Text kind="body/regular/sm">
                Settings changed since last evaluation. Run an evaluation to check
                accuracy with current settings.
              </Text>
            </Banner>
          )}

          {triggers.icl_growth.is_active && !dismissedBanners.has("icl_growth") && (
            <Banner
              testId="trigger-icl-growth"
              icon={<Info size={14} />}
              actions={
                <>
                  <Button
                    kind="secondary"
                    onClick={() => {
                      startEval.mutate();
                      handleDismiss("icl_growth");
                    }}
                    disabled={isRunning || startEval.isPending}
                    data-testid="trigger-icl-growth-evaluate"
                  >
                    Evaluate
                  </Button>
                  <DismissBtn
                    testId="trigger-icl-growth"
                    onClick={() => handleDismiss("icl_growth")}
                  />
                </>
              }
            >
              <Text kind="body/regular/sm">{triggers.icl_growth.message}</Text>
            </Banner>
          )}
        </>
      )}

      {/* ── Coverage warnings ──────────────────────────────────────── */}
      {isCompleted &&
        latestRun?.coverage_gaps &&
        latestRun.coverage_gaps.length > 0 &&
        !dismissedBanners.has(runBannerKey("coverage")) && (
          <Banner
            testId="coverage-warning"
            tone="warning"
            border="edge"
            actions={
              <DismissBtn
                testId="coverage-warning"
                onClick={() =>
                  setDismissedBanners((p) => new Set(p).add(runBannerKey("coverage")))
                }
              />
            }
          >
            <Text kind="body/regular/sm">
              {/* Per-field gaps separated by
                 middle-dot, values double-quoted (`primary_damage = "leak",
                 "tear"  ·  severity = 0, 4`). Single-quote / semicolon form
                 reads as a run-on. */}
              Test pool has no examples with:{" "}
              {latestRun.coverage_gaps
                .map(
                  (g) =>
                    `${g.field_name} = ${g.missing_values.map((v) => `"${v}"`).join(", ")}`,
                )
                .join("  ·  ")}
              . Evaluation results don&apos;t cover these values.
            </Text>
          </Banner>
        )}
    </div>
  );
}

// ── Shared small components ────────────────────────────────────────────────

/**
 * Banner — text on left, action buttons on right.
 *
 * Adapter over the shared <InfoBanner/>: contributes the per-banner
 * testid and the pre-built icon element (spinner or sized lucide icon);
 * all banner chrome — glass-info vs toast-error surface, two-column
 * layout, py-3 padding matching InlineNotices' NoticeBanner so trigger
 * recommendations read at the same weight as the schema refinement
 * reminder — lives in InfoBanner. The strip bar itself (the row above)
 * stays at py-2 — it's the denser chrome row and reads that way on purpose.
 */
function Banner({
  children,
  testId,
  icon,
  actions,
  tone = "info",
  border,
}: {
  children: React.ReactNode;
  testId: string;
  /** Pre-built icon element; omit to use InfoBanner's tone default. */
  icon?: React.ReactElement;
  actions?: React.ReactNode;
  /**
   * Visual tone. `info` is the default neutral glass surface used for
   * recommendations. `success` (green) marks a completed evaluation,
   * `warning` (amber) marks diagnostic-only results and coverage gaps —
   * agreeing with ResultsPanel's amber for the same states. `error`
   * switches to the red toast surface (shared with ProposalFailure.tsx)
   * so hard failures read as distinct from info.
   */
  tone?: "info" | "warning" | "success" | "error";
  border?: "full" | "edge";
}) {
  return (
    <InfoBanner
      tone={tone}
      align="center"
      icon={icon}
      actions={actions ?? null}
      border={border}
      data-testid={testId}
    >
      {children}
    </InfoBanner>
  );
}

function DismissBtn({ testId, onClick }: { testId: string; onClick: () => void }) {
  return (
    <Button
      kind="tertiary"
      size="tiny"
      onClick={onClick}
      aria-label="Dismiss"
      data-testid={`${testId}-dismiss`}
    >
      <X size={14} style={{ color: "var(--text-muted)" }} />
    </Button>
  );
}

function DeltaBadge({ current, previous }: { current: number; previous: number }) {
  const delta = current - previous;
  const color = delta >= 0 ? "var(--accent-green)" : "var(--error-red-text)";
  // Subsequent-run banner: "(+3 pts vs previous on same images)" —
  // the delta compares the Returning subset against the previous run.
  return (
    <span style={{ color, marginLeft: 4 }} data-testid="delta-badge">
      ({formatDeltaPoints(delta)} vs previous on same images)
    </span>
  );
}
