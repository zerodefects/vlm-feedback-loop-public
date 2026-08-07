// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Per-variant card on the Compare & Benchmark screen.
 *
 * Top section is a selection checkbox + identity row (base, quantization,
 * GPU, profile). Below are the quality column (TAO-rescored, no ICL —
 * Students deploy bare) and the serving column (latency × concurrency
 * matrix). The action button at the bottom flips between:
 *
 *   - ``[Benchmark]``         — when current AIPerf serving validation is
 *                               absent and NIM preflight has not failed.
 *   - ``[Deploy for serving validation]`` — when
 *     ``nim_preflight_status="failed"``.
 *                               Expands a ``student_nim_deploy`` Action Request
 *                               so infrastructure can stand up an evaluation
 *                               NIM container externally.
 *   - ``[Request Production Deployment]`` — when both ``quality_status``
 *                               and ``serving_status`` are ``"validated"``
 *                               and the serving run satisfies the current
 *                               AIPerf workload contract.
 *                               Expands a ``deployment_handoff`` Action Request
 *                               via the gated Student-scoped endpoint. The
 *                               "Production" qualifier disambiguates this from
 *                               the temporary-eval
 *                               ``[Deploy for serving validation]``
 *                               affordance above.
 *
 * The card is intentionally dumb about the queue state — the parent
 * passes down a ``stage`` value during a benchmark so the card swaps in
 * a ``BenchmarkStageStrip``. All data comes in via props; mutations
 * raise callbacks back to the parent.
 */

import { useState } from "react";
import { Badge, Button, Spinner, Text } from "@kui/react";

import { ActionRequestPanel } from "@/components/ActionRequestPanel";
import { InfoBanner } from "@/components/common/InfoBanner";
import { deploymentBundleUrl, requestDeploymentHandoff } from "@/api/students";
import { formatDeltaPoints, formatMetricPct } from "@/lib/format-percent";
import { quantizationDisplayName } from "@/lib/model-display";
import type {
  EvaluationMetrics,
  EvaluationRunResponse,
  MetricsBucket,
  NimBenchmarkStage,
} from "@/types/evaluation";
import type { SchemaFieldResponse } from "@/types/guidance";
import type { StudentModel } from "@/types/training";

import { BenchmarkStageStrip } from "./BenchmarkStageStrip";
import type { MetricSelection } from "./CompareScopeBar";
import { PerFieldMetricsBlock } from "./PerFieldMetricsBlock";
import { ServingMatrix } from "./ServingMatrix";

export interface StudentVariantCardProps {
  projectId: string;
  student: StudentModel;
  /** Friendly base model name for the card title (e.g.
      ``Cosmos Reason2 8B``). The parent resolves this from the project's
      model catalog. */
  baseModelLabel: string;

  /** TAO-rescored quality run referenced by
      ``student.quality_evaluation_run_id``. ``null`` while loading. */
  qualityRun: EvaluationRunResponse | null | undefined;

  /** Serving run referenced by ``student.serving_evaluation_run_id`` —
      carries the ``metrics.benchmarks`` latency table written by the NIM
      benchmark lifecycle. ``null`` if not yet benchmarked. */
  servingRun: EvaluationRunResponse | null;

  /** Historical Teacher-baseline Exact Match used for the comparison delta. */
  teacherOverallExactMatch: number | null;

  coreFields: SchemaFieldResponse[];
  metricSelection: MetricSelection;
  perFieldMatchThreshold: number;
  minPerValueF1Threshold: number;
  concurrencies: number[];

  selected: boolean;
  onToggleSelected: () => void;

  /** Stage of an active local NIM benchmark for this variant. ``null`` when
      no benchmark is in flight. */
  benchmarkStage: NimBenchmarkStage | null;
  benchmarkElapsedMs: number;
  /** Backend-served NIM startup budget for the health_poll stage label. */
  benchmarkStartupBudgetS?: number;
  benchmarkConcurrency?: number | null;
  benchmarkEvalProgress?: { processed: number; total: number } | null;

  /** Disabled while the queue is busy with another variant. */
  busy: boolean;
  /** Error message from the most recent ``:deploy_nim`` attempt, surfaced
      inline if present. */
  benchmarkError?: string | null;
  onBenchmark: () => void;

  /** The host has exactly one GPU (one-NIM-per-GPU applies).
      When true, the [Benchmark] CTA renders
      a one-line displacement-preview message explaining that the
      benchmark will displace the resident Teacher (if local) and
      auto-restore it after. Multi-GPU hosts have no contention; the
      preview is suppressed. */
  singleGpuHost?: boolean;
}

export function StudentVariantCard({
  projectId,
  student,
  baseModelLabel,
  qualityRun,
  servingRun,
  teacherOverallExactMatch,
  coreFields,
  metricSelection,
  perFieldMatchThreshold,
  minPerValueF1Threshold,
  concurrencies,
  selected,
  onToggleSelected,
  benchmarkStage,
  benchmarkElapsedMs,
  benchmarkStartupBudgetS,
  benchmarkConcurrency,
  benchmarkEvalProgress,
  busy,
  benchmarkError,
  onBenchmark,
  singleGpuHost = false,
}: StudentVariantCardProps) {
  // Inline panel state — only one of the two AR panels is open at a time
  // per card (the deploy-fallback and deployment-handoff panels are
  // mutually exclusive UI states).
  const [fallbackOpen, setFallbackOpen] = useState(false);
  const [handoffOpen, setHandoffOpen] = useState(false);

  const qualityMetrics: EvaluationMetrics | null = qualityRun?.metrics ?? null;
  const overall: MetricsBucket | null = qualityMetrics?.overall ?? null;
  const exactMatch = overall?.exact_match_rate ?? null;

  const teacherDelta =
    exactMatch != null && teacherOverallExactMatch != null
      ? exactMatch - teacherOverallExactMatch
      : null;

  // Action button selection gates.
  const preflightFailed = student.nim_preflight_status === "failed";
  const servingBenchmarkCurrent = student.serving_benchmark_current === true;
  const dualValidated =
    student.quality_status === "validated" &&
    student.serving_status === "validated" &&
    servingBenchmarkCurrent;
  // ``serving_status=pending`` is durable backend authority that the
  // Student NIM lifecycle is already running. The local queue/stage state is
  // intentionally ephemeral, so a refreshed or reopened Compare screen must
  // still suppress the deploy CTA instead of inviting a duplicate request.
  const serverPending = student.serving_status === "pending";
  const benchmarking = benchmarkStage != null || serverPending;
  // `partial` quality is informational; production handoff still
  // requires `validated`. Card shows a yellow badge + helper line so
  // operators immediately see "the model serves but did not produce
  // parseable output on every example."
  const isPartialQuality = student.quality_status === "partial";
  const isPendingQuality = student.quality_status === "pending";

  const legacyServingValidation =
    student.serving_status === "validated" && !servingBenchmarkCurrent;
  const showBenchmark =
    !preflightFailed &&
    student.serving_status !== "pending" &&
    !servingBenchmarkCurrent;
  const showFallbackRequestDeploy = preflightFailed && !dualValidated;
  const showHandoffRequestDeploy = dualValidated;

  return (
    <div
      className="glass-card p-6 flex flex-col gap-3"
      style={{ overflowX: "hidden" }}
      data-testid={`student-variant-card-${student.student_model_id}`}
      data-quality-status={student.quality_status}
      data-serving-status={student.serving_status}
      data-nim-preflight-status={student.nim_preflight_status ?? ""}
    >
      {/* Identity row + selection checkbox */}
      <div className="flex items-center justify-between gap-3">
        <label
          className={`flex items-center gap-2 ${busy ? "cursor-not-allowed opacity-60" : "cursor-pointer"}`}
        >
          <input
            type="checkbox"
            className="glass-input"
            checked={selected}
            onChange={onToggleSelected}
            disabled={busy}
            data-testid={`variant-checkbox-${student.student_model_id}`}
            aria-label={`Select ${baseModelLabel} for benchmarking`}
          />
          <Text kind="label/bold/sm">
            Student: {baseModelLabel} ·{" "}
            {quantizationDisplayName(student.quantization_method)}
          </Text>
        </label>
        <Text kind="label/regular/xs" style={{ color: "var(--text-muted)" }}>
          {student.gpu_count != null && student.gpu_type
            ? `Benchmark hardware: ${student.gpu_count}× ${student.gpu_type}`
            : "Benchmark hardware: —"}
          {student.nim_model_profile_selected
            ? ` · profile ${student.nim_model_profile_selected}`
            : ""}
        </Text>
      </div>

      {/* Pending makes the cold-start action explicit; partial explains that
          serving worked without enough parseable results for production.
          Validated needs no extra badge and failed uses its dedicated card. */}
      {isPendingQuality ? (
        <div
          className="flex items-center gap-2"
          data-testid="quality-status-pending-row"
        >
          <Badge color="gray" kind="solid" data-testid="quality-status-badge-pending">
            Quality: Pending
          </Badge>
          <Text kind="label/regular/xs" style={{ color: "var(--text-muted)" }}>
            Run the first NIM validation to measure quality and serving performance.
          </Text>
        </div>
      ) : isPartialQuality ? (
        <div
          className="flex items-center gap-2"
          data-testid="quality-status-partial-row"
        >
          <Badge
            color="yellow"
            kind="solid"
            data-testid="quality-status-badge-partial"
            title="Partial validation — model serves but did not produce parseable output on every example. Production deployment requires re-evaluation."
          >
            Quality: Partial
          </Badge>
          <Text kind="label/regular/xs" style={{ color: "var(--text-muted)" }}>
            Re-run NIM evaluation to promote to validated before requesting production
            handoff.
          </Text>
        </div>
      ) : null}

      {/* Quality column — the TAO-rescored Exact Match + per-field
          block, with the vs-Teacher delta. Students are evaluated bare,
          so there is exactly one quality layout. */}
      <div className="flex flex-col gap-3" data-testid="quality-column">
        <ExactMatchRow
          value={exactMatch}
          delta={teacherDelta}
          deltaLabel="vs Teacher baseline"
          testid="exact-match"
        />
        {overall && (
          <PerFieldMetricsBlock
            overall={overall}
            coreFields={coreFields}
            metricSelection={metricSelection}
            perFieldMatchThreshold={perFieldMatchThreshold}
            minPerValueF1Threshold={minPerValueF1Threshold}
            data-testid-prefix={`variant-${student.student_model_id}`}
          />
        )}
      </div>

      {/* Serving column */}
      <div className="flex flex-col gap-1">
        <Text
          kind="label/regular/xs"
          className="section-eyebrow"
          style={{ color: "var(--text-muted)" }}
        >
          Serving
        </Text>
        {benchmarkStage ? (
          <BenchmarkStageStrip
            stage={benchmarkStage}
            elapsedMs={benchmarkElapsedMs}
            concurrency={benchmarkConcurrency}
            evaluationProgress={benchmarkEvalProgress ?? null}
            startupBudgetS={benchmarkStartupBudgetS}
          />
        ) : serverPending ? (
          <div className="flex items-center gap-2" data-testid="benchmark-reconciled">
            <Spinner size="small" aria-label="Serving validation in progress" />
            <Text kind="body/regular/sm" style={{ color: "var(--text-primary)" }}>
              Serving validation is running. Live stage details resume when events
              arrive.
            </Text>
          </div>
        ) : servingRun ? (
          <ServingMatrix run={servingRun} concurrencies={concurrencies} />
        ) : preflightFailed ? (
          <Text
            kind="body/regular/sm"
            style={{ color: "var(--warning-amber, #f59e0b)" }}
            data-testid="serving-nim-not-available"
          >
            NIM not available
          </Text>
        ) : (
          <Text
            kind="body/regular/sm"
            style={{ color: "var(--text-muted)" }}
            data-testid="serving-not-benchmarked"
          >
            Not benchmarked
          </Text>
        )}

        {benchmarkError && (
          <Text
            kind="body/regular/sm"
            style={{ color: "var(--error-red-text)" }}
            data-testid="benchmark-error"
          >
            {benchmarkError}
          </Text>
        )}

        {legacyServingValidation && !benchmarking && (
          <InfoBanner
            tone="warning"
            heading="AIPerf revalidation required"
            body="This historical serving result predates the required real Test Pool AIPerf workload. Re-run validation before requesting production deployment."
            data-testid="serving-aiperf-revalidation-required"
          />
        )}
      </div>

      {/*
       * One-NIM-per-GPU: on single-GPU hosts, Student NIM benchmarking takes
       * the GPU for ~6 minutes — the lifecycle stops the resident
       * Teacher (step 0) and best-effort auto-restores it after step
       * 9. Surface this honestly above the [Benchmark] CTA so the
       * SME isn't surprised mid-flow. The project-selected Teacher may be
       * hosted while a different local Teacher or embedding resident owns the
       * GPU, so this message names the actual resource class rather than the
       * selected Teacher config. Suppressed during an active benchmark and on
       * multi-GPU hosts.
       */}
      {!benchmarking && showBenchmark && singleGpuHost && (
        <Text
          kind="body/regular/xs"
          style={{ color: "var(--text-muted)" }}
          data-testid="benchmark-displacement-preview"
        >
          Benchmarking takes the GPU for ~6 minutes. If another local NIM is using this
          GPU, the Blueprint pauses it during the benchmark and best-effort restores it
          afterward.
        </Text>
      )}

      {/* Action buttons row */}
      {!benchmarking && (
        <div className="flex items-center justify-end gap-2">
          {showBenchmark && (
            <Button
              kind="secondary"
              onClick={onBenchmark}
              disabled={busy}
              data-testid={`benchmark-button-${student.student_model_id}`}
            >
              {isPendingQuality
                ? "Deploy and benchmark"
                : legacyServingValidation
                  ? "Revalidate with AIPerf"
                  : "Benchmark"}
            </Button>
          )}
          {showFallbackRequestDeploy && (
            <Button
              kind="secondary"
              onClick={() => setFallbackOpen((v) => !v)}
              data-testid={`request-deployment-fallback-${student.student_model_id}`}
              aria-pressed={fallbackOpen}
            >
              Deploy for serving validation
            </Button>
          )}
          {showHandoffRequestDeploy && (
            <Button
              kind="primary"
              className="nvidia-green-button"
              onClick={() => setHandoffOpen((v) => !v)}
              data-testid={`request-deployment-handoff-${student.student_model_id}`}
              aria-pressed={handoffOpen}
            >
              Request Production Deployment
            </Button>
          )}
        </div>
      )}

      {/* Fallback — student_nim_deploy Action Request via generic
          ``:generate`` endpoint. The generic dispatcher's
          ``student_nim_deploy`` generator is gate-less by design (it's a
          docker-run handoff for an already-failed preflight), so the
          generic path is correct here. */}
      {fallbackOpen && (
        <ActionRequestPanel
          projectId={projectId}
          requestType="student_nim_deploy"
          context={{ student_model_id: student.student_model_id }}
          onClose={() => setFallbackOpen(false)}
        />
      )}

      {/* Deployment Handoff — gated Student-scoped endpoint
          (``requestDeploymentHandoff``). Surfacing 409s inline is critical
          for the INFERENCE_CONTRACT_MISMATCH path. */}
      {handoffOpen && (
        <ActionRequestPanel
          projectId={projectId}
          requestType="deployment_handoff"
          mutationFn={() =>
            requestDeploymentHandoff(projectId, student.student_model_id)
          }
          deploymentBundleHref={deploymentBundleUrl(
            projectId,
            student.student_model_id,
          )}
          onClose={() => setHandoffOpen(false)}
        />
      )}
    </div>
  );
}

interface ExactMatchRowProps {
  value: number | null;
  delta: number | null;
  deltaLabel: string;
  testid: string;
}

function ExactMatchRow({ value, delta, deltaLabel, testid }: ExactMatchRowProps) {
  const formatted = formatMetricPct(value);
  return (
    <div className="flex items-baseline gap-2">
      <Text
        kind="label/regular/xs"
        className="section-eyebrow"
        style={{ color: "var(--text-muted)" }}
      >
        Exact Match
      </Text>
      <Text
        kind="label/bold/lg"
        style={{ color: "var(--accent-green)" }}
        data-testid={testid}
      >
        {formatted}
      </Text>
      {delta != null && (
        <Text
          kind="label/regular/xs"
          style={{
            color: delta >= 0 ? "var(--accent-green)" : "var(--error-red-text)",
          }}
          data-testid={`${testid}-delta`}
        >
          ({formatDeltaPoints(delta)} {deltaLabel})
        </Text>
      )}
    </div>
  );
}
