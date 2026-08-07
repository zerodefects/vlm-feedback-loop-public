// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Compare & Benchmark page.
 *
 * Surfaces every quality-validated Student variant for the project plus
 * the Teacher accuracy baseline. Drives:
 *
 *   - Per-card metric drill-down (Match rate · Per-value F1 · Per-value
 *     Precision · Per-value Recall)
 *   - Toggleable grouped bar chart over the selected metric
 *   - Sequential local NIM benchmark queue with a deploy-request fallback
 *   - ``deployment_handoff`` Action Request with the gated 409 path
 *
 * The Teacher baseline is the most recent completed Teacher evaluation
 * run (Students deploy and are evaluated bare — no Student+ICL arm);
 * each variant's serving latency comes from the run referenced by its
 * ``serving_evaluation_run_id``.
 *
 * Backend pieces consumed:
 *   - GET .../student_models
 *   - GET .../evaluation_runs                (Teacher baseline lookup)
 *   - GET .../evaluation_runs/{id}
 *   - GET .../guidance/{id}                 (schema fields)
 *   - GET .../model_configs                 (friendly labels)
 *   - POST .../student_models/{id}:deploy_nim
 *   - POST .../student_models/{id}:deployment_handoff
 *
 * SSE events consumed:
 *   - ``nim_benchmark_progress`` / ``nim_benchmark_completed`` /
 *     ``run_failed`` (with ``run_type="student_nim_deploy"``) — drive the
 *     benchmark queue + cache invalidation. While a benchmark is active
 *     the student list also polls, and a reconciliation effect advances
 *     the queue from REST state when a terminal event is missed.
 *   - ``evaluation_progress`` — the (N / M) counter for the active
 *     benchmark's Test-Pool evaluation stage.
 *
 * State scope is component-local (selection set, metric dropdown, chart
 * toggle, benchmark queue, per-variant stage). No Zustand —
 * the screen is single-route and queue resets on unmount are correct.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Button, Spinner, Text } from "@kui/react";
import {
  useMutation,
  useQueries,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query";

import { ApiError } from "@/api/client";
import { getEvaluationRun, listEvaluationRuns } from "@/api/evaluation";
import { fetchGuidance } from "@/api/guidance";
import { fetchModelConfigs } from "@/api/model-configs";
import { deployNim, listStudentModels } from "@/api/students";
import { listTrainingSuites } from "@/api/training";
import {
  evaluationKeys,
  guidanceKeys,
  modelConfigKeys,
  studentModelKeys,
  trainingKeys,
} from "@/api/query-keys";
import { PageContainer } from "@/components/common/PageContainer";
import { CompareScopeBar } from "@/components/compare/CompareScopeBar";
import type { MetricSelection } from "@/components/compare/CompareScopeBar";
import { localTeacherDisplayName, quantizationDisplayName } from "@/lib/model-display";
import { CHART_TEACHER_COLOR, variantColor } from "@/lib/chart-palette";
import { GroupedBarChart } from "@/components/compare/GroupedBarChart";
import type { ChartGroup, ChartSeries } from "@/components/compare/GroupedBarChart";
import { chartGroupKey } from "@/lib/chart-group";
import { QualityFailedVariantCard } from "@/components/compare/QualityFailedVariantCard";
import { StudentVariantCard } from "@/components/compare/StudentVariantCard";
import { TeacherBaselineCard } from "@/components/compare/TeacherBaselineCard";
import { TrainingRunGroup } from "@/components/compare/TrainingRunGroup";
import { useProjectSSE } from "@/hooks/useProjectSSE";
import { useEnvironmentSetupContext } from "@/pages/setup-context";
import { formatTimestamp } from "@/lib/format-date";
import { titleCasePreset } from "@/lib/formatPreset";
import type {
  EvaluationMetrics,
  EvaluationRunResponse,
  NimBenchmarkStage,
  PerValueMetric,
} from "@/types/evaluation";
import type { GuidanceResponse, SchemaFieldResponse } from "@/types/guidance";
import type { ModelConfigResponse } from "@/types/nim";
import type {
  StudentModel,
  StudentModelListResponse,
  TrainingSuite,
  TrainingSuiteListResponse,
} from "@/types/training";

interface BenchmarkState {
  stage: NimBenchmarkStage;
  startedAt: number;
  elapsedMs: number;
  concurrency: number | null;
  evalProgress: { processed: number; total: number } | null;
}

interface StudentRunGroupData {
  key: string;
  suite: TrainingSuite | null;
  students: StudentModel[];
  latest: boolean;
}

function normalizedInferenceContract(
  contract: Record<string, unknown> | null,
): string | null {
  if (!contract) return null;
  return JSON.stringify({
    output_field_mode: contract.output_field_mode ?? null,
    icl_field_mode: contract.icl_field_mode ?? null,
    icl_max_examples: contract.icl_max_examples ?? null,
  });
}

/** A delta is only meaningful when both runs measured the same task contract
 * and frozen Test Pool. Unknown provenance fails closed. */
function evaluationContextsMatch(
  studentRun: EvaluationRunResponse | null,
  teacherRun: EvaluationRunResponse | null,
): boolean {
  if (!studentRun || !teacherRun) return false;
  if (!studentRun.pool_version_id || !teacherRun.pool_version_id) return false;
  return (
    studentRun.guidance_id === teacherRun.guidance_id &&
    studentRun.pool_version_id === teacherRun.pool_version_id &&
    normalizedInferenceContract(studentRun.inference_contract) ===
      normalizedInferenceContract(teacherRun.inference_contract)
  );
}

// Poll cadence for the student list while a benchmark is active — the
// REST fallback that advances the queue when a terminal SSE event is
// missed (outage, backend restart).
const BENCHMARK_RECONCILE_POLL_MS = 5_000;

export function CompareBenchmarkPage() {
  const { projectId, project, environment } = useEnvironmentSetupContext();
  // One-NIM-per-GPU: on single-GPU hosts, Student NIM
  // benchmarking displaces the resident
  // Teacher (replace semantics) and best-effort auto-
  // restores it after. Surface this expectation on the
  // [Benchmark] CTA so the SME isn't surprised mid-flow. Multi-GPU
  // hosts have no contention; preview is suppressed.
  const singleGpuHost = environment.gpus.length === 1;
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { lastEvent } = useProjectSSE(projectId);

  // ── Page-level state ──────────────────────────────────────────────────────
  const [metricSelection, setMetricSelection] = useState<MetricSelection>("match_rate");
  const [chartOpen, setChartOpen] = useState(false);
  // The chart mounts below all variant cards, so toggling [Chart]
  // used to give zero visible feedback until the user scrolled. Scroll the
  // chart into view when the toggle opens (keyed on chartOpen alone —
  // chartData refreshes on SSE invalidations and must not re-scroll).
  const chartRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    if (!chartOpen) return;
    const id = requestAnimationFrame(() => {
      chartRef.current?.scrollIntoView?.({ behavior: "smooth", block: "start" });
    });
    return () => cancelAnimationFrame(id);
  }, [chartOpen]);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [benchmarkQueue, setBenchmarkQueue] = useState<string[]>([]);
  const [activeBenchmarkId, setActiveBenchmarkId] = useState<string | null>(null);
  const [stageByStudentId, setStageByStudentId] = useState<
    Record<string, BenchmarkState>
  >({});
  const [benchmarkErrorByStudentId, setBenchmarkErrorByStudentId] = useState<
    Record<string, string>
  >({});

  // ── Server data ──────────────────────────────────────────────────────────
  const {
    data: studentList,
    isLoading: studentsLoading,
    dataUpdatedAt: studentListUpdatedAt,
  } = useQuery<StudentModelListResponse>({
    queryKey: studentModelKeys.list(projectId),
    queryFn: () => listStudentModels(projectId, { limit: 200 }),
    // REST is authoritative; SSE is a hint channel. While a benchmark is
    // active, poll so a missed terminal event still reaches the
    // reconciliation effect below instead of stalling the queue.
    refetchInterval: (query) => {
      const current = query.state.data;
      const hasDurablePending = current?.items.some(
        (student) => student.serving_status === "pending",
      );
      return activeBenchmarkId != null || hasDurablePending
        ? BENCHMARK_RECONCILE_POLL_MS
        : false;
    },
  });

  const { data: completedRuns, isLoading: completedRunsLoading } = useQuery({
    queryKey: evaluationKeys.completedList(projectId),
    queryFn: () => listEvaluationRuns(projectId, { status: "completed", limit: 100 }),
  });

  const { data: trainingSuiteList, isLoading: trainingSuitesLoading } =
    useQuery<TrainingSuiteListResponse>({
      queryKey: trainingKeys.suites(projectId),
      queryFn: () => listTrainingSuites(projectId),
    });

  const { data: guidance, isLoading: guidanceLoading } = useQuery<GuidanceResponse>({
    queryKey: guidanceKeys.detail(projectId, project.active_guidance_id ?? ""),
    queryFn: () => fetchGuidance(projectId, project.active_guidance_id!),
    enabled: !!project.active_guidance_id,
  });

  const { data: modelConfigs, isLoading: modelConfigsLoading } = useQuery({
    queryKey: modelConfigKeys.list(projectId, undefined),
    queryFn: () => fetchModelConfigs(projectId),
  });

  // Historical Students must render against the Guidance schema they were
  // trained under, not whichever Guidance happens to be active today. The
  // current Guidance query above remains the Teacher/current-project source;
  // this deduplicated set fills the immutable Training Run headings and card
  // field definitions.
  const historicalGuidanceIds = useMemo(() => {
    const ids = new Set<string>();
    for (const suite of trainingSuiteList?.items ?? []) {
      if (suite.guidance_id !== project.active_guidance_id) ids.add(suite.guidance_id);
    }
    for (const student of studentList?.items ?? []) {
      if (student.guidance_id !== project.active_guidance_id)
        ids.add(student.guidance_id);
    }
    return Array.from(ids).sort();
  }, [project.active_guidance_id, studentList, trainingSuiteList]);

  const historicalGuidanceQueries = useQueries({
    queries: historicalGuidanceIds.map((guidanceId) => ({
      queryKey: guidanceKeys.detail(projectId, guidanceId),
      queryFn: () => fetchGuidance(projectId, guidanceId),
    })),
  });

  const guidanceById = useMemo(() => {
    const byId: Record<string, GuidanceResponse> = {};
    if (guidance) byId[guidance.guidance_id] = guidance;
    historicalGuidanceIds.forEach((guidanceId, index) => {
      const historical = historicalGuidanceQueries[index]?.data;
      if (historical) byId[guidanceId] = historical;
    });
    return byId;
  }, [guidance, historicalGuidanceIds, historicalGuidanceQueries]);

  // Keep every packaged Student that can run the NIM lifecycle visible.
  // ``pending`` is the cold-start path: its first NIM evaluation supplies
  // both quality and serving evidence. ``validated`` is the production-
  // handoff bar; ``partial`` is informational —
  // partial Students render with the yellow "Quality: Partial" badge so
  // the SME can compare them and decide whether to re-run NIM eval to
  // promote to validated. The [Request Production Deployment] gate on
  // each card stays at strict ``validated && validated``, so partial
  // Students see the disabled deployment button + a partial-quality helper line.
  // Don't also gate on ``serving_status="validated"`` — unbenchmarked
  // variants must render with "Not benchmarked" so the SME can
  // [Benchmark] them.
  const variants = useMemo<StudentModel[]>(() => {
    return (studentList?.items ?? []).filter(
      (s) =>
        s.quality_status === "pending" ||
        s.quality_status === "validated" ||
        s.quality_status === "partial",
    );
  }, [studentList]);

  // Quality-``failed`` Students stay visible as compact notices with
  // their failure reason and a [Re-score quality] affordance, instead of
  // silently vanishing from the page (a serving-validated Student used to
  // disappear with no explanation and no UI remediation path).
  const failedVariants = useMemo<StudentModel[]>(() => {
    return (studentList?.items ?? []).filter((s) => s.quality_status === "failed");
  }, [studentList]);

  // One project-wide comparison, visually partitioned by immutable Start
  // Training actions. Existing databases are backfilled with
  // training_suite_id; student_model_ids is a second authoritative link for
  // records created while a migration/recovery boundary was in flight.
  const runGroups = useMemo<StudentRunGroupData[]>(() => {
    const suites = trainingSuiteList?.items ?? [];
    const suiteByStudentId = new Map<string, string>();
    for (const suite of suites) {
      for (const studentId of suite.student_model_ids) {
        suiteByStudentId.set(studentId, suite.training_suite_id);
      }
    }

    const studentsBySuite = new Map<string, StudentModel[]>();
    const unassigned: StudentModel[] = [];
    for (const student of studentList?.items ?? []) {
      const suiteId =
        student.training_suite_id ?? suiteByStudentId.get(student.student_model_id);
      if (!suiteId) {
        unassigned.push(student);
        continue;
      }
      const groupStudents = studentsBySuite.get(suiteId) ?? [];
      groupStudents.push(student);
      studentsBySuite.set(suiteId, groupStudents);
    }

    const groups: StudentRunGroupData[] = [];
    for (const suite of suites) {
      const students = studentsBySuite.get(suite.training_suite_id);
      if (!students?.length) continue;
      groups.push({
        key: suite.training_suite_id,
        suite,
        students,
        latest: groups.length === 0,
      });
      studentsBySuite.delete(suite.training_suite_id);
    }

    // A Student may reference a suite beyond the first 100 history rows. Keep
    // it visible and actionable instead of dropping it from Models & Results.
    for (const [suiteId, students] of studentsBySuite) {
      groups.push({ key: suiteId, suite: null, students, latest: false });
    }
    if (unassigned.length > 0) {
      groups.push({
        key: "unassigned",
        suite: null,
        students: unassigned,
        latest: groups.length === 0,
      });
    }
    return groups;
  }, [studentList, trainingSuiteList]);

  // Fetch each referenced evaluation run exactly once. Several variants can
  // legitimately share a quality or serving run. Passing duplicate query
  // keys to one ``useQueries`` observer produces an explicit React Query
  // warning and can make observer result sharing ambiguous, so deduplicate
  // the union before constructing the queries.
  const variantEvaluationRunIds = useMemo(() => {
    const ids = new Set<string>();
    for (const variant of variants) {
      if (variant.quality_evaluation_run_id) {
        ids.add(variant.quality_evaluation_run_id);
      }
      if (variant.serving_evaluation_run_id) {
        ids.add(variant.serving_evaluation_run_id);
      }
    }
    return Array.from(ids);
  }, [variants]);

  const variantEvaluationRunQueries = useQueries({
    queries: variantEvaluationRunIds.map((runId) => ({
      queryKey: evaluationKeys.detail(projectId, runId),
      queryFn: () => getEvaluationRun(projectId, runId),
    })),
  });

  const comparisonDataLoading =
    studentsLoading ||
    completedRunsLoading ||
    trainingSuitesLoading ||
    modelConfigsLoading ||
    (!!project.active_guidance_id && guidanceLoading) ||
    historicalGuidanceQueries.some((query) => query.isLoading) ||
    variantEvaluationRunQueries.some((query) => query.isLoading);

  const variantEvaluationRunById = useMemo(() => {
    const runs: Record<string, EvaluationRunResponse | null> = {};
    variantEvaluationRunIds.forEach((runId, index) => {
      runs[runId] = variantEvaluationRunQueries[index]?.data ?? null;
    });
    return runs;
  }, [variantEvaluationRunIds, variantEvaluationRunQueries]);

  const qualityRunByStudentId = useMemo(() => {
    const m: Record<string, EvaluationMetrics | null> = {};
    const fullByStudentId: Record<
      string,
      Awaited<ReturnType<typeof getEvaluationRun>> | null
    > = {};
    for (const v of variants) {
      const result = v.quality_evaluation_run_id
        ? (variantEvaluationRunById[v.quality_evaluation_run_id] ?? null)
        : null;
      fullByStudentId[v.student_model_id] = result;
      m[v.student_model_id] = (result?.metrics ?? null) as EvaluationMetrics | null;
    }
    return { metricsByStudentId: m, runByStudentId: fullByStudentId };
  }, [variants, variantEvaluationRunById]);

  // Teacher baseline = the newest completed run made with the fixed
  // Teacher contract (all/core_only/null — schemas/inference_contract.py,
  // pinned by test_evaluation_service::test_teacher_contract_constant).
  // ``student_model_config_id == null`` is the primary discriminator: a
  // Student trained on an ``export_field_mode="all"`` dataset carries a
  // contract identical to the Teacher's, so contract-match alone could
  // pick a Student serving run. The list endpoint returns newest-first
  // with ``status`` filtered server-side, so ``.find()`` picks the
  // latest match. The 100-run window is an accepted edge: a project
  // needs >100 completed Student runs newer than its last Teacher run
  // to miss the baseline, and the card's empty state (run an
  // evaluation) is the right nudge in that state anyway.
  const teacherBaselineRun = useMemo<EvaluationRunResponse | null>(() => {
    const items = completedRuns?.items ?? [];
    return (
      items.find((run) => {
        if (run.student_model_config_id != null) return false;
        const c = run.inference_contract;
        return (
          !!c &&
          c.output_field_mode === "all" &&
          c.icl_field_mode === "core_only" &&
          (c.icl_max_examples ?? null) === null
        );
      }) ?? null
    );
  }, [completedRuns]);

  const teacherBaselineMetrics =
    (teacherBaselineRun?.metrics as EvaluationMetrics | null) ?? null;

  const teacherOverallExactMatch =
    teacherBaselineMetrics?.overall?.exact_match_rate ?? null;

  // Per-variant serving runs carry ``metrics.benchmarks`` (latency ×
  // concurrency) written by the NIM benchmark lifecycle. They come from the
  // shared, deduplicated run map above.
  const servingRunByStudentId = useMemo(() => {
    const m: Record<string, EvaluationRunResponse | null> = {};
    for (const v of variants) {
      m[v.student_model_id] = v.serving_evaluation_run_id
        ? (variantEvaluationRunById[v.serving_evaluation_run_id] ?? null)
        : null;
    }
    return m;
  }, [variants, variantEvaluationRunById]);

  // ── Model labels ──────────────────────────────────────────────────────────
  // Two label maps: ``labelByModelConfigId`` keeps the canonical
  // provider-prefixed name (e.g. ``nvidia/cosmos-reason2-8b``) used
  // on the Teacher card. The
  // ``friendlyLabelByModelConfigId`` map applies the shared product-model
  // display names so Student variant cards
  // and chart legend entries render as ``Cosmos Reason2 8B``
  // — the human-readable form the comparison surface requires.
  const labelByModelConfigId = useMemo(() => {
    const m: Record<string, string> = {};
    for (const mc of (modelConfigs?.items ?? []) as ModelConfigResponse[]) {
      m[mc.model_config_id] = mc.model_name;
    }
    return m;
  }, [modelConfigs]);

  const friendlyLabelByModelConfigId = useMemo(() => {
    const m: Record<string, string> = {};
    for (const mc of (modelConfigs?.items ?? []) as ModelConfigResponse[]) {
      m[mc.model_config_id] = localTeacherDisplayName(mc.model_name);
    }
    return m;
  }, [modelConfigs]);

  const teacherBaselineLabel = useMemo(() => {
    const modelConfigId = teacherBaselineRun?.model_config_id;
    if (!modelConfigId) return "Unknown Teacher model";
    return labelByModelConfigId[modelConfigId] ?? modelConfigId;
  }, [labelByModelConfigId, teacherBaselineRun]);

  const runLabelByStudentId = useMemo(() => {
    const labels: Record<string, string> = {};
    for (const group of runGroups) {
      const timestamp =
        formatTimestamp(
          group.suite?.started_at ??
            group.suite?.created_at ??
            group.students[0]?.created_at,
        ) ?? "Unknown date";
      for (const student of group.students)
        labels[student.student_model_id] = timestamp;
    }
    return labels;
  }, [runGroups]);

  // Chart series names must remain unique when the same base/precision is
  // trained more than once. Include the immutable run timestamp, then add a
  // candidate suffix only for the rare duplicate within one suite.
  const chartLabelByStudentId = useMemo(() => {
    const rawById: Record<string, string> = {};
    const totals = new Map<string, number>();
    for (const student of variants) {
      const model =
        friendlyLabelByModelConfigId[student.student_base_model_config_id] ??
        student.student_base_model_config_id;
      const raw = `${model} · ${quantizationDisplayName(student.quantization_method)} · ${runLabelByStudentId[student.student_model_id] ?? "Unknown run"}`;
      rawById[student.student_model_id] = raw;
      totals.set(raw, (totals.get(raw) ?? 0) + 1);
    }
    const seen = new Map<string, number>();
    const labels: Record<string, string> = {};
    for (const student of variants) {
      const raw = rawById[student.student_model_id];
      const ordinal = (seen.get(raw) ?? 0) + 1;
      seen.set(raw, ordinal);
      labels[student.student_model_id] =
        (totals.get(raw) ?? 0) > 1 ? `${raw} · Candidate ${ordinal}` : raw;
    }
    return labels;
  }, [friendlyLabelByModelConfigId, runLabelByStudentId, variants]);

  const teacherComparableByStudentId = useMemo(() => {
    const comparable: Record<string, boolean> = {};
    for (const student of variants) {
      comparable[student.student_model_id] = evaluationContextsMatch(
        qualityRunByStudentId.runByStudentId[student.student_model_id] ?? null,
        teacherBaselineRun,
      );
    }
    return comparable;
  }, [qualityRunByStudentId, teacherBaselineRun, variants]);

  const latestSuite = runGroups.find((group) => group.suite != null)?.suite ?? null;

  const warningByRunKey = useMemo(() => {
    const warnings: Record<string, string | null> = {};
    for (const group of runGroups) {
      const messages: string[] = [];
      if (!group.suite) {
        messages.push(
          "Training Run provenance is unavailable. The model remains fully actionable, but cross-run score differences are directional.",
        );
      } else if (
        latestSuite &&
        group.suite.training_suite_id !== latestSuite.training_suite_id
      ) {
        const differences: string[] = [];
        if (group.suite.guidance_id !== latestSuite.guidance_id) {
          differences.push("Guidance");
        }
        if (group.suite.export_field_mode !== latestSuite.export_field_mode) {
          differences.push("output contract");
        }
        const groupChecksum = group.suite.evaluation_dataset_checksum_sha256;
        const latestChecksum = latestSuite.evaluation_dataset_checksum_sha256;
        if (groupChecksum && latestChecksum) {
          if (groupChecksum !== latestChecksum) differences.push("evaluation set");
        } else if (
          group.suite.evaluation_dataset_export_id !==
          latestSuite.evaluation_dataset_export_id
        ) {
          differences.push("unverified evaluation set");
        }
        if (differences.length > 0) {
          messages.push(
            `Different ${differences.join(", ")} from the latest training run. Cross-run score differences are directional.`,
          );
        }
      }

      const hasTeacherMismatch = group.students.some((student) => {
        if (!student.quality_evaluation_run_id) return false;
        return !teacherComparableByStudentId[student.student_model_id];
      });
      if (hasTeacherMismatch && teacherBaselineRun) {
        messages.push(
          "The Teacher baseline uses a different Guidance, Test Pool, or output contract; Student-vs-Teacher deltas are hidden.",
        );
      }
      warnings[group.key] = messages.length > 0 ? messages.join(" ") : null;
    }
    return warnings;
  }, [latestSuite, runGroups, teacherBaselineRun, teacherComparableByStudentId]);

  const chartContextWarning = Object.values(warningByRunKey).some(Boolean)
    ? "This chart includes results from different or unverified evaluation contexts. Use it for directional review; rely on deltas only where the run cards confirm compatible provenance."
    : null;

  const modelSummaryByRunKey = useMemo(() => {
    const summaries: Record<string, string> = {};
    for (const group of runGroups) {
      const selectedIds =
        group.suite?.selected_student_base_model_config_ids ??
        group.students.map((student) => student.student_base_model_config_id);
      const names = Array.from(
        new Set(
          selectedIds.map(
            (modelConfigId) =>
              friendlyLabelByModelConfigId[modelConfigId] ?? modelConfigId,
          ),
        ),
      );
      summaries[group.key] = `Models: ${names.join(" · ") || "Unavailable"}`;
    }
    return summaries;
  }, [friendlyLabelByModelConfigId, runGroups]);

  // ── Selection helpers ─────────────────────────────────────────────────────
  const toggleSelected = (id: string) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  // ── Benchmark queue ──────────────────────────────────────────────────────
  const deployMutation = useMutation({
    mutationFn: (id: string) => deployNim(projectId, id, {}),
    onError: (error, id) => {
      const msg =
        error instanceof ApiError
          ? `${error.status}: ${error.body}`
          : (error as Error).message;
      setBenchmarkErrorByStudentId((prev) => ({ ...prev, [id]: msg }));
      // Pop this variant from the queue and continue.
      setActiveBenchmarkId(null);
      setBenchmarkQueue((q) => q.slice(1));
    },
  });

  // Dispatch timestamp for the REST reconciliation effect below: list
  // snapshots fetched before the deploy POST may still carry a prior
  // attempt's terminal serving_status and must not settle the new run.
  const benchmarkDispatchedAtRef = useRef(0);

  const startNextInQueue = (queue: string[]) => {
    if (queue.length === 0) {
      setActiveBenchmarkId(null);
      return;
    }
    const next = queue[0];
    benchmarkDispatchedAtRef.current = Date.now();
    setActiveBenchmarkId(next);
    // Optimistic stage so the strip lights up before the first SSE event.
    setStageByStudentId((prev) => ({
      ...prev,
      [next]: {
        stage: "preflight",
        startedAt: Date.now(),
        elapsedMs: 0,
        concurrency: null,
        evalProgress: null,
      },
    }));
    setBenchmarkErrorByStudentId((prev) => {
      const { [next]: _omit, ...rest } = prev;
      return rest;
    });
    deployMutation.mutate(next);
  };

  // Watch the queue length — when items appear and nothing is active,
  // start the head.
  useEffect(() => {
    if (activeBenchmarkId == null && benchmarkQueue.length > 0) {
      startNextInQueue(benchmarkQueue);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeBenchmarkId, benchmarkQueue]);

  const enqueueBenchmark = (ids: string[]) => {
    if (ids.length === 0) return;
    setBenchmarkQueue((q) => Array.from(new Set([...q, ...ids])));
  };

  // Terminal transition shared by the SSE handlers and the REST
  // reconciler below: drop the variant's stage entry, record a failure
  // message when given, pop the queue head, and clear the active id.
  // Cache invalidations stay at the call sites — they differ per event.
  const settleBenchmark = useCallback((sid: string, failureMsg?: string) => {
    setStageByStudentId((prev) => {
      const { [sid]: _omit, ...rest } = prev;
      return rest;
    });
    if (failureMsg !== undefined) {
      setBenchmarkErrorByStudentId((prev) => ({ ...prev, [sid]: failureMsg }));
    }
    setBenchmarkQueue((q) => (q[0] === sid ? q.slice(1) : q));
    setActiveBenchmarkId((curr) => (curr === sid ? null : curr));
  }, []);

  // ── SSE consumption ───────────────────────────────────────────────────────
  useEffect(() => {
    if (!lastEvent) return;
    const { type, data } = lastEvent;
    if (type === "nim_benchmark_progress") {
      const sid = data.student_model_id as string | undefined;
      if (!sid) return;
      setStageByStudentId((prev) => {
        const prior = prev[sid];
        const startedAt = prior?.startedAt ?? Date.now();
        return {
          ...prev,
          [sid]: {
            stage: data.stage as NimBenchmarkStage,
            startedAt,
            elapsedMs: typeof data.elapsed_ms === "number" ? data.elapsed_ms : 0,
            concurrency: typeof data.concurrency === "number" ? data.concurrency : null,
            evalProgress: prior?.evalProgress ?? null,
          },
        };
      });
      return;
    }
    if (type === "evaluation_progress") {
      // The benchmark's Test-Pool evaluation reports its (N / M) counter
      // via the standard evaluation_progress event — the eval executor
      // doesn't know it runs under a benchmark, and nim_benchmark_progress
      // never carries processed/total. The queue is sequential, so at most
      // one variant is in its evaluation stage; starting a competing
      // evaluation would supersede (cancel) the benchmark's run, so the
      // counter can only describe that variant.
      const processed = (data as { processed?: unknown }).processed;
      const total = (data as { total?: unknown }).total;
      if (typeof processed !== "number" || typeof total !== "number") return;
      setStageByStudentId((prev) => {
        const evaluating = Object.entries(prev).filter(
          ([, st]) => st.stage === "evaluation",
        );
        if (evaluating.length !== 1) return prev;
        const [sid, prior] = evaluating[0];
        return {
          ...prev,
          [sid]: { ...prior, evalProgress: { processed, total } },
        };
      });
      return;
    }
    if (type === "nim_benchmark_completed") {
      const sid = data.student_model_id as string | undefined;
      if (!sid) return;
      settleBenchmark(sid);
      // Refresh student + evaluation caches so the card flips to the
      // benchmarked layout. The evaluation invalidation must cover run
      // details: the lifecycle writes ``metrics.benchmarks`` onto the
      // serving run AFTER it terminalizes, so a detail cached during the
      // eval stage would lack the latency table.
      queryClient.invalidateQueries({
        queryKey: studentModelKeys.list(projectId),
      });
      queryClient.invalidateQueries({
        queryKey: evaluationKeys.all(projectId),
      });
      return;
    }
    if (
      type === "run_failed" &&
      (data as { run_type?: string }).run_type === "student_nim_deploy"
    ) {
      const sid = data.student_model_id as string | undefined;
      if (!sid) return;
      const stage = (data as { failure_stage?: string }).failure_stage;
      const errorRef = (data as { error_ref?: string }).error_ref;
      settleBenchmark(
        sid,
        `Benchmark failed at stage ${stage ?? "?"}${errorRef ? `: ${errorRef}` : ""}`,
      );
      queryClient.invalidateQueries({
        queryKey: studentModelKeys.list(projectId),
      });
      return;
    }
  }, [lastEvent, projectId, queryClient, settleBenchmark]);

  // ── REST reconciliation of the benchmark queue ───────────────────────────
  // A terminal SSE event dropped during an outage (or a backend restart
  // mid-benchmark, which terminalizes the Student as failed) would leave
  // the queue stuck on the active variant until unmount. When a
  // post-dispatch list snapshot shows the active variant terminal while
  // its local stage entry still exists, perform the SSE terminal
  // handler's transitions from REST state.
  useEffect(() => {
    const sid = activeBenchmarkId;
    if (sid == null || !stageByStudentId[sid]) return;
    if (studentListUpdatedAt <= benchmarkDispatchedAtRef.current) return;
    const active = (studentList?.items ?? []).find((s) => s.student_model_id === sid);
    if (!active) return;
    const failed =
      active.serving_status === "failed" || active.nim_preflight_status === "failed";
    if (!failed && active.serving_status !== "validated") return;
    if (failed) {
      const failureStage = (
        active.nim_preflight_details as { failure_stage?: string } | null
      )?.failure_stage;
      settleBenchmark(sid, `Benchmark failed at stage ${failureStage ?? "?"}`);
    } else {
      settleBenchmark(sid);
      // Validated without the SSE event — refresh evaluation caches so
      // the serving run's metrics.benchmarks land on the card.
      queryClient.invalidateQueries({
        queryKey: evaluationKeys.all(projectId),
      });
    }
  }, [
    studentList,
    studentListUpdatedAt,
    activeBenchmarkId,
    stageByStudentId,
    projectId,
    queryClient,
    settleBenchmark,
  ]);

  // ── Stale Teacher baseline note ───────────────────────────────────────────
  // The baseline run snapshots its Teacher and Guidance. When either live
  // project selection has moved on, surface a small caption while retaining
  // the historical run's exact model identity on every metric surface.
  const staleTeacherBaselineNote: string | null = useMemo(() => {
    if (!teacherBaselineRun) return null;
    const changed: string[] = [];
    if (
      teacherBaselineRun.model_config_id &&
      teacherBaselineRun.model_config_id !== project.teacher_model_config_id
    ) {
      changed.push("Teacher");
    }
    if (
      teacherBaselineRun.guidance_id &&
      project.active_guidance_id &&
      teacherBaselineRun.guidance_id !== project.active_guidance_id
    ) {
      changed.push("Guidance");
    }
    return changed.length > 0
      ? `Teacher baseline predates the current ${changed
          .map((field) => {
            if (field !== "Teacher") return field;
            const currentId = project.teacher_model_config_id;
            const currentLabel = currentId
              ? (friendlyLabelByModelConfigId[currentId] ?? currentId)
              : "not selected";
            return `Teacher (${currentLabel})`;
          })
          .join(" and ")}. Run a new evaluation to refresh the baseline.`
      : null;
  }, [
    teacherBaselineRun,
    project.active_guidance_id,
    project.teacher_model_config_id,
    friendlyLabelByModelConfigId,
  ]);

  // ── Chart data ───────────────────────────────────────────────────────────
  const chartData = useMemo(() => {
    if (!chartOpen) return null;

    // A historical Guidance can rename or replace Core fields. Build a union
    // so every retained Student keeps its measured fields visible; the run
    // warnings above explain when those columns are directional rather than
    // directly comparable.
    const coreFieldByName = new Map<string, SchemaFieldResponse>();
    const addCoreFields = (candidate: GuidanceResponse | undefined) => {
      for (const field of candidate?.schema_fields ?? []) {
        if (field.role === "core" && !coreFieldByName.has(field.field_name)) {
          coreFieldByName.set(field.field_name, field);
        }
      }
    };
    addCoreFields(guidance);
    for (const variant of variants) addCoreFields(guidanceById[variant.guidance_id]);
    const coreFields = Array.from(coreFieldByName.values());
    if (coreFields.length === 0) return null;

    const teacherSeries: ChartSeries = {
      label: `Teacher · ${teacherBaselineLabel}`,
      color: CHART_TEACHER_COLOR,
      values: {},
    };

    const variantSeries: ChartSeries[] = variants.map((v, idx) => ({
      label: chartLabelByStudentId[v.student_model_id],
      color: variantColor(idx),
      values: {},
    }));

    const groups: ChartGroup[] = [];
    let title = "";

    if (metricSelection === "match_rate") {
      title = "Per-field Match Rate";
      const teacherOverall = teacherBaselineMetrics?.overall ?? null;
      for (const f of coreFields) {
        const group: ChartGroup = { label: f.field_name };
        groups.push(group);
        const key = chartGroupKey(group);
        teacherSeries.values[key] =
          teacherOverall?.per_field_match_rates?.[f.field_name] ?? null;
        for (let i = 0; i < variants.length; i++) {
          const overall =
            qualityRunByStudentId.metricsByStudentId[variants[i].student_model_id]
              ?.overall ?? null;
          variantSeries[i].values[key] =
            overall?.per_field_match_rates?.[f.field_name] ?? null;
        }
      }
    } else {
      const valueKey: keyof PerValueMetric =
        metricSelection === "per_value_f1"
          ? "f1"
          : metricSelection === "per_value_precision"
            ? "precision"
            : "recall";
      title = `Per-value ${
        valueKey === "f1" ? "F1" : valueKey === "precision" ? "Precision" : "Recall"
      }`;
      const categoricalFields = coreFields.filter((f) =>
        ["enum", "enum_set", "boolean"].includes(f.type),
      );
      for (const f of categoricalFields) {
        const labels =
          f.type === "boolean" ? ["true", "false"] : (f.allowed_values ?? []);
        for (const v of labels) {
          // Series values are keyed by the (field, value) pair — bare
          // value labels repeat across fields (two boolean Core fields
          // both emit "true"/"false") and would overwrite each other.
          const group: ChartGroup = { cluster: f.field_name, label: v };
          groups.push(group);
          const key = chartGroupKey(group);
          const teacherOverall = teacherBaselineMetrics?.overall ?? null;
          const tpv = teacherOverall?.per_value_metrics?.[f.field_name]?.[v] ?? null;
          teacherSeries.values[key] = tpv ? tpv[valueKey] : null;
          for (let i = 0; i < variants.length; i++) {
            const sv = variants[i];
            const overall =
              qualityRunByStudentId.metricsByStudentId[sv.student_model_id]?.overall ??
              null;
            const m =
              (
                overall?.per_value_metrics as
                  | Record<string, Record<string, PerValueMetric>>
                  | undefined
              )?.[f.field_name]?.[v] ?? null;
            variantSeries[i].values[key] = m ? m[valueKey] : null;
          }
        }
      }
    }

    return {
      title,
      groups,
      series: [teacherSeries, ...variantSeries],
    };
  }, [
    chartOpen,
    guidance,
    guidanceById,
    metricSelection,
    variants,
    teacherBaselineMetrics,
    qualityRunByStudentId,
    chartLabelByStudentId,
    teacherBaselineLabel,
  ]);

  // ── Derived UX state ──────────────────────────────────────────────────────
  const coreFields: SchemaFieldResponse[] = useMemo(() => {
    if (!guidance) return [];
    return guidance.schema_fields.filter((f) => f.role === "core");
  }, [guidance]);

  const unbenchmarkedIds = useMemo(
    () =>
      variants
        .filter(
          (v) =>
            (v.serving_status !== "validated" ||
              v.serving_benchmark_current !== true) &&
            v.serving_status !== "pending" &&
            v.nim_preflight_status !== "failed",
        )
        .map((v) => v.student_model_id),
    [variants],
  );

  // ``busy`` disables the scope-bar action buttons and the per-card
  // checkboxes while an NIM benchmark is in flight — a sequential
  // workload where re-firing the trigger would double-fire the queue.
  const durablePendingBenchmark = variants.some(
    (student) => student.serving_status === "pending",
  );
  const busy = activeBenchmarkId != null || durablePendingBenchmark;

  // When no production handoff is available, promote the first failed-quality
  // Student that still needs serving validation as the primary recovery action.
  const primaryFailedRecoveryId = useMemo<string | null>(() => {
    const hasProductionHandoff = variants.some(
      (v) =>
        v.quality_status === "validated" &&
        v.serving_status === "validated" &&
        v.serving_benchmark_current === true,
    );
    if (hasProductionHandoff) return null;
    return (
      failedVariants.find((v) => v.serving_status !== "validated")?.student_model_id ??
      null
    );
  }, [failedVariants, variants]);

  // ── Render ───────────────────────────────────────────────────────────────
  if (comparisonDataLoading) {
    return (
      <PageContainer data-testid="compare-benchmark-page-loading">
        <div className="flex items-center justify-center py-20">
          <Spinner aria-label="Loading comparison data" size="large" />
        </div>
      </PageContainer>
    );
  }

  if (variants.length === 0 && failedVariants.length === 0) {
    return (
      <PageContainer data-testid="compare-benchmark-page-empty">
        <div className="flex min-w-0 flex-col gap-1">
          <Text kind="title/md">Compare and Benchmark</Text>
          <Text
            kind="body/regular/sm"
            style={{ color: "var(--text-secondary)" }}
            data-testid="compare-project-name"
          >
            Project: {project.name}
          </Text>
        </div>
        <div
          className="glass-card p-6 flex flex-col gap-2"
          data-testid="compare-empty-state"
        >
          <Text kind="body/regular/sm">
            No quality-validated Students yet. Train one via{" "}
            <button
              type="button"
              className="underline"
              onClick={() => navigate(`/projects/${projectId}/scale-up`)}
            >
              Scale-Up Hub
            </button>
            .
          </Text>
        </div>
        <div className="flex items-center justify-end pt-2">
          <Button
            kind="secondary"
            onClick={() => navigate(`/projects/${projectId}/labeling`)}
            data-testid="back-to-labeling"
          >
            Back to Labeling
          </Button>
        </div>
      </PageContainer>
    );
  }

  return (
    <PageContainer
      // This page deliberately overrides PageContainer's `max-w-5xl` default
      // via the documented escape hatch. The comparison surface needs a wider
      // canvas to hold the per-card per-field block (3 fields × per-value
      // breakdown), the latency × concurrency ServingMatrix, and the grouped
      // bar chart legend without visible compression. Sibling dashboard
      // screens (Student Training, Training Job Monitor) keep `max-w-5xl`
      // because their content is narrower.
      maxWidthClass="max-w-6xl"
      data-testid="compare-benchmark-page"
    >
      <div className="flex items-end justify-between gap-3">
        <div className="flex min-w-0 flex-col gap-1">
          <Text kind="title/md">Compare and Benchmark</Text>
          <Text
            kind="body/regular/sm"
            style={{ color: "var(--text-secondary)" }}
            data-testid="compare-project-name"
          >
            Project: {project.name}
          </Text>
        </div>
        {teacherBaselineRun && (
          <Text
            kind="body/regular/sm"
            style={{ color: "var(--text-muted)" }}
            data-testid="compare-page-subtitle"
          >
            {(() => {
              const exampleCount =
                teacherBaselineMetrics?.overall?.example_count ?? null;
              const segments: string[] = [];
              if (exampleCount != null) {
                segments.push(
                  `Test Pool: ${exampleCount} image${exampleCount === 1 ? "" : "s"}`,
                );
              }
              if (teacherBaselineRun.visual_budget_preset_key) {
                segments.push(
                  `Visual budget: ${teacherBaselineRun.visual_budget_preset_key}`,
                );
              }
              if (guidance) {
                segments.push(`Guidance v${guidance.version_number}`);
              }
              return segments.join(" · ");
            })()}
          </Text>
        )}
      </div>

      <CompareScopeBar
        metricSelection={metricSelection}
        onMetricChange={setMetricSelection}
        chartOpen={chartOpen}
        onToggleChart={() => setChartOpen((v) => !v)}
        selectedCount={selectedIds.size}
        unbenchmarkedCount={unbenchmarkedIds.length}
        onBenchmarkAll={() => {
          enqueueBenchmark(unbenchmarkedIds);
        }}
        onBenchmarkSelected={() => {
          enqueueBenchmark([...selectedIds]);
          setSelectedIds(new Set());
        }}
        busy={busy}
      />

      <TeacherBaselineCard
        modelLabel={teacherBaselineLabel}
        metrics={teacherBaselineMetrics}
        baselineStatus={teacherBaselineRun ? "available" : "not_available"}
        staleNote={staleTeacherBaselineNote}
        coreFields={coreFields}
        metricSelection={metricSelection}
        perFieldMatchThreshold={project.scaleup_per_field_match_threshold}
        minPerValueF1Threshold={project.scaleup_min_per_value_f1_threshold}
      />

      {runGroups.map((group) => {
        const suite = group.suite;
        const groupGuidance =
          guidanceById[suite?.guidance_id ?? group.students[0]?.guidance_id];
        return (
          <TrainingRunGroup
            key={group.key}
            groupKey={group.key}
            latest={group.latest}
            unassigned={suite == null}
            startedAt={formatTimestamp(
              suite?.started_at ?? suite?.created_at ?? group.students[0]?.created_at,
            )}
            presetLabel={
              suite
                ? titleCasePreset(suite.training_preset)
                : group.students[0]
                  ? titleCasePreset(group.students[0].training_preset)
                  : null
            }
            guidanceVersion={groupGuidance?.version_number ?? null}
            trainingExampleCount={suite?.training_example_count ?? null}
            evaluationExampleCount={suite?.evaluation_example_count ?? null}
            modelSummary={modelSummaryByRunKey[group.key]}
            warning={warningByRunKey[group.key]}
          >
            <div className="flex flex-col gap-3">
              {group.students.map((student) => {
                const stage = stageByStudentId[student.student_model_id];
                const baseLabel =
                  friendlyLabelByModelConfigId[student.student_base_model_config_id] ??
                  student.student_base_model_config_id;
                if (student.quality_status === "failed") {
                  return (
                    <QualityFailedVariantCard
                      key={student.student_model_id}
                      projectId={projectId}
                      student={student}
                      baseModelLabel={baseLabel}
                      benchmarkStage={stage?.stage ?? null}
                      benchmarkStartupBudgetS={environment.nim_startup_timeout_s}
                      benchmarkElapsedMs={stage?.elapsedMs ?? 0}
                      benchmarkConcurrency={stage?.concurrency ?? null}
                      benchmarkEvalProgress={stage?.evalProgress ?? null}
                      benchmarkError={
                        benchmarkErrorByStudentId[student.student_model_id] ?? null
                      }
                      benchmarkEmphasis={
                        primaryFailedRecoveryId === student.student_model_id
                          ? "primary"
                          : "secondary"
                      }
                      busy={busy && activeBenchmarkId !== student.student_model_id}
                      onBenchmark={() => enqueueBenchmark([student.student_model_id])}
                    />
                  );
                }

                const qualityFullRun =
                  qualityRunByStudentId.runByStudentId[student.student_model_id] ??
                  null;
                const studentGuidance = guidanceById[student.guidance_id] ?? guidance;
                const studentCoreFields =
                  studentGuidance?.schema_fields.filter(
                    (field) => field.role === "core",
                  ) ?? [];
                return (
                  <StudentVariantCard
                    key={student.student_model_id}
                    projectId={projectId}
                    student={student}
                    baseModelLabel={baseLabel}
                    qualityRun={qualityFullRun}
                    servingRun={servingRunByStudentId[student.student_model_id] ?? null}
                    teacherOverallExactMatch={
                      teacherComparableByStudentId[student.student_model_id]
                        ? teacherOverallExactMatch
                        : null
                    }
                    coreFields={studentCoreFields}
                    metricSelection={metricSelection}
                    perFieldMatchThreshold={project.scaleup_per_field_match_threshold}
                    minPerValueF1Threshold={project.scaleup_min_per_value_f1_threshold}
                    concurrencies={environment.student_latency_test_concurrencies}
                    selected={selectedIds.has(student.student_model_id)}
                    onToggleSelected={() => toggleSelected(student.student_model_id)}
                    benchmarkStage={stage?.stage ?? null}
                    benchmarkStartupBudgetS={environment.nim_startup_timeout_s}
                    benchmarkElapsedMs={stage?.elapsedMs ?? 0}
                    benchmarkConcurrency={stage?.concurrency ?? null}
                    benchmarkEvalProgress={stage?.evalProgress ?? null}
                    busy={busy && activeBenchmarkId !== student.student_model_id}
                    benchmarkError={
                      benchmarkErrorByStudentId[student.student_model_id] ?? null
                    }
                    onBenchmark={() => enqueueBenchmark([student.student_model_id])}
                    singleGpuHost={singleGpuHost}
                  />
                );
              })}
            </div>
          </TrainingRunGroup>
        );
      })}

      {chartOpen && chartData && (
        <div ref={chartRef}>
          {chartContextWarning && (
            <div
              className="glass-card glass-card--static px-4 py-3 mb-3"
              data-testid="chart-context-warning"
              role="note"
            >
              <Text kind="body/regular/xs" style={{ color: "var(--warning-amber)" }}>
                {chartContextWarning}
              </Text>
            </div>
          )}
          <GroupedBarChart
            title={chartData.title}
            groups={chartData.groups}
            series={chartData.series}
          />
        </div>
      )}

      <div className="flex items-center justify-end gap-3 pt-2">
        <Button
          kind="secondary"
          onClick={() => navigate(`/projects/${projectId}/labeling`)}
          data-testid="back-to-labeling"
        >
          Back to Labeling
        </Button>
      </div>
    </PageContainer>
  );
}
