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
import {
  evaluationKeys,
  guidanceKeys,
  modelConfigKeys,
  studentModelKeys,
} from "@/api/query-keys";
import { PageContainer } from "@/components/common/PageContainer";
import { CompareScopeBar } from "@/components/compare/CompareScopeBar";
import type { MetricSelection } from "@/components/compare/CompareScopeBar";
import { formatModelDisplayName } from "@/lib/model-display";
import { CHART_TEACHER_COLOR, variantColor } from "@/lib/chart-palette";
import { chartGroupKey, GroupedBarChart } from "@/components/compare/GroupedBarChart";
import type { ChartGroup, ChartSeries } from "@/components/compare/GroupedBarChart";
import { QualityFailedVariantCard } from "@/components/compare/QualityFailedVariantCard";
import { StudentVariantCard } from "@/components/compare/StudentVariantCard";
import { TeacherBaselineCard } from "@/components/compare/TeacherBaselineCard";
import { useProjectSSE } from "@/hooks/useProjectSSE";
import { useSetupContext } from "@/pages/ProjectSetupLayout";
import type {
  EvaluationMetrics,
  EvaluationRunResponse,
  NimBenchmarkStage,
  PerValueMetric,
} from "@/types/evaluation";
import type { GuidanceResponse, SchemaFieldResponse } from "@/types/guidance";
import type { ModelConfigResponse } from "@/types/nim";
import type { StudentModel, StudentModelListResponse } from "@/types/training";

interface BenchmarkState {
  stage: NimBenchmarkStage;
  startedAt: number;
  elapsedMs: number;
  concurrency: number | null;
  evalProgress: { processed: number; total: number } | null;
}

// Poll cadence for the student list while a benchmark is active — the
// REST fallback that advances the queue when a terminal SSE event is
// missed (outage, backend restart).
const BENCHMARK_RECONCILE_POLL_MS = 5_000;

export function CompareBenchmarkPage() {
  const { projectId, project, environment } = useSetupContext();
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
    refetchInterval: activeBenchmarkId != null ? BENCHMARK_RECONCILE_POLL_MS : false,
  });

  const { data: completedRuns } = useQuery({
    queryKey: evaluationKeys.completedList(projectId),
    queryFn: () => listEvaluationRuns(projectId, { status: "completed", limit: 100 }),
  });

  const { data: guidance } = useQuery<GuidanceResponse>({
    queryKey: guidanceKeys.detail(projectId, project.active_guidance_id ?? ""),
    queryFn: () => fetchGuidance(projectId, project.active_guidance_id!),
    enabled: !!project.active_guidance_id,
  });

  const { data: modelConfigs } = useQuery({
    queryKey: modelConfigKeys.list(projectId, undefined),
    queryFn: () => fetchModelConfigs(projectId),
  });

  // Filter to Students with usable quality validation. ``validated`` is
  // the production-handoff bar; ``partial`` is informational —
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
      (s) => s.quality_status === "validated" || s.quality_status === "partial",
    );
  }, [studentList]);

  // Quality-``failed`` Students stay visible as compact notices with
  // their failure reason and a [Re-score quality] affordance, instead of
  // silently vanishing from the page (a serving-validated Student used to
  // disappear with no explanation and no UI remediation path).
  const failedVariants = useMemo<StudentModel[]>(() => {
    return (studentList?.items ?? []).filter((s) => s.quality_status === "failed");
  }, [studentList]);

  // Per-variant lazy fetch of the ``quality_evaluation_run_id`` for
  // without-ICL Exact Match + per-field metrics.
  const qualityRunQueries = useQueries({
    queries: variants
      .filter((v) => v.quality_evaluation_run_id != null)
      .map((v) => ({
        queryKey: evaluationKeys.detail(projectId, v.quality_evaluation_run_id!),
        queryFn: () => getEvaluationRun(projectId, v.quality_evaluation_run_id!),
      })),
  });

  const qualityRunByStudentId = useMemo(() => {
    const m: Record<string, EvaluationMetrics | null> = {};
    const fullByStudentId: Record<
      string,
      Awaited<ReturnType<typeof getEvaluationRun>> | null
    > = {};
    let q = 0;
    for (const v of variants) {
      if (v.quality_evaluation_run_id == null) {
        m[v.student_model_id] = null;
        fullByStudentId[v.student_model_id] = null;
        continue;
      }
      const result = qualityRunQueries[q]?.data ?? null;
      fullByStudentId[v.student_model_id] = result;
      m[v.student_model_id] = (result?.metrics ?? null) as EvaluationMetrics | null;
      q += 1;
    }
    return { metricsByStudentId: m, runByStudentId: fullByStudentId };
  }, [variants, qualityRunQueries]);

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

  // Per-variant lazy fetch of the serving run referenced by
  // ``serving_evaluation_run_id`` — carries ``metrics.benchmarks``
  // (latency × concurrency) written by the NIM benchmark lifecycle.
  // React Query dedupes with the quality fetch above when the two run
  // ids coincide (NIM-validated Students).
  const servingRunQueries = useQueries({
    queries: variants
      .filter((v) => v.serving_evaluation_run_id != null)
      .map((v) => ({
        queryKey: evaluationKeys.detail(projectId, v.serving_evaluation_run_id!),
        queryFn: () => getEvaluationRun(projectId, v.serving_evaluation_run_id!),
      })),
  });

  const servingRunByStudentId = useMemo(() => {
    const m: Record<string, EvaluationRunResponse | null> = {};
    let q = 0;
    for (const v of variants) {
      if (v.serving_evaluation_run_id == null) {
        m[v.student_model_id] = null;
        continue;
      }
      m[v.student_model_id] = servingRunQueries[q]?.data ?? null;
      q += 1;
    }
    return m;
  }, [variants, servingRunQueries]);

  // ── Model labels ──────────────────────────────────────────────────────────
  // Two label maps: ``labelByModelConfigId`` keeps the canonical
  // provider-prefixed name (e.g. ``nvidia/cosmos-reason2-8b``) used
  // on the Teacher card. The
  // ``friendlyLabelByModelConfigId`` map applies
  // ``formatModelDisplayName`` so Student variant cards
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
      m[mc.model_config_id] = formatModelDisplayName(mc.model_name);
    }
    return m;
  }, [modelConfigs]);

  const currentTeacherLabel = useMemo(() => {
    return (
      labelByModelConfigId[project.teacher_model_config_id] ??
      project.teacher_model_config_id
    );
  }, [labelByModelConfigId, project]);

  const teacherBaselineLabel = useMemo(() => {
    const modelConfigId = teacherBaselineRun?.model_config_id;
    if (!modelConfigId) return "Unknown Teacher model";
    return labelByModelConfigId[modelConfigId] ?? modelConfigId;
  }, [labelByModelConfigId, teacherBaselineRun]);

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
      ? `Teacher baseline predates the current ${changed.join(" and ")}`
      : null;
  }, [teacherBaselineRun, project.active_guidance_id, project.teacher_model_config_id]);

  // ── Chart data ───────────────────────────────────────────────────────────
  const chartData = useMemo(() => {
    if (!chartOpen || !guidance) return null;

    const coreFields = guidance.schema_fields.filter((f) => f.role === "core");

    const teacherSeries: ChartSeries = {
      label: `Teacher · ${teacherBaselineLabel}`,
      color: CHART_TEACHER_COLOR,
      values: {},
    };

    const variantSeries: ChartSeries[] = variants.map((v, idx) => ({
      label: `${
        friendlyLabelByModelConfigId[v.student_base_model_config_id] ??
        v.student_base_model_config_id
      }${v.quantization_method ? ` · ${v.quantization_method}` : ""}`,
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
    metricSelection,
    variants,
    teacherBaselineMetrics,
    qualityRunByStudentId,
    friendlyLabelByModelConfigId,
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
            v.serving_status !== "validated" && v.nim_preflight_status !== "failed",
        )
        .map((v) => v.student_model_id),
    [variants],
  );

  // ``busy`` disables the scope-bar action buttons and the per-card
  // checkboxes while an NIM benchmark is in flight — a sequential
  // workload where re-firing the trigger would double-fire the queue.
  const busy = activeBenchmarkId != null;

  // Among dual-validated variants, only the one with the best quality
  // Exact Match keeps the full-strength green handoff CTA (first in list
  // order on ties); the rest demote to secondary. Several identical
  // primaries down the page would leave nothing reading as "the" action.
  const primaryHandoffId = useMemo<string | null>(() => {
    let best: { id: string; rate: number } | null = null;
    for (const v of variants) {
      if (v.quality_status !== "validated" || v.serving_status !== "validated") {
        continue;
      }
      const rate =
        qualityRunByStudentId.metricsByStudentId[v.student_model_id]?.overall
          ?.exact_match_rate ?? -1;
      if (best == null || rate > best.rate) {
        best = { id: v.student_model_id, rate };
      }
    }
    return best?.id ?? null;
  }, [variants, qualityRunByStudentId]);

  // ── Render ───────────────────────────────────────────────────────────────
  if (studentsLoading) {
    return (
      <PageContainer data-testid="compare-benchmark-page-loading">
        <div className="flex items-center justify-center py-20">
          <Spinner aria-label="Loading Student variants" size="large" />
        </div>
      </PageContainer>
    );
  }

  if (variants.length === 0) {
    return (
      <PageContainer data-testid="compare-benchmark-page-empty">
        <Text kind="title/md">Compare and Benchmark</Text>
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
      <div className="flex items-baseline justify-between gap-3">
        <Text kind="title/md">Compare and Benchmark</Text>
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
      />

      {variants.map((v) => {
        const stage = stageByStudentId[v.student_model_id];
        const baseLabel =
          friendlyLabelByModelConfigId[v.student_base_model_config_id] ??
          v.student_base_model_config_id;
        const qualityFullRun =
          qualityRunByStudentId.runByStudentId[v.student_model_id] ?? null;
        return (
          <StudentVariantCard
            key={v.student_model_id}
            projectId={projectId}
            student={v}
            baseModelLabel={baseLabel}
            qualityRun={qualityFullRun}
            servingRun={servingRunByStudentId[v.student_model_id] ?? null}
            teacherOverallExactMatch={teacherOverallExactMatch}
            coreFields={coreFields}
            metricSelection={metricSelection}
            concurrencies={environment.student_latency_test_concurrencies}
            selected={selectedIds.has(v.student_model_id)}
            onToggleSelected={() => toggleSelected(v.student_model_id)}
            benchmarkStage={stage?.stage ?? null}
            benchmarkStartupBudgetS={environment.nim_startup_timeout_s}
            benchmarkElapsedMs={stage?.elapsedMs ?? 0}
            benchmarkConcurrency={stage?.concurrency ?? null}
            benchmarkEvalProgress={stage?.evalProgress ?? null}
            busy={busy && activeBenchmarkId !== v.student_model_id}
            benchmarkError={benchmarkErrorByStudentId[v.student_model_id] ?? null}
            onBenchmark={() => enqueueBenchmark([v.student_model_id])}
            singleGpuHost={singleGpuHost}
            teacherLabel={currentTeacherLabel}
            handoffEmphasis={
              primaryHandoffId === null || primaryHandoffId === v.student_model_id
                ? "primary"
                : "secondary"
            }
          />
        );
      })}

      {failedVariants.map((v) => (
        <QualityFailedVariantCard
          key={v.student_model_id}
          projectId={projectId}
          student={v}
          baseModelLabel={
            friendlyLabelByModelConfigId[v.student_base_model_config_id] ??
            v.student_base_model_config_id
          }
        />
      ))}

      {chartOpen && chartData && (
        <div ref={chartRef}>
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
