// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Acceptance tests for the Compare & Benchmark screen: Teacher baseline
 * (re-sourced from the latest completed Teacher evaluation run) and
 * per-variant cards, benchmark stage progression over SSE, per-value metric
 * drill-down (including a 25-value enum without overflow), the grouped bar
 * chart, the serving latency matrix (from the Student's serving run), and the
 * deployment-handoff Action Request (5-section content, read-only panel,
 * copy audit, 409 conflict path) plus the single-GPU displacement preview.
 */

import { beforeEach, describe, expect, it, vi } from "vitest";
import { StrictMode } from "react";
import { installEventSourceMock } from "@/test/event-source-mock";
import { makeProjectResponse } from "@/test/fixtures";
import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter, Outlet, Route, Routes } from "react-router-dom";

import { ApiError } from "@/api/client";
import { useSSEStore } from "@/stores/sse-store";

import { CompareBenchmarkPage } from "../CompareBenchmarkPage";

// ── Mocks ──────────────────────────────────────────────────────────────────

vi.mock("@/api/students", () => ({
  listStudentModels: vi.fn(),
  deployNim: vi.fn(),
  requestDeploymentHandoff: vi.fn(),
  rerescoreStudentModel: vi.fn(),
  deploymentBundleUrl: vi.fn(
    (projectId: string, studentId: string) =>
      `/v1/projects/${projectId}/student_models/${studentId}/deployment_bundle`,
  ),
}));

vi.mock("@/api/evaluation", () => ({
  getEvaluationRun: vi.fn(),
  createEvaluationRun: vi.fn(),
  listEvaluationRuns: vi.fn(),
  cancelEvaluationRun: vi.fn(),
  fetchTriggerStatus: vi.fn(),
  dismissTrigger: vi.fn(),
  fetchScaleUpGate: vi.fn(),
}));

vi.mock("@/api/guidance", () => ({
  fetchGuidance: vi.fn(),
  createGuidance: vi.fn(),
  validateDraft: vi.fn(),
  listGuidances: vi.fn(),
  editGuidancePreview: vi.fn(),
  editGuidanceExecute: vi.fn(),
}));

vi.mock("@/api/model-configs", () => ({
  fetchModelConfigs: vi.fn(),
  updateProject: vi.fn(),
}));

vi.mock("@/api/training", () => ({
  listTrainingSuites: vi.fn(),
}));

vi.mock("@/api/nim", () => ({
  generateActionRequest: vi.fn(),
  logActionRequestCopy: vi.fn(),
  fetchEnvironment: vi.fn(),
  testConnection: vi.fn(),
  runPreflight: vi.fn(),
  deployLocalNim: vi.fn(),
}));

vi.mock("@/pages/setup-context", () => ({
  useEnvironmentSetupContext: vi.fn(),
}));

installEventSourceMock();

// Clipboard stub for the Copy-to-Clipboard happy path.
const writeTextMock = vi.fn(() => Promise.resolve());
Object.defineProperty(navigator, "clipboard", {
  value: { writeText: writeTextMock },
  writable: true,
  configurable: true,
});

import {
  deployNim,
  listStudentModels,
  requestDeploymentHandoff,
  rerescoreStudentModel,
} from "@/api/students";
import { getEvaluationRun, listEvaluationRuns } from "@/api/evaluation";
import { fetchGuidance } from "@/api/guidance";
import { fetchModelConfigs } from "@/api/model-configs";
import { logActionRequestCopy } from "@/api/nim";
import { listTrainingSuites } from "@/api/training";
import { useEnvironmentSetupContext } from "@/pages/setup-context";

const mockListStudents = listStudentModels as ReturnType<typeof vi.fn>;
const mockDeployNim = deployNim as ReturnType<typeof vi.fn>;
const mockRerescore = rerescoreStudentModel as ReturnType<typeof vi.fn>;
const mockRequestDeploymentHandoff = requestDeploymentHandoff as ReturnType<
  typeof vi.fn
>;
const mockGetEvalRun = getEvaluationRun as ReturnType<typeof vi.fn>;
const mockListEvalRuns = listEvaluationRuns as ReturnType<typeof vi.fn>;
const mockFetchGuidance = fetchGuidance as ReturnType<typeof vi.fn>;
const mockFetchModelConfigs = fetchModelConfigs as ReturnType<typeof vi.fn>;
const mockListTrainingSuites = listTrainingSuites as ReturnType<typeof vi.fn>;
const mockLogCopy = logActionRequestCopy as ReturnType<typeof vi.fn>;
const mockSetup = useEnvironmentSetupContext as ReturnType<typeof vi.fn>;

// ── Fixtures ───────────────────────────────────────────────────────────────

const PROJECT = makeProjectResponse({
  project_id: "pid-1",
  name: "RPS",
  project_dir: "/tmp/p",
  created_at: "2026-04-29T00:00:00Z",
  updated_at: "2026-04-29T00:00:00Z",
  embedding_provider: "hosted_nvclip",
  counts: {
    verified: 124,
    unlabeled: 60,
    auto_labeled: 248,
    omitted: 0,
    pending_relabel: 0,
    prior_relabeled: 0,
  },
});

// Backend-served deployment config values (EnvironmentResponse) the page
// consumes verbatim — benchmark column sweep + NIM startup budget.
const ENVIRONMENT = {
  nim_startup_timeout_s: 1200,
  student_latency_test_concurrencies: [1, 8, 24],
};

function makeStudent(overrides: Partial<Record<string, unknown>>) {
  return {
    student_model_id: overrides.student_model_id ?? "sm-1",
    project_id: "pid-1",
    training_suite_id:
      "training_suite_id" in overrides ? overrides.training_suite_id : "suite-1",
    student_base_model_config_id: overrides.student_base_model_config_id ?? "mc-8b",
    tao_job_id: overrides.tao_job_id ?? "tao-train-1",
    guidance_id: "g-1",
    dataset_export_ids: ["de-train", "de-eval"],
    training_preset: "standard",
    lora_config: { enable_lora: true, lora_rank: 16 },
    created_at: "2026-04-29T00:00:00Z",
    checkpoint_packaging_status: "validated",
    nim_checkpoint_ref: overrides.nim_checkpoint_ref ?? "/ck/sm-1",
    quality_status: overrides.quality_status ?? "validated",
    quality_evaluation_run_id: overrides.quality_evaluation_run_id ?? "rr-q-1",
    serving_status: overrides.serving_status ?? "not_attempted",
    serving_evaluation_run_id: overrides.serving_evaluation_run_id ?? null,
    serving_benchmark_current:
      overrides.serving_benchmark_current ?? overrides.serving_status === "validated",
    serving_benchmark_blocker: overrides.serving_benchmark_blocker ?? null,
    nim_preflight_status: overrides.nim_preflight_status ?? null,
    nim_preflight_details: null,
    nim_preflight_at: null,
    nim_deployment_mode: null,
    nim_container_id: null,
    nim_endpoint_url: null,
    nim_vlm_release_version: "1.6.0",
    nim_model_profile_requested: null,
    nim_model_profile_selected: overrides.nim_model_profile_selected ?? null,
    nim_profile_metadata: null,
    gpu_type: overrides.gpu_type ?? "A100 80GB",
    gpu_count: overrides.gpu_count ?? 1,
    quantization_method: overrides.quantization_method ?? null,
    quantize_tao_job_id: overrides.quantize_tao_job_id ?? null,
    ...overrides,
  };
}

const STUDENT_BASELINE = makeStudent({
  student_model_id: "sm-baseline",
  quantization_method: null,
});
const STUDENT_FP8 = makeStudent({
  student_model_id: "sm-fp8",
  quantization_method: "FP8_DYNAMIC",
  quantize_tao_job_id: "tao-q-fp8",
});
const STUDENT_W4 = makeStudent({
  student_model_id: "sm-w4",
  quantization_method: "W4A16",
  quantize_tao_job_id: "tao-q-w4",
});

const TRAINING_SUITE = {
  training_suite_id: "suite-1",
  project_id: "pid-1",
  idempotency_key: "idem-1",
  guidance_id: "g-1",
  training_preset: "standard",
  export_field_mode: "all",
  include_auto_labeled: true,
  quantization_schemes: ["FP8_DYNAMIC", "W4A16"],
  training_dataset_export_id: "de-train",
  evaluation_dataset_export_id: "de-eval",
  training_example_count: 372,
  evaluation_example_count: 20,
  evaluation_dataset_checksum_sha256: "sha256:test-pool-1",
  selected_student_base_model_config_ids: ["mc-8b"],
  chain_ids_ordered: ["chain-1"],
  chains: [
    {
      chain_id: "chain-1",
      student_base_model_config_id: "mc-8b",
      base_model_name: "nvidia/cosmos-reason2-8b",
      jobs: [],
    },
  ],
  student_model_ids: ["sm-baseline", "sm-fp8", "sm-w4"],
  provisioning_run_id: null,
  provisioning_model_names: [],
  setup_error_ref: null,
  status: "completed",
  created_at: "2026-04-29T00:00:00Z",
  started_at: "2026-04-29T00:00:00Z",
  completed_at: "2026-04-29T03:00:00Z",
};

const GUIDANCE_RPS = {
  guidance_id: "g-1",
  project_id: "pid-1",
  version_number: 3,
  description: "Classify the gesture.",
  schema_fields: [
    {
      field_id: "f-gesture",
      field_name: "gesture",
      type: "enum",
      role: "core",
      allowed_values: ["rock", "paper", "scissors"],
      display_order: 0,
    },
  ],
  rules: "",
  derived_json_schema: {},
  generation_order: ["rationale_note", "gesture"],
  schema_hash: "sha256:abc",
  created_at: "2026-04-28T00:00:00Z",
};

// Guidance variant with 25 allowed values for the wide-enum drill-down test.
const GUIDANCE_LARGE_ENUM = {
  ...GUIDANCE_RPS,
  guidance_id: "g-large",
  schema_fields: [
    {
      field_id: "f-class",
      field_name: "class",
      type: "enum",
      role: "core",
      allowed_values: Array.from({ length: 25 }, (_, i) => `cls_${i}`),
      display_order: 0,
    },
  ],
  generation_order: ["rationale_note", "class"],
};

const MODEL_CONFIGS = {
  items: [
    {
      model_config_id: "mc-teacher",
      project_id: "pid-1",
      endpoint_id: "ep",
      model_name: "mistralai/mistral-large-3-675b-instruct-2512",
      context_window_tokens: 262144,
      eligible_roles: ["teacher"],
      supports_image_input: true,
      structured_generation_support: "supported",
      thinking_toggle_mode: "none",
      thinking_toggle_support: "supported",
      visual_budget_mode: "none",
      visual_budget_support: "unsupported",
      model_quantization: null,
      nim_model_profile: null,
      nim_profile_metadata: null,
      local_deploy_metadata: null,
      created_at: "2026-04-17T00:00:00Z",
    },
    {
      model_config_id: "mc-8b",
      project_id: "pid-1",
      endpoint_id: "ep",
      model_name: "nvidia/cosmos-reason2-8b",
      context_window_tokens: 256000,
      eligible_roles: ["teacher", "student_base"],
      supports_image_input: true,
      structured_generation_support: "supported",
      thinking_toggle_mode: "qwen_enable_thinking",
      thinking_toggle_support: "supported",
      visual_budget_mode: "mm_processor_size",
      visual_budget_support: "supported",
      model_quantization: null,
      nim_model_profile: null,
      nim_profile_metadata: null,
      local_deploy_metadata: null,
      created_at: "2026-04-17T00:00:00Z",
    },
  ],
  next_cursor: null,
};

const QUALITY_RUN_85 = {
  run_id: "rr-q-1",
  run_type: "evaluation_run",
  status: "completed",
  status_reason: null,
  pool_version_id: "pv-1",
  guidance_id: "g-1",
  model_config_id: "mc-8b",
  icl_mode: "disabled",
  evaluation_source: "tao",
  generation_preset_key: "precise",
  thinking_mode_effective: "on",
  visual_budget_preset_key: "balanced",
  structured_generation_mode_effective: "auto",
  // Teacher-identical contract on a STUDENT run (export_field_mode="all"
  // training) — the baseline predicate must reject it via
  // student_model_config_id, not the contract shape.
  inference_contract: { output_field_mode: "all", icl_field_mode: "core_only" },
  student_model_config_id: "mc-8b",
  icl_eligible_count_at_start: 0,
  icl_eligible_count_at_completion: 0,
  progress: { processed: 20, total: 20 },
  metrics: {
    overall: {
      exact_match_rate: 0.85,
      example_count: 20,
      per_field_match_rates: { gesture: 0.85 },
      per_value_metrics: {
        gesture: {
          rock: { precision: 0.9, recall: 0.85, f1: 0.87 },
          paper: { precision: 0.88, recall: 0.85, f1: 0.86 },
          scissors: { precision: 0.7, recall: 0.65, f1: 0.67 },
        },
      },
    },
    returning: null,
    new: null,
  },
  previous_pool_version: null,
  returning_example_keys: null,
  new_example_keys: null,
  previous_overall_exact_match: null,
  coverage_gaps: [],
  created_at: "2026-04-29T00:00:00Z",
  started_at: "2026-04-29T00:00:00Z",
  completed_at: "2026-04-29T00:00:00Z",
};

const TEACHER_BASELINE_METRICS = {
  overall: {
    exact_match_rate: 0.88,
    example_count: 20,
    per_field_match_rates: { gesture: 0.88 },
    per_value_metrics: {
      gesture: {
        rock: { precision: 0.95, recall: 0.9, f1: 0.92 },
        paper: { precision: 0.9, recall: 0.88, f1: 0.89 },
        scissors: { precision: 0.78, recall: 0.7, f1: 0.74 },
      },
    },
  },
  returning: null,
  new: null,
};

// The most recent completed run with the fixed Teacher contract and no
// Student provenance — what the page picks as the Teacher baseline.
const TEACHER_RUN = {
  ...QUALITY_RUN_85,
  run_id: "rr-t-1",
  guidance_id: "g-1",
  model_config_id: "mc-teacher",
  icl_mode: "enabled",
  evaluation_source: null,
  inference_contract: {
    output_field_mode: "all",
    icl_field_mode: "core_only",
    icl_max_examples: null,
  },
  student_model_config_id: null,
  metrics: TEACHER_BASELINE_METRICS,
  created_at: "2026-04-28T12:00:00Z",
};

// Serving run referenced by ``student.serving_evaluation_run_id`` — the
// NIM lifecycle writes the ``metrics.benchmarks`` latency table onto it
// after the run terminalizes.
const SERVING_RUN_BASELINE = {
  ...QUALITY_RUN_85,
  run_id: "rr-s-1",
  icl_mode: "disabled",
  evaluation_source: "nim",
  student_model_config_id: "mc-8b",
  created_at: "2026-04-29T02:00:00Z",
  metrics: {
    overall: {
      exact_match_rate: 0.85,
      example_count: 20,
      per_field_match_rates: { gesture: 0.85 },
      per_value_metrics: {},
    },
    benchmarks: [
      {
        concurrency: 1,
        status: "passed",
        latency_p50_ms: 300,
        latency_p90_ms: 500,
        latency_p99_ms: 800,
        request_throughput_rps: 2.5,
        failure_rate: 0,
      },
      {
        concurrency: 8,
        status: "passed",
        latency_p50_ms: 800,
        latency_p90_ms: 1200,
        latency_p99_ms: 1800,
        request_throughput_rps: 7.25,
        failure_rate: 0,
      },
      {
        concurrency: 24,
        status: "passed",
        latency_p50_ms: 1500,
        latency_p90_ms: 2400,
        latency_p99_ms: 3600,
        request_throughput_rps: 9.75,
        failure_rate: 0,
      },
    ],
    benchmark_workload: {
      version: "production_vlm_v1",
      workload_hash: "0123456789abcdef0123456789abcdef",
      pool_member_count: 120,
      selected_count: 120,
      inference_contract: { output_field_mode: "core_only" },
      output_limit_mode: "uncapped",
      kv_cache_reuse: "disabled",
      driver: { name: "aiperf", version: "0.10.0" },
    },
  },
};

let qc: QueryClient;

beforeEach(() => {
  vi.clearAllMocks();
  writeTextMock.mockClear();
  // Reset the SSE store between tests so the previous test's lastEvent
  // doesn't leak into the next render.
  useSSEStore.setState({ connections: {} });
  qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  mockSetup.mockReturnValue({
    projectId: "pid-1",
    project: PROJECT,
    // environment.gpus drives the displacement-preview affordance on
    // [Benchmark]. Default to empty gpus[] so the preview is
    // suppressed in unrelated tests; per-test overrides set
    // gpus: [GPU] for single-GPU coverage.
    environment: { ...ENVIRONMENT, gpus: [] },
  });
  mockFetchGuidance.mockResolvedValue(GUIDANCE_RPS);
  mockFetchModelConfigs.mockResolvedValue(MODEL_CONFIGS);
  mockListTrainingSuites.mockResolvedValue({
    items: [TRAINING_SUITE],
    next_cursor: null,
  });
  // Per-run-id routing: quality fetches resolve the TAO-rescored run,
  // serving fetches resolve the benchmarked run.
  mockGetEvalRun.mockImplementation((_pid: string, runId: string) =>
    Promise.resolve(runId === "rr-s-1" ? SERVING_RUN_BASELINE : QUALITY_RUN_85),
  );
  // Newest-first, as the endpoint returns: the Student quality run is
  // NEWER than the Teacher run and carries a Teacher-identical
  // contract — the baseline predicate must skip it.
  mockListEvalRuns.mockResolvedValue({
    items: [QUALITY_RUN_85, TEACHER_RUN],
    next_cursor: null,
  });
  mockListStudents.mockResolvedValue({
    items: [STUDENT_BASELINE, STUDENT_FP8, STUDENT_W4],
    next_cursor: null,
  });
  mockDeployNim.mockResolvedValue({
    student_model_id: "sm-baseline",
    nim_deployment_mode: "local",
    serving_status: "pending",
    task_id: "task-1",
    created_at: "2026-04-29T02:00:00Z",
  });
  mockRequestDeploymentHandoff.mockResolvedValue({
    request_type: "deployment_handoff",
    generated_at: "2026-04-29T03:00:00Z",
    project_name: "RPS",
    technical_requirements: {
      nim_container_image: "nvcr.io/nim/nvidia/cosmos-reason2-8b:1.6.0",
      checkpoint_reference: "/ck/sm-1",
      gpu_requirements: "1× A100 (≥56 GB)",
    },
    current_environment: {
      quality_status: "validated",
      serving_status: "validated",
    },
    rendered_text:
      "Deployment Handoff Request\n\n" +
      "Project: RPS\n" +
      "Student: sm-baseline\n" +
      "Base: Cosmos Reason2 8B · Quantization: none · GPU: 1× A100 80GB\n\n" +
      "Checkpoint\n    Packaging status: validated\n\n" +
      "NIM Configuration\n" +
      "Production deployment command:\n    docker run …\n\n" +
      "Endpoint health: GET /v1/health/ready\n\n" +
      "Model\n    Quantization: none\n\n" +
      "Evaluation\n" +
      "Quality (TAO-rescored):\n    Overall Exact Match: 0.85\n\n" +
      "Serving (NIM-validated):\n    Overall Exact Match: 0.85\n\n" +
      "Training Lineage\n    Training TAO job: tao-train-1\n",
  });
  mockLogCopy.mockResolvedValue({ audit_event_id: "ae-1" });
});

function renderPage({ strict = false }: { strict?: boolean } = {}) {
  const page = (
    <QueryClientProvider client={qc}>
      <MemoryRouter initialEntries={["/projects/pid-1/compare"]}>
        <Routes>
          <Route path="/projects/:projectId" element={<Outlet />}>
            <Route path="compare" element={<CompareBenchmarkPage />} />
            <Route path="labeling" element={<div data-testid="labeling-page" />} />
            <Route path="scale-up" element={<div data-testid="scale-up-page" />} />
          </Route>
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>
  );
  return render(strict ? <StrictMode>{page}</StrictMode> : page);
}

function emitSse(projectId: string, type: string, data: Record<string, unknown>) {
  act(() => {
    useSSEStore.setState((s) => ({
      connections: {
        ...s.connections,
        [projectId]: {
          ...(s.connections[projectId] ?? {
            lastEvent: null,
            eventSource: null,
            pollingIntervalId: null,
          }),
          lastEvent: {
            type,
            data,
          },
        },
      },
    }));
  });
}

// ── Acceptance tests ──────────────────────────────────────────────────────

describe("CompareBenchmarkPage", () => {
  // ── Default view ──────────────────────────────────────────────────────
  it("keeps loading until human-readable model identities are ready", async () => {
    let resolveModelConfigs: (value: typeof MODEL_CONFIGS) => void = () => undefined;
    mockFetchModelConfigs.mockReturnValue(
      new Promise<typeof MODEL_CONFIGS>((resolve) => {
        resolveModelConfigs = resolve;
      }),
    );

    renderPage();
    await waitFor(() => expect(mockListStudents).toHaveBeenCalled());

    expect(screen.getByTestId("compare-benchmark-page-loading")).toBeInTheDocument();
    expect(screen.getByLabelText("Loading comparison data")).toBeInTheDocument();
    expect(screen.queryByTestId("student-variant-card-sm-baseline")).toBeNull();
    expect(screen.queryByText("mc-8b")).toBeNull();

    act(() => resolveModelConfigs(MODEL_CONFIGS));

    const card = await screen.findByTestId("student-variant-card-sm-baseline");
    expect(card).toHaveTextContent("Cosmos Reason2 8B");
    expect(screen.getByTestId("teacher-baseline-card")).toHaveTextContent(
      "mistralai/mistral-large-3-675b-instruct-2512",
    );
  });

  it("deduplicates evaluation run queries shared by multiple variants", async () => {
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => undefined);
    try {
      renderPage();
      await screen.findByTestId("student-variant-card-sm-w4");

      const warnings = warnSpy.mock.calls.flat().join("\n");
      expect(warnings).not.toContain("[QueriesObserver]: Duplicate Queries found");
    } finally {
      warnSpy.mockRestore();
    }
  });

  it("default view renders Teacher baseline + per-variant cards + scope controls + 'Not benchmarked'", async () => {
    renderPage();
    await screen.findByTestId("compare-benchmark-page");

    // Teacher accuracy baseline.
    const teacher = await screen.findByTestId("teacher-baseline-card");
    expect(teacher.textContent).toContain("Teacher (accuracy baseline)");
    expect(screen.getByTestId("teacher-baseline-exact-match").textContent).toContain(
      "88%",
    );

    // Per-variant cards present (3 trained variants).
    expect(screen.getByTestId("student-variant-card-sm-baseline")).toBeTruthy();
    expect(screen.getByTestId("student-variant-card-sm-fp8")).toBeTruthy();
    expect(screen.getByTestId("student-variant-card-sm-w4")).toBeTruthy();

    // Each variant shows its quality Exact Match (mocked to 85% across).
    const baselineEm = await screen.findAllByTestId("exact-match");
    expect(baselineEm.length).toBeGreaterThanOrEqual(3);
    expect(baselineEm[0].textContent).toContain("85%");

    // Delta vs the historical Teacher baseline (-3 since 85 - 88 = -3).
    const deltas = await screen.findAllByTestId("exact-match-delta");
    expect(deltas[0].textContent).toContain("-3 pts vs Teacher baseline");

    // Subtitle summarizes the baseline run's Test Pool + visual budget.
    const subtitle = screen.getByTestId("compare-page-subtitle");
    expect(subtitle.textContent).toContain("Test Pool: 20 images");
    expect(subtitle.textContent).toContain("Visual budget: balanced");
    expect(screen.getByTestId("compare-project-name").textContent).toBe("Project: RPS");

    // FP8 + W4 not yet benchmarked → "Not benchmarked".
    const fp8Card = screen.getByTestId("student-variant-card-sm-fp8");
    expect(fp8Card.textContent).toContain("Not benchmarked");
    const w4Card = screen.getByTestId("student-variant-card-sm-w4");
    expect(w4Card.textContent).toContain("Not benchmarked");

    // Scope controls.
    expect(screen.getByTestId("benchmark-all-button")).toBeTruthy();
    expect(screen.getByTestId("benchmark-selected-button")).toBeTruthy();
    expect(screen.getByRole("combobox", { name: "Per-field metric:" })).toBeTruthy();
    expect(screen.getByTestId("chart-toggle-button")).toBeTruthy();

    const runGroup = screen.getByTestId("training-run-group-suite-1");
    expect(runGroup).toHaveTextContent("Latest training run");
    expect(runGroup).toHaveTextContent("Standard preset");
    expect(runGroup).toHaveTextContent("Guidance v3");
    expect(runGroup).toHaveTextContent("372 training images");
    expect(runGroup).toHaveTextContent("20 Test Pool images");
    expect(runGroup).toHaveTextContent("Models: Cosmos Reason2 8B");
  });

  it("keeps all runs visible while marking incompatible history and disambiguating chart series", async () => {
    const olderStudent = makeStudent({
      student_model_id: "sm-older-8b",
      training_suite_id: "suite-older",
      quality_evaluation_run_id: "rr-q-old",
      created_at: "2026-04-20T12:00:00Z",
    });
    const olderSuite = {
      ...TRAINING_SUITE,
      training_suite_id: "suite-older",
      idempotency_key: "idem-older",
      evaluation_dataset_export_id: "de-eval-old",
      evaluation_dataset_checksum_sha256: "sha256:test-pool-old",
      training_example_count: 240,
      evaluation_example_count: 18,
      student_model_ids: ["sm-older-8b"],
      created_at: "2026-04-20T12:00:00Z",
      started_at: "2026-04-20T12:00:00Z",
      completed_at: "2026-04-20T14:00:00Z",
    };
    mockListStudents.mockResolvedValue({
      items: [STUDENT_BASELINE, olderStudent],
      next_cursor: null,
    });
    mockListTrainingSuites.mockResolvedValue({
      items: [{ ...TRAINING_SUITE, student_model_ids: ["sm-baseline"] }, olderSuite],
      next_cursor: null,
    });
    mockGetEvalRun.mockImplementation((_pid: string, runId: string) =>
      Promise.resolve(
        runId === "rr-q-old"
          ? { ...QUALITY_RUN_85, run_id: runId, pool_version_id: "pv-old" }
          : QUALITY_RUN_85,
      ),
    );

    renderPage();
    const latest = await screen.findByTestId("training-run-group-suite-1");
    const older = screen.getByTestId("training-run-group-suite-older");
    expect(latest).toHaveTextContent("Latest training run");
    expect(older).toHaveTextContent("Training run");
    expect(older).toHaveTextContent("Different evaluation set");
    expect(older).toHaveTextContent("Student-vs-Teacher deltas are hidden");
    expect(within(older).queryByTestId("exact-match-delta")).toBeNull();
    expect(within(latest).getByTestId("exact-match-delta")).toBeInTheDocument();

    fireEvent.click(screen.getByTestId("chart-toggle-button"));
    const legend = await screen.findByTestId("chart-legend");
    expect(legend).toHaveTextContent("Apr 29");
    expect(legend).toHaveTextContent("Apr 20");
    expect(screen.getByTestId("chart-context-warning")).toHaveTextContent(
      "different or unverified evaluation contexts",
    );
  });

  it("keeps a packaged quality-pending Student visible for its first NIM validation", async () => {
    mockListStudents.mockResolvedValueOnce({
      items: [
        makeStudent({
          student_model_id: "sm-pending-quality",
          quality_status: "pending",
          quality_evaluation_run_id: null,
          serving_status: "not_attempted",
          serving_evaluation_run_id: null,
        }),
      ],
      next_cursor: null,
    });

    renderPage();
    const card = await screen.findByTestId("student-variant-card-sm-pending-quality");
    expect(card).toHaveTextContent("Quality: Pending");
    expect(card).toHaveTextContent(
      "Run the first NIM validation to measure quality and serving performance.",
    );
    expect(screen.getByTestId("exact-match")).toHaveTextContent("—");

    const deploy = screen.getByTestId("benchmark-button-sm-pending-quality");
    expect(deploy).toHaveTextContent("Deploy and benchmark");
    fireEvent.click(deploy);
    await waitFor(() =>
      expect(mockDeployNim).toHaveBeenCalledWith(
        "pid-1",
        "sm-pending-quality",
        expect.objectContaining({}),
      ),
    );
  });

  it("reconciles a normal Student's pending NIM validation after reopen", async () => {
    mockListStudents.mockResolvedValue({
      items: [
        makeStudent({
          student_model_id: "sm-pending-serving",
          quality_status: "pending",
          quality_evaluation_run_id: null,
          serving_status: "pending",
          serving_evaluation_run_id: null,
        }),
        makeStudent({
          student_model_id: "sm-other",
          serving_status: "not_attempted",
        }),
      ],
      next_cursor: null,
    });

    renderPage();
    const pendingCard = await screen.findByTestId(
      "student-variant-card-sm-pending-serving",
    );

    expect(within(pendingCard).getByTestId("benchmark-reconciled")).toHaveTextContent(
      "Serving validation is running. Live stage details resume when events arrive.",
    );
    expect(
      within(pendingCard).queryByTestId("benchmark-button-sm-pending-serving"),
    ).toBeNull();
    expect(screen.getByTestId("benchmark-all-button")).toBeDisabled();
    expect(screen.getByTestId("benchmark-button-sm-other")).toBeDisabled();
  });

  // ── Fallback (preflight failed) ─────────────────────────────────────────
  it("when nim_preflight_status='failed', card shows [Deploy for serving validation] and expands student_nim_deploy AR", async () => {
    mockListStudents.mockResolvedValueOnce({
      items: [
        makeStudent({
          student_model_id: "sm-fail",
          quality_status: "validated",
          serving_status: "failed",
          nim_preflight_status: "failed",
        }),
      ],
      next_cursor: null,
    });
    const { generateActionRequest } = await import("@/api/nim");
    (generateActionRequest as ReturnType<typeof vi.fn>).mockResolvedValue({
      request_type: "student_nim_deploy",
      generated_at: "2026-04-29T05:00:00Z",
      project_name: "RPS",
      technical_requirements: {},
      current_environment: {},
      rendered_text:
        "Student NIM Deployment Request\n\nDocker run line:\n    docker run nvcr.io/nim/nvidia/cosmos-reason2-8b:1.6.0",
    });

    renderPage();
    const card = await screen.findByTestId("student-variant-card-sm-fail");
    const fallbackButton = (await screen.findByTestId(
      "request-deployment-fallback-sm-fail",
    )) as HTMLButtonElement;
    expect(fallbackButton).toBeTruthy();
    expect(
      card.querySelector("[data-testid='request-deployment-handoff-sm-fail']"),
    ).toBeNull();

    fireEvent.click(fallbackButton);
    await waitFor(() =>
      expect(generateActionRequest).toHaveBeenCalledWith(
        "pid-1",
        expect.objectContaining({
          request_type: "student_nim_deploy",
          context: { student_model_id: "sm-fail" },
        }),
      ),
    );
    await screen.findByTestId("action-request-ready");
  });

  // ── Benchmark stages ────────────────────────────────────────────────────
  it("sequential stages render: docker_run → health_poll → smoke_inference → evaluation → benchmark → stopping; only the active variant updates", async () => {
    renderPage();
    await screen.findByTestId("compare-benchmark-page");

    // Start a benchmark on sm-baseline.
    fireEvent.click(screen.getByTestId("benchmark-button-sm-baseline"));
    await waitFor(() =>
      expect(mockDeployNim).toHaveBeenCalledWith(
        "pid-1",
        "sm-baseline",
        expect.objectContaining({}),
      ),
    );
    // Optimistic preflight stage shows immediately.
    const baselineCard = screen.getByTestId("student-variant-card-sm-baseline");
    await waitFor(() =>
      expect(
        baselineCard.querySelector("[data-testid='benchmark-stage-strip']"),
      ).toBeTruthy(),
    );

    // Walk through stages.
    const stages: string[] = [
      "docker_run",
      "health_poll",
      "smoke_inference",
      "registering_endpoint",
      "evaluation",
      "benchmark",
      "stopping",
    ];
    for (let i = 0; i < stages.length; i++) {
      emitSse("pid-1", "nim_benchmark_progress", {
        student_model_id: "sm-baseline",
        stage: stages[i],
        elapsed_ms: 1000 * (i + 1),
        ...(stages[i] === "benchmark" ? { concurrency: 1 } : {}),
      });
      await waitFor(() => {
        const strip = baselineCard.querySelector(
          "[data-testid='benchmark-stage-strip']",
        );
        expect(strip?.getAttribute("data-stage")).toBe(stages[i]);
      });
      if (stages[i] === "evaluation") {
        // The (N / M) counter rides the standard evaluation_progress
        // event — nim_benchmark_progress never carries processed/total.
        emitSse("pid-1", "evaluation_progress", {
          run_id: "eval-run-1",
          processed: 8,
          total: 20,
        });
        await waitFor(() => {
          const suffix = baselineCard.querySelector(
            "[data-testid='benchmark-stage-suffix']",
          );
          expect(suffix?.textContent).toBe("(8 / 20)");
        });
      }
    }

    // Other cards are untouched throughout.
    expect(
      screen
        .getByTestId("student-variant-card-sm-fp8")
        .querySelector("[data-testid='benchmark-stage-strip']"),
    ).toBeNull();
  });

  // ── Per-value drill-down ────────────────────────────────────────────────
  it("Per-value F1 expands categorical Core fields across all cards; switching to Precision/Recall updates simultaneously", async () => {
    renderPage();
    await screen.findByTestId("compare-benchmark-page");

    // Default = Match rate; per-field row labels visible.
    expect(screen.getByTestId("teacher-baseline-block-match-rate")).toBeTruthy();

    // Switch to Per-value F1.
    fireEvent.change(screen.getByTestId("compare-metric-select"), {
      target: { value: "per_value_f1" },
    });
    await waitFor(() =>
      expect(screen.getByTestId("teacher-baseline-block-per-value")).toBeTruthy(),
    );
    // Teacher: scissors F1 = 0.74 → "74%".
    const teacherScissors = screen.getByTestId(
      "teacher-baseline-value-gesture-scissors",
    );
    expect(teacherScissors.textContent).toContain("74%");

    // Variant card baseline shows scissors F1 = 0.67.
    const variantScissors = screen.getByTestId(
      "variant-sm-baseline-value-gesture-scissors",
    );
    expect(variantScissors.textContent).toContain("67%");

    // Switch to Per-value Precision.
    fireEvent.change(screen.getByTestId("compare-metric-select"), {
      target: { value: "per_value_precision" },
    });
    await waitFor(() =>
      expect(
        screen.getByTestId("teacher-baseline-value-gesture-scissors").textContent,
      ).toContain("78%"),
    );
    // sm-baseline scissors precision = 0.7 → "70%".
    expect(
      screen.getByTestId("variant-sm-baseline-value-gesture-scissors").textContent,
    ).toContain("70%");
  });

  it("uses project thresholds only for their authoritative quality metrics", async () => {
    mockSetup.mockReturnValue({
      projectId: "pid-1",
      project: {
        ...PROJECT,
        scaleup_per_field_match_threshold: 0.86,
        scaleup_min_per_value_f1_threshold: 0.68,
      },
      environment: { ...ENVIRONMENT, gpus: [] },
    });
    renderPage();
    await screen.findByTestId("compare-benchmark-page");

    const variantFieldRate = screen
      .getByTestId("variant-sm-baseline-row-gesture")
      .querySelector("[data-threshold-status]");
    const teacherFieldRate = screen
      .getByTestId("teacher-baseline-row-gesture")
      .querySelector("[data-threshold-status]");
    expect(variantFieldRate).toHaveAttribute("data-threshold-status", "below");
    expect(teacherFieldRate).toHaveAttribute("data-threshold-status", "meets");

    fireEvent.change(screen.getByTestId("compare-metric-select"), {
      target: { value: "per_value_f1" },
    });
    const variantScissors = await screen.findByTestId(
      "variant-sm-baseline-value-gesture-scissors",
    );
    expect(variantScissors.querySelector("[data-threshold-status]")).toHaveAttribute(
      "data-threshold-status",
      "below",
    );

    fireEvent.change(screen.getByTestId("compare-metric-select"), {
      target: { value: "per_value_precision" },
    });
    expect(variantScissors.querySelector("[data-threshold-status]")).toHaveAttribute(
      "data-threshold-status",
      "diagnostic",
    );
  });

  it("per-value drill-down at 25 allowed values renders inside a wrapping grid (no horizontal overflow)", async () => {
    mockFetchGuidance.mockResolvedValue(GUIDANCE_LARGE_ENUM);
    // Quality run with 25 per-value buckets.
    const largeRun = {
      ...QUALITY_RUN_85,
      metrics: {
        overall: {
          exact_match_rate: 0.5,
          example_count: 20,
          per_field_match_rates: { class: 0.5 },
          per_value_metrics: {
            class: Object.fromEntries(
              GUIDANCE_LARGE_ENUM.schema_fields[0].allowed_values!.map((v) => [
                v,
                { precision: 0.5, recall: 0.5, f1: 0.5 },
              ]),
            ),
          },
        },
        returning: null,
        new: null,
      },
    };
    mockGetEvalRun.mockResolvedValue(largeRun);
    mockListEvalRuns.mockResolvedValue({
      items: [{ ...TEACHER_RUN, metrics: largeRun.metrics }],
      next_cursor: null,
    });
    renderPage();
    await screen.findByTestId("compare-benchmark-page");

    fireEvent.change(screen.getByTestId("compare-metric-select"), {
      target: { value: "per_value_f1" },
    });
    await waitFor(() =>
      expect(screen.getByTestId("variant-sm-baseline-row-class-values")).toBeTruthy(),
    );
    const valuesGrid = screen.getByTestId("variant-sm-baseline-row-class-values");
    // 25 buckets all rendered.
    expect(
      valuesGrid.querySelectorAll(
        "[data-testid^='variant-sm-baseline-value-class-cls_']",
      ).length,
    ).toBe(25);
    // CSS grid wrap layout (auto-fill + minmax).
    const style = (valuesGrid as HTMLElement).getAttribute("style") ?? "";
    expect(style).toContain("auto-fill");
    expect(style).toContain("minmax(140px");
    // Card wrapper enforces no horizontal overflow.
    const card = screen.getByTestId("student-variant-card-sm-baseline") as HTMLElement;
    expect(card.style.overflowX).toBe("hidden");
  });

  // ── Chart ──────────────────────────────────────────────────────────────
  it("[Chart] toggle renders grouped bar chart with Teacher + variants; metric dropdown drives bar values; bars carry exact %; toggle hides", async () => {
    renderPage();
    await screen.findByTestId("compare-benchmark-page");
    // Chart hidden by default.
    expect(screen.queryByTestId("grouped-bar-chart")).toBeNull();

    // The toggle is a stable-label pressed pill: the label stays "Chart"
    // in both states and the open state is announced via aria-pressed.
    const chartToggle = screen.getByTestId("chart-toggle-button");
    expect(chartToggle.textContent).toContain("Chart");
    expect(chartToggle.getAttribute("aria-pressed")).toBe("false");

    fireEvent.click(chartToggle);
    await screen.findByTestId("grouped-bar-chart");
    expect(chartToggle.getAttribute("aria-pressed")).toBe("true");
    expect(chartToggle.textContent).toContain("Chart");
    // Teacher legend entry present.
    const legend = screen.getByTestId("chart-legend").textContent ?? "";
    expect(legend).toContain("Teacher");

    // Bar present for teacher × gesture (88%).
    await waitFor(() => {
      const bar = screen
        .getByTestId("grouped-bar-chart")
        .querySelector("[data-testid^='chart-bar-Teacher · '][data-percent]");
      expect(bar).toBeTruthy();
      expect(bar?.getAttribute("data-percent")).toBe("88");
    });

    // Switch metric → chart updates (per-value bars now grouped per value).
    fireEvent.change(screen.getByTestId("compare-metric-select"), {
      target: { value: "per_value_f1" },
    });
    await waitFor(() => {
      const bars = screen
        .getByTestId("grouped-bar-chart")
        .querySelectorAll("[data-testid^='chart-bar-Teacher · ']");
      // 3 enum values × 1 series = 3 teacher bars.
      expect(bars.length).toBe(3);
    });

    // Toggle off hides chart.
    fireEvent.click(screen.getByTestId("chart-toggle-button"));
    await waitFor(() => expect(screen.queryByTestId("grouped-bar-chart")).toBeNull());
  });

  // ── Teacher baseline selection ──────────────────────────────────────────
  it("baseline empty state renders when no completed Teacher-contract run exists", async () => {
    // Only Student runs (teacher-identical contract but student
    // provenance) — the predicate must match none of them.
    mockListEvalRuns.mockResolvedValue({
      items: [QUALITY_RUN_85],
      next_cursor: null,
    });
    renderPage();
    await screen.findByTestId("compare-benchmark-page");
    const empty = await screen.findByTestId("teacher-baseline-empty");
    expect(empty.textContent).toMatch(/No completed Teacher evaluation yet/);
    expect(screen.queryByTestId("compare-page-subtitle")).toBeNull();
  });

  it("stale note renders when the baseline run's Guidance is not the active one", async () => {
    mockListEvalRuns.mockResolvedValue({
      items: [{ ...TEACHER_RUN, guidance_id: "g-0" }],
      next_cursor: null,
    });
    renderPage();
    await screen.findByTestId("compare-benchmark-page");
    const note = await screen.findByTestId("teacher-baseline-stale-note");
    expect(note.textContent).toContain("predates the current Guidance");
  });

  it("attributes historical metrics to their Teacher while current-Teacher operations stay current", async () => {
    mockSetup.mockReturnValue({
      projectId: "pid-1",
      project: { ...PROJECT, teacher_model_config_id: "mc-8b" },
      environment: {
        ...ENVIRONMENT,
        gpus: [{ name: "NVIDIA A100", memory_total_gb: 80 }],
      },
    });

    renderPage();
    await screen.findByTestId("compare-benchmark-page");

    // TEACHER_RUN belongs to Mistral even though the project now selects
    // Cosmos. Every metric surface must retain the run's immutable identity.
    const baselineLabel = screen.getByTestId("teacher-baseline-model-label");
    expect(baselineLabel).toHaveTextContent(
      "mistralai/mistral-large-3-675b-instruct-2512",
    );
    const note = screen.getByTestId("teacher-baseline-stale-note");
    expect(note).toHaveTextContent("predates the current Teacher (Cosmos Reason2 8B)");
    expect(note).toHaveTextContent("Run a new evaluation to refresh the baseline.");
    expect(screen.getAllByTestId("exact-match-delta")[0]).toHaveTextContent(
      "-3 pts vs Teacher baseline",
    );

    fireEvent.click(screen.getByTestId("chart-toggle-button"));
    const legend = await screen.findByTestId("chart-legend");
    expect(legend).toHaveTextContent(
      "Teacher · mistralai/mistral-large-3-675b-instruct-2512",
    );

    // Displacement copy names the actual local-resident resource class. The
    // selected Teacher can be hosted, so neither current nor historical model
    // identity belongs in this warning.
    const preview = screen.getAllByTestId("benchmark-displacement-preview")[0];
    expect(preview).toHaveTextContent(/another local NIM/i);
    expect(preview).not.toHaveTextContent("nvidia/cosmos-reason2-8b");
    expect(preview).not.toHaveTextContent("mistralai");
  });

  it("humanizes quantization and labels persisted benchmark hardware", async () => {
    mockListStudents.mockResolvedValue({
      items: [
        makeStudent({
          student_model_id: "sm-fp8",
          quality_evaluation_run_id: "rr-q-fp8",
          serving_status: "validated",
          serving_evaluation_run_id: "rr-s-1",
          quantization_method: "FP8_DYNAMIC",
          gpu_type: "RTX PRO 6000",
          gpu_count: 1,
        }),
      ],
      next_cursor: null,
    });
    renderPage();
    const card = await screen.findByTestId("student-variant-card-sm-fp8");
    expect(card).toHaveTextContent("FP8 Dynamic");
    expect(card).not.toHaveTextContent("FP8_DYNAMIC");
    expect(card).toHaveTextContent("Benchmark hardware: 1× RTX PRO 6000");
  });

  it("retains a historical baseline ID after its catalog row is removed", async () => {
    mockSetup.mockReturnValue({
      projectId: "pid-1",
      project: { ...PROJECT, teacher_model_config_id: "mc-8b" },
      environment: { ...ENVIRONMENT, gpus: [] },
    });
    mockFetchModelConfigs.mockResolvedValue({
      ...MODEL_CONFIGS,
      items: MODEL_CONFIGS.items.filter(
        (model) => model.model_config_id !== TEACHER_RUN.model_config_id,
      ),
    });

    renderPage();
    await screen.findByTestId("compare-benchmark-page");

    // Historical runs can outlive retired catalog entries. The immutable raw ID is still honest;
    // falling back to the current Teacher would silently change provenance.
    expect(screen.getByTestId("teacher-baseline-model-label")).toHaveTextContent(
      "mc-teacher",
    );
    fireEvent.click(screen.getByTestId("chart-toggle-button"));
    expect(await screen.findByTestId("chart-legend")).toHaveTextContent(
      "Teacher · mc-teacher",
    );
  });

  // ── Serving latency matrix ──────────────────────────────────────────────
  it("serving-validated variant renders the c={1,8,24} × {p50,p90,p99} latency matrix from its serving run", async () => {
    mockListStudents.mockResolvedValue({
      items: [
        makeStudent({
          student_model_id: "sm-baseline",
          serving_status: "validated",
          serving_evaluation_run_id: "rr-s-1",
        }),
      ],
      next_cursor: null,
    });
    renderPage();
    await screen.findByTestId("compare-benchmark-page");

    await screen.findByTestId("serving-matrix");
    for (const c of [1, 8, 24]) {
      expect(screen.getByTestId(`serving-matrix-cell-c${c}-p50`)).toBeTruthy();
      expect(screen.getByTestId(`serving-matrix-cell-c${c}-p90`)).toBeTruthy();
      expect(screen.getByTestId(`serving-matrix-cell-c${c}-p99`)).toBeTruthy();
    }
    // Milliseconds preserve useful distinctions instead of rounding every
    // sub-second result into a handful of tenths.
    expect(screen.getByTestId("serving-matrix-cell-c1-p50").textContent).toContain(
      "300 ms",
    );
    expect(screen.getByTestId("serving-matrix-cell-c1-p99").textContent).toContain(
      "800 ms",
    );
    expect(screen.getByText("2.50")).toBeTruthy();
    expect(screen.getAllByText("0.0%").length).toBe(3);

    fireEvent.click(screen.getByTestId("serving-workload-toggle"));
    expect(screen.getByTestId("serving-workload-details").textContent).toContain(
      "120 real Test Pool images",
    );
    expect(screen.getByTestId("serving-workload-details").textContent).toContain(
      "Uncapped",
    );
    expect(screen.getByTestId("serving-workload-details").textContent).toContain(
      "AIPerf 0.10.0",
    );
  });

  it("legacy validated serving evidence requires AIPerf revalidation before handoff", async () => {
    mockListStudents.mockResolvedValue({
      items: [
        makeStudent({
          student_model_id: "sm-legacy",
          serving_status: "validated",
          serving_evaluation_run_id: "rr-s-1",
          serving_benchmark_current: false,
          serving_benchmark_blocker: "aiperf_workload_missing",
        }),
      ],
      next_cursor: null,
    });

    renderPage();
    const warning = await screen.findByTestId("serving-aiperf-revalidation-required");
    expect(warning).toHaveTextContent("AIPerf revalidation required");
    expect(warning).toHaveTextContent("real Test Pool AIPerf workload");
    expect(screen.queryByTestId("request-deployment-handoff-sm-legacy")).toBeNull();

    const revalidate = screen.getByTestId("benchmark-button-sm-legacy");
    expect(revalidate).toHaveTextContent("Revalidate with AIPerf");
    fireEvent.click(revalidate);
    await waitFor(() =>
      expect(mockDeployNim).toHaveBeenCalledWith("pid-1", "sm-legacy", {}),
    );
  });

  // ── Deployment handoff happy path + audit ───────────────────────────────
  it("[Request Production Deployment] emits one handoff under production StrictMode, expands all sections, and audits Copy", async () => {
    // Dual-validated student fixture.
    mockListStudents.mockResolvedValue({
      items: [
        makeStudent({
          student_model_id: "sm-baseline",
          serving_status: "validated",
          serving_evaluation_run_id: "rr-s-1",
        }),
      ],
      next_cursor: null,
    });

    renderPage({ strict: true });
    await screen.findByTestId("compare-benchmark-page");

    // Handoff button visible (dual-validated). Fallback button is NOT.
    const handoffBtn = await screen.findByTestId(
      "request-deployment-handoff-sm-baseline",
    );
    expect(screen.queryByTestId("request-deployment-fallback-sm-baseline")).toBeNull();

    fireEvent.click(handoffBtn);
    await waitFor(() =>
      expect(mockRequestDeploymentHandoff).toHaveBeenCalledWith("pid-1", "sm-baseline"),
    );
    const ar = await screen.findByTestId("action-request-ready");
    expect(mockRequestDeploymentHandoff).toHaveBeenCalledTimes(1);

    // Five logical sections present in the rendered text: checkpoint,
    // NIM config, model metadata, evaluation snapshot, training lineage.
    const text = ar.textContent ?? "";
    expect(text).toContain("Project: RPS");
    expect(text).toContain("Production deployment command:");
    expect(text).toContain("docker run");
    expect(text).toMatch(/Quantization:.+(none|FP8|W4)/);
    expect(text).toContain("Quality (TAO-rescored):");
    expect(text).toContain("Serving (NIM-validated):");
    expect(text).toContain("Training Lineage");
    expect(text).toContain("Training TAO job");

    // Read-only — no SME form fields. The panel only contains buttons
    // and Text nodes; no inputs/textareas.
    expect(ar.querySelectorAll("input,textarea,select").length).toBe(0);

    // No secrets in rendered text.
    expect(text.toLowerCase()).not.toContain("nvapi-");
    expect(text.toLowerCase()).not.toContain("ngc-");
    expect(text.toLowerCase()).not.toContain("api_key=");

    // The portable artifact is the primary action and uses a browser-native
    // streamed download instead of buffering a multi-GB bundle in JS.
    const anchorClick = vi
      .spyOn(HTMLAnchorElement.prototype, "click")
      .mockImplementation(() => undefined);
    fireEvent.click(
      screen.getByRole("button", {
        name: "Download portable NIM deployment bundle",
      }),
    );
    expect(anchorClick).toHaveBeenCalledTimes(1);
    expect((anchorClick.mock.instances[0] as HTMLAnchorElement).href).toContain(
      "/v1/projects/pid-1/student_models/sm-baseline/deployment_bundle",
    );
    anchorClick.mockRestore();

    // Copy → logActionRequestCopy fires (audited).
    fireEvent.click(screen.getByRole("button", { name: "Copy to Clipboard" }));
    await waitFor(() =>
      expect(mockLogCopy).toHaveBeenCalledWith(
        "pid-1",
        expect.objectContaining({ request_type: "deployment_handoff" }),
      ),
    );
    expect(writeTextMock).toHaveBeenCalled();
  });

  // ── Deployment handoff CTAs ──────────────────────────────────────────────
  it("gives every dual-validated variant a green production handoff CTA", async () => {
    mockListStudents.mockResolvedValue({
      items: [
        makeStudent({
          student_model_id: "sm-a",
          serving_status: "validated",
          serving_evaluation_run_id: "rr-s-1",
        }),
        makeStudent({
          student_model_id: "sm-b",
          serving_status: "validated",
          serving_evaluation_run_id: "rr-s-1",
        }),
      ],
      next_cursor: null,
    });

    renderPage();
    await screen.findByTestId("compare-benchmark-page");
    const handoffA = await screen.findByTestId("request-deployment-handoff-sm-a");
    const handoffB = await screen.findByTestId("request-deployment-handoff-sm-b");

    expect(handoffA).toHaveClass("nv-button--kind-primary", "nvidia-green-button");
    expect(handoffB).toHaveClass("nv-button--kind-primary", "nvidia-green-button");
    expect((handoffA as HTMLButtonElement).disabled).toBe(false);
    expect((handoffB as HTMLButtonElement).disabled).toBe(false);
    expect(document.querySelectorAll(".nvidia-green-button")).toHaveLength(2);
  });

  it("keeps serving recovery secondary when a production handoff is available", async () => {
    mockListStudents.mockResolvedValue({
      items: [
        makeStudent({
          student_model_id: "sm-ready",
          serving_status: "validated",
          serving_evaluation_run_id: "rr-s-1",
        }),
        makeStudent({
          student_model_id: "sm-needs-recovery",
          quality_status: "failed",
          quality_evaluation_run_id: null,
          serving_status: "not_attempted",
          serving_evaluation_run_id: null,
        }),
      ],
      next_cursor: null,
    });

    renderPage();
    const handoff = await screen.findByTestId("request-deployment-handoff-sm-ready");
    const recovery = await screen.findByTestId(
      "failed-quality-deploy-sm-needs-recovery",
    );

    expect(handoff).toHaveClass("nv-button--kind-primary");
    expect(recovery).toHaveClass("nv-button--kind-secondary");
    expect(document.querySelectorAll(".nvidia-green-button")).toHaveLength(1);
  });

  it("promotes only the first serving-recovery action when no handoff is ready", async () => {
    mockListStudents.mockResolvedValue({
      items: [
        makeStudent({
          student_model_id: "sm-recovery-a",
          quality_status: "failed",
          quality_evaluation_run_id: null,
          serving_status: "not_attempted",
        }),
        makeStudent({
          student_model_id: "sm-recovery-b",
          quality_status: "failed",
          quality_evaluation_run_id: null,
          serving_status: "not_attempted",
        }),
      ],
      next_cursor: null,
    });

    renderPage();
    const recoveryA = await screen.findByTestId("failed-quality-deploy-sm-recovery-a");
    const recoveryB = await screen.findByTestId("failed-quality-deploy-sm-recovery-b");

    expect(recoveryA).toHaveClass("nv-button--kind-primary");
    expect(recoveryB).toHaveClass("nv-button--kind-secondary");
    expect(document.querySelectorAll(".nvidia-green-button")).toHaveLength(1);
  });

  // ── Deployment handoff 409 conflict path ────────────────────────────────
  it("409 from :deployment_handoff is surfaced inline with a plain-language explanation", async () => {
    // Dual-validated fixture so the button is rendered.
    mockListStudents.mockResolvedValue({
      items: [
        makeStudent({
          student_model_id: "sm-baseline",
          serving_status: "validated",
          serving_evaluation_run_id: "rr-s-1",
        }),
      ],
      next_cursor: null,
    });
    mockRequestDeploymentHandoff.mockRejectedValue(
      new ApiError(409, "conflict: INFERENCE_CONTRACT_MISMATCH"),
    );

    renderPage();
    await screen.findByTestId("compare-benchmark-page");
    fireEvent.click(
      await screen.findByTestId("request-deployment-handoff-sm-baseline"),
    );

    const conflict = await screen.findByTestId("action-request-conflict");
    expect(conflict.textContent).toContain(
      "training-time and serving-time Inference Contracts disagree",
    );
    expect(screen.getByTestId("action-request-conflict-detail").textContent).toContain(
      "INFERENCE_CONTRACT_MISMATCH",
    );
  });

  it("explains a stale-screen AIPerf handoff rejection without hiding the backend reason", async () => {
    mockListStudents.mockResolvedValue({
      items: [
        makeStudent({
          student_model_id: "sm-stale",
          serving_status: "validated",
          serving_evaluation_run_id: "rr-s-1",
          serving_benchmark_current: true,
        }),
      ],
      next_cursor: null,
    });
    mockRequestDeploymentHandoff.mockRejectedValue(
      new ApiError(409, "conflict: serving_benchmark_requires_aiperf"),
    );

    renderPage();
    fireEvent.click(await screen.findByTestId("request-deployment-handoff-sm-stale"));

    const conflict = await screen.findByTestId("action-request-conflict");
    expect(conflict).toHaveTextContent("Revalidate with AIPerf");
    expect(screen.getByTestId("action-request-conflict-detail")).toHaveTextContent(
      "serving_benchmark_requires_aiperf",
    );
  });

  // ── Selection persistence ───────────────────────────────────────────────
  it("selection state survives a benchmark-completion refetch (selectedIds keyed by student_model_id)", async () => {
    renderPage();
    await screen.findByTestId("compare-benchmark-page");

    fireEvent.click(screen.getByTestId("variant-checkbox-sm-fp8"));
    fireEvent.click(screen.getByTestId("variant-checkbox-sm-w4"));
    expect(
      (screen.getByTestId("variant-checkbox-sm-fp8") as HTMLInputElement).checked,
    ).toBe(true);
    expect(
      (screen.getByTestId("variant-checkbox-sm-w4") as HTMLInputElement).checked,
    ).toBe(true);

    // Simulate an unrelated cache invalidation by re-resolving the list.
    // Selection state is page-local React state so refetching does NOT
    // reset checkboxes (this verifies the keyed-by-id design).
    mockListStudents.mockResolvedValueOnce({
      items: [STUDENT_BASELINE, STUDENT_FP8, STUDENT_W4],
      next_cursor: null,
    });
    await act(async () => {
      qc.invalidateQueries({ queryKey: ["studentModels", "pid-1"] });
      await new Promise((r) => setTimeout(r, 0));
    });
    expect(
      (screen.getByTestId("variant-checkbox-sm-fp8") as HTMLInputElement).checked,
    ).toBe(true);
    expect(
      (screen.getByTestId("variant-checkbox-sm-w4") as HTMLInputElement).checked,
    ).toBe(true);
  });

  // ── Sequential queue ───────────────────────────────────────────────────
  it("[Benchmark Selected] runs sequentially — second deployNim only fires after the first nim_benchmark_completed", async () => {
    renderPage();
    await screen.findByTestId("compare-benchmark-page");

    fireEvent.click(screen.getByTestId("variant-checkbox-sm-fp8"));
    fireEvent.click(screen.getByTestId("variant-checkbox-sm-w4"));
    fireEvent.click(screen.getByTestId("benchmark-selected-button"));

    // The first :deploy_nim fires for the head of the queue.
    await waitFor(() => expect(mockDeployNim).toHaveBeenCalledTimes(1));
    const firstCall = mockDeployNim.mock.calls[0];
    const firstStudentId = firstCall[1] as string;

    // While the queue is busy, no second call goes out yet. The advance is
    // purely SSE-event-driven (no timers in the page), so draining the
    // already-queued micro/macrotasks is a deterministic negative check —
    // no wall-clock sleep needed.
    await act(async () => {
      await new Promise((r) => setTimeout(r, 0));
    });
    expect(mockDeployNim).toHaveBeenCalledTimes(1);

    // Complete the first benchmark; queue advances and the second
    // :deploy_nim fires.
    emitSse("pid-1", "nim_benchmark_completed", {
      student_model_id: firstStudentId,
      evaluation_run_id: "rr-2",
      exact_match: 0.85,
      per_field_match_rates: { gesture: 0.85 },
      benchmarks: [],
      skipped_concurrencies: [],
      serving_status: "validated",
      elapsed_ms: 1234,
    });
    await waitFor(() => expect(mockDeployNim).toHaveBeenCalledTimes(2));
    expect(mockDeployNim.mock.calls[1][1]).not.toBe(firstStudentId);
  });

  // ── Empty state ────────────────────────────────────────────────────────
  it("shows an empty state with a Scale-Up Hub link when no Students are packaged", async () => {
    mockListStudents.mockResolvedValue({ items: [], next_cursor: null });
    renderPage();
    await screen.findByTestId("compare-benchmark-page-empty");
    expect(screen.getByTestId("compare-empty-state").textContent).toContain(
      "Scale-Up Hub",
    );
    expect(screen.getByTestId("compare-project-name").textContent).toBe("Project: RPS");
  });

  // ── Back to Labeling navigates to /labeling ────────────────────────────
  it("[Back to Labeling] navigates to /projects/:projectId/labeling", async () => {
    renderPage();
    await screen.findByTestId("compare-benchmark-page");
    fireEvent.click(screen.getByTestId("back-to-labeling"));
    await screen.findByTestId("labeling-page");
  });

  // ── Action Request CTA placement ────────────────────────────────────────
  it("Action Request CTAs are inline within the variant card, not in a settings page", async () => {
    mockListStudents.mockResolvedValue({
      items: [
        makeStudent({
          student_model_id: "sm-baseline",
          serving_status: "validated",
          serving_evaluation_run_id: "rr-s-1",
        }),
      ],
      next_cursor: null,
    });
    renderPage();
    await screen.findByTestId("compare-benchmark-page");
    const card = screen.getByTestId("student-variant-card-sm-baseline");
    const handoff = await screen.findByTestId("request-deployment-handoff-sm-baseline");
    // Handoff CTA is a descendant of the variant card (not a sibling
    // somewhere else on the page).
    expect(card.contains(handoff)).toBe(true);
  });

  // ── Partial quality_status ──────────────────────────────────────────────

  it("quality_status='partial' renders yellow badge with helper line", async () => {
    mockListStudents.mockResolvedValue({
      items: [
        makeStudent({
          student_model_id: "sm-partial",
          quality_status: "partial",
          quality_evaluation_run_id: "rr-q-partial",
          serving_status: "validated",
          serving_evaluation_run_id: "rr-s-1",
        }),
      ],
      next_cursor: null,
    });
    renderPage();
    await screen.findByTestId("compare-benchmark-page");
    const badge = await screen.findByTestId("quality-status-badge-partial");
    expect(badge).toBeTruthy();
    expect(badge.textContent).toContain("Quality: Partial");
    // Helper line is rendered alongside the badge.
    const row = screen.getByTestId("quality-status-partial-row");
    expect(row.textContent).toMatch(/Re-run NIM evaluation/i);
    // The card's data-attribute mirrors the partial status (surfaced
    // for tests); the badge is the visual signal.
    const card = screen.getByTestId("student-variant-card-sm-partial");
    expect(card.getAttribute("data-quality-status")).toBe("partial");
  });

  it("quality_status='partial' does NOT enable the deployment handoff button", async () => {
    mockListStudents.mockResolvedValue({
      items: [
        makeStudent({
          student_model_id: "sm-partial",
          quality_status: "partial",
          quality_evaluation_run_id: "rr-q-partial",
          serving_status: "validated",
          serving_evaluation_run_id: "rr-s-1",
        }),
      ],
      next_cursor: null,
    });
    renderPage();
    await screen.findByTestId("compare-benchmark-page");
    // [Request Production Deployment] is gated by ``dualValidated`` —
    // quality_status="partial" must NOT satisfy the gate.
    expect(screen.queryByTestId("request-deployment-handoff-sm-partial")).toBeNull();
  });

  it("quality_status='validated' does NOT render the partial badge", async () => {
    mockListStudents.mockResolvedValue({
      items: [
        makeStudent({
          student_model_id: "sm-validated",
          quality_status: "validated",
          serving_status: "validated",
          serving_evaluation_run_id: "rr-s-1",
        }),
      ],
      next_cursor: null,
    });
    renderPage();
    await screen.findByTestId("compare-benchmark-page");
    expect(screen.queryByTestId("quality-status-badge-partial")).toBeNull();
    expect(screen.queryByTestId("quality-status-partial-row")).toBeNull();
  });

  // ── Quality-failed variants stay visible with reason + remediation ───────

  function mockOneFailedStudent() {
    mockListStudents.mockResolvedValue({
      items: [
        STUDENT_BASELINE,
        makeStudent({
          student_model_id: "sm-dead",
          quality_status: "failed",
          quality_evaluation_run_id: null,
          serving_status: "validated",
          serving_evaluation_run_id: "rr-s-1",
          nim_preflight_details: {
            quality_failure_reason:
              "evaluate job failed: does NOT match a known upstream-loader gap",
          },
        }),
      ],
      next_cursor: null,
    });
  }

  it("quality_status='failed' renders the failed-variant notice with its reason instead of hiding", async () => {
    mockOneFailedStudent();
    renderPage();
    await screen.findByTestId("compare-benchmark-page");

    const notice = await screen.findByTestId("quality-failed-variant-card");
    expect(notice.textContent).toMatch(/excluded from comparison/i);
    expect(notice.textContent).toMatch(/serving is validated/i);
    expect(screen.getByTestId("quality-failure-reason").textContent).toContain(
      "does NOT match a known upstream-loader gap",
    );
    // No comparison card for the failed variant — it has no quality
    // metrics to compare; the validated sibling still renders normally.
    expect(screen.queryByTestId("student-variant-card-sm-dead")).toBeNull();
    expect(screen.getByTestId("student-variant-card-sm-baseline")).toBeTruthy();
  });

  it("humanizes a persisted quality failure token instead of leaking snake_case", async () => {
    mockListStudents.mockResolvedValue({
      items: [
        makeStudent({
          student_model_id: "sm-token",
          quality_status: "failed",
          serving_status: "failed",
          nim_preflight_details: {
            quality_failure_reason: "tao_evaluate_failed",
          },
        }),
      ],
      next_cursor: null,
    });

    renderPage();
    const reason = await screen.findByTestId("quality-failure-reason");
    expect(reason.textContent).toBe("TAO quality evaluation failed.");
    expect(reason.textContent).not.toContain("tao_evaluate_failed");
  });

  it("[Re-score quality] replays the rescore for the failed Student", async () => {
    mockOneFailedStudent();
    mockRerescore.mockResolvedValue({ quality_status: "validated", run_id: "rr-x" });
    renderPage();
    await screen.findByTestId("quality-failed-variant-card");
    expect(screen.queryByTestId("compare-empty-state")).toBeNull();

    fireEvent.click(screen.getByTestId("rerescore-button"));
    await waitFor(() => expect(mockRerescore).toHaveBeenCalledWith("pid-1", "sm-dead"));
  });

  it("quality-failed packaged Students can enter the NIM validation lifecycle", async () => {
    mockListStudents.mockResolvedValue({
      items: [
        makeStudent({
          student_model_id: "sm-loader-gap",
          quality_status: "failed",
          quality_evaluation_run_id: null,
          serving_status: "not_attempted",
          serving_evaluation_run_id: null,
          checkpoint_packaging_status: "validated",
          nim_preflight_details: {
            quality_failure_reason: "TAO evaluator hit a known upstream loader gap",
          },
        }),
      ],
      next_cursor: null,
    });
    renderPage();
    await screen.findByTestId("quality-failed-variant-card");

    expect(screen.getByTestId("quality-failed-nim-fallback-help").textContent).toMatch(
      /persisted lineage determines/i,
    );
    fireEvent.click(screen.getByTestId("failed-quality-deploy-sm-loader-gap"));
    await waitFor(() =>
      expect(mockDeployNim).toHaveBeenCalledWith("pid-1", "sm-loader-gap", {}),
    );
    expect(await screen.findByTestId("quality-failed-benchmark-stage")).toHaveAttribute(
      "data-stage",
      "preflight",
    );
  });

  it("reconciles a quality-failed Student's pending NIM validation after reopen", async () => {
    mockListStudents.mockResolvedValue({
      items: [
        makeStudent({
          student_model_id: "sm-reopened",
          quality_status: "failed",
          quality_evaluation_run_id: null,
          serving_status: "pending",
          serving_evaluation_run_id: null,
          checkpoint_packaging_status: "validated",
        }),
      ],
      next_cursor: null,
    });
    renderPage();
    await screen.findByTestId("quality-failed-variant-card");

    expect(
      screen.getByTestId("quality-failed-benchmark-reconciled").textContent,
    ).toMatch(/serving validation is running/i);
    expect(screen.queryByTestId("failed-quality-deploy-sm-reopened")).toBeNull();
  });

  it("rerescore refusal renders the backend detail inline", async () => {
    mockOneFailedStudent();
    mockRerescore.mockRejectedValue(
      new ApiError(
        400,
        JSON.stringify({
          detail: "No paired evaluate TAOJob found for this Student",
        }),
      ),
    );
    renderPage();
    await screen.findByTestId("quality-failed-variant-card");

    fireEvent.click(screen.getByTestId("rerescore-button"));
    const err = await screen.findByTestId("rerescore-error");
    expect(err.textContent).toContain("No paired evaluate TAOJob found");
  });

  // ── [Chart] scrolls the below-the-fold chart into view ───────────────────

  it("opening the chart scrolls it into view", async () => {
    const scrollSpy = vi.fn();
    Element.prototype.scrollIntoView = scrollSpy;
    renderPage();
    await screen.findByTestId("compare-benchmark-page");
    await screen.findByTestId("student-variant-card-sm-baseline");

    fireEvent.click(screen.getByTestId("chart-toggle-button"));
    await screen.findByTestId("grouped-bar-chart");
    await waitFor(() => expect(scrollSpy).toHaveBeenCalled());
  });

  // ── Displacement preview on the [Benchmark] CTA when single-GPU ─────────
  // (one-NIM-per-GPU replace semantics)

  it("single-GPU host renders displacement preview above [Benchmark]", async () => {
    // The Student NIM lifecycle stops the resident Teacher first and
    // best-effort auto-restores it after. Setup 2A's pHash
    // advisory tells the SME up front about the labeling-time
    // tradeoff; this preview tells them about the benchmark-time
    // tradeoff so they aren't surprised mid-flow.
    mockSetup.mockReturnValue({
      projectId: "pid-1",
      project: PROJECT,
      environment: {
        ...ENVIRONMENT,
        gpus: [{ name: "NVIDIA A100", memory_total_gb: 80 }],
      },
    });
    renderPage();
    await screen.findByTestId("compare-benchmark-page");
    // Preview renders for each variant that still has a [Benchmark]
    // button (the baseline + the two quant variants here, all
    // unbenchmarked). Use getAllByTestId because three variants render.
    const previews = screen.getAllByTestId("benchmark-displacement-preview");
    expect(previews.length).toBeGreaterThan(0);
    expect(previews[0]).toHaveTextContent(/~6 minutes/);
    expect(previews[0]).toHaveTextContent(/another local NIM is using this GPU/i);
    expect(previews[0]).not.toHaveTextContent(/minimax/i);
  });

  it("multi-GPU host suppresses the displacement preview", async () => {
    // Multi-GPU has no one-NIM-per-GPU contention (Teacher → device 0,
    // embedding → device 1, Student → device 2 etc. via deterministic
    // GPU placement). Preview is suppressed.
    mockSetup.mockReturnValue({
      projectId: "pid-1",
      project: PROJECT,
      environment: {
        ...ENVIRONMENT,
        gpus: [
          { name: "NVIDIA A100", memory_total_gb: 80 },
          { name: "NVIDIA A100", memory_total_gb: 80 },
        ],
      },
    });
    renderPage();
    await screen.findByTestId("compare-benchmark-page");
    expect(screen.queryByTestId("benchmark-displacement-preview")).toBeNull();
  });

  it("already-validated variant (no [Benchmark]) omits the displacement preview", async () => {
    // The preview is gated on ``showBenchmark`` so it disappears
    // when the variant is already serving-validated. Prevents
    // dangling copy on variants the SME has already benchmarked.
    mockSetup.mockReturnValue({
      projectId: "pid-1",
      project: PROJECT,
      environment: {
        ...ENVIRONMENT,
        gpus: [{ name: "NVIDIA A100", memory_total_gb: 80 }],
      },
    });
    mockListStudents.mockResolvedValue({
      items: [
        makeStudent({
          student_model_id: "sm-validated",
          quality_status: "validated",
          serving_status: "validated",
          serving_evaluation_run_id: "rr-s-1",
        }),
      ],
      next_cursor: null,
    });
    renderPage();
    await screen.findByTestId("compare-benchmark-page");
    // No [Benchmark] button → no preview.
    expect(screen.queryByTestId("benchmark-button-sm-validated")).toBeNull();
    expect(screen.queryByTestId("benchmark-displacement-preview")).toBeNull();
  });

  // ── REST reconciliation of the benchmark queue ─────────────────────────

  it("advances the benchmark queue from REST when the terminal SSE event is missed", async () => {
    // REST is authoritative; SSE is a hint channel. If the terminal SSE
    // event lands during an EventSource outage (or a backend restart
    // kills the stream), the refreshed student list — which the SSE
    // store's reconnect handling invalidates — must clear the busy state
    // and dispatch the next queued variant instead of stalling forever.
    renderPage();
    await screen.findByTestId("compare-benchmark-page");

    fireEvent.click(screen.getByTestId("variant-checkbox-sm-fp8"));
    fireEvent.click(screen.getByTestId("variant-checkbox-sm-w4"));
    fireEvent.click(screen.getByTestId("benchmark-selected-button"));
    await waitFor(() => expect(mockDeployNim).toHaveBeenCalledTimes(1));
    const firstStudentId = mockDeployNim.mock.calls[0][1] as string;

    // No SSE event ever arrives. REST now reports the active variant
    // terminal (validated).
    mockListStudents.mockResolvedValue({
      items: [STUDENT_BASELINE, STUDENT_FP8, STUDENT_W4].map((s) =>
        s.student_model_id === firstStudentId
          ? { ...s, serving_status: "validated", serving_evaluation_run_id: "rr-s-1" }
          : s,
      ),
      next_cursor: null,
    });
    // Reconciliation only trusts list snapshots fetched after the deploy
    // dispatched — let the clock tick past the dispatch timestamp first.
    await act(async () => {
      await new Promise((r) => setTimeout(r, 10));
    });
    await act(async () => {
      await qc.invalidateQueries({ queryKey: ["studentModels", "pid-1"] });
    });

    // Queue advances from REST state: the second variant dispatches.
    await waitFor(() => expect(mockDeployNim).toHaveBeenCalledTimes(2));
    expect(mockDeployNim.mock.calls[1][1]).not.toBe(firstStudentId);
  });

  // ── Per-value chart keying ──────────────────────────────────────────────

  it("per-value chart keys series by field+value so same-labeled values across fields don't collide", async () => {
    // Two boolean Core fields both emit "true"/"false" buckets. A chart
    // keyed by bare value label would render the later field's metrics
    // for both clusters — misrepresenting the evaluation on the surface
    // that feeds the Student deployment decision.
    const GUIDANCE_TWO_BOOLEANS = {
      ...GUIDANCE_RPS,
      schema_fields: [
        {
          field_id: "f-a",
          field_name: "is_damaged",
          type: "boolean",
          role: "core",
          allowed_values: null,
          display_order: 0,
        },
        {
          field_id: "f-b",
          field_name: "is_rusty",
          type: "boolean",
          role: "core",
          allowed_values: null,
          display_order: 1,
        },
      ],
      generation_order: ["rationale_note", "is_damaged", "is_rusty"],
    };
    mockFetchGuidance.mockResolvedValue(GUIDANCE_TWO_BOOLEANS);
    const twoFieldMetrics = {
      overall: {
        exact_match_rate: 0.8,
        example_count: 20,
        per_field_match_rates: { is_damaged: 0.9, is_rusty: 0.4 },
        per_value_metrics: {
          is_damaged: {
            true: { precision: 0.9, recall: 0.9, f1: 0.9 },
            false: { precision: 0.85, recall: 0.85, f1: 0.85 },
          },
          is_rusty: {
            true: { precision: 0.4, recall: 0.4, f1: 0.4 },
            false: { precision: 0.35, recall: 0.35, f1: 0.35 },
          },
        },
      },
      returning: null,
      new: null,
    };
    mockListEvalRuns.mockResolvedValue({
      items: [{ ...TEACHER_RUN, metrics: twoFieldMetrics }],
      next_cursor: null,
    });

    renderPage();
    await screen.findByTestId("compare-benchmark-page");

    fireEvent.change(screen.getByTestId("compare-metric-select"), {
      target: { value: "per_value_f1" },
    });
    fireEvent.click(screen.getByTestId("chart-toggle-button"));
    await screen.findByTestId("grouped-bar-chart");

    const teacherSeries = "Teacher · mistralai/mistral-large-3-675b-instruct-2512";
    const damagedTrue = screen.getByTestId(
      `chart-bar-${teacherSeries}-is_damaged::true`,
    );
    const rustyTrue = screen.getByTestId(`chart-bar-${teacherSeries}-is_rusty::true`);
    // Each cluster carries its OWN field's F1 — the two same-labeled
    // groups must differ.
    expect(damagedTrue.getAttribute("data-percent")).toBe("90");
    expect(rustyTrue.getAttribute("data-percent")).toBe("40");
  });
});
