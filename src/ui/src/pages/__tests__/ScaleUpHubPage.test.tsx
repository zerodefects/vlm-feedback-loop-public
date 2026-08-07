// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, it, expect, vi, beforeEach } from "vitest";
import { installEventSourceMock } from "@/test/event-source-mock";
import { makeProjectResponse } from "@/test/fixtures";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter, Routes, Route, Outlet } from "react-router-dom";

import { ScaleUpHubPage } from "../ScaleUpHubPage";

// ── Mocks ──────────────────────────────────────────────────────────────────

vi.mock("@/api/evaluation", () => ({
  fetchScaleUpGate: vi.fn(),
}));

vi.mock("@/api/model-configs", () => ({
  fetchModelConfigs: vi.fn().mockResolvedValue({
    items: [
      {
        model_config_id: "mc-1",
        model_name: "nvidia/cosmos-reason2-8b",
        eligible_roles: ["teacher"],
        supports_image_input: true,
        context_window_tokens: 256_000,
        thinking_toggle_mode: "qwen_enable_thinking",
        visual_budget_mode: "mm_processor_size",
      },
    ],
  }),
  updateProject: vi.fn(),
}));

vi.mock("@/api/guidance", () => ({
  fetchGuidance: vi.fn().mockResolvedValue({
    guidance_id: "g-1",
    version_number: 3,
    description: "Test guidance",
    rules: "",
    schema: { fields: [] },
  }),
}));

vi.mock("@/api/training", () => ({
  listStudentBaseModelConfigs: vi.fn().mockResolvedValue({ items: [] }),
  runTrainingPreflight: vi.fn(),
}));

vi.mock("@/api/nim", () => ({
  generateActionRequest: vi.fn().mockResolvedValue({
    request_type: "tao_setup",
    rendered_text: "TAO Setup Request\n...",
    generated_at: "2026-04-16T00:00:00Z",
    project_name: "Test",
    technical_requirements: {},
    current_environment: {},
  }),
  logActionRequestCopy: vi.fn().mockResolvedValue({ audit_event_id: "ae-1" }),
}));

vi.mock("@/pages/setup-context", () => ({
  useSetupContext: vi.fn(),
}));

installEventSourceMock();

import { fetchScaleUpGate } from "@/api/evaluation";
import { listStudentBaseModelConfigs, runTrainingPreflight } from "@/api/training";
import { generateActionRequest, logActionRequestCopy } from "@/api/nim";
import { useSetupContext } from "@/pages/setup-context";

const mockFetchGate = fetchScaleUpGate as ReturnType<typeof vi.fn>;
const mockListStudentBases = listStudentBaseModelConfigs as ReturnType<typeof vi.fn>;
const mockRunTrainingPreflight = runTrainingPreflight as ReturnType<typeof vi.fn>;
const mockGenerateActionRequest = generateActionRequest as ReturnType<typeof vi.fn>;
const mockLogActionRequestCopy = logActionRequestCopy as ReturnType<typeof vi.fn>;
const mockSetup = useSetupContext as ReturnType<typeof vi.fn>;

const writeTextMock = vi.fn(() => Promise.resolve());
Object.defineProperty(navigator, "clipboard", {
  value: { writeText: writeTextMock },
  writable: true,
  configurable: true,
});

const PROJECT = makeProjectResponse({
  project_id: "pid-1",
  name: "Test",
  active_guidance_id: "g-1",
  teacher_model_config_id: "mc-1",
  labeling_generation_preset_key: "precise",
  thinking_default_on: true,
  visual_budget_preset_key: "balanced",
  counts: {
    verified: 5,
    unlabeled: 10,
    auto_labeled: 0,
    omitted: 2,
  },
});

let qc: QueryClient;

beforeEach(() => {
  vi.clearAllMocks();
  writeTextMock.mockClear();
  qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  mockSetup.mockReturnValue({
    projectId: "pid-1",
    project: PROJECT,
    environment: {},
  });
  mockRunTrainingPreflight.mockResolvedValue({
    status: "passed",
    checks: [
      {
        check_name: "verified_train_examples",
        passed: true,
        message: "3 Verified training examples available (Test Pool excluded).",
        model_config_id: null,
        provisioning_required: false,
        remediation: null,
      },
      {
        check_name: "min_test_pool_size",
        passed: true,
        message: "Test Pool has 2 held-out evaluation examples (need 2).",
        model_config_id: null,
        provisioning_required: false,
        remediation: null,
      },
    ],
    data_summary: {
      verified_training_count: 3,
      test_pool_count: 2,
      required_test_pool_count: 2,
      auto_labeled_eligible_count: 0,
      auto_labeled_included_count: 0,
      excluded_test_pool_count: 2,
      excluded_auto_labeled_count: 0,
      usable_training_count: 3,
    },
    resolved_presets: {},
  });
});

function renderPage() {
  return render(
    <QueryClientProvider client={qc}>
      <MemoryRouter initialEntries={["/projects/pid-1/scale-up"]}>
        <Routes>
          <Route path="/projects/:projectId" element={<Outlet />}>
            <Route path="scale-up" element={<ScaleUpHubPage />} />
            <Route path="training" element={<div data-testid="training-page" />} />
          </Route>
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

// ── Tests ──────────────────────────────────────────────────────────────────

describe("ScaleUpHubPage", () => {
  it("shows 'Not ready' when gate not ready", async () => {
    mockFetchGate.mockResolvedValue({
      gate_status: "not_ready",
      criteria: [
        {
          criterion_name: "overall_exact_match",
          passed: false,
          current_value: 0.6,
          threshold: 0.8,
          message: "Model accuracy: 60% (need 80%).",
          details: null,
        },
        {
          criterion_name: "min_test_pool_size",
          passed: false,
          current_value: 5,
          threshold: 20,
          message: "Test Pool: 5 (need 20).",
          details: null,
        },
      ],
      evaluated_at: "2026-04-16T00:00:00Z",
    });
    renderPage();
    await waitFor(() => expect(screen.getByText(/Not ready/)).toBeInTheDocument());
    // The message surfaces in both the gate's next-steps and in the
    // Teacher-readiness card's Evaluation line — either is sufficient.
    expect(screen.getAllByText(/Model accuracy/).length).toBeGreaterThan(0);
  });

  it("shows 'Ready' when gate ready and enables button", async () => {
    mockFetchGate.mockResolvedValue({
      gate_status: "ready",
      criteria: [
        {
          criterion_name: "overall_exact_match",
          passed: true,
          current_value: 0.85,
          threshold: 0.8,
          message: "Passed.",
          details: null,
        },
      ],
      evaluated_at: "2026-04-16T00:00:00Z",
    });
    renderPage();
    await waitFor(() =>
      expect(screen.getByText(/Ready for Batch/)).toBeInTheDocument(),
    );
    const btn = screen.getByTestId("run-batch-labeling");
    expect(btn).not.toBeDisabled();
  });

  it("disables Run Batch Labeling when gate not ready", async () => {
    mockFetchGate.mockResolvedValue({
      gate_status: "not_ready",
      criteria: [],
      evaluated_at: "2026-04-16T00:00:00Z",
    });
    renderPage();
    await waitFor(() => {
      const btn = screen.getByTestId("run-batch-labeling");
      expect(btn).toBeDisabled();
    });
  });

  it("expands criteria details on click", async () => {
    mockFetchGate.mockResolvedValue({
      gate_status: "not_ready",
      criteria: [
        {
          criterion_name: "pool_size",
          passed: true,
          current_value: 25,
          threshold: 20,
          message: "Pool size OK.",
          details: null,
        },
      ],
      evaluated_at: "2026-04-16T00:00:00Z",
    });
    renderPage();
    await waitFor(() => screen.getByTestId("details-toggle"));
    fireEvent.click(screen.getByTestId("details-toggle"));
    expect(screen.getByTestId("criteria-list")).toBeInTheDocument();
    expect(screen.getByText(/Pool size OK/)).toBeInTheDocument();
  });

  it("keeps Train a Student navigation available when TAO is not ready", async () => {
    mockListStudentBases.mockResolvedValueOnce({
      items: [
        {
          model_config_id: "student-1",
          tao_base_experiment_id: "base-1",
          tao_base_experiment_pull_status: "pull_complete",
        },
      ],
    });
    mockRunTrainingPreflight.mockResolvedValueOnce({
      status: "failed",
      checks: [
        {
          check_name: "tao_reachable",
          passed: false,
          message: "TAO_API_BASE_URL is not configured.",
          model_config_id: null,
          provisioning_required: false,
          remediation: null,
        },
      ],
      data_summary: {
        verified_training_count: 3,
        test_pool_count: 2,
        required_test_pool_count: 2,
        auto_labeled_eligible_count: 0,
        auto_labeled_included_count: 0,
        excluded_test_pool_count: 2,
        excluded_auto_labeled_count: 0,
        usable_training_count: 3,
      },
      resolved_presets: {},
    });
    mockFetchGate.mockResolvedValue({
      gate_status: "ready",
      criteria: [],
      evaluated_at: "2026-04-16T00:00:00Z",
    });
    renderPage();
    await screen.findByText(/Student Training: infrastructure setup required/);
    const btn = screen.getByTestId("train-a-student");
    expect(btn).not.toBeDisabled();
    expect(btn).not.toHaveClass("nvidia-green-button");
    fireEvent.click(btn);
    await screen.findByTestId("training-page");
  });

  it("reports Student Training ready only after authoritative preflight", async () => {
    mockListStudentBases.mockResolvedValueOnce({
      items: [
        {
          model_config_id: "student-unavailable",
          tao_base_experiment_id: null,
          tao_base_experiment_pull_status: null,
        },
        {
          model_config_id: "student-ready",
          tao_base_experiment_id: "base-ready",
          tao_base_experiment_pull_status: "pull_complete",
        },
      ],
    });
    mockFetchGate.mockResolvedValue({
      gate_status: "ready",
      criteria: [],
      evaluated_at: "2026-04-16T00:00:00Z",
    });

    renderPage();

    expect(await screen.findByTestId("train-a-student")).not.toBeDisabled();
    await waitFor(() =>
      expect(screen.getByTestId("readiness-capability")).toHaveTextContent(
        "Student Training: ready",
      ),
    );
    expect(screen.getByTestId("readiness-capability")).toHaveTextContent(
      "3 usable training examples; 2 held out for evaluation; TAO preflight passed.",
    );
    expect(screen.getByTestId("train-a-student")).toHaveClass("nvidia-green-button");
    expect(mockRunTrainingPreflight).toHaveBeenCalledWith(
      "pid-1",
      ["student-unavailable", "student-ready"],
      true,
    );
  });

  it("keeps a model-specific method mismatch out of TAO setup", async () => {
    mockFetchGate.mockResolvedValue({
      gate_status: "ready",
      criteria: [],
      evaluated_at: "2026-04-16T00:00:00Z",
    });
    mockListStudentBases.mockResolvedValueOnce({
      items: [
        {
          model_config_id: "student-super",
          tao_base_experiment_id: "base-super",
          tao_base_experiment_pull_status: "pull_complete",
        },
      ],
    });
    mockRunTrainingPreflight.mockResolvedValueOnce({
      status: "failed",
      checks: [
        {
          check_name: "training_mode_compatible",
          passed: false,
          message: "Cosmos 3 Super requires Full-weight training.",
          model_config_id: "student-super",
          provisioning_required: false,
          remediation: "Choose Full-weight.",
        },
      ],
      data_summary: {
        verified_training_count: 180,
        test_pool_count: 120,
        required_test_pool_count: 60,
        auto_labeled_eligible_count: 0,
        auto_labeled_included_count: 0,
        excluded_test_pool_count: 120,
        excluded_auto_labeled_count: 0,
        usable_training_count: 180,
      },
      resolved_presets: {},
    });

    renderPage();
    expect(
      await screen.findByText("Student Training: ready to configure"),
    ).toBeInTheDocument();
    expect(screen.getByTestId("readiness-capability")).toHaveTextContent(
      "Select a compatible model and training method",
    );
    expect(screen.queryByTestId("tao-action-request")).not.toBeInTheDocument();
    expect(screen.getByTestId("train-a-student")).not.toBeDisabled();
  });

  it("generates, copies, and audits the TAO setup request exactly once", async () => {
    mockFetchGate.mockResolvedValue({
      gate_status: "ready",
      criteria: [],
      evaluated_at: "2026-04-16T00:00:00Z",
    });
    mockListStudentBases.mockResolvedValueOnce({
      items: [
        {
          model_config_id: "student-1",
          tao_base_experiment_id: "base-1",
          tao_base_experiment_pull_status: "pull_complete",
        },
      ],
    });
    mockRunTrainingPreflight.mockResolvedValueOnce({
      status: "failed",
      checks: [
        {
          check_name: "tao_reachable",
          passed: false,
          message: "TAO endpoint is not configured.",
          model_config_id: null,
          provisioning_required: false,
          remediation: "Configure TAO and retry.",
        },
      ],
      data_summary: {
        verified_training_count: 3,
        test_pool_count: 2,
        required_test_pool_count: 2,
        auto_labeled_eligible_count: 0,
        auto_labeled_included_count: 0,
        excluded_test_pool_count: 2,
        excluded_auto_labeled_count: 0,
        usable_training_count: 3,
      },
      resolved_presets: {},
    });

    renderPage();
    fireEvent.click(await screen.findByTestId("request-tao-setup"));
    const request = await screen.findByTestId("action-request-ready");
    expect(request).toHaveTextContent("TAO Setup Request");
    expect(mockGenerateActionRequest).toHaveBeenCalledTimes(1);

    fireEvent.click(screen.getByRole("button", { name: "Copy to Clipboard" }));
    await screen.findByRole("button", { name: "Copied" });
    expect(writeTextMock).toHaveBeenCalledWith("TAO Setup Request\n...");
    await waitFor(() =>
      expect(mockLogActionRequestCopy).toHaveBeenCalledWith("pid-1", {
        request_type: "tao_setup",
        rendered_text: "TAO Setup Request\n...",
      }),
    );
    expect(mockLogActionRequestCopy).toHaveBeenCalledTimes(1);
  });

  // ── Readiness cards ───────────────────────────────────────────────
  // Teacher / Data / Capability cards render between gate and CTAs.

  it("renders the three Readiness cards (Teacher/Data/Capability)", async () => {
    mockFetchGate.mockResolvedValue({
      gate_status: "not_ready",
      criteria: [
        {
          criterion_name: "accept_rate",
          passed: false,
          current_value: 0.6,
          threshold: 0.8,
          message: "Low accept.",
          details: null,
        },
        {
          criterion_name: "min_test_pool_size",
          passed: false,
          current_value: 5,
          threshold: 20,
          message: "Pool too small.",
          details: null,
        },
      ],
      evaluated_at: "2026-04-16T00:00:00Z",
    });
    mockRunTrainingPreflight.mockResolvedValueOnce({
      status: "failed",
      checks: [
        {
          check_name: "verified_train_examples",
          passed: false,
          message: "No Verified training examples yet. Continue labeling.",
          model_config_id: null,
          provisioning_required: false,
          remediation: null,
        },
      ],
      data_summary: {
        verified_training_count: 0,
        test_pool_count: 0,
        required_test_pool_count: 1,
        auto_labeled_eligible_count: 0,
        auto_labeled_included_count: 0,
        excluded_test_pool_count: 0,
        excluded_auto_labeled_count: 0,
        usable_training_count: 0,
      },
      resolved_presets: {},
    });
    renderPage();
    await waitFor(() =>
      expect(screen.getByTestId("readiness-cards")).toBeInTheDocument(),
    );
    expect(screen.getByTestId("readiness-teacher")).toBeInTheDocument();
    expect(screen.getByTestId("readiness-data")).toBeInTheDocument();
    expect(screen.getByTestId("readiness-capability")).toBeInTheDocument();
    // Teacher card surfaces the active model + guidance version.
    expect(screen.getByText(/nvidia\/cosmos-reason2-8b/)).toBeInTheDocument();
    expect(screen.getByText(/v3/)).toBeInTheDocument();
    // Data card surfaces the counts as labeled stat blocks.
    const verified = screen.getByTestId("data-readiness-verified");
    expect(verified).toHaveTextContent("Verified");
    expect(verified).toHaveTextContent("5");
    const unlabeled = screen.getByTestId("data-readiness-unlabeled");
    expect(unlabeled).toHaveTextContent("Unlabeled");
    expect(unlabeled).toHaveTextContent("10");
    // Omitted appears only because the fixture has omitted > 0.
    expect(screen.getByTestId("data-readiness-omitted")).toHaveTextContent("2");
  });

  // ── blocked_by criteria not in next-steps ─────────────────────────

  it("excludes blocked-by criteria from the next-steps list", async () => {
    mockFetchGate.mockResolvedValue({
      gate_status: "not_ready",
      criteria: [
        {
          criterion_name: "overall_exact_match",
          passed: false,
          current_value: 0,
          threshold: 0.8,
          message:
            "No completed evaluation run found. Run an evaluation to measure quality.",
          // The backend marks the no-eval state structurally; the UI keys
          // its pending rendering on this flag, not the message copy.
          details: { no_completed_run: true },
        },
        {
          criterion_name: "per_field_match",
          passed: false,
          current_value: 0,
          threshold: 0.8,
          message: "Depends on evaluation results.",
          details: { blocked_by: "overall_exact_match" },
        },
        {
          criterion_name: "min_per_value_f1",
          passed: false,
          current_value: 0,
          threshold: 0.8,
          message: "Depends on evaluation results.",
          details: { blocked_by: "overall_exact_match" },
        },
      ],
      evaluated_at: "2026-04-16T00:00:00Z",
    });
    renderPage();
    await waitFor(() => expect(screen.getByText(/Not ready/)).toBeInTheDocument());
    // Root criterion appears in next-steps.
    expect(
      screen.getAllByText(/Run an evaluation to measure quality/).length,
    ).toBeGreaterThan(0);
    // "Depends on evaluation results." must NOT appear in the next-steps
    // region. It will still appear in the Details expander, but we don't
    // expand it here.
    expect(
      screen.queryByText(/Depends on evaluation results\./),
    ).not.toBeInTheDocument();
  });

  // ── Actionable next steps surface in user-action priority order ────

  it("orders 'Continue labeling' before 'Run an evaluation' in the next-steps list", async () => {
    // When both min_test_pool_size and overall_exact_match are
    // failing-and-actionable, the pool-too-small message MUST surface
    // above "Run an evaluation". Backend criterion enumeration order is
    // the inverse, so the UI sort is what holds this contract.
    mockFetchGate.mockResolvedValue({
      gate_status: "not_ready",
      criteria: [
        {
          criterion_name: "overall_exact_match",
          passed: false,
          current_value: 0,
          threshold: 0.8,
          message: "No evaluation run found. Run an evaluation to measure quality.",
          details: null,
        },
        {
          criterion_name: "min_test_pool_size",
          passed: false,
          current_value: 3,
          threshold: 20,
          message: "Continue labeling — the test pool needs 17 more images.",
          details: null,
        },
      ],
      evaluated_at: "2026-04-16T00:00:00Z",
    });
    renderPage();
    await waitFor(() => expect(screen.getByText(/Not ready/)).toBeInTheDocument());
    // Both messages may surface twice (next-steps list + Data
    // readiness card / Teacher readiness card). The next-steps list
    // is the FIRST in document order because the gate card renders
    // above the readiness-cards row, so picking [0] of each is the
    // correct anchor for the ordering assertion.
    const pool = screen.getAllByText(
      /Continue labeling — the test pool needs 17 more images\./,
    )[0];
    const eval_ = screen.getAllByText(
      /No evaluation run found\. Run an evaluation to measure quality\./,
    )[0];
    // compareDocumentPosition returns Node.DOCUMENT_POSITION_FOLLOWING (4)
    // when the argument is positioned AFTER the receiver in document
    // order. We expect pool to come first, then eval.
    expect(
      pool.compareDocumentPosition(eval_) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
  });

  // ── Details disclosure groups failing rows above passing rows ──────

  it("sorts criteria fail-first inside the Details disclosure", async () => {
    mockFetchGate.mockResolvedValue({
      gate_status: "not_ready",
      criteria: [
        {
          criterion_name: "overall_exact_match",
          passed: false,
          current_value: 0,
          threshold: 0.8,
          message: "No evaluation run.",
          details: null,
        },
        {
          // Backend enumeration places accept_rate (the only PASS) in the
          // middle. After sort it must move to the end of the rendered list.
          criterion_name: "accept_rate",
          passed: true,
          current_value: 0.85,
          threshold: 0.8,
          message: "Accept rate: 85%. Passed.",
          details: null,
        },
        {
          criterion_name: "min_test_pool_size",
          passed: false,
          current_value: 5,
          threshold: 20,
          message: "Test Pool: 5 (need 20).",
          details: null,
        },
      ],
      evaluated_at: "2026-04-16T00:00:00Z",
    });
    renderPage();
    await waitFor(() => screen.getByTestId("details-toggle"));
    fireEvent.click(screen.getByTestId("details-toggle"));
    const list = await screen.findByTestId("criteria-list");
    const rows = Array.from(list.children) as HTMLElement[];
    expect(rows.length).toBe(3);
    // The sole passing row (Accept rate) MUST be the last child, regardless
    // of its position in the backend enumeration.
    expect(rows[2].textContent ?? "").toMatch(/Accept rate/);
    // Both failing rows precede it.
    expect(rows[0].textContent ?? "").not.toMatch(/Accept rate/);
    expect(rows[1].textContent ?? "").not.toMatch(/Accept rate/);
  });

  // ── Details toggle shares the verdict row in the ready state ───────

  it("places the [Details] toggle on the verdict row in the ready state", async () => {
    mockFetchGate.mockResolvedValue({
      gate_status: "ready",
      criteria: [
        {
          criterion_name: "overall_exact_match",
          passed: true,
          current_value: 0.85,
          threshold: 0.8,
          message: "Model accuracy: 85%. Passed.",
          details: null,
        },
      ],
      evaluated_at: "2026-04-16T00:00:00Z",
    });
    renderPage();
    await waitFor(() =>
      expect(screen.getByText(/Ready for Batch Labeling/)).toBeInTheDocument(),
    );
    // The verdict heading and [Details] occupy the same row. We assert
    // the testid'd row wrapper exists AND the toggle sits inside it
    // (rather than orphaned on a sibling row).
    const row = screen.getByTestId("ready-verdict-row");
    const heading = screen.getByText(/Ready for Batch Labeling/);
    const toggle = screen.getByTestId("details-toggle");
    expect(row.contains(heading)).toBe(true);
    expect(row.contains(toggle)).toBe(true);
  });

  // ── Capability card distinguishes TAO vs no-train-data ────────────

  it("reports missing training data without blocking navigation", async () => {
    // With zero Verified examples, the capability card identifies the
    // data-side blocker. Navigation stays available because Start Training
    // owns authoritative server-side validation.
    mockSetup.mockReturnValue({
      projectId: "pid-1",
      project: makeProjectResponse({
        ...PROJECT,
        counts: { ...PROJECT.counts, verified: 0 },
      }),
      environment: {},
    });
    mockListStudentBases.mockResolvedValueOnce({
      items: [
        {
          model_config_id: "student-1",
          tao_base_experiment_id: "base-1",
          tao_base_experiment_pull_status: "pull_complete",
        },
      ],
    });
    mockFetchGate.mockResolvedValue({
      gate_status: "ready",
      criteria: [],
      evaluated_at: "2026-04-16T00:00:00Z",
    });
    mockRunTrainingPreflight.mockResolvedValueOnce({
      status: "failed",
      checks: [
        {
          check_name: "verified_train_examples",
          passed: false,
          message: "No Verified training examples yet. Continue labeling.",
          model_config_id: null,
          provisioning_required: false,
          remediation: null,
        },
        {
          check_name: "min_test_pool_size",
          passed: false,
          message:
            "Test Pool has 0 of 60 required held-out evaluation examples. Continue labeling to grow the pool.",
          model_config_id: null,
          provisioning_required: false,
          remediation: null,
        },
      ],
      data_summary: {
        verified_training_count: 0,
        test_pool_count: 0,
        required_test_pool_count: 60,
        auto_labeled_eligible_count: 0,
        auto_labeled_included_count: 0,
        excluded_test_pool_count: 0,
        excluded_auto_labeled_count: 0,
        usable_training_count: 0,
      },
      resolved_presets: {},
    });
    renderPage();
    await screen.findByText(/Student Training: more Verified data needed/);
    expect(
      screen.getByText(/No Verified training examples yet\. Continue labeling\./),
    ).toBeInTheDocument();
    expect(screen.getByTestId("train-a-student")).not.toBeDisabled();
  });

  // ── Not ready headline uses warning color ─────────────────────────

  it("renders the 'Not ready' headline with warning-amber color", async () => {
    mockFetchGate.mockResolvedValue({
      gate_status: "not_ready",
      criteria: [],
      evaluated_at: "2026-04-16T00:00:00Z",
    });
    renderPage();
    await waitFor(() =>
      expect(screen.getByText(/Not ready for Batch Labeling/)).toBeInTheDocument(),
    );
    const headline = screen.getByText(/Not ready for Batch Labeling/);
    // Inline style must reference the warning token so the headline is
    // prominent instead of muted (matches the 'Ready' sibling's pattern).
    expect(headline.getAttribute("style") ?? "").toContain("warning-amber");
  });
});
