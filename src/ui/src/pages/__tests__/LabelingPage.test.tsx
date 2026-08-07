// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Unit tests for the Labeling screen — proposal states (loading, core-valid,
 * rationale-visible, schema-invalid, timeout, endpoint error, missing image),
 * route guards, auto-advance, and persistent elements.
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { installEventSourceMock } from "@/test/event-source-mock";
import { makeEnvironmentResponse, makeProjectResponse } from "@/test/fixtures";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter, Routes, Route, Outlet } from "react-router-dom";
import type { ReactNode } from "react";
import { LabelingPage } from "@/pages/LabelingPage";

// ---------------------------------------------------------------------------
// Mock API modules
// ---------------------------------------------------------------------------

const mockFetchNextReviewItem = vi.fn();
const mockCreateProposal = vi.fn();
const mockSaveLabel = vi.fn();
const mockSkipExample = vi.fn();
const mockRegenerateRationale = vi.fn();
const mockRestoreOmitted = vi.fn();

vi.mock("@/api/labeling", () => ({
  fetchNextReviewItem: (...args: unknown[]) => mockFetchNextReviewItem(...args),
  createProposal: (...args: unknown[]) => mockCreateProposal(...args),
  saveLabel: (...args: unknown[]) => mockSaveLabel(...args),
  skipExample: (...args: unknown[]) => mockSkipExample(...args),
  regenerateRationale: (...args: unknown[]) => mockRegenerateRationale(...args),
  restoreOmitted: (...args: unknown[]) => mockRestoreOmitted(...args),
  imageUrl: (pid: string, key: string) => `/v1/projects/${pid}/examples/${key}/image`,
}));

const mockFetchGuidance = vi.fn();
const mockListGuidances = vi.fn();
const mockFetchReminderStatus = vi.fn();
const mockDismissReminder = vi.fn();
vi.mock("@/api/guidance", () => ({
  fetchGuidance: (...args: unknown[]) => mockFetchGuidance(...args),
  listGuidances: (...args: unknown[]) => mockListGuidances(...args),
  fetchReminderStatus: (...args: unknown[]) => mockFetchReminderStatus(...args),
  dismissReminder: (...args: unknown[]) => mockDismissReminder(...args),
}));

const mockFetchModelConfigs = vi.fn();
const mockUpdateProject = vi.fn();
vi.mock("@/api/model-configs", () => ({
  fetchModelConfigs: (...args: unknown[]) => mockFetchModelConfigs(...args),
  updateProject: (...args: unknown[]) => mockUpdateProject(...args),
}));

const mockFetchProject = vi.fn();
const mockFetchProjectList = vi.fn();
vi.mock("@/api/projects", () => ({
  fetchProject: (...args: unknown[]) => mockFetchProject(...args),
  fetchProjectList: (...args: unknown[]) => mockFetchProjectList(...args),
}));

const mockFetchEnvironment = vi.fn();
const mockGenerateActionRequest = vi.fn();
const mockLogActionRequestCopy = vi.fn();
vi.mock("@/api/nim", () => ({
  fetchEnvironment: (...args: unknown[]) => mockFetchEnvironment(...args),
  generateActionRequest: (...args: unknown[]) => mockGenerateActionRequest(...args),
  logActionRequestCopy: (...args: unknown[]) => mockLogActionRequestCopy(...args),
}));

const mockListStudentModels = vi.fn();
vi.mock("@/api/students", () => ({
  listStudentModels: (...args: unknown[]) => mockListStudentModels(...args),
}));

// EvaluationStrip pulls trigger-status + run list via @/api/evaluation. The
// default mock returns an all-inactive trigger envelope so no banner renders
// in tests that don't opt in (the real module errors silently in JSDOM).
const mockFetchTriggerStatus = vi.fn();
const mockFetchScaleUpGate = vi.fn();
const mockListEvaluationRuns = vi.fn();
const mockCreateEvaluationRun = vi.fn();
const mockCancelEvaluationRun = vi.fn();
const mockDismissTrigger = vi.fn();
vi.mock("@/api/evaluation", () => ({
  fetchScaleUpGate: (...args: unknown[]) => mockFetchScaleUpGate(...args),
  fetchTriggerStatus: (...args: unknown[]) => mockFetchTriggerStatus(...args),
  listEvaluationRuns: (...args: unknown[]) => mockListEvaluationRuns(...args),
  createEvaluationRun: (...args: unknown[]) => mockCreateEvaluationRun(...args),
  cancelEvaluationRun: (...args: unknown[]) => mockCancelEvaluationRun(...args),
  dismissTrigger: (...args: unknown[]) => mockDismissTrigger(...args),
}));

// Mock EventSource
installEventSourceMock();

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const PROJECT = makeProjectResponse({
  name: "Damage Inspection",
  description: "Surface damage classification",
  created_at: "2026-04-14T00:00:00Z",
  updated_at: "2026-04-14T00:00:00Z",
});

const ENVIRONMENT = {
  ...makeEnvironmentResponse(),
  embedding_deployment: null,
};

const GUIDANCE = {
  guidance_id: "g-1",
  project_id: "test-pid",
  version_number: 1,
  description: "Classify damage",
  schema_fields: [
    {
      field_id: "f-1",
      field_name: "damage_type",
      type: "enum",
      role: "core",
      allowed_values: ["dent", "scratch", "crack"],
      minimum: null,
      maximum: null,
      min_length: null,
      max_length: null,
      display_order: 0,
    },
    {
      field_id: "f-2",
      field_name: "is_severe",
      type: "boolean",
      role: "core",
      allowed_values: null,
      minimum: null,
      maximum: null,
      min_length: null,
      max_length: null,
      display_order: 1,
    },
    {
      field_id: "f-rat",
      field_name: "rationale_note",
      type: "string",
      role: "aux",
      allowed_values: null,
      minimum: null,
      maximum: null,
      min_length: null,
      max_length: null,
      display_order: 0,
    },
    {
      field_id: "f-3",
      field_name: "confidence",
      type: "integer",
      role: "aux",
      allowed_values: null,
      minimum: 1,
      maximum: 5,
      min_length: null,
      max_length: null,
      display_order: 1,
    },
  ],
  rules: "",
  derived_json_schema: {},
  generation_order: ["rationale_note", "confidence", "damage_type", "is_severe"],
  schema_hash: "abc123",
  created_at: "2026-04-14T00:00:00Z",
};

const MODEL_CONFIGS = {
  items: [
    {
      model_config_id: "mc-teacher",
      project_id: "test-pid",
      endpoint_id: "ep-1",
      model_name: "nvidia/cosmos-reason2-8b",
      context_window_tokens: 256000,
      eligible_roles: ["teacher"],
      supports_image_input: true,
      structured_generation_support: "unknown" as const,
      thinking_toggle_mode: "qwen_enable_thinking",
      thinking_toggle_support: "supported" as const,
      visual_budget_mode: "mm_processor_size",
      visual_budget_support: "unknown" as const,
      model_quantization: null,
      nim_model_profile: null,
      nim_profile_metadata: null,
      local_deploy_metadata: null,
      created_at: "2026-04-14T00:00:00Z",
    },
  ],
  next_cursor: null,
};

const MODEL_CONFIG_MINIMAL = {
  model_config_id: "mc-mistral",
  project_id: "test-pid",
  endpoint_id: "ep-1",
  model_name: "mistralai/mistral-large-3-675b-instruct-2512",
  context_window_tokens: 262144,
  eligible_roles: ["teacher"],
  supports_image_input: true,
  structured_generation_support: "unknown" as const,
  thinking_toggle_mode: "none",
  thinking_toggle_support: "unsupported" as const,
  visual_budget_mode: "none",
  visual_budget_support: "unsupported" as const,
  model_quantization: null,
  nim_model_profile: null,
  nim_profile_metadata: null,
  local_deploy_metadata: null,
  created_at: "2026-04-14T00:00:00Z",
};

const GUIDANCE_LIST = {
  items: [GUIDANCE],
  next_cursor: null,
};

const PROJECT_LIST = {
  items: [
    {
      project_id: "test-pid",
      name: "Damage Inspection",
      description: null,
      created_at: "2026-04-14T00:00:00Z",
      updated_at: "2026-04-14T00:00:00Z",
      counts: {
        verified: 42,
        unlabeled: 103,
        auto_labeled: 0,
        omitted: 5,
        pending_relabel: 0,
      },
    },
  ],
  next_cursor: null,
};

const REVIEW_NEXT = {
  example_key: "img_001",
  example_state: "Unlabeled",
  has_existing_label: false,
  selection_mode: "phash_diverse",
  queue_empty: false,
  storage_ref: "/data/img_001.jpg",
  prior_verified_label_ref: null,
};

const REVIEW_NEXT_EMPTY = {
  example_key: null,
  example_state: null,
  has_existing_label: false,
  selection_mode: "phash_diverse",
  queue_empty: true,
  storage_ref: null,
  prior_verified_label_ref: null,
};

// Backend-owned schema refinement reminder state. Default: nothing
// active — tests that exercise the reminder banner override active_reminder.
const REMINDER_STATUS_NONE = {
  active_reminder: null,
  verified_count: 42,
  threshold_1: 10,
  threshold_2: 35,
  dismissed_count: 0,
};

const PROPOSAL_SUCCESS = {
  inference_invocation_id: "inv-1",
  example_key: "img_001",
  proposal_json: {
    rationale_note: "Visible dent on front-left corner.",
    damage_type: "dent",
    is_severe: false,
    confidence: 3,
  },
  schema_valid_core: true,
  validation_errors_core: [],
  validation_errors_aux: [],
  invocation_status: "success",
  latency_ms_end_to_end: 1500,
  used_existing_label: false,
};

const PROPOSAL_INVALID = {
  ...PROPOSAL_SUCCESS,
  invocation_status: "schema_invalid",
  schema_valid_core: false,
  validation_errors_core: [
    'damage_type: value "break" not in allowed values',
    "is_severe: expected boolean, got string",
  ],
  validation_errors_aux: ["confidence: out of range"],
};

const PROPOSAL_TIMEOUT = {
  ...PROPOSAL_SUCCESS,
  invocation_status: "timeout",
  proposal_json: null,
  schema_valid_core: false,
};

const PROPOSAL_ENDPOINT_ERROR = {
  ...PROPOSAL_SUCCESS,
  invocation_status: "endpoint_error",
  proposal_json: null,
  schema_valid_core: false,
};

// ---------------------------------------------------------------------------
// Wrapper — renders inside ProjectSetupLayout context
// ---------------------------------------------------------------------------

function createWrapper(projectOverrides?: Record<string, unknown>) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false, staleTime: 0 } },
  });

  const projectData = { ...PROJECT, ...projectOverrides };

  function MockSetupLayout() {
    return (
      <Outlet
        context={{
          projectId: "test-pid",
          project: projectData,
          environment: ENVIRONMENT,
        }}
      />
    );
  }

  function Wrapper({ children }: { children: ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>
        <MemoryRouter initialEntries={["/projects/test-pid/labeling"]}>
          <Routes>
            <Route path="/projects/:projectId" element={<MockSetupLayout />}>
              <Route path="labeling" element={children} />
              <Route
                path="create-guidance"
                element={<div data-testid="create-guidance-page" />}
              />
              <Route path="ready" element={<div data-testid="image-ingest-page" />} />
              <Route path="compare" element={<div data-testid="compare-page" />} />
            </Route>
          </Routes>
        </MemoryRouter>
      </QueryClientProvider>
    );
  }

  return { Wrapper, queryClient };
}

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------

beforeEach(() => {
  // Reset queued `mockResolvedValueOnce` implementations as well as call
  // history. Several proposal-state tests intentionally enqueue multiple
  // review cycles; clearing calls alone lets an unused response leak into the
  // next test and makes route/auto-advance behavior depend on execution timing.
  vi.resetAllMocks();
  localStorage.clear();

  // Default mocks for shared queries
  mockFetchProject.mockResolvedValue(PROJECT);
  mockFetchEnvironment.mockResolvedValue(ENVIRONMENT);
  mockFetchGuidance.mockResolvedValue(GUIDANCE);
  mockFetchModelConfigs.mockResolvedValue(MODEL_CONFIGS);
  mockFetchProjectList.mockResolvedValue(PROJECT_LIST);
  mockListGuidances.mockResolvedValue(GUIDANCE_LIST);
  mockUpdateProject.mockResolvedValue(PROJECT);
  mockFetchReminderStatus.mockResolvedValue(REMINDER_STATUS_NONE);
  mockDismissReminder.mockResolvedValue({ dismissed_count: 1 });
  mockRestoreOmitted.mockResolvedValue({ restored_count: 0 });
  mockListStudentModels.mockResolvedValue({ items: [], next_cursor: null });

  // Default: fetchNextReviewItem returns REVIEW_NEXT unless overridden
  mockFetchNextReviewItem.mockResolvedValue(REVIEW_NEXT);
  // Default: createProposal returns success unless overridden
  mockCreateProposal.mockResolvedValue(PROPOSAL_SUCCESS);

  // Default: evaluation API returns an all-inactive trigger envelope + no
  // runs. Tests that want banner behavior override with mockResolvedValueOnce.
  mockFetchTriggerStatus.mockResolvedValue({
    auto_evaluate_enabled: false,
    first_pool_threshold: {
      is_active: false,
      dismissed: false,
      message: "",
      context: null,
    },
    configuration_change: {
      is_active: false,
      dismissed: false,
      message: "",
      context: null,
    },
    icl_growth: { is_active: false, dismissed: false, message: "", context: null },
    updated_at: "2026-04-23T00:00:00Z",
  });
  mockFetchScaleUpGate.mockResolvedValue({
    gate_status: "not_ready",
    criteria: [
      {
        criterion_name: "min_test_pool_size",
        passed: false,
        current_value: 7,
        threshold: 20,
        message: "Test Pool: 7 (need 20).",
        details: { pool_target: 20 },
      },
    ],
    evaluated_at: "2026-04-23T00:00:00Z",
  });
  mockListEvaluationRuns.mockResolvedValue({ items: [], next_cursor: null });
  mockCreateEvaluationRun.mockResolvedValue({ run_id: "new-run", status: "queued" });
  mockDismissTrigger.mockResolvedValue({
    trigger_type: "first_pool_threshold",
    dismissed: true,
  });
});

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("LabelingPage", () => {
  // ── Loading state ─────────────────────────────────────────────────────

  it("shows loading spinners while fetching next example and proposal", async () => {
    // fetchNextReviewItem resolves, then createProposal hangs
    mockFetchNextReviewItem.mockResolvedValueOnce(REVIEW_NEXT);
    mockCreateProposal.mockImplementation(() => new Promise(() => {})); // never resolves

    const { Wrapper } = createWrapper();
    render(<LabelingPage />, { wrapper: Wrapper });

    // Top bar and status bar should render immediately
    await waitFor(() => {
      expect(screen.getByTestId("labeling-page")).toBeInTheDocument();
    });

    expect(screen.getByTestId("labeling-top-bar")).toBeInTheDocument();
    expect(screen.getByTestId("labeling-status-bar")).toBeInTheDocument();

    // Proposal panel shows loading
    await waitFor(() => {
      expect(screen.getByTestId("proposal-loading")).toBeInTheDocument();
    });
  });

  // ── Core Valid (proposal success) ─────────────────────────────────────

  it("displays proposal with Core fields, Aux fields, rationale hidden", async () => {
    mockFetchNextReviewItem.mockResolvedValueOnce(REVIEW_NEXT);
    mockCreateProposal.mockResolvedValueOnce(PROPOSAL_SUCCESS);

    const { Wrapper } = createWrapper();
    render(<LabelingPage />, { wrapper: Wrapper });

    // Wait for the proposal form to appear
    await waitFor(() => {
      expect(screen.getByTestId("proposal-form")).toBeInTheDocument();
    });

    // Core fields visible
    expect(screen.getByTestId("field-damage_type")).toBeInTheDocument();
    expect(screen.getByTestId("field-is_severe")).toBeInTheDocument();

    // Aux field visible (confidence)
    expect(screen.getByTestId("field-confidence")).toBeInTheDocument();

    // Rationale note HIDDEN (anti-anchoring default is true)
    expect(screen.queryByTestId("rationale-note-display")).not.toBeInTheDocument();

    // Action buttons visible
    expect(screen.getByTestId("save-btn")).toBeInTheDocument();
    expect(screen.getByTestId("skip-btn")).toBeInTheDocument();
    expect(screen.getByTestId("retry-btn")).toBeInTheDocument();

    // Status bar shows counts
    expect(screen.getByTestId("count-verified")).toHaveTextContent("Verified: 42");
    expect(screen.getByTestId("count-unlabeled")).toHaveTextContent("Unlabeled: 103");
    expect(screen.getByTestId("count-omitted")).toHaveTextContent("Omitted: 5");

    // Top bar shows teacher model name
    expect(screen.getByTestId("labeling-top-bar")).toHaveTextContent(
      "nvidia/cosmos-reason2-8b",
    );

    // Image panel visible
    expect(screen.getByTestId("image-panel")).toBeInTheDocument();

    // Footer with Add Images
    expect(screen.getByTestId("labeling-footer")).toBeInTheDocument();
    expect(screen.getByTestId("add-images-btn")).toBeInTheDocument();
    expect(screen.queryByTestId("models-results-indicator")).not.toBeInTheDocument();
  });

  it("offers Models & Results beside Scale-Up when a Student exists", async () => {
    mockListStudentModels.mockResolvedValue({
      items: [{ student_model_id: "student-1" }],
      next_cursor: null,
    });

    const { Wrapper } = createWrapper();
    render(<LabelingPage />, { wrapper: Wrapper });

    const modelsButton = await screen.findByTestId("models-results-indicator");
    expect(modelsButton).toHaveTextContent("Models & Results");
    expect(screen.getByTestId("scaleup-indicator")).toBeInTheDocument();

    fireEvent.click(modelsButton);
    expect(await screen.findByTestId("compare-page")).toBeInTheDocument();
  });

  // ── Rationale Visible ─────────────────────────────────────────────────

  it("shows rationale_note when rationale_anti_anchoring is false", async () => {
    mockFetchNextReviewItem.mockResolvedValueOnce(REVIEW_NEXT);
    mockCreateProposal.mockResolvedValueOnce(PROPOSAL_SUCCESS);

    const { Wrapper } = createWrapper({ rationale_anti_anchoring: false });
    render(<LabelingPage />, { wrapper: Wrapper });

    await waitFor(() => {
      expect(screen.getByTestId("proposal-form")).toBeInTheDocument();
    });

    // Rationale note IS visible
    const rationaleEl = screen.getByTestId("rationale-note-display");
    expect(rationaleEl).toBeInTheDocument();
    expect(rationaleEl).toHaveTextContent(/Visible dent on front-left corner/);
  });

  // ── Failure: Schema-Invalid ───────────────────────────────────────────

  it("shows schema-invalid failure with Core errors listed", async () => {
    mockFetchNextReviewItem.mockResolvedValueOnce(REVIEW_NEXT);
    mockCreateProposal.mockResolvedValueOnce(PROPOSAL_INVALID);

    const { Wrapper } = createWrapper();
    render(<LabelingPage />, { wrapper: Wrapper });

    await waitFor(() => {
      expect(screen.getByTestId("proposal-failure-schema")).toBeInTheDocument();
    });

    // Error message and Core errors listed
    expect(screen.getByText(/Schema-invalid/)).toBeInTheDocument();
    expect(
      screen.getByText(/damage_type: value "break" not in allowed values/),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/is_severe: expected boolean, got string/),
    ).toBeInTheDocument();

    // Action buttons: Skip and Retry remain actionable; Save is disabled
    // because there is no valid proposal JSON to save.
    expect(screen.getByTestId("skip-btn")).toBeInTheDocument();
    expect(screen.getByTestId("retry-btn")).toBeInTheDocument();
    expect(screen.getByTestId("save-btn")).toBeDisabled();
  });

  // ── Failure: Timeout ──────────────────────────────────────────────────

  it("shows timeout message with helper text", async () => {
    mockFetchNextReviewItem.mockResolvedValueOnce(REVIEW_NEXT);
    mockCreateProposal.mockResolvedValueOnce(PROPOSAL_TIMEOUT);

    const { Wrapper } = createWrapper();
    render(<LabelingPage />, { wrapper: Wrapper });

    await waitFor(() => {
      expect(screen.getByTestId("proposal-failure-timeout")).toBeInTheDocument();
    });

    expect(screen.getByText(/Timeout/)).toBeInTheDocument();
    expect(screen.getByText(/did not respond within the deadline/)).toBeInTheDocument();
    expect(screen.getByText(/check with your administrator/)).toBeInTheDocument();
    // Skip/Retry remain; Save disabled on any failure mode.
    expect(screen.getByTestId("skip-btn")).toBeInTheDocument();
    expect(screen.getByTestId("retry-btn")).toBeInTheDocument();
    expect(screen.getByTestId("save-btn")).toBeDisabled();
  });

  // ── Failure: Endpoint Error ───────────────────────────────────────────

  it("shows endpoint error with Report NIM Issue button", async () => {
    mockFetchNextReviewItem.mockResolvedValueOnce(REVIEW_NEXT);
    mockCreateProposal.mockResolvedValueOnce(PROPOSAL_ENDPOINT_ERROR);

    const { Wrapper } = createWrapper();
    render(<LabelingPage />, { wrapper: Wrapper });

    await waitFor(() => {
      expect(screen.getByTestId("proposal-failure-endpoint")).toBeInTheDocument();
    });

    expect(screen.getByText(/Endpoint error/)).toBeInTheDocument();
    expect(screen.getByText(/could not reach the NIM endpoint/)).toBeInTheDocument();
    expect(screen.getByTestId("report-nim-issue-btn")).toBeInTheDocument();
    // Skip/Retry remain; Save disabled on any failure mode.
    expect(screen.getByTestId("skip-btn")).toBeInTheDocument();
    expect(screen.getByTestId("retry-btn")).toBeInTheDocument();
    expect(screen.getByTestId("save-btn")).toBeDisabled();
  });

  // The retry panel replaces the action buttons in the right column — only
  // the retry panel content shows, not the failure card above it. So when
  // the user clicks Retry while a failure is displayed, the failure card
  // must hide.
  it("hides failure card when retry panel opens from a failure state", async () => {
    mockFetchNextReviewItem.mockResolvedValueOnce(REVIEW_NEXT);
    mockCreateProposal.mockResolvedValueOnce(PROPOSAL_ENDPOINT_ERROR);

    const { Wrapper } = createWrapper();
    const user = userEvent.setup();
    render(<LabelingPage />, { wrapper: Wrapper });

    await waitFor(() => {
      expect(screen.getByTestId("proposal-failure-endpoint")).toBeInTheDocument();
    });

    // Open the retry panel from the failure view
    await user.click(screen.getByTestId("retry-btn"));

    await waitFor(() => {
      expect(screen.getByTestId("retry-panel")).toBeInTheDocument();
    });

    // Failure card must NOT co-render with the retry panel
    expect(screen.queryByTestId("proposal-failure-endpoint")).not.toBeInTheDocument();
    expect(screen.queryByTestId("report-nim-issue-btn")).not.toBeInTheDocument();
  });

  // ── Missing Image ─────────────────────────────────────────────────────

  it("offers image recovery while keeping Skip as the only label action", async () => {
    mockFetchNextReviewItem.mockResolvedValueOnce(REVIEW_NEXT);
    mockCreateProposal.mockResolvedValueOnce(PROPOSAL_SUCCESS);

    const { Wrapper } = createWrapper();
    render(<LabelingPage />, { wrapper: Wrapper });

    await waitFor(() => {
      expect(screen.getByTestId("image-panel")).toBeInTheDocument();
    });

    // Simulate image load error
    const img = screen.getByTestId("labeling-image");
    fireEvent.error(img);

    await waitFor(() => {
      expect(screen.getByTestId("image-panel-missing")).toBeInTheDocument();
    });

    // Missing image state elements
    expect(
      screen.getByText(/Image not found at original location/),
    ).toBeInTheDocument();
    expect(screen.getByTestId("retry-image-btn")).toBeInTheDocument();
    expect(screen.getByTestId("report-missing-files-btn")).toBeInTheDocument();

    // Only Skip is available (no Save or Retry)
    const actionsContainer = screen.getByTestId("labeling-actions-missing");
    expect(
      actionsContainer.querySelector('[data-testid="skip-btn"]'),
    ).toBeInTheDocument();

    // Save and Retry should NOT be in the missing-image action bar
    expect(screen.queryByTestId("labeling-actions")).not.toBeInTheDocument();

    await userEvent.setup().click(screen.getByTestId("retry-image-btn"));
    const retriedImage = await screen.findByTestId("labeling-image");
    expect(retriedImage).toHaveAttribute("src", expect.stringContaining("?retry=1"));
    fireEvent.load(retriedImage);

    await waitFor(() => {
      expect(screen.queryByTestId("image-panel-missing")).not.toBeInTheDocument();
      expect(screen.getByTestId("proposal-form")).toBeInTheDocument();
    });
  });

  // ── Auto-Advance: Save ────────────────────────────────────────────────

  it("auto-advances to next image after Save", async () => {
    mockSaveLabel.mockResolvedValueOnce({
      example_key: "img_001",
      label_status: "verified",
      verified_outcome: "Accept",
      verified_at: "2026-04-14T12:00:00Z",
      edited_core_fields: [],
      edited_aux_fields: [],
      pool_assignment: null,
    });

    const { Wrapper, queryClient } = createWrapper();
    const invalidateSpy = vi.spyOn(queryClient, "invalidateQueries");
    const user = userEvent.setup();
    render(<LabelingPage />, { wrapper: Wrapper });

    // Wait for first proposal
    await waitFor(() => {
      expect(screen.getByTestId("proposal-form")).toBeInTheDocument();
    });

    const callCountBefore = mockFetchNextReviewItem.mock.calls.length;

    // Click Save
    await user.click(screen.getByTestId("save-btn"));

    // Assert saveLabel was called
    await waitFor(() => {
      expect(mockSaveLabel).toHaveBeenCalledWith(
        "test-pid",
        expect.objectContaining({
          example_key: "img_001",
          inference_invocation_id: "inv-1",
          rationale_source: "teacher_proposal",
        }),
      );
    });

    // Assert auto-advance: fetchNextReviewItem called at least once more
    await waitFor(() => {
      expect(mockFetchNextReviewItem.mock.calls.length).toBeGreaterThan(
        callCountBefore,
      );
    });

    const invalidatedKeys = invalidateSpy.mock.calls
      .map((call) => call[0]?.queryKey as readonly unknown[] | undefined)
      .filter((key): key is readonly unknown[] => Array.isArray(key));
    expect(invalidatedKeys).toEqual(
      expect.arrayContaining([["evaluations", "test-pid", "gate"]]),
    );
  });

  it("shows context selection instead of a false cold-start state between proposals", async () => {
    mockSaveLabel.mockResolvedValueOnce({
      example_key: "img_001",
      label_status: "verified",
      verified_outcome: "Edit",
      verified_at: "2026-04-14T12:00:00Z",
      edited_core_fields: ["damage_type"],
      edited_aux_fields: [],
      pool_assignment: null,
    });
    mockCreateProposal
      .mockResolvedValueOnce(PROPOSAL_SUCCESS)
      .mockImplementationOnce(() => new Promise(() => {}));

    const { Wrapper } = createWrapper();
    const user = userEvent.setup();
    render(<LabelingPage />, { wrapper: Wrapper });

    await waitFor(() => {
      expect(screen.getByTestId("proposal-form")).toBeInTheDocument();
    });
    await user.click(screen.getByTestId("save-btn"));

    await waitFor(() => {
      expect(screen.getByTestId("icl-chip-pending")).toHaveTextContent(
        "ICL: selecting context…",
      );
    });
    expect(screen.queryByText("ICL: no edits yet")).not.toBeInTheDocument();
  });

  // ── Auto-Advance: Skip ────────────────────────────────────────────────

  it("auto-advances to next image after Skip", async () => {
    mockFetchNextReviewItem.mockResolvedValueOnce(REVIEW_NEXT);
    mockCreateProposal.mockResolvedValueOnce(PROPOSAL_SUCCESS);
    mockSkipExample.mockResolvedValueOnce({
      example_key: "img_001",
      state: "Omitted",
      omitted_at: "2026-04-14T12:00:00Z",
    });

    // Second cycle
    mockFetchNextReviewItem.mockResolvedValueOnce({
      ...REVIEW_NEXT,
      example_key: "img_002",
    });
    mockCreateProposal.mockResolvedValueOnce({
      ...PROPOSAL_SUCCESS,
      example_key: "img_002",
    });

    const { Wrapper } = createWrapper();
    const user = userEvent.setup();
    render(<LabelingPage />, { wrapper: Wrapper });

    await waitFor(() => {
      expect(screen.getByTestId("proposal-form")).toBeInTheDocument();
    });

    await user.click(screen.getByTestId("skip-btn"));

    await waitFor(() => {
      expect(mockSkipExample).toHaveBeenCalledWith("test-pid", "img_001");
    });

    await waitFor(() => {
      expect(mockFetchNextReviewItem).toHaveBeenCalledTimes(2);
    });
  });

  // ── Route Guard: No Guidance ──────────────────────────────────────────

  it("redirects to create-guidance when active_guidance_id is null", async () => {
    const { Wrapper } = createWrapper({ active_guidance_id: null });
    render(<LabelingPage />, { wrapper: Wrapper });

    await waitFor(
      () => {
        expect(screen.getByTestId("create-guidance-page")).toBeInTheDocument();
      },
      { timeout: 5000 },
    );
  });

  // ── Route Guard: No Examples (queue empty with 0 total) ───────────────

  it("redirects to image ingestion when queue is empty and no examples exist", async () => {
    mockFetchNextReviewItem.mockResolvedValue(REVIEW_NEXT_EMPTY);
    mockFetchProjectList.mockResolvedValue({
      items: [
        {
          ...PROJECT_LIST.items[0],
          counts: {
            verified: 0,
            unlabeled: 0,
            auto_labeled: 0,
            omitted: 0,
            pending_relabel: 0,
          },
        },
      ],
      next_cursor: null,
    });

    const { Wrapper } = createWrapper();
    render(<LabelingPage />, { wrapper: Wrapper });

    await waitFor(
      () => {
        expect(screen.getByTestId("image-ingest-page")).toBeInTheDocument();
      },
      { timeout: 5000 },
    );
  });

  // ── Retry calls createProposal with retry_of ──────────────────────────

  it("Retry opens panel, then re-fetches proposal with retry_of_inference_invocation_id", async () => {
    const { Wrapper } = createWrapper();
    const user = userEvent.setup();
    render(<LabelingPage />, { wrapper: Wrapper });

    await waitFor(() => {
      expect(screen.getByTestId("retry-btn")).toBeInTheDocument();
    });

    // Click Retry → opens the retry panel
    await user.click(screen.getByTestId("retry-btn"));
    await waitFor(() => {
      expect(screen.getByTestId("retry-panel")).toBeInTheDocument();
    });

    // Click Retry button in the panel to confirm
    await user.click(screen.getByTestId("retry-confirm-btn"));

    await waitFor(() => {
      expect(mockCreateProposal).toHaveBeenCalledTimes(2);
      expect(mockCreateProposal).toHaveBeenLastCalledWith(
        "test-pid",
        expect.objectContaining({
          example_key: "img_001",
          retry_of_inference_invocation_id: "inv-1",
        }),
      );
    });
  });

  // =========================================================================
  // Edit flow and rationale panel
  // =========================================================================

  describe("Edit flow and rationale panel", () => {
    // Helper: render the page with a successful proposal and wait for the form
    async function renderWithProposal(projectOverrides?: Record<string, unknown>) {
      const { Wrapper } = createWrapper(projectOverrides);
      const user = userEvent.setup();
      render(<LabelingPage />, { wrapper: Wrapper });

      await waitFor(() => {
        expect(screen.getByTestId("proposal-form")).toBeInTheDocument();
      });

      return { user };
    }

    // ── Panel expansion on field edit ──────────────────────────────────

    it("expands rationale panel when a field is modified, Save disabled", async () => {
      const { user } = await renderWithProposal();

      // Initially: no rationale panel, Save enabled
      expect(screen.queryByTestId("rationale-panel")).not.toBeInTheDocument();
      expect(screen.getByTestId("save-btn")).not.toBeDisabled();

      // Modify a Core field
      const select = screen.getByTestId("field-input-damage_type");
      await user.selectOptions(select, "scratch");

      // Rationale panel appears
      await waitFor(() => {
        expect(screen.getByTestId("rationale-panel")).toBeInTheDocument();
      });

      // Save disabled, helper text visible
      expect(screen.getByTestId("save-btn")).toBeDisabled();
      expect(screen.getByTestId("rationale-helper")).toHaveTextContent(
        /Update the rationale/,
      );

      // Textarea pre-filled with original rationale
      const textarea = screen.getByTestId("rationale-textarea") as HTMLTextAreaElement;
      expect(textarea.value).toBe("Visible dent on front-left corner.");

      // [Generate AI Rationale] button visible
      expect(screen.getByTestId("generate-ai-rationale-btn")).toBeInTheDocument();

      // [Reset] button visible
      expect(screen.getByTestId("reset-btn")).toBeInTheDocument();
    });

    it("bypasses rationale review when Guidance has rationale notes disabled", async () => {
      const disabledGuidance = {
        ...GUIDANCE,
        schema_fields: GUIDANCE.schema_fields.filter(
          (field) => field.field_name !== "rationale_note",
        ),
        generation_order: ["confidence", "damage_type", "is_severe"],
      };
      mockFetchGuidance.mockResolvedValue(disabledGuidance);
      mockCreateProposal.mockResolvedValue({
        ...PROPOSAL_SUCCESS,
        proposal_json: {
          damage_type: "dent",
          is_severe: false,
          confidence: 3,
        },
      });
      mockSaveLabel.mockResolvedValue({
        example_key: "img_001",
        label_status: "verified",
        verified_outcome: "Edit",
        verified_at: "2026-04-14T12:00:00Z",
        edited_core_fields: ["damage_type"],
        edited_aux_fields: [],
        pool_assignment: null,
      });

      const { user } = await renderWithProposal();
      await user.selectOptions(
        screen.getByTestId("field-input-damage_type"),
        "scratch",
      );

      expect(screen.queryByTestId("rationale-panel")).not.toBeInTheDocument();
      expect(screen.queryByTestId("rationale-note-display")).not.toBeInTheDocument();
      expect(screen.getByTestId("save-btn")).not.toBeDisabled();

      await user.click(screen.getByTestId("save-btn"));
      await waitFor(() => {
        expect(mockSaveLabel).toHaveBeenCalled();
      });
      const payload = mockSaveLabel.mock.calls[0][1];
      expect(payload.label_json).toEqual({
        damage_type: "scratch",
        is_severe: false,
        confidence: 3,
      });
      expect(payload).not.toHaveProperty("rationale_source");
      expect(payload).not.toHaveProperty("rationale_regeneration_invocation_id");
      expect(mockRegenerateRationale).not.toHaveBeenCalled();
    });

    // ── Rationale regeneration loading ─────────────────────────────────

    it("shows loading state during rationale regeneration", async () => {
      // Make regeneration hang
      mockRegenerateRationale.mockImplementation(() => new Promise(() => {}));

      const { user } = await renderWithProposal();

      // Modify a field to show the rationale panel
      await user.selectOptions(
        screen.getByTestId("field-input-damage_type"),
        "scratch",
      );

      await waitFor(() => {
        expect(screen.getByTestId("generate-ai-rationale-btn")).toBeInTheDocument();
      });

      // Click Generate AI Rationale
      await user.click(screen.getByTestId("generate-ai-rationale-btn"));

      // Loading state: spinner visible, no action buttons
      await waitFor(() => {
        expect(screen.getByTestId("rationale-regenerating")).toBeInTheDocument();
      });
      expect(screen.queryByTestId("generate-ai-rationale-btn")).not.toBeInTheDocument();
      expect(screen.queryByTestId("approve-ai-rationale-btn")).not.toBeInTheDocument();
      expect(screen.getByTestId("save-btn")).toBeDisabled();
    });

    // ── AI-regenerated, review required ────────────────────────────────

    it("shows AI-regenerated text with Approve button, Save still disabled", async () => {
      mockRegenerateRationale.mockResolvedValueOnce({
        inference_invocation_id: "regen-inv-1",
        rationale_note: "AI-generated rationale for corrected label.",
        invocation_status: "success",
      });

      const { user } = await renderWithProposal();

      // Modify field → show panel → click Generate
      await user.selectOptions(
        screen.getByTestId("field-input-damage_type"),
        "scratch",
      );
      await waitFor(() => {
        expect(screen.getByTestId("generate-ai-rationale-btn")).toBeInTheDocument();
      });
      await user.click(screen.getByTestId("generate-ai-rationale-btn"));

      await waitFor(() => {
        expect(mockRegenerateRationale).toHaveBeenCalledWith("test-pid", "img_001", {});
      });

      // After regen completes: AI text in textarea, Save still disabled
      await waitFor(() => {
        expect(screen.getByTestId("rationale-textarea")).toBeInTheDocument();
      });

      const textarea = screen.getByTestId("rationale-textarea") as HTMLTextAreaElement;
      expect(textarea.value).toBe("AI-generated rationale for corrected label.");
      expect(screen.getByTestId("save-btn")).toBeDisabled();

      // [Approve AI Rationale] button visible
      expect(screen.getByTestId("approve-ai-rationale-btn")).toBeInTheDocument();

      // State badge shows AI-generated
      expect(screen.getByTestId("rationale-state-badge")).toHaveTextContent(
        /AI-generated, review required/,
      );
    });

    // ── Edited state — Save enabled ────────────────────────────────────

    it("enables Save when rationale is meaningfully edited", async () => {
      const { user } = await renderWithProposal();

      // Modify a field
      await user.selectOptions(
        screen.getByTestId("field-input-damage_type"),
        "scratch",
      );
      await waitFor(() => {
        expect(screen.getByTestId("rationale-textarea")).toBeInTheDocument();
      });

      // Edit the rationale text meaningfully
      const textarea = screen.getByTestId("rationale-textarea");
      await user.clear(textarea);
      await user.type(textarea, "Scratch damage visible on lower edge.");

      // Save should be enabled, state badge shows "Edited"
      await waitFor(() => {
        expect(screen.getByTestId("save-btn")).not.toBeDisabled();
      });
      expect(screen.getByTestId("rationale-state-badge")).toHaveTextContent("Edited");
    });

    // ── Approved state — Save enabled ──────────────────────────────────

    it("enables Save when AI rationale is approved", async () => {
      mockRegenerateRationale.mockResolvedValueOnce({
        inference_invocation_id: "regen-inv-1",
        rationale_note: "AI-generated rationale.",
        invocation_status: "success",
      });

      const { user } = await renderWithProposal();

      // Modify → Generate → Approve
      await user.selectOptions(
        screen.getByTestId("field-input-damage_type"),
        "scratch",
      );
      await waitFor(() => {
        expect(screen.getByTestId("generate-ai-rationale-btn")).toBeInTheDocument();
      });
      await user.click(screen.getByTestId("generate-ai-rationale-btn"));
      await waitFor(() => {
        expect(screen.getByTestId("approve-ai-rationale-btn")).toBeInTheDocument();
      });
      await user.click(screen.getByTestId("approve-ai-rationale-btn"));

      // Save enabled, state badge shows "Approved"
      await waitFor(() => {
        expect(screen.getByTestId("save-btn")).not.toBeDisabled();
      });
      expect(screen.getByTestId("rationale-state-badge")).toHaveTextContent("Approved");
    });

    // ── Whitespace-only rule ───────────────────────────────────────────

    it("whitespace-only rationale changes do not enable Save", async () => {
      const { user } = await renderWithProposal();

      // Modify a field to trigger rationale panel
      await user.selectOptions(
        screen.getByTestId("field-input-damage_type"),
        "scratch",
      );
      await waitFor(() => {
        expect(screen.getByTestId("rationale-textarea")).toBeInTheDocument();
      });

      // Add whitespace only to the rationale
      const textarea = screen.getByTestId("rationale-textarea");
      await user.type(textarea, "   ");

      // Save should still be disabled
      expect(screen.getByTestId("save-btn")).toBeDisabled();
    });

    // ── Reset restores all fields ──────────────────────────────────────

    it("Reset restores all fields, collapses rationale panel, re-enables Save", async () => {
      const { user } = await renderWithProposal();

      // Modify a field → rationale panel appears
      await user.selectOptions(
        screen.getByTestId("field-input-damage_type"),
        "scratch",
      );
      await waitFor(() => {
        expect(screen.getByTestId("rationale-panel")).toBeInTheDocument();
        expect(screen.getByTestId("reset-btn")).toBeInTheDocument();
      });

      // Click Reset
      await user.click(screen.getByTestId("reset-btn"));

      // Rationale panel collapses
      await waitFor(() => {
        expect(screen.queryByTestId("rationale-panel")).not.toBeInTheDocument();
      });

      // Save re-enabled (one-click)
      expect(screen.getByTestId("save-btn")).not.toBeDisabled();

      // Reset button gone (no dirty fields)
      expect(screen.queryByTestId("reset-btn")).not.toBeInTheDocument();

      // Field restored to original value
      const select = screen.getByTestId("field-input-damage_type") as HTMLSelectElement;
      expect(select.value).toBe("dent");
    });

    // ── Save with sme_edited rationale ─────────────────────────────────

    it("saves with rationale_source sme_edited when rationale is manually edited", async () => {
      mockSaveLabel.mockResolvedValueOnce({
        example_key: "img_001",
        label_status: "verified",
        verified_outcome: "Edit",
        verified_at: "2026-04-14T12:00:00Z",
        edited_core_fields: ["damage_type"],
        edited_aux_fields: [],
        pool_assignment: null,
      });

      const { user } = await renderWithProposal();

      // Modify a field
      await user.selectOptions(
        screen.getByTestId("field-input-damage_type"),
        "scratch",
      );
      await waitFor(() => {
        expect(screen.getByTestId("rationale-textarea")).toBeInTheDocument();
      });

      // Edit rationale
      const textarea = screen.getByTestId("rationale-textarea");
      await user.clear(textarea);
      await user.type(textarea, "Scratch mark on surface.");

      await waitFor(() => {
        expect(screen.getByTestId("save-btn")).not.toBeDisabled();
      });

      // Save
      await user.click(screen.getByTestId("save-btn"));

      await waitFor(() => {
        expect(mockSaveLabel).toHaveBeenCalledWith(
          "test-pid",
          expect.objectContaining({
            rationale_source: "sme_edited",
            label_json: expect.objectContaining({
              rationale_note: "Scratch mark on surface.",
              damage_type: "scratch",
            }),
          }),
        );
      });
    });

    // ── Save with teacher_regenerated_approved ──────────────────────────

    it("saves with rationale_source teacher_regenerated_approved and invocation ID", async () => {
      mockRegenerateRationale.mockResolvedValueOnce({
        inference_invocation_id: "regen-inv-1",
        rationale_note: "AI rationale text.",
        invocation_status: "success",
      });
      mockSaveLabel.mockResolvedValueOnce({
        example_key: "img_001",
        label_status: "verified",
        verified_outcome: "Edit",
        verified_at: "2026-04-14T12:00:00Z",
        edited_core_fields: ["damage_type"],
        edited_aux_fields: [],
        pool_assignment: null,
      });

      const { user } = await renderWithProposal();

      // Modify → Generate → Approve → Save
      await user.selectOptions(
        screen.getByTestId("field-input-damage_type"),
        "scratch",
      );
      await waitFor(() => {
        expect(screen.getByTestId("generate-ai-rationale-btn")).toBeInTheDocument();
      });
      await user.click(screen.getByTestId("generate-ai-rationale-btn"));
      await waitFor(() => {
        expect(screen.getByTestId("approve-ai-rationale-btn")).toBeInTheDocument();
      });
      await user.click(screen.getByTestId("approve-ai-rationale-btn"));

      await waitFor(() => {
        expect(screen.getByTestId("save-btn")).not.toBeDisabled();
      });
      await user.click(screen.getByTestId("save-btn"));

      await waitFor(() => {
        expect(mockSaveLabel).toHaveBeenCalledWith(
          "test-pid",
          expect.objectContaining({
            rationale_source: "teacher_regenerated_approved",
            rationale_regeneration_invocation_id: "regen-inv-1",
            label_json: expect.objectContaining({
              rationale_note: "AI rationale text.",
            }),
          }),
        );
      });
    });

    // ── No cross-example rationale leak on proposal failure ─────────────

    it("resets the rationale editor when the next example's proposal fails", async () => {
      // Save an Edit with an SME rationale on example 1, then serve a
      // FAILED proposal (null proposal_json) for example 2. The rationale
      // editor must reset: carrying the previous image's text and
      // 'edited' state forward would let it be saved verbatim into a
      // different image's Verified Edit and flow into ICL.
      mockSaveLabel.mockResolvedValueOnce({
        example_key: "img_001",
        label_status: "verified",
        verified_outcome: "Edit",
        verified_at: "2026-04-14T12:00:00Z",
        edited_core_fields: ["damage_type"],
        edited_aux_fields: [],
        pool_assignment: null,
      });
      mockFetchNextReviewItem
        .mockResolvedValueOnce(REVIEW_NEXT)
        .mockResolvedValueOnce({ ...REVIEW_NEXT, example_key: "img_002" });
      mockCreateProposal
        .mockResolvedValueOnce(PROPOSAL_SUCCESS)
        .mockResolvedValueOnce({ ...PROPOSAL_TIMEOUT, example_key: "img_002" });

      const { user } = await renderWithProposal();

      // Example 1: edit a field, write a rationale, save the Edit.
      await user.selectOptions(
        screen.getByTestId("field-input-damage_type"),
        "scratch",
      );
      await waitFor(() => {
        expect(screen.getByTestId("rationale-textarea")).toBeInTheDocument();
      });
      const textarea = screen.getByTestId("rationale-textarea");
      await user.clear(textarea);
      await user.type(textarea, "Rationale written for the previous image.");
      await waitFor(() => {
        expect(screen.getByTestId("save-btn")).not.toBeDisabled();
      });
      await user.click(screen.getByTestId("save-btn"));

      // Example 2 lands with a failed proposal (manual-label flow).
      await screen.findByTestId("proposal-failure-timeout");

      // The stale 'edited' panel must not survive into the new example.
      await waitFor(() => {
        expect(screen.queryByTestId("rationale-panel")).not.toBeInTheDocument();
      });

      // Labeling manually re-opens the panel EMPTY, and Save stays gated
      // until a fresh rationale is written for THIS image.
      await user.selectOptions(screen.getByTestId("field-input-damage_type"), "dent");
      await waitFor(() => {
        expect(screen.getByTestId("rationale-textarea")).toBeInTheDocument();
      });
      expect(
        (screen.getByTestId("rationale-textarea") as HTMLTextAreaElement).value,
      ).toBe("");
      expect(screen.getByTestId("save-btn")).toBeDisabled();
    });
  });

  // =========================================================================
  // Retry and top bar controls
  // =========================================================================

  describe("Retry and top bar controls", () => {
    // Helper
    async function renderWithProposal(
      projectOverrides?: Record<string, unknown>,
      modelConfigsOverride?: typeof MODEL_CONFIGS,
      proposalOverride?: typeof PROPOSAL_SUCCESS,
    ) {
      if (modelConfigsOverride)
        mockFetchModelConfigs.mockResolvedValue(modelConfigsOverride);
      if (proposalOverride) mockCreateProposal.mockResolvedValue(proposalOverride);

      const { Wrapper } = createWrapper(projectOverrides);
      const user = userEvent.setup();
      render(<LabelingPage />, { wrapper: Wrapper });

      await waitFor(() => {
        expect(screen.getByTestId("proposal-form")).toBeInTheDocument();
      });

      return { user };
    }

    // ── Top bar all-controls (Cosmos Reason2) ──────────────────────────

    it("shows all top bar controls when teacher supports thinking + visual budget", async () => {
      await renderWithProposal();

      // Output Stability visible
      expect(screen.getByTestId("output-stability")).toBeInTheDocument();

      // Thinking visible (cosmos-reason-2 has qwen_enable_thinking)
      expect(screen.getByTestId("thinking-toggle")).toBeInTheDocument();

      // Visual Budget visible (mm_processor_size)
      expect(screen.getByTestId("visual-budget")).toBeInTheDocument();
    });

    // ── Top bar minimal (Mistral Large 3 — no thinking, no visual budget) ───

    it("hides Thinking and Visual Budget when teacher does not support them", async () => {
      const minimalConfigs = {
        items: [MODEL_CONFIG_MINIMAL],
        next_cursor: null,
      };
      await renderWithProposal(
        { teacher_model_config_id: "mc-mistral" },
        minimalConfigs,
      );

      // Output Stability still visible
      expect(screen.getByTestId("output-stability")).toBeInTheDocument();

      // Thinking hidden
      expect(screen.queryByTestId("thinking-toggle")).not.toBeInTheDocument();

      // Visual Budget hidden
      expect(screen.queryByTestId("visual-budget")).not.toBeInTheDocument();
    });

    // ── Retry panel opens with controls ─────────────────────────────────

    it("opens retry panel with 5 controls pre-populated from settings", async () => {
      const { user } = await renderWithProposal();

      await user.click(screen.getByTestId("retry-btn"));

      await waitFor(() => {
        expect(screen.getByTestId("retry-panel")).toBeInTheDocument();
      });

      // 5 controls visible
      expect(screen.getByTestId("retry-teacher")).toBeInTheDocument();
      expect(screen.getByTestId("retry-guidance")).toBeInTheDocument();
      expect(screen.getByTestId("retry-stability")).toBeInTheDocument();
      expect(screen.getByTestId("retry-thinking")).toBeInTheDocument();
      expect(screen.getByTestId("retry-visual-budget")).toBeInTheDocument();

      // Cancel and Retry buttons
      expect(screen.getByTestId("retry-cancel-btn")).toBeInTheDocument();
      expect(screen.getByTestId("retry-confirm-btn")).toBeInTheDocument();

      // Cancel returns to normal action buttons
      await user.click(screen.getByTestId("retry-cancel-btn"));
      await waitFor(() => {
        expect(screen.queryByTestId("retry-panel")).not.toBeInTheDocument();
        expect(screen.getByTestId("labeling-actions")).toBeInTheDocument();
      });
    });

    // ── Retry panel hides controls for non-capable teacher ──────────────

    it("hides Thinking in retry panel when selected teacher has mode=none", async () => {
      // Add the minimal model to the teacher list
      const multiConfigs = {
        items: [MODEL_CONFIGS.items[0], MODEL_CONFIG_MINIMAL],
        next_cursor: null,
      };
      const { user } = await renderWithProposal(undefined, multiConfigs);

      await user.click(screen.getByTestId("retry-btn"));
      await waitFor(() => {
        expect(screen.getByTestId("retry-panel")).toBeInTheDocument();
      });

      // Initially thinking is visible (cosmos-reason-2 selected)
      expect(screen.getByTestId("retry-thinking")).toBeInTheDocument();

      // Switch to mistral model
      await user.selectOptions(
        screen.getByTestId("retry-teacher-select"),
        "mc-mistral",
      );

      // Thinking should hide
      await waitFor(() => {
        expect(screen.queryByTestId("retry-thinking")).not.toBeInTheDocument();
      });
    });

    // ── Top-bar Teacher model picker ───────────────────────────────────

    it("threads model configs and current teacher into the picker", async () => {
      // Option rendering and highlight logic are pinned in
      // TeacherModelPicker.test.tsx — this checks the page wires the
      // catalog entries and project.teacher_model_config_id through.
      const multi = {
        items: [MODEL_CONFIGS.items[0], MODEL_CONFIG_MINIMAL],
        next_cursor: null,
      };
      await renderWithProposal({ teacher_model_config_id: "mc-teacher" }, multi);

      const select = screen.getByTestId(
        "teacher-model-picker-select",
      ) as HTMLSelectElement;
      const options = Array.from(select.options).map((o) => o.value);
      expect(options).toEqual(["mc-teacher", "mc-mistral"]);
      expect(select.value).toBe("mc-teacher");
    });

    it("PATCHes project.teacher_model_config_id when a different model is picked", async () => {
      const multi = {
        items: [MODEL_CONFIGS.items[0], MODEL_CONFIG_MINIMAL],
        next_cursor: null,
      };
      mockUpdateProject.mockResolvedValue({
        ...PROJECT,
        teacher_model_config_id: "mc-mistral",
      });

      const { user } = await renderWithProposal(undefined, multi);
      await user.selectOptions(
        screen.getByTestId("teacher-model-picker-select"),
        "mc-mistral",
      );

      await waitFor(() => {
        expect(mockUpdateProject).toHaveBeenCalledWith("test-pid", {
          teacher_model_config_id: "mc-mistral",
        });
      });
    });

    it("does not PATCH when the picker fires with the current value", async () => {
      await renderWithProposal();
      mockUpdateProject.mockClear();

      // Selecting the already-current option — should be a no-op PATCH-wise.
      await userEvent
        .setup()
        .selectOptions(screen.getByTestId("teacher-model-picker-select"), "mc-teacher");

      expect(mockUpdateProject).not.toHaveBeenCalled();
    });

    it("updates the project cache and invalidates trigger status after a successful Teacher PATCH", async () => {
      // The capability-gated controls (Thinking / Visual Budget) hide/show
      // based on `project.teacher_model_config_id`, so the page MUST
      // put the authoritative PATCH response in the project cache so the
      // project context drives the re-render without another request. The config-change nudge
      // is also refreshed via the evaluation_trigger_status cache. Use
      // the page's queryClient to spy on both cache operations.
      const { Wrapper, queryClient } = createWrapper();
      const invalidateSpy = vi.spyOn(queryClient, "invalidateQueries");
      const setQueryDataSpy = vi.spyOn(queryClient, "setQueryData");
      mockUpdateProject.mockResolvedValue({
        ...PROJECT,
        teacher_model_config_id: "mc-mistral",
      });
      mockFetchModelConfigs.mockResolvedValue({
        items: [MODEL_CONFIGS.items[0], MODEL_CONFIG_MINIMAL],
        next_cursor: null,
      });

      const user = userEvent.setup();
      render(<LabelingPage />, { wrapper: Wrapper });
      await waitFor(() => {
        expect(screen.getByTestId("proposal-form")).toBeInTheDocument();
      });

      await user.selectOptions(
        screen.getByTestId("teacher-model-picker-select"),
        "mc-mistral",
      );

      await waitFor(() => {
        expect(mockUpdateProject).toHaveBeenCalledWith("test-pid", {
          teacher_model_config_id: "mc-mistral",
        });
      });

      expect(setQueryDataSpy).toHaveBeenCalledWith(
        ["project", "test-pid"],
        expect.objectContaining({ teacher_model_config_id: "mc-mistral" }),
      );

      await waitFor(() => {
        const invalidatedKeys = invalidateSpy.mock.calls
          .map((call) => call[0]?.queryKey as readonly unknown[] | undefined)
          .filter((k): k is readonly unknown[] => Array.isArray(k));
        expect(invalidatedKeys).toEqual(
          expect.arrayContaining([["evaluations", "test-pid", "triggerStatus"]]),
        );
      });
    });

    it("persists every header control through the shared project-setting path", async () => {
      const multi = {
        items: [MODEL_CONFIGS.items[0], MODEL_CONFIG_MINIMAL],
        next_cursor: null,
      };
      const { user } = await renderWithProposal(undefined, multi);

      await user.click(screen.getByTestId("output-stability-explore"));
      await waitFor(() => {
        expect(mockUpdateProject).toHaveBeenCalledWith("test-pid", {
          labeling_generation_preset_key: "explore",
        });
      });

      await user.click(screen.getByTestId("thinking-toggle-off"));
      await waitFor(() => {
        expect(mockUpdateProject).toHaveBeenCalledWith("test-pid", {
          thinking_default_on: false,
        });
      });

      await user.click(screen.getByTestId("visual-budget-fast"));
      await waitFor(() => {
        expect(mockUpdateProject).toHaveBeenCalledWith("test-pid", {
          visual_budget_preset_key: "fast",
        });
      });

      await user.selectOptions(
        screen.getByTestId("teacher-model-picker-select"),
        "mc-mistral",
      );
      await waitFor(() => {
        expect(mockUpdateProject).toHaveBeenCalledWith("test-pid", {
          teacher_model_config_id: "mc-mistral",
        });
      });
    });

    it("gates label actions until a header setting is persisted", async () => {
      let resolveUpdate!: (value: typeof PROJECT) => void;
      mockUpdateProject.mockReturnValueOnce(
        new Promise<typeof PROJECT>((resolve) => {
          resolveUpdate = resolve;
        }),
      );
      const { user } = await renderWithProposal();

      await user.click(screen.getByTestId("output-stability-explore"));
      await waitFor(() => {
        expect(mockUpdateProject).toHaveBeenCalledWith("test-pid", {
          labeling_generation_preset_key: "explore",
        });
      });

      expect(screen.getByTestId("save-btn")).toBeDisabled();
      expect(screen.getByTestId("skip-btn")).toBeDisabled();
      expect(screen.getByTestId("retry-btn")).toBeDisabled();
      expect(screen.getByTestId("thinking-toggle-off")).toBeDisabled();
      expect(screen.getByTestId("teacher-model-picker-select")).toBeDisabled();

      resolveUpdate({ ...PROJECT, labeling_generation_preset_key: "explore" });
      await waitFor(() => {
        expect(screen.getByTestId("save-btn")).not.toBeDisabled();
        expect(screen.getByTestId("skip-btn")).not.toBeDisabled();
        expect(screen.getByTestId("retry-btn")).not.toBeDisabled();
      });
    });

    it("shows a header-setting failure instead of silently falling back", async () => {
      mockUpdateProject.mockRejectedValueOnce(new Error("network unavailable"));
      const { user } = await renderWithProposal();

      await user.click(screen.getByTestId("output-stability-explore"));

      expect(await screen.findByTestId("top-bar-settings-error")).toHaveTextContent(
        "Could not save proposal settings: network unavailable",
      );
      expect(screen.getByTestId("output-stability-precise")).toHaveAttribute(
        "aria-pressed",
        "true",
      );
    });

    // ── Retry skip when top-bar already changed ──────────────────────────
    //
    // A top-bar PATCH (Teacher, Stability, Thinking, Visual Budget) flips
    // a ``topBarDirty`` flag. Click Retry while dirty → skip the panel and
    // fire the retry immediately against the now-current project state.
    // The panel-confirm path is reserved for the case where the SME has
    // not yet expressed any preference change.

    it("Retry skips the panel and re-fetches immediately after a top-bar Teacher change", async () => {
      const multi = {
        items: [MODEL_CONFIGS.items[0], MODEL_CONFIG_MINIMAL],
        next_cursor: null,
      };
      mockUpdateProject.mockResolvedValue({
        ...PROJECT,
        teacher_model_config_id: "mc-mistral",
      });
      const { user } = await renderWithProposal(undefined, multi);

      // Initial proposal fetch fires once on load.
      expect(mockCreateProposal).toHaveBeenCalledTimes(1);

      // SME flips Teacher in the top bar — flag becomes dirty.
      await user.selectOptions(
        screen.getByTestId("teacher-model-picker-select"),
        "mc-mistral",
      );
      await waitFor(() => {
        expect(mockUpdateProject).toHaveBeenCalledWith("test-pid", {
          teacher_model_config_id: "mc-mistral",
        });
      });

      // Click Retry → the panel must NOT open; a new proposal fetch
      // fires directly against current project state.
      await user.click(screen.getByTestId("retry-btn"));
      await waitFor(() => {
        expect(mockCreateProposal).toHaveBeenCalledTimes(2);
      });
      expect(screen.queryByTestId("retry-panel")).not.toBeInTheDocument();
    });

    it("Retry opens the panel when no top-bar control has been changed", async () => {
      const { user } = await renderWithProposal();
      expect(mockCreateProposal).toHaveBeenCalledTimes(1);

      // No top-bar interaction — click Retry → panel opens (the SME's
      // chance to vary settings explicitly).
      await user.click(screen.getByTestId("retry-btn"));
      await waitFor(() => {
        expect(screen.getByTestId("retry-panel")).toBeInTheDocument();
      });
      // No retry fetch should fire until the SME confirms in the panel.
      expect(mockCreateProposal).toHaveBeenCalledTimes(1);
    });
  });

  // =========================================================================
  // Inline notices, prior-label hints, and queue empty
  // =========================================================================

  describe("Inline notices, prior-label hints, and queue empty", () => {
    // ── Cold start notice ───────────────────────────────────────────────

    it("shows cold start notice when Verified=0 and dismisses it", async () => {
      // Zero verified count
      mockFetchProjectList.mockResolvedValue({
        items: [
          {
            ...PROJECT_LIST.items[0],
            counts: {
              verified: 0,
              unlabeled: 50,
              auto_labeled: 0,
              omitted: 0,
              pending_relabel: 0,
            },
          },
        ],
        next_cursor: null,
      });

      const { Wrapper } = createWrapper();
      const user = userEvent.setup();
      render(<LabelingPage />, { wrapper: Wrapper });

      await waitFor(() => {
        expect(screen.getByTestId("cold-start-notice")).toBeInTheDocument();
      });

      expect(screen.getByTestId("cold-start-notice")).toHaveTextContent(
        /first label.*no examples to learn from.*Accuracy improves immediately/,
      );

      // Dismiss
      await user.click(screen.getByTestId("cold-start-notice-dismiss"));

      await waitFor(() => {
        expect(screen.queryByTestId("cold-start-notice")).not.toBeInTheDocument();
      });
    });

    // ── Schema refinement reminder ──────────────────────────────────────

    it("shows the reminder the backend reports active and dismisses via the endpoint", async () => {
      // Backend decides eligibility — the UI renders reminder #1
      // because reminder_status says so, not from any client threshold.
      mockFetchReminderStatus.mockResolvedValue({
        active_reminder: 1,
        verified_count: 12,
        threshold_1: 10,
        threshold_2: 35,
        dismissed_count: 0,
      });

      const { Wrapper } = createWrapper();
      const user = userEvent.setup();
      render(<LabelingPage />, { wrapper: Wrapper });

      await waitFor(() => {
        expect(screen.getByTestId("proposal-form")).toBeInTheDocument();
      });

      await waitFor(() => {
        expect(screen.getByTestId("schema-reminder-1")).toBeInTheDocument();
      });

      expect(screen.getByTestId("schema-reminder-1")).toHaveTextContent(
        /adjust your schema.*Fewer labels/,
      );
      expect(screen.getByTestId("review-schema-btn")).toBeInTheDocument();

      // X-dismiss goes through POST guidance:dismiss_reminder; once the
      // refetched status reports no active reminder, the banner leaves.
      mockFetchReminderStatus.mockResolvedValue({
        active_reminder: null,
        verified_count: 12,
        threshold_1: 10,
        threshold_2: 35,
        dismissed_count: 1,
      });
      await user.click(screen.getByTestId("schema-reminder-1-dismiss"));

      await waitFor(() => {
        expect(mockDismissReminder).toHaveBeenCalledWith("test-pid");
      });
      await waitFor(() => {
        expect(screen.queryByTestId("schema-reminder-1")).not.toBeInTheDocument();
      });
    });

    it("shows reminder #2 with the backend's verified count when both thresholds crossed", async () => {
      // Backend higher-of-two rule: when both thresholds were
      // crossed before either fired, only reminder #2 shows.
      mockFetchReminderStatus.mockResolvedValue({
        active_reminder: 2,
        verified_count: 40,
        threshold_1: 10,
        threshold_2: 35,
        dismissed_count: 0,
      });

      const { Wrapper } = createWrapper();
      render(<LabelingPage />, { wrapper: Wrapper });

      await waitFor(() => {
        expect(screen.getByTestId("schema-reminder-2")).toBeInTheDocument();
      });
      expect(screen.queryByTestId("schema-reminder-1")).not.toBeInTheDocument();
      expect(screen.getByTestId("schema-reminder-2")).toHaveTextContent(
        /You have 40 labels.*more images to re-label/,
      );
    });

    it("suppresses the first-pool evaluation banner while schema reminder #1 is visible", async () => {
      // Collision rule: when both the schema
      // refinement reminder and the first-pool evaluation banner would fire
      // together, the schema reminder wins. The SME sees one nudge at a
      // time; the first-pool banner returns on the next render after the
      // reminder is dismissed or acted on. Reminder eligibility comes from
      // the backend's reminder_status.
      mockFetchReminderStatus.mockResolvedValue({
        active_reminder: 1,
        verified_count: 12,
        threshold_1: 10,
        threshold_2: 35,
        dismissed_count: 0,
      });
      mockFetchTriggerStatus.mockResolvedValue({
        auto_evaluate_enabled: false,
        first_pool_threshold: {
          is_active: true,
          dismissed: false,
          message:
            "5 images reserved for testing. Run an evaluation to measure quality.",
          context: null,
        },
        configuration_change: {
          is_active: false,
          dismissed: false,
          message: "",
          context: null,
        },
        icl_growth: { is_active: false, dismissed: false, message: "", context: null },
        updated_at: "2026-04-23T00:00:00Z",
      });

      const { Wrapper } = createWrapper();
      render(<LabelingPage />, { wrapper: Wrapper });

      await waitFor(() => {
        expect(screen.getByTestId("schema-reminder-1")).toBeInTheDocument();
      });

      // Give the trigger-status query time to resolve. If suppression were
      // broken, the banner would appear within the next render.
      await waitFor(() => {
        expect(screen.queryByTestId("trigger-first-pool")).toBeNull();
      });
    });

    // ── Prior-label hints ───────────────────────────────────────────────

    it("shows prior-label annotations when example has prior_verified_label_ref", async () => {
      const PRIOR_SNAPSHOT = {
        label_json: {
          damage_type: "scratch",
          is_severe: true,
          confidence: 4,
          rationale_note: "old rationale",
        },
        vlm_proposal_json: { damage_type: "dent", is_severe: false, confidence: 3 },
        edited_core_fields: ["damage_type"],
        edited_aux_fields: [],
        verified_outcome: "Edit",
        rationale_note: "old rationale",
        guidance_id: "g-old",
      };

      // The review-selector next response carries the prior-label
      // snapshot — no examples-list scan.
      mockFetchNextReviewItem.mockResolvedValue({
        ...REVIEW_NEXT,
        prior_verified_label_ref: JSON.stringify(PRIOR_SNAPSHOT),
      });

      const { Wrapper } = createWrapper();
      const user = userEvent.setup();
      render(<LabelingPage />, { wrapper: Wrapper });

      await waitFor(() => {
        expect(screen.getByTestId("proposal-form")).toBeInTheDocument();
      });

      // Prior-label hints should appear for damage_type (which was in the prior snapshot)
      await waitFor(() => {
        expect(screen.getByTestId("prior-hint-damage_type")).toBeInTheDocument();
      });

      // "you edited" badge for damage_type
      expect(screen.getByTestId("prior-edited-damage_type")).toBeInTheDocument();

      // [Adopt prior] button should be present for schema-valid prior values
      const adoptBtn = screen.getByTestId("adopt-prior-damage_type");
      expect(adoptBtn).toBeInTheDocument();

      // Click Adopt prior → field value changes
      await user.click(adoptBtn);

      // After adopting, rationale panel should expand (dirty state triggers it)
      await waitFor(() => {
        expect(screen.getByTestId("rationale-panel")).toBeInTheDocument();
      });
    });

    // ── Prior-label progress indicator ────────────────────────────────

    it("shows prior-label progress banner when pendingRelabel > 0", async () => {
      // Override project list to show pending_relabel + prior_relabeled counts.
      // Progress copy: "Prior labels: N of M re-labeled (prior edits first).",
      // with M = prior_relabeled + pending_relabel (= 4 + 8 = 12 here).
      mockFetchProjectList.mockResolvedValue({
        items: [
          {
            ...PROJECT_LIST.items[0],
            counts: {
              verified: 10,
              unlabeled: 40,
              auto_labeled: 0,
              omitted: 5,
              pending_relabel: 8,
              prior_relabeled: 4,
            },
          },
        ],
        next_cursor: null,
      });

      const { Wrapper } = createWrapper();
      render(<LabelingPage />, { wrapper: Wrapper });

      await waitFor(() => {
        expect(screen.getByTestId("prior-label-progress")).toBeInTheDocument();
      });

      expect(screen.getByTestId("prior-label-progress")).toHaveTextContent(
        /Prior labels:\s*4 of 12 re-labeled\s*\(prior edits first\)/,
      );
    });

    // ── Queue empty ─────────────────────────────────────────────────────

    it("shows enhanced queue-empty state with action buttons when omitted > 0", async () => {
      mockFetchNextReviewItem.mockResolvedValue(REVIEW_NEXT_EMPTY);

      const { Wrapper } = createWrapper();
      render(<LabelingPage />, { wrapper: Wrapper });

      await waitFor(
        () => {
          expect(screen.getByTestId("queue-empty")).toBeInTheDocument();
        },
        { timeout: 5000 },
      );

      const queueEmpty = screen.getByTestId("queue-empty");
      expect(queueEmpty).toHaveTextContent("All images have been reviewed.");
      expect(screen.getByTestId("icl-chip-idle")).toHaveTextContent(
        "ICL: no active proposal",
      );
      expect(screen.queryByTestId("icl-chip-pending")).not.toBeInTheDocument();
      // Counts render as a single meta line — the status bar on the same
      // screen already carries the full pill counters.
      expect(queueEmpty).toHaveTextContent(/Verified 42/);
      expect(queueEmpty).toHaveTextContent(/Omitted 5/);

      // Action buttons
      expect(screen.getByTestId("queue-add-images-btn")).toBeInTheDocument();
      expect(screen.getByTestId("restore-omitted-btn")).toBeInTheDocument();
      expect(screen.getByTestId("restore-omitted-btn")).toHaveTextContent(
        /Restore 5 Omitted/,
      );
    });

    // ── Restore Omitted ─────────────────────────────────────────────────

    it("calls restoreOmitted and restarts cycle when Restore clicked", async () => {
      mockFetchNextReviewItem.mockResolvedValueOnce(REVIEW_NEXT_EMPTY);
      mockRestoreOmitted.mockResolvedValueOnce({ restored_count: 5 });
      // After restore, re-enter labeling cycle
      mockFetchNextReviewItem.mockResolvedValueOnce(REVIEW_NEXT);

      const { Wrapper } = createWrapper();
      const user = userEvent.setup();
      render(<LabelingPage />, { wrapper: Wrapper });

      await waitFor(
        () => {
          expect(screen.getByTestId("restore-omitted-btn")).toBeInTheDocument();
        },
        { timeout: 5000 },
      );

      await user.click(screen.getByTestId("restore-omitted-btn"));

      await waitFor(() => {
        expect(mockRestoreOmitted).toHaveBeenCalledWith("test-pid");
      });

      // Cycle should restart — fetchNextReviewItem called again
      await waitFor(() => {
        expect(mockFetchNextReviewItem.mock.calls.length).toBeGreaterThanOrEqual(2);
      });
    });
  });
});
