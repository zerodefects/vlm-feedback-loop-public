// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

import { EvaluationStrip } from "../EvaluationStrip";

// ── Mocks ──────────────────────────────────────────────────────────────────

vi.mock("@/api/evaluation", () => ({
  fetchTriggerStatus: vi.fn(),
  listEvaluationRuns: vi.fn(),
  createEvaluationRun: vi.fn(),
  cancelEvaluationRun: vi.fn(),
  dismissTrigger: vi.fn(),
}));

vi.mock("@/api/model-configs", () => ({
  updateProject: vi.fn(),
}));

import {
  fetchTriggerStatus,
  listEvaluationRuns,
  createEvaluationRun,
  dismissTrigger,
} from "@/api/evaluation";

const mockTriggers = fetchTriggerStatus as ReturnType<typeof vi.fn>;
const mockListRuns = listEvaluationRuns as ReturnType<typeof vi.fn>;
const mockCreateRun = createEvaluationRun as ReturnType<typeof vi.fn>;
const mockDismiss = dismissTrigger as ReturnType<typeof vi.fn>;

function makeTriggers(overrides: Record<string, unknown> = {}) {
  return {
    auto_evaluate_enabled: false,
    first_pool_threshold: {
      is_active: false,
      dismissed: false,
      message: "pool msg",
      context: null,
    },
    configuration_change: {
      is_active: false,
      dismissed: false,
      message: "config msg",
      context: null,
    },
    icl_growth: {
      is_active: false,
      dismissed: false,
      message: "icl msg",
      context: null,
    },
    updated_at: "2026-04-15T10:00:00Z",
    ...overrides,
  };
}

function makeRun(overrides: Record<string, unknown> = {}) {
  return {
    run_id: "run-1",
    run_type: "evaluation_run",
    status: "completed",
    status_reason: null,
    pool_version_id: null,
    guidance_id: null,
    model_config_id: null,
    metrics: {
      overall: {
        exact_match_rate: 0.85,
        example_count: 10,
        per_field_match_rates: {},
        per_value_metrics: {},
      },
      returning: null,
      new: null,
    },
    previous_overall_exact_match: null,
    coverage_gaps: [],
    created_at: "2026-04-15T10:00:00Z",
    started_at: null,
    completed_at: null,
    ...overrides,
  };
}

// ── Setup ──────────────────────────────────────────────────────────────────

let queryClient: QueryClient;

beforeEach(() => {
  vi.clearAllMocks();
  queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  mockTriggers.mockResolvedValue(makeTriggers());
  mockListRuns.mockResolvedValue({ items: [], next_cursor: null });
  mockCreateRun.mockResolvedValue({ run_id: "new-run", status: "queued" });
  mockDismiss.mockResolvedValue({ trigger_type: "icl_growth", dismissed: true });
});

function renderStrip(
  poolCount = 5,
  opts: { suppressFirstPoolTrigger?: boolean; poolTarget?: number | null } = {},
) {
  return render(
    <QueryClientProvider client={queryClient}>
      <EvaluationStrip
        projectId="test-pid"
        poolCount={poolCount}
        poolTarget={opts.poolTarget ?? null}
        onShowResults={vi.fn()}
        suppressFirstPoolTrigger={opts.suppressFirstPoolTrigger}
      />
    </QueryClientProvider>,
  );
}

// ── Tests ──────────────────────────────────────────────────────────────────

describe("EvaluationStrip", () => {
  it("renders nothing when pool count is 0", () => {
    const { container } = renderStrip(0);
    expect(container.querySelector("[data-testid='evaluation-strip']")).toBeNull();
  });

  it("renders the strip bar with Evaluate button and auto-evaluate toggle", async () => {
    renderStrip(5);
    await waitFor(() => {
      expect(screen.getByTestId("eval-strip-bar")).toBeInTheDocument();
    });
    expect(screen.getByText(/Test Pool: 5/)).toBeInTheDocument();
    expect(screen.getByTestId("evaluate-btn")).toBeInTheDocument();
    expect(screen.getByTestId("auto-evaluate-toggle")).toBeInTheDocument();
  });

  it("lists only gate-basis runs — a Student benchmark is never its 'current evaluation'", async () => {
    renderStrip(5);
    await waitFor(() => {
      expect(mockListRuns).toHaveBeenCalled();
    });
    expect(mockListRuns).toHaveBeenCalledWith("test-pid", {
      basis: "gate",
      limit: 5,
    });
  });

  it("discloses impending pool growth when the target exceeds the count", async () => {
    renderStrip(120, { poolTarget: 1933 });
    await waitFor(() => {
      expect(screen.getByTestId("eval-strip-bar")).toBeInTheDocument();
    });
    const note = screen.getByTestId("pool-growth-note");
    expect(note.textContent).toContain("grows to 1933 as you verify");
    expect(note).toHaveAttribute(
      "title",
      expect.stringContaining("test_pool_fraction"),
    );
  });

  it("shows no growth note when the pool is at or above its target", async () => {
    renderStrip(120, { poolTarget: 96 });
    await waitFor(() => {
      expect(screen.getByTestId("eval-strip-bar")).toBeInTheDocument();
    });
    expect(screen.queryByTestId("pool-growth-note")).toBeNull();
    expect(screen.getByText(/Test Pool: 120/)).toBeInTheDocument();
  });

  it("shows first pool threshold banner when active", async () => {
    mockTriggers.mockResolvedValue(
      makeTriggers({
        first_pool_threshold: {
          is_active: true,
          dismissed: false,
          message: "5 images reserved",
          context: null,
        },
      }),
    );
    renderStrip(5);
    await waitFor(() => {
      expect(screen.getByTestId("trigger-first-pool")).toBeInTheDocument();
    });
  });

  it("first-pool banner renders an Evaluate CTA that triggers createEvaluationRun", async () => {
    // The banner carries [Evaluate] alongside [Dismiss] so the SME can
    // act on the nudge without locating the strip's top-right button.
    mockTriggers.mockResolvedValue(
      makeTriggers({
        first_pool_threshold: {
          is_active: true,
          dismissed: false,
          message:
            "5 images reserved for testing. Run an evaluation to measure quality.",
          context: null,
        },
      }),
    );
    renderStrip(5);
    const cta = await screen.findByTestId("trigger-first-pool-evaluate");
    expect(cta).toBeInTheDocument();
    // Banner CTAs stay secondary — the hero card's Save is the labeling
    // screen's sole green primary.
    expect(cta.className).not.toContain("nvidia-green-button");
    await userEvent.click(cta);
    expect(mockCreateRun).toHaveBeenCalledWith("test-pid", {});
  });

  it("first-pool banner auto-dismisses when [Evaluate] is clicked", async () => {
    // The banner has nothing left to communicate once the run is in
    // flight — the strip-bar takes over with the running progress UI.
    // Without auto-dismiss the SME is left clicking [X] on a dead nudge
    // after acting on it. Same pattern applies to any banner where the
    // primary action carries the user past the banner's reason for
    // existing (no further banner state). The dismiss MUST also fire
    // the dismiss API call so the banner doesn't come back on next
    // refetch.
    mockTriggers.mockResolvedValue(
      makeTriggers({
        first_pool_threshold: {
          is_active: true,
          dismissed: false,
          message:
            "5 images reserved for testing. Run an evaluation to measure quality.",
          context: null,
        },
      }),
    );
    renderStrip(5);
    const banner = await screen.findByTestId("trigger-first-pool");
    expect(banner).toBeInTheDocument();
    const cta = await screen.findByTestId("trigger-first-pool-evaluate");
    await userEvent.click(cta);

    // Banner must be gone after Evaluate click — without a separate
    // [X] click — and the dismiss API must have been called so the
    // server-side trigger state matches.
    await waitFor(() => {
      expect(screen.queryByTestId("trigger-first-pool")).toBeNull();
    });
    expect(mockDismiss).toHaveBeenCalledWith("test-pid", "first_pool_threshold");
    expect(mockCreateRun).toHaveBeenCalledWith("test-pid", {});
  });

  it("suppresses the first-pool banner when suppressFirstPoolTrigger=true", async () => {
    // Collision rule: while a schema refinement reminder is on
    // screen, the first-pool banner MUST NOT render — the SME
    // sees one nudge at a time. Config-change and ICL-growth are unaffected
    // by this rule because neither fires at the low verified counts where
    // the schema reminder lives.
    mockTriggers.mockResolvedValue(
      makeTriggers({
        first_pool_threshold: {
          is_active: true,
          dismissed: false,
          message:
            "5 images reserved for testing. Run an evaluation to measure quality.",
          context: null,
        },
      }),
    );
    renderStrip(5, { suppressFirstPoolTrigger: true });
    await waitFor(() => {
      expect(screen.getByTestId("eval-strip-bar")).toBeInTheDocument();
    });
    expect(screen.queryByTestId("trigger-first-pool")).toBeNull();
  });

  it("first-pool message must not leak the raw threshold (no-jargon rule)", async () => {
    // SME-facing eval UI MUST NOT expose raw metric names or
    // thresholds. The active message is the mandated copy.
    mockTriggers.mockResolvedValue(
      makeTriggers({
        first_pool_threshold: {
          is_active: true,
          dismissed: false,
          message:
            "7 images reserved for testing. Run an evaluation to measure quality.",
          context: null,
        },
      }),
    );
    renderStrip(7);
    const banner = await screen.findByTestId("trigger-first-pool");
    expect(banner).toHaveTextContent("Run an evaluation to measure quality");
    expect(banner.textContent ?? "").not.toMatch(/threshold:/i);
  });

  it("shows config change banner when active", async () => {
    mockTriggers.mockResolvedValue(
      makeTriggers({
        configuration_change: {
          is_active: true,
          dismissed: false,
          message: "changed",
          context: null,
        },
      }),
    );
    renderStrip(5);
    await waitFor(() => {
      expect(screen.getByTestId("trigger-config-change")).toBeInTheDocument();
    });
  });

  it("shows ICL growth banner when active", async () => {
    mockTriggers.mockResolvedValue(
      makeTriggers({
        icl_growth: {
          is_active: true,
          dismissed: false,
          message: "12 edits",
          context: null,
        },
      }),
    );
    renderStrip(5);
    await waitFor(() => {
      expect(screen.getByTestId("trigger-icl-growth")).toBeInTheDocument();
    });
  });

  it("shows completion notification with accuracy", async () => {
    mockListRuns.mockResolvedValue({ items: [makeRun()], next_cursor: null });
    renderStrip(5);
    await waitFor(() => {
      expect(screen.getByTestId("eval-complete")).toBeInTheDocument();
    });
    expect(screen.getByText(/85%/)).toBeInTheDocument();
    // A finished run is good news — success tone (green check), not
    // another neutral info nudge.
    expect(screen.getByTestId("eval-complete")).toHaveAttribute("data-tone", "success");
  });

  it("suppresses the completion banner while a configuration-change nudge is due", async () => {
    // Once settings changed, the completed run's accuracy headline
    // describes a stale configuration — the config-change nudge takes
    // the slot instead of stacking an identical banner above it.
    mockTriggers.mockResolvedValue(
      makeTriggers({
        configuration_change: {
          is_active: true,
          dismissed: false,
          message: "changed",
          context: null,
        },
      }),
    );
    mockListRuns.mockResolvedValue({ items: [makeRun()], next_cursor: null });
    renderStrip(5);
    await waitFor(() => {
      expect(screen.getByTestId("trigger-config-change")).toBeInTheDocument();
    });
    expect(screen.queryByTestId("eval-complete")).toBeNull();
  });

  it("restores the completion banner when the config-change nudge is dismissed", async () => {
    // The nudge only borrows the banner slot — dismissing it must bring
    // back the completed run's View Results affordance, not leave the
    // strip with neither banner.
    mockTriggers.mockResolvedValue(
      makeTriggers({
        configuration_change: {
          is_active: true,
          dismissed: false,
          message: "changed",
          context: null,
        },
      }),
    );
    mockListRuns.mockResolvedValue({ items: [makeRun()], next_cursor: null });
    renderStrip(5);
    await waitFor(() => {
      expect(screen.getByTestId("trigger-config-change")).toBeInTheDocument();
    });
    expect(screen.queryByTestId("eval-complete")).toBeNull();

    await userEvent.click(screen.getByTestId("trigger-config-change-dismiss"));

    await waitFor(() => {
      expect(screen.getByTestId("eval-complete")).toBeInTheDocument();
    });
    expect(screen.getByTestId("view-results-btn")).toBeInTheDocument();
    expect(screen.queryByTestId("trigger-config-change")).toBeNull();
  });

  it("config-change nudge carries an inline Evaluate CTA that starts a run and dismisses", async () => {
    // The banner tells the SME to evaluate, so it must offer the action
    // in place rather than sending them hunting for the strip button.
    mockTriggers.mockResolvedValue(
      makeTriggers({
        configuration_change: {
          is_active: true,
          dismissed: false,
          message: "changed",
          context: null,
        },
      }),
    );
    renderStrip(5);
    const cta = await screen.findByTestId("trigger-config-change-evaluate");
    await userEvent.click(cta);
    expect(mockCreateRun).toHaveBeenCalledWith("test-pid", {});
    await waitFor(() => {
      expect(screen.queryByTestId("trigger-config-change")).toBeNull();
    });
    expect(mockDismiss).toHaveBeenCalledWith("test-pid", "configuration_change");
  });

  it("delta badge compares the Returning subset against the previous run", async () => {
    // Overall 85% on 20, but the "on same images" claim must use the
    // Returning subset: 13/15 = 86.7% vs previous 80% -> +7, not the
    // overall delta (+5).
    mockListRuns.mockResolvedValue({
      items: [
        makeRun({
          previous_overall_exact_match: 0.8,
          metrics: {
            overall: {
              exact_match_rate: 0.85,
              example_count: 20,
              per_field_match_rates: {},
              per_value_metrics: {},
            },
            returning: {
              exact_match_rate: 13 / 15,
              example_count: 15,
              per_field_match_rates: {},
              per_value_metrics: {},
            },
            new: null,
          },
        }),
      ],
      next_cursor: null,
    });
    renderStrip(5);
    await waitFor(() => {
      expect(screen.getByTestId("delta-badge")).toBeInTheDocument();
    });
    expect(screen.getByTestId("delta-badge")).toHaveTextContent(
      "(+7 pts vs previous on same images)",
    );
  });

  it("omits the delta badge when the run has no Returning split", async () => {
    // A previous rate without a Returning subset cannot honestly claim a
    // same-images delta — render the plain image count instead.
    mockListRuns.mockResolvedValue({
      items: [makeRun({ previous_overall_exact_match: 0.8 })],
      next_cursor: null,
    });
    renderStrip(5);
    await waitFor(() => {
      expect(screen.getByTestId("eval-complete")).toBeInTheDocument();
    });
    expect(screen.queryByTestId("delta-badge")).not.toBeInTheDocument();
    expect(screen.getByText(/\(10 images\)/)).toBeInTheDocument();
  });

  it("shows coverage warning when gaps exist", async () => {
    mockListRuns.mockResolvedValue({
      items: [
        makeRun({
          coverage_gaps: [
            { field_name: "severity", field_type: "enum", missing_values: ["low"] },
          ],
        }),
      ],
      next_cursor: null,
    });
    renderStrip(5);
    await waitFor(() => {
      expect(screen.getByTestId("coverage-warning")).toBeInTheDocument();
    });
  });

  it("shows incomplete warning", async () => {
    mockListRuns.mockResolvedValue({
      items: [makeRun({ status: "incomplete" })],
      next_cursor: null,
    });
    renderStrip(5);
    await waitFor(() => {
      expect(screen.getByTestId("eval-incomplete")).toBeInTheDocument();
    });
    // Diagnostic-only results carry the amber warning tone, matching
    // ResultsPanel's treatment of the same state.
    expect(screen.getByTestId("eval-incomplete")).toHaveAttribute(
      "data-tone",
      "warning",
    );
  });

  it("shows failed banner with error tone", async () => {
    mockListRuns.mockResolvedValue({
      items: [
        makeRun({
          status: "failed",
          status_reason: "structured_generation_rejected",
        }),
      ],
      next_cursor: null,
    });
    renderStrip(5);
    await waitFor(() => {
      expect(screen.getByTestId("eval-failed")).toBeInTheDocument();
    });
    // Error-tone distinction: the banner MUST use the error surface so
    // a hard failure is visually distinct from dismissable info nudges.
    expect(screen.getByTestId("eval-failed")).toHaveAttribute("data-tone", "error");
    expect(screen.getByTestId("restart-prompt-only-btn")).toBeInTheDocument();
    // Recovery CTA stays secondary — the hero card's Save is the labeling
    // screen's sole green primary.
    expect(screen.getByTestId("restart-prompt-only-btn").className).not.toContain(
      "nvidia-green-button",
    );
    // Explainer text must accompany the banner so the SME can see
    // *why* the run failed — and what the recovery button actually
    // does — without opening Results.
    expect(screen.getByTestId("eval-failed")).toHaveTextContent(
      "The model rejected json_schema output for this run. Prompt-only asks for JSON in the prompt instead of enforcing a schema.",
    );
  });

  it("keeps the failure banner and surfaces the error when the prompt-only restart is rejected", async () => {
    // The restart button used to dismiss the failed banner BEFORE the
    // create call resolved and then swallow the rejection — a failed
    // restart left the SME with no banner, no error, and no new run.
    mockListRuns.mockResolvedValue({
      items: [
        makeRun({
          status: "failed",
          status_reason: "structured_generation_rejected",
        }),
      ],
      next_cursor: null,
    });
    mockCreateRun.mockRejectedValue(new Error("API 409: evaluation already in flight"));
    renderStrip(5);
    const user = userEvent.setup();

    await waitFor(() => {
      expect(screen.getByTestId("restart-prompt-only-btn")).toBeInTheDocument();
    });
    await user.click(screen.getByTestId("restart-prompt-only-btn"));

    const err = await screen.findByTestId("restart-prompt-only-error");
    expect(err).toHaveTextContent(/Could not restart the evaluation/);
    // The failed banner is only dismissed on a successful restart.
    expect(screen.getByTestId("eval-failed")).toBeInTheDocument();
  });

  it("re-shows the completion banner for a NEW run after dismissing an old one", async () => {
    // Banner dismissals are scoped per run: dismissing run-1's completion
    // banner must not suppress run-2's — otherwise one [X] click silences
    // every later evaluation's notification for the whole session.
    mockListRuns.mockResolvedValue({ items: [makeRun()], next_cursor: null });
    renderStrip(5);
    await waitFor(() => {
      expect(screen.getByTestId("eval-complete")).toBeInTheDocument();
    });
    await userEvent.click(screen.getByTestId("eval-complete-dismiss"));
    expect(screen.queryByTestId("eval-complete")).toBeNull();

    // A newer completed run arrives on the next poll.
    mockListRuns.mockResolvedValue({
      items: [makeRun({ run_id: "run-2" })],
      next_cursor: null,
    });
    await queryClient.refetchQueries();
    await waitFor(() => {
      expect(screen.getByTestId("eval-complete")).toBeInTheDocument();
    });
  });

  it("surfaces a dismissible error banner when the Evaluate request fails", async () => {
    // A rejected POST previously produced no feedback at all — the
    // Evaluate button just looked dead.
    mockCreateRun.mockRejectedValue(new Error("API 409: evaluation busy"));
    renderStrip(5);
    await waitFor(() => {
      expect(screen.getByTestId("evaluate-btn")).toBeInTheDocument();
    });
    await userEvent.click(screen.getByTestId("evaluate-btn"));
    await waitFor(() => {
      expect(screen.getByTestId("eval-start-error")).toBeInTheDocument();
    });
    expect(screen.getByTestId("eval-start-error")).toHaveTextContent(
      "Could not start evaluation",
    );
    await userEvent.click(screen.getByTestId("eval-start-error-dismiss"));
    expect(screen.queryByTestId("eval-start-error")).toBeNull();
  });

  it("does not crash when completed run has null metrics", async () => {
    mockListRuns.mockResolvedValue({
      items: [makeRun({ metrics: null })],
      next_cursor: null,
    });
    renderStrip(5);
    await waitFor(() => {
      expect(screen.getByTestId("eval-strip-bar")).toBeInTheDocument();
    });
    expect(screen.queryByTestId("eval-complete")).toBeNull();
  });

  it("does not crash when completed run has metrics without overall", async () => {
    mockListRuns.mockResolvedValue({
      items: [
        makeRun({
          metrics: {
            overall: undefined as unknown as never,
            returning: null,
            new: null,
          },
        }),
      ],
      next_cursor: null,
    });
    renderStrip(5);
    await waitFor(() => {
      expect(screen.getByTestId("eval-strip-bar")).toBeInTheDocument();
    });
    expect(screen.queryByTestId("eval-complete")).toBeNull();
  });
});
