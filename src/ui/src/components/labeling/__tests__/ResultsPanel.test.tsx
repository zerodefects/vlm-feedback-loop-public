// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";

import { ResultsPanel } from "../ResultsPanel";
import type { EvaluationRunResponse } from "@/types/evaluation";

function makeRun(
  overrides: Partial<EvaluationRunResponse> = {},
): EvaluationRunResponse {
  return {
    run_id: "run-1",
    run_type: "evaluation_run",
    status: "completed",
    status_reason: null,
    pool_version_id: "pool-1",
    guidance_id: "guid-1",
    model_config_id: "mc-1",
    icl_mode: "enabled",
    evaluation_source: "nim",
    generation_preset_key: "precise",
    thinking_mode_effective: "on",
    visual_budget_preset_key: "balanced",
    structured_generation_mode_effective: "auto",
    inference_contract: null,
    icl_eligible_count_at_start: 5,
    icl_eligible_count_at_completion: 8,
    metrics: {
      overall: {
        exact_match_rate: 0.85,
        example_count: 10,
        per_field_match_rates: { severity: 0.9, damaged: 0.8 },
        per_value_metrics: {
          severity: {
            high: { precision: 0.9, recall: 0.85, f1: 0.87 },
            low: { precision: 0.7, recall: 0.65, f1: 0.67 },
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
    created_at: "2026-04-15T10:00:00Z",
    started_at: "2026-04-15T10:00:01Z",
    completed_at: "2026-04-15T10:01:00Z",
    ...overrides,
  };
}

function renderResultsPanel(
  run: EvaluationRunResponse = makeRun(),
  onClose = vi.fn(),
  minPerValueF1Threshold = 0.6,
) {
  return render(
    <ResultsPanel
      run={run}
      onClose={onClose}
      minPerValueF1Threshold={minPerValueF1Threshold}
    />,
  );
}

describe("ResultsPanel", () => {
  it("names the icon-only close action", () => {
    const onClose = vi.fn();
    renderResultsPanel(makeRun(), onClose);

    const close = screen.getByRole("button", { name: "Close evaluation results" });
    fireEvent.click(close);
    expect(onClose).toHaveBeenCalledOnce();
  });

  it("renders accuracy section", () => {
    renderResultsPanel();
    expect(screen.getByTestId("results-panel")).toBeInTheDocument();
    expect(screen.getByTestId("accuracy-section")).toBeInTheDocument();
    expect(screen.getByText(/85%/)).toBeInTheDocument();
  });

  it("preserves meaningful fractional precision in measured results", () => {
    const run = makeRun();
    if (run.metrics?.overall) run.metrics.overall.exact_match_rate = 0.908333;
    renderResultsPanel(run);
    expect(screen.getByText(/90\.8%/)).toBeInTheDocument();
  });

  it("shows per-field match rates", () => {
    renderResultsPanel();
    expect(screen.getByTestId("per-field-section")).toBeInTheDocument();
    expect(screen.getByText("severity")).toBeInTheDocument();
    expect(screen.getByText("damaged")).toBeInTheDocument();
  });

  it("expands per-value breakdown on click", async () => {
    renderResultsPanel();
    const toggle = screen.getByTestId("toggle-per-value");
    fireEvent.click(toggle);
    expect(screen.getByText("high")).toBeInTheDocument();
    expect(screen.getByText("low")).toBeInTheDocument();
  });

  it("uses the project's per-value F1 threshold", () => {
    renderResultsPanel(makeRun(), vi.fn(), 0.68);
    fireEvent.click(screen.getByTestId("toggle-per-value"));

    expect(screen.getAllByText("below 68%")).toHaveLength(1);
    expect(screen.queryByLabelText("below 68%")).not.toBeInTheDocument();
    expect(screen.queryByText("below 60%")).not.toBeInTheDocument();
  });

  it("shows incomplete warning", () => {
    renderResultsPanel(makeRun({ status: "incomplete" }));
    expect(screen.getByTestId("incomplete-warning")).toBeInTheDocument();
  });

  it("shows coverage gaps", () => {
    const run = makeRun({
      coverage_gaps: [
        { field_name: "severity", field_type: "enum", missing_values: ["medium"] },
      ],
    });
    renderResultsPanel(run);
    expect(screen.getByTestId("coverage-gaps")).toBeInTheDocument();
    expect(screen.getByText(/medium/)).toBeInTheDocument();
  });

  it("does not crash when metrics is null", () => {
    const run = makeRun({ metrics: null });
    renderResultsPanel(run);
    expect(screen.getByTestId("results-panel")).toBeInTheDocument();
  });

  it("does not crash when metrics.overall is missing", () => {
    const run = makeRun({
      metrics: { overall: undefined as unknown as never, returning: null, new: null },
    });
    renderResultsPanel(run);
    expect(screen.getByTestId("results-panel")).toBeInTheDocument();
  });

  it("does not crash when per_field_match_rates is missing", () => {
    const run = makeRun({
      metrics: {
        overall: {
          exact_match_rate: 0.5,
          example_count: 2,
          per_field_match_rates: undefined as unknown as Record<string, number>,
          per_value_metrics: undefined as unknown as Record<
            string,
            Record<string, never>
          >,
        },
        returning: null,
        new: null,
      },
    });
    renderResultsPanel(run);
    expect(screen.getByTestId("results-panel")).toBeInTheDocument();
    expect(screen.getByTestId("per-field-section")).toBeInTheDocument();
  });
});
