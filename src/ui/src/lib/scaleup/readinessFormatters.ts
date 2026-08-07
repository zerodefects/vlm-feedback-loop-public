// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Formatters for Scale-Up Hub Readiness cards.
 *
 * Pure helpers that map ``GateCriterion`` objects into display-ready
 * ``{status, label, detail}`` tuples.  No React imports — 100% unit-testable.
 */

import { formatPct } from "@/lib/format-percent";
import type { GateCriterion } from "@/types/evaluation";

export type ReadinessStatus = "pass" | "fail" | "pending";

export interface ReadinessLine {
  status: ReadinessStatus;
  label: string;
  detail: string;
}

/**
 * Summarise a single criterion for display on a Readiness card.
 *
 * - ``pass`` when the criterion's ``passed`` flag is true.
 * - ``pending`` when the criterion depends on another (``details.blocked_by``)
 *   — the UI should render this as "waiting" rather than a hard fail.
 * - ``fail`` otherwise.
 */
export function summariseCriterion(c: GateCriterion, label: string): ReadinessLine {
  const blockedBy =
    c.details && typeof c.details === "object" && "blocked_by" in c.details
      ? String(c.details["blocked_by"])
      : null;

  if (c.passed) {
    return { status: "pass", label, detail: c.message };
  }
  if (blockedBy) {
    return { status: "pending", label, detail: c.message };
  }
  return { status: "fail", label, detail: c.message };
}

/** Locate a criterion by name, or return null when absent. */
export function findCriterion(
  criteria: GateCriterion[],
  name: string,
): GateCriterion | null {
  return criteria.find((c) => c.criterion_name === name) ?? null;
}

/**
 * Format the Test Pool size line for the Data-readiness card.
 *
 * Uses the ``min_test_pool_size`` criterion when present so the threshold
 * is the same the backend gate enforces.  Falls back to a neutral display
 * when the criterion is unavailable.
 */
export function formatPoolLine(
  criteria: GateCriterion[],
  verifiedCount: number,
): ReadinessLine {
  const c = findCriterion(criteria, "min_test_pool_size");
  if (!c) {
    return {
      status: "pending",
      label: "Test Pool",
      detail: `${verifiedCount} Verified labels so far`,
    };
  }
  return summariseCriterion(
    c,
    `Test Pool: ${formatValue(c.current_value)} / ${formatValue(c.threshold)} min`,
  );
}

/**
 * Format the Accept-rate line for the Teacher-readiness card.
 */
export function formatAcceptLine(criteria: GateCriterion[]): ReadinessLine {
  const c = findCriterion(criteria, "accept_rate");
  if (!c) {
    return {
      status: "pending",
      label: "Accept rate",
      detail: "Not yet measured",
    };
  }
  return summariseCriterion(
    c,
    `Accept rate: ${formatPct(c.current_value)} (need ${formatPct(c.threshold)})`,
  );
}

/**
 * Format the latest-evaluation line for the Teacher-readiness card.
 *
 * Renders "pending" status when the criterion is present but the
 * evaluation run doesn't exist yet.
 */
export function formatEvalLine(criteria: GateCriterion[]): ReadinessLine {
  const c = findCriterion(criteria, "overall_exact_match");
  if (!c) {
    return {
      status: "pending",
      label: "Evaluation",
      detail: "Not run",
    };
  }
  // The backend uses current_value=0 for BOTH "no eval yet" AND a genuine
  // 0% Exact-Match run (plausible under strict Core-field matching), so
  // current_value alone can't tell them apart — keying on it masked a real
  // failing evaluation as "not run". The gate authority marks the no-eval
  // state structurally via details.no_completed_run; the human-readable
  // message is display copy, not a wire contract.
  if (c.details?.no_completed_run === true) {
    return {
      status: "pending",
      label: "Evaluation",
      detail: "No evaluation run yet",
    };
  }
  // "Model accuracy" matches the backend gate message and verdict banner —
  // the UI never exposes the raw metric name.
  const base = summariseCriterion(
    c,
    `Model accuracy: ${formatPct(c.current_value)} (need ${formatPct(c.threshold)})`,
  );
  const evaluatedModel =
    typeof c.details?.evaluated_model_name === "string"
      ? c.details.evaluated_model_name
      : null;
  if (!evaluatedModel) return base;

  const score = `Model accuracy: ${formatPct(c.current_value)} (need ${formatPct(c.threshold)}).`;
  if (c.details?.current_configuration_differs === true) {
    return {
      status: "pending",
      label: `Evaluation: ${evaluatedModel}`,
      detail: `${score} Current settings differ; run a new evaluation to measure the selected setup.`,
    };
  }
  return {
    ...base,
    label: `Evaluation: ${evaluatedModel}`,
  };
}

function formatValue(v: number): string {
  return Number.isInteger(v) ? String(v) : v.toFixed(2);
}
