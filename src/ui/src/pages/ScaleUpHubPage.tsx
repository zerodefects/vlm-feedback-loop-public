// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Scale-Up Hub screen.
 *
 * Gate-first landing page for the optional Scale-Up path. Shows a
 * plain-language verdict with 1-2 next steps and two primary CTAs:
 * [Run Batch Labeling] (gated) and [Train a Student]. The latter always
 * opens the configuration screen; its readiness styling reflects the
 * authoritative preflight while Start Training owns final validation.
 */

import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { Button, Spinner, Text } from "@kui/react";
import {
  AlertTriangle,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  CircleX,
} from "lucide-react";

import { fetchScaleUpGate } from "@/api/evaluation";
import { listStudentBaseModelConfigs, runTrainingPreflight } from "@/api/training";
import { evaluationKeys, trainingKeys } from "@/api/query-keys";
import { PageContainer } from "@/components/common/PageContainer";
import { ActionRequestPanel } from "@/components/ActionRequestPanel";
import { SectionCard } from "@/components/common/SectionCard";
import { SectionHeading } from "@/components/common/SectionHeading";
import { StatBlock } from "@/components/common/StatBlock";
import {
  formatAcceptLine,
  formatEvalLine,
  formatPoolLine,
  type ReadinessLine,
  type ReadinessStatus,
} from "@/lib/scaleup/readinessFormatters";
import {
  isTrainingConfigurationCheck,
  isTrainingDataReadinessCheck,
} from "@/lib/training/preflight";
import { useSetupContext } from "@/pages/setup-context";
import { useTeacherAndGuidance } from "@/hooks/useTeacherAndGuidance";
import type { GateCriterion } from "@/types/evaluation";

// User-action priority for the gate verdict's "Next step(s):" line.
// Concrete labeling actions outrank "run an evaluation" which outranks
// guidance refinement ("Continue labeling —
// the test pool needs N more images" precedes "Run an evaluation to
// measure quality"). The Details disclosure below preserves backend
// criterion enumeration order so the full breakdown stays
// deterministic regardless of which subset surfaces as actionable.
const NEXT_STEP_PRIORITY: Record<string, number> = {
  min_test_pool_size: 0,
  overall_exact_match: 1,
  accept_rate: 2,
  per_field_match: 3,
  min_per_value_f1: 4,
};

export function ScaleUpHubPage() {
  const { projectId, project } = useSetupContext();
  const navigate = useNavigate();
  const [showDetails, setShowDetails] = useState(false);
  const [showTaoRequest, setShowTaoRequest] = useState(false);

  const { data: gate, isLoading } = useQuery({
    queryKey: evaluationKeys.gate(projectId),
    queryFn: () => fetchScaleUpGate(projectId),
  });

  // Readiness cards need the active Teacher + Guidance names; both are
  // lightweight lookups that share caches with other screens.
  const { teacherName, activeGuidance } = useTeacherAndGuidance(projectId, project);

  // Catalog availability for the Capability card. TAO/data validation runs
  // server-side only after the SME submits Student Training.
  const { data: studentBases } = useQuery({
    queryKey: trainingKeys.studentBases(projectId),
    queryFn: () => listStudentBaseModelConfigs(projectId),
  });
  const studentBaseIds = (studentBases?.items ?? []).map((m) => m.model_config_id);
  const {
    data: trainingPreflight,
    isFetching: trainingPreflightLoading,
    isError: trainingPreflightError,
    refetch: refetchTrainingPreflight,
  } = useQuery({
    queryKey: trainingKeys.preflight(projectId, studentBaseIds, true),
    queryFn: () => runTrainingPreflight(projectId, studentBaseIds, true),
    enabled: studentBaseIds.length > 0,
    retry: false,
  });

  if (isLoading) {
    return (
      <PageContainer data-testid="scaleup-hub-page">
        <div className="flex items-center justify-center py-20">
          <Spinner aria-label="Loading gate status" />
        </div>
      </PageContainer>
    );
  }

  const isReady = gate?.gate_status === "ready";
  const criteria = gate?.criteria ?? [];
  const passCount = criteria.filter((c) => c.passed).length;
  const failingCriteria = criteria.filter((c) => !c.passed);

  // Pick 1-2 most important next steps from failing criteria.  A criterion
  // with ``details.blocked_by`` is downstream of another failing criterion
  // (e.g., ``per_field_match`` + ``min_per_value_f1`` when no evaluation
  // has completed yet — both are blocked on ``overall_exact_match``).
  // Filtering them out keeps the actionable next-steps list focused while
  // the full Details expander still shows every criterion.
  // Sort by NEXT_STEP_PRIORITY (above) so concrete labeling actions surface
  // ahead of "run an evaluation".
  // Unknown criterion_names fall to the end (?? 99).
  const actionableNextSteps = failingCriteria
    .filter((c) => !(c.details && "blocked_by" in c.details))
    .slice() // shallow copy — sort mutates in place
    .sort(
      (a, b) =>
        (NEXT_STEP_PRIORITY[a.criterion_name] ?? 99) -
        (NEXT_STEP_PRIORITY[b.criterion_name] ?? 99),
    );
  const nextSteps = actionableNextSteps.slice(0, 2);

  // Within the Details disclosure, group failing rows above passing rows
  // for scannability when status is mixed.  Stable sort on `passed`
  // preserves backend enumeration order within each group.
  const detailsCriteria = [...criteria].sort(
    (a, b) => Number(a.passed) - Number(b.passed),
  );

  // ── Readiness-card inputs (pure derivations) ─────────────────────────
  const evalLine = formatEvalLine(criteria);
  const acceptLine = formatAcceptLine(criteria);
  const poolLine = formatPoolLine(criteria, project.counts.verified);
  const trainingDataFailures =
    trainingPreflight?.checks.filter(
      (check) => !check.passed && isTrainingDataReadinessCheck(check.check_name),
    ) ?? [];
  const taoSetupFailures =
    trainingPreflight?.checks.filter(
      (check) =>
        !check.passed &&
        !isTrainingDataReadinessCheck(check.check_name) &&
        !isTrainingConfigurationCheck(check.check_name),
    ) ?? [];
  const configurationFailures =
    trainingPreflight?.checks.filter(
      (check) => !check.passed && isTrainingConfigurationCheck(check.check_name),
    ) ?? [];
  const taoSetupRequired = taoSetupFailures.length > 0;
  let capabilityTaoLine: ReadinessLine;
  if (studentBases && studentBaseIds.length === 0) {
    capabilityTaoLine = {
      status: "fail",
      label: "Student Training: no Student bases",
      detail: "The model catalog has no student_base entries.",
    };
  } else if (!studentBases) {
    capabilityTaoLine = {
      status: "pending",
      label: "Student Training",
      detail: "Loading Student bases…",
    };
  } else if (trainingPreflightError) {
    capabilityTaoLine = {
      status: "fail",
      label: "Student Training: check unavailable",
      detail: "TAO readiness could not be checked. Retry before training.",
    };
  } else if (trainingPreflightLoading || !trainingPreflight) {
    capabilityTaoLine = {
      status: "pending",
      label: "Student Training: checking",
      detail: "Checking TAO, workspace, Student bases, and usable data…",
    };
  } else if (trainingDataFailures.length > 0) {
    capabilityTaoLine = {
      status: "fail",
      label: "Student Training: more Verified data needed",
      detail: trainingDataFailures[0]?.message ?? "Continue labeling.",
    };
  } else if (taoSetupRequired) {
    capabilityTaoLine = {
      status: "fail",
      label: "Student Training: infrastructure setup required",
      detail: taoSetupFailures[0]?.message ?? "TAO setup is incomplete.",
    };
  } else if (configurationFailures.length > 0) {
    capabilityTaoLine = {
      status: "pass",
      label: "Student Training: ready to configure",
      detail:
        "TAO and data checks passed. Select a compatible model and training method on Student Training.",
    };
  } else {
    capabilityTaoLine = {
      status: "pass",
      label: "Student Training: ready",
      detail: `${trainingPreflight.data_summary.usable_training_count} usable training example${
        trainingPreflight.data_summary.usable_training_count === 1 ? "" : "s"
      }; ${trainingPreflight.data_summary.test_pool_count} held out for evaluation; TAO preflight passed.`,
    };
  }
  const studentTrainingReady = capabilityTaoLine.status === "pass";

  return (
    <PageContainer data-testid="scaleup-hub-page">
      <Text kind="title/md">Scale Up</Text>

      {/* ── Gate Verdict ──────────────────────────────────────────── */}
      <div className="glass-card p-6 space-y-4">
        {isReady ? (
          // "Ready for Batch Labeling." and
          // [Details] share the same row, with [Details] right-aligned.
          // justify-between spreads the two halves to opposite edges of
          // the gate-card content area.
          <div
            className="flex items-center justify-between gap-3"
            data-testid="ready-verdict-row"
          >
            <div className="flex items-center gap-3">
              <CheckCircle2 size={22} style={{ color: "var(--accent-green)" }} />
              <Text kind="title/sm" style={{ color: "var(--accent-green)" }}>
                Ready for Batch Labeling.
              </Text>
            </div>
            <DetailsToggle
              open={showDetails}
              passCount={passCount}
              total={criteria.length}
              onToggle={() => setShowDetails((v) => !v)}
            />
          </div>
        ) : (
          <>
            <div className="flex items-center gap-3">
              <AlertTriangle size={22} style={{ color: "var(--warning-amber)" }} />
              <Text kind="title/sm" style={{ color: "var(--warning-amber)" }}>
                Not ready for Batch Labeling.
              </Text>
            </div>

            {nextSteps.length > 0 && (
              <div className="space-y-2 pl-9">
                <Text kind="label/bold/sm" style={{ color: "var(--text-muted)" }}>
                  {nextSteps.length === 1 ? "Next step:" : "Next steps:"}
                </Text>
                {/* Each next step in its own block so consecutive messages
                    never run together as a single sentence. */}
                {nextSteps.map((c) => (
                  <div key={c.criterion_name}>
                    <Text
                      kind="body/regular/sm"
                      style={{ color: "var(--text-secondary)" }}
                    >
                      {c.message}
                    </Text>
                  </div>
                ))}
              </div>
            )}

            {/* [Go to Labeling] and [Details] are peer
                controls on the same row. Details is a disclosure chip
                rather than a KUI button so it reads as the subordinate
                of the two. */}
            <div className="flex items-center gap-3 pl-9">
              <Button
                kind="secondary"
                onClick={() => navigate("../labeling")}
                data-testid="go-to-labeling"
              >
                Go to Labeling
              </Button>
              <DetailsToggle
                open={showDetails}
                passCount={passCount}
                total={criteria.length}
                onToggle={() => setShowDetails((v) => !v)}
              />
            </div>
          </>
        )}

        {/* Ready state: DetailsToggle is inlined on the verdict row
            above, so no separate row here. */}

        {showDetails && (
          <div className="space-y-2 pl-9" data-testid="criteria-list">
            {detailsCriteria.map((c: GateCriterion) => (
              <div key={c.criterion_name} className="flex items-start gap-2">
                {c.passed ? (
                  <CheckCircle2
                    size={14}
                    className="mt-0.5 flex-shrink-0"
                    style={{ color: "var(--accent-green)" }}
                  />
                ) : (
                  <CircleX size={14} className="mt-0.5 flex-shrink-0 text-error" />
                )}
                <Text kind="body/regular/sm">{c.message}</Text>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* ── Readiness cards ─────────────────────────────────────────
          Three glass-card panels providing informational context
          alongside the gate verdict: Teacher, Data, Capability. All
          source data is already queried above — no new backend calls. */}
      <div
        className="grid grid-cols-1 gap-4 md:grid-cols-3"
        data-testid="readiness-cards"
      >
        {/* Teacher readiness */}
        <SectionCard data-testid="readiness-teacher">
          <SectionHeading>Teacher readiness</SectionHeading>
          <div className="space-y-1">
            <Text className="block" kind="body/regular/sm">
              Teacher:{" "}
              <Text
                kind="body/semibold/sm"
                className="break-words"
                style={{ color: "var(--text-primary)" }}
              >
                {teacherName ?? "—"}
              </Text>
            </Text>
            <Text className="block" kind="body/regular/sm">
              Guidance:{" "}
              <Text kind="body/semibold/sm" style={{ color: "var(--text-primary)" }}>
                {activeGuidance ? `v${activeGuidance.version_number}` : "—"}
              </Text>
            </Text>
          </div>
          <ReadinessRow line={acceptLine} />
          <ReadinessRow line={evalLine} />
        </SectionCard>

        {/* Data readiness */}
        <SectionCard data-testid="readiness-data">
          <SectionHeading>Data readiness</SectionHeading>
          <div className="flex flex-wrap gap-6">
            <StatBlock
              label="Verified"
              value={project.counts.verified}
              tone="green"
              data-testid="data-readiness-verified"
            />
            <StatBlock
              label="Unlabeled"
              value={project.counts.unlabeled}
              data-testid="data-readiness-unlabeled"
            />
            <StatBlock
              label="Auto-Labeled"
              value={project.counts.auto_labeled}
              data-testid="data-readiness-auto-labeled"
            />
            {project.counts.omitted > 0 && (
              <StatBlock
                label="Omitted"
                value={project.counts.omitted}
                data-testid="data-readiness-omitted"
              />
            )}
          </div>
          <ReadinessRow line={poolLine} />
        </SectionCard>

        {/* Capability readiness */}
        <SectionCard data-testid="readiness-capability">
          <SectionHeading>Capability readiness</SectionHeading>
          <ReadinessRow
            line={{
              status: isReady ? "pass" : "fail",
              label: `Batch Labeling: ${isReady ? "ready" : "not ready"}`,
              detail: isReady
                ? "All quality gates passed."
                : "Gate criteria above must pass.",
            }}
          />
          <ReadinessRow line={capabilityTaoLine} />
          {(trainingPreflightError || taoSetupRequired) && (
            <div className="flex flex-wrap gap-2 pt-2">
              <Button
                kind="secondary"
                onClick={() => void refetchTrainingPreflight()}
                disabled={trainingPreflightLoading}
                data-testid="retry-training-preflight"
              >
                Retry TAO check
              </Button>
              {taoSetupRequired && (
                <Button
                  kind="tertiary"
                  onClick={() => setShowTaoRequest((value) => !value)}
                  data-testid="request-tao-setup"
                >
                  {showTaoRequest ? "Hide setup request" : "Request TAO setup"}
                </Button>
              )}
            </div>
          )}
        </SectionCard>
      </div>

      {showTaoRequest && taoSetupRequired && (
        <div className="glass-card p-6" data-testid="tao-setup-action-request">
          <ActionRequestPanel
            projectId={projectId}
            requestType="tao_setup"
            onClose={() => setShowTaoRequest(false)}
          />
        </div>
      )}

      {/* ── Primary CTAs (side-by-side) ─────────────────────────────
          The two CTAs sit at opposite edges of
          the row ("[Run Batch Labeling] ... [Train a Student]") so the
          page bottom anchors visually. md:justify-between spreads them
          across the container.

          No nested glass-card wrapper: the CTAs sit at the page level
          like other summary screens' action rows, so the Scale-Up
          Hub doesn't stack two glass cards where sibling screens use
          one. */}
      <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
        {/* Run Batch Labeling */}
        <div className="space-y-2">
          <Button
            kind="primary"
            className="nvidia-green-button"
            disabled={!isReady}
            onClick={() => navigate("../batch-prerun")}
            data-testid="run-batch-labeling"
          >
            Run Batch Labeling
          </Button>
          {!isReady && (
            <Text
              className="block"
              kind="body/regular/sm"
              style={{ color: "var(--text-muted)" }}
            >
              Gate not ready.
            </Text>
          )}
        </div>

        {/* Train a Student */}
        <div className="space-y-2 md:flex md:flex-col md:items-end">
          <Button
            kind={studentTrainingReady ? "primary" : "secondary"}
            className={studentTrainingReady ? "nvidia-green-button" : undefined}
            onClick={() => navigate("../training")}
            data-testid="train-a-student"
          >
            Train a Student
          </Button>
        </div>
      </div>
    </PageContainer>
  );
}

// ── Readiness row (small helper) ──────────────────────────────────────

const STATUS_COLORS: Record<ReadinessStatus, string> = {
  pass: "var(--accent-green)",
  fail: "var(--warning-amber)",
  pending: "var(--text-muted)",
};

// ── Details disclosure (gate criterion breakdown) ─────────────────────

function DetailsToggle({
  open,
  passCount,
  total,
  onToggle,
}: {
  open: boolean;
  passCount: number;
  total: number;
  onToggle: () => void;
}) {
  return (
    <button
      className="flex items-center gap-1 text-sm"
      style={{ color: "var(--text-muted)" }}
      onClick={onToggle}
      data-testid="details-toggle"
    >
      {open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
      Details ({passCount}/{total} criteria passed)
    </button>
  );
}

function ReadinessRow({ line }: { line: ReadinessLine }) {
  const color = STATUS_COLORS[line.status];
  const Icon =
    line.status === "pass"
      ? CheckCircle2
      : line.status === "fail"
        ? CircleX
        : ChevronRight;
  return (
    <div className="flex items-start gap-2">
      <Icon size={14} className="mt-0.5 flex-shrink-0" style={{ color }} />
      <div>
        <Text
          className="block"
          kind="body/regular/sm"
          style={{ color: "var(--text-primary)" }}
        >
          {line.label}
        </Text>
        <Text
          className="block"
          kind="body/regular/sm"
          style={{ color: "var(--text-muted)" }}
        >
          {line.detail}
        </Text>
      </div>
    </div>
  );
}
